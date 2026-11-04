// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#include <folly/ScopeGuard.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "cachinglayer/lrucache/DList.h"
#include "cachinglayer/lrucache/LRUCacheCore.h"
#include "common/TrackingStdAllocator.h"
#include "syncpoint/sync_point.h"

namespace milvus::cachinglayer::internal {
namespace {
using StoredKey = std::variant<uint64_t, std::string>;

struct KeyHash {
    using is_transparent = void;

    template <typename Key>
    size_t
    operator()(const Key& key) const noexcept {
        return std::visit(
            [](const auto& value) {
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, uint64_t>) {
                    return std::hash<uint64_t>{}(value);
                } else {
                    return std::hash<std::string_view>{}(value);
                }
            },
            key);
    }
};

struct KeyEqual {
    using is_transparent = void;

    template <typename Left, typename Right>
    bool
    operator()(const Left& left, const Right& right) const noexcept {
        if (left.index() != right.index()) {
            return false;
        }
        if (left.index() == 0) {
            return std::get<0>(left) == std::get<0>(right);
        }
        return std::string_view(std::get<1>(left)) == std::string_view(std::get<1>(right));
    }
};
}  // namespace

struct LRUCacheCore::Entry final : ListNode {
    explicit Entry(CacheState* owner);

    void
    unload() override;
    std::unique_ptr<ListNode>
    detach_evicted() override;
    std::string
    key() const override;
    ResourceUsage
    select_for_eviction() noexcept override;
    void
    clear_eviction_selection() noexcept override;

    CacheState* const owner;
    const StoredKey* index_key{nullptr};
    ValuePtr value{nullptr, nullptr};
    ResourceUsage value_bytes;
    int64_t metadata_bytes{0};
    bool replacing{false};
};

struct LRUCacheCore::CacheState {
    using MapValue = std::pair<const StoredKey, std::unique_ptr<Entry>>;
    using MapAllocator = TrackingStdAllocator<MapValue>;
    using Map = std::unordered_map<StoredKey, std::unique_ptr<Entry>, KeyHash, KeyEqual, MapAllocator>;
    explicit CacheState(std::shared_ptr<DList> domain) : dlist(std::move(domain)) {
        if (!dlist || !dlist->event_base_thread_) {
            ThrowInfo(ErrorCode::InvalidParameter, "Keyed cache requires an initialized eviction-enabled DList");
        }
    }

    // All entry/index operations use the domain mutex. Get never exports an
    // unpinned node, so eviction can reclaim metadata before admitting work.
    void
    PinLocked(Entry& entry) {
        if (entry.pin_count_.fetch_add(1) == 0) {
            dlist->freezeItem(&entry, entry.loaded_size_);
        }
    }

    void
    TouchLocked(Entry& entry) {
        dlist->evictable_size_ += entry.loaded_size_;
        const auto now = std::chrono::steady_clock::now();
        if (entry.state_ != ListNode::State::CACHED ||
            now - entry.last_touch_ > dlist->eviction_config_.cache_touch_window) {
            dlist->popItem(&entry);
            dlist->pushHead(&entry);
            entry.last_touch_ = now;
        }
        entry.state_ = ListNode::State::CACHED;
    }

    void
    EraseLocked(Entry& entry, const bool was_evictable = true) {
        const auto bytes = entry.loaded_size_;
        if (dlist->popItem(&entry) && was_evictable) {
            dlist->evictable_size_ -= bytes;
        }
        entry.unload();
        entries.erase(*entry.index_key);
        dlist->total_loaded_size_ -= bytes;
    }

    void
    DropEmptyBuckets() {
        if (!entries.empty() || put_active || bucket_bytes == 0) {
            return;
        }
        {
            Map empty(0, KeyHash{}, KeyEqual{}, entries.get_allocator());
            entries.swap(empty);
        }
        dlist->total_loaded_size_ -= ResourceUsage{bucket_bytes, 0};
        dlist->index_overhead_ -= ResourceUsage{bucket_bytes, 0};
        monitor::cache_kv_metadata_bytes.Decrement(bucket_bytes);
        bucket_bytes = 0;
    }

    void
    UnpinLocked(Entry& entry) {
        if (entry.pin_count_.fetch_sub(1) != 1) {
            return;
        }
        if (closed) {
            // This node was frozen; it has not reentered evictable_size_.
            EraseLocked(entry, false);
        } else {
            TouchLocked(entry);
        }
    }

    void
    Release(Entry& entry) {
        {
            std::lock_guard lock(dlist->list_mtx_);
            UnpinLocked(entry);
            DropEmptyBuckets();
        }
        dlist->notifyWaitingRequests();
    }

    void
    Close() {
        {
            std::lock_guard lock(dlist->list_mtx_);
            closed = true;
            for (auto it = entries.begin(); it != entries.end();) {
                auto& entry = *it++->second;
                if (entry.pin_count_ == 0) {
                    EraseLocked(entry);
                }
            }
            DropEmptyBuckets();
        }
        dlist->notifyWaitingRequests();
    }

    const std::shared_ptr<DList> dlist;
    // Never acquired by Get, unpin, Close or DList eviction. No Domain -> Put lock edge.
    std::mutex put_mutex;
    Map entries;
    int64_t bucket_bytes{0};
    size_t selected_entries{0};
    bool put_active{false};
    bool closed{false};
};

LRUCacheCore::Entry::Entry(CacheState* owner) : ListNode(owner->dlist.get(), true), owner(owner) {
}

void
LRUCacheCore::Entry::unload() {
    value.reset();
    monitor::cache_kv_entry_count.Decrement();
    monitor::cache_kv_metadata_bytes.Decrement(metadata_bytes);
    monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Decrement(value_bytes.memory_bytes);
    monitor::cache_kv_loaded_bytes(StorageType::DISK).Decrement(value_bytes.file_bytes);
    loaded_size_ = {};
    ListNode::unload();
}

std::unique_ptr<ListNode>
LRUCacheCore::Entry::detach_evicted() {
    auto node = owner->entries.extract(*index_key);
    auto result = std::move(node.mapped());
    result->index_key = nullptr;
    node = {};  // Free the map node and key before refunding their charge.
    owner->DropEmptyBuckets();
    return result;
}

std::string
LRUCacheCore::Entry::key() const {
    return fmt::format("keyed-entry:{}", fmt::ptr(this));
}

ResourceUsage
LRUCacheCore::Entry::select_for_eviction() noexcept {
    ++owner->selected_entries;
    return !owner->put_active && owner->selected_entries == owner->entries.size()
               ? ResourceUsage{owner->bucket_bytes, 0}
               : ResourceUsage{};
}

void
LRUCacheCore::Entry::clear_eviction_selection() noexcept {
    --owner->selected_entries;
}

LRUCacheCore::PinState::PinState() = default;

LRUCacheCore::PinState::PinState(std::shared_ptr<CacheState> state, Entry* entry)
    : state_(std::move(state)), entry_(entry) {
}

LRUCacheCore::PinState::PinState(PinState&& other) noexcept
    : state_(std::move(other.state_)), entry_(std::exchange(other.entry_, nullptr)) {
}

LRUCacheCore::PinState&
LRUCacheCore::PinState::operator=(PinState&& other) noexcept {
    if (this != &other) {
        if (entry_) {
            state_->Release(*entry_);
        }
        state_ = std::move(other.state_);
        entry_ = std::exchange(other.entry_, nullptr);
    }
    return *this;
}

LRUCacheCore::PinState::~PinState() {
    if (entry_) {
        state_->Release(*entry_);
    }
}

LRUCacheCore::PinState::operator bool() const noexcept {
    return entry_ != nullptr;
}

const void*
LRUCacheCore::PinState::get() const {
    return entry_->value.get();
}

LRUCacheCore::LRUCacheCore(std::shared_ptr<DList> dlist) : state_(std::make_shared<CacheState>(std::move(dlist))) {
}

LRUCacheCore::~LRUCacheCore() {
    state_->Close();
}

LRUCacheCore::PinState
LRUCacheCore::Get(const KeyView& key) const {
    auto state = state_;
    std::lock_guard lock(state->dlist->list_mtx_);
    const auto it = state->entries.find(key);
    if (state->closed || it == state->entries.end()) {
        return {};
    }
    auto* entry = it->second.get();
    state->PinLocked(*entry);
    return PinState(std::move(state), entry);
}

LRUCacheCore::Result
LRUCacheCore::Put(const KeyView& key, ValuePtr& value, const ResourceUsage& bytes) {
    if (!value || !bytes.AllGEZero()) {
        return folly::makeUnexpected(LRUCacheError::InvalidArgument);
    }
    auto state = state_;
    const auto& dlist = state->dlist;
    std::unique_lock writer(state->put_mutex);
    Entry* old = nullptr;
    std::optional<CacheState::Map> pending;
    ResourceUsage reservation;
    bool reserved = false;
    {
        std::lock_guard lock(dlist->list_mtx_);
        if (state->closed) {
            return folly::makeUnexpected(LRUCacheError::Closed);
        }
        const auto it = state->entries.find(key);
        if (it != state->entries.end()) {
            old = it->second.get();
            if (old->pin_count_ != 0) {
                return folly::makeUnexpected(LRUCacheError::Pinned);
            }
            state->PinLocked(*old);
            old->replacing = true;
        }
        state->put_active = true;
    }
    auto finish = folly::makeGuard([&] {
        pending.reset();
        {
            std::lock_guard lock(dlist->list_mtx_);
            if (old) {
                old->replacing = false;
                state->UnpinLocked(*old);
            }
            state->put_active = false;
            state->DropEmptyBuckets();
        }
        // Completion callbacks may reenter Put; release every cache lock first.
        writer.unlock();
        if (reserved) {
            dlist->ReleaseLoadingResource(reservation);
        } else {
            dlist->notifyWaitingRequests();
        }
    });

    bool grow = false;
    int64_t bucket_bytes = 0;
    int64_t metadata_bytes = 0;
    Entry* entry = nullptr;
    {
        std::lock_guard lock(dlist->list_mtx_);
        const auto allocator = state->entries.get_allocator();
        pending.emplace(0, KeyHash{}, KeyEqual{}, allocator);
        const auto count = state->entries.size() + (old ? 0 : 1);
        // An empty libstdc++ map may report one bucket without allocating it.
        grow = state->bucket_bytes == 0 || count > state->entries.bucket_count();
        auto before = allocator.total_allocated();
        pending->reserve(grow ? std::max<size_t>(16, count * 2) : 1);
        bucket_bytes = static_cast<int64_t>(allocator.total_allocated() - before);
        before = allocator.total_allocated();
        auto stored_key = std::visit(
            [](const auto& value) -> StoredKey {
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, uint64_t>) {
                    return value;
                } else {
                    return std::string(value);
                }
            },
            key);
        auto it = pending->emplace(std::move(stored_key), std::make_unique<Entry>(state.get())).first;
        entry = it->second.get();
        entry->index_key = &it->first;
        metadata_bytes = static_cast<int64_t>(sizeof(Entry) + allocator.total_allocated() - before);
        if (const auto* text = std::get_if<std::string>(&it->first)) {
            // Conservative for SSO: no allocator-specific string layout assumptions.
            metadata_bytes += static_cast<int64_t>(text->capacity() + 1);
        }
    }
    if (bytes.memory_bytes > std::numeric_limits<int64_t>::max() - metadata_bytes - bucket_bytes) {
        return folly::makeUnexpected(LRUCacheError::InvalidArgument);
    }
    reservation = bytes + ResourceUsage{metadata_bytes + bucket_bytes, 0};
    TEST_SYNC_POINT("LRUCache::Put:BeforeReserve");
    reserved = SemiInlineGet(dlist->ReserveLoadingResourceWithTimeout(reservation, std::chrono::milliseconds(0)));
    if (!reserved) {
        return folly::makeUnexpected(LRUCacheError::InsufficientResource);
    }
    TEST_SYNC_POINT("LRUCache::Put:BeforePublish");

    {
        std::lock_guard lock(dlist->list_mtx_);
        if (state->closed) {
            return folly::makeUnexpected(LRUCacheError::Closed);
        }
        if (old && old->pin_count_ != 1) {
            return folly::makeUnexpected(LRUCacheError::Pinned);
        }
        if (old) {
            state->EraseLocked(*old, false);  // The internal pin kept it out of eviction.
            old = nullptr;
        }
        if (grow) {
            // Integral/string hash and equality do not throw. Capacity and all
            // nodes were allocated before admission; merge transfers ownership.
            pending->merge(state->entries);
            state->entries.swap(*pending);
            pending.reset();
            dlist->total_loaded_size_ -= ResourceUsage{state->bucket_bytes, 0};
            dlist->index_overhead_ += ResourceUsage{bucket_bytes - state->bucket_bytes, 0};
            monitor::cache_kv_metadata_bytes.Decrement(state->bucket_bytes);
            state->bucket_bytes = bucket_bytes;
            dlist->ChargeLoadedResource({bucket_bytes, 0});
            monitor::cache_kv_metadata_bytes.Increment(bucket_bytes);
        } else {
            state->entries.insert(pending->extract(pending->begin()));
            pending.reset();
        }
        entry->value = std::move(value);
        entry->value_bytes = bytes;
        entry->metadata_bytes = metadata_bytes;
        entry->loaded_size_ = bytes + ResourceUsage{metadata_bytes, 0};
        entry->state_ = ListNode::State::LOADED;
        dlist->ChargeLoadedResource(entry->loaded_size_);
        state->TouchLocked(*entry);
        monitor::cache_kv_entry_count.Increment();
        monitor::cache_kv_metadata_bytes.Increment(metadata_bytes);
        monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Increment(bytes.memory_bytes);
        monitor::cache_kv_loaded_bytes(StorageType::DISK).Increment(bytes.file_bytes);
    }
    return folly::unit;
}

LRUCacheCore::Result
LRUCacheCore::Erase(const KeyView& key) {
    auto state = state_;
    {
        std::lock_guard lock(state->dlist->list_mtx_);
        const auto it = state->entries.find(key);
        if (it == state->entries.end()) {
            return folly::makeUnexpected(LRUCacheError::NotFound);
        }
        auto& entry = *it->second;
        if (entry.pin_count_ > (entry.replacing ? 1 : 0)) {
            return folly::makeUnexpected(LRUCacheError::Pinned);
        }
        if (entry.replacing) {
            return folly::makeUnexpected(LRUCacheError::Conflict);
        }
        state->EraseLocked(entry);
        state->DropEmptyBuckets();
    }
    state->dlist->notifyWaitingRequests();
    return folly::unit;
}

size_t
LRUCacheCore::Size() const {
    std::lock_guard lock(state_->dlist->list_mtx_);
    return state_->entries.size();
}

}  // namespace milvus::cachinglayer::internal
