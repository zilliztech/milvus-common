// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <folly/Expected.h>
#include <folly/ScopeGuard.h>
#include <folly/Unit.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "cachinglayer/lrucache/ChainedHashTable.h"
#include "cachinglayer/lrucache/DList.h"
#include "syncpoint/sync_point.h"

namespace milvus::cachinglayer {

enum class CacheMapError { Pinned, NotFound, InsufficientResource, InvalidArgument, Conflict, Closed };

// Key operations must not reenter the cache. Key/Value destructors must not
// throw or reenter a cache. Get never loads; recency follows the shared
// domain's unpin/touch-window policy.
template <CacheMapKey Key, typename Value>
class CacheMap {
    struct CacheState;
    struct Entry;

 public:
    // A move-only pin bound to this cache's Key and Value types. The typed
    // state and Entry references keep the Value alive after cache destruction.
    class Pin {
     public:
        Pin() = default;
        Pin(const Pin&) = delete;
        Pin&
        operator=(const Pin&) = delete;
        Pin(Pin&& other) noexcept : state_(std::move(other.state_)), entry_(std::move(other.entry_)) {
        }
        Pin&
        operator=(Pin&& other) noexcept {
            if (this != &other) {
                if (entry_) {
                    state_->Release(*entry_);
                }
                entry_.reset();
                state_ = std::move(other.state_);
                entry_ = std::move(other.entry_);
            }
            return *this;
        }
        ~Pin() {
            if (entry_) {
                state_->Release(*entry_);
            }
        }
        explicit
        operator bool() const noexcept {
            return entry_ != nullptr;
        }
        const Value&
        operator*() const {
            return *entry_->value;
        }
        const Value*
        operator->() const {
            return entry_->value.get();
        }

     private:
        friend class CacheMap;
        Pin(std::shared_ptr<CacheState> state, std::shared_ptr<Entry> entry)
            : state_(std::move(state)), entry_(std::move(entry)) {
        }
        std::shared_ptr<CacheState> state_;
        std::shared_ptr<Entry> entry_;
    };

    using Result = folly::Expected<folly::Unit, CacheMapError>;

    using IndexConfig = internal::ChainedHashTable::Config;

    explicit CacheMap(std::shared_ptr<internal::DList> dlist, const IndexConfig index_config = {})
        : state_(std::make_shared<CacheState>(std::move(dlist), index_config)) {
    }
    CacheMap(const CacheMap&) = delete;
    CacheMap&
    operator=(const CacheMap&) = delete;
    ~CacheMap() {
        state_->Close();
    }

    [[nodiscard]] Pin
    Get(const Key& key) const {
        auto state = state_;
        auto entry = state->entries.find(key);
        if (!entry) {
            return {};
        }
        // The index lock is already released. This reference keeps the wrapper
        // alive even if eviction detaches it before we acquire the node lock.
        TEST_SYNC_POINT("CacheMap::Get:AfterLookup");
        std::shared_lock lock(entry->mtx_);
        if (state->closed || entry->state_ == internal::ListNode::State::NOT_LOADED) {
            return {};
        }
        // Do not use ListNode::pin(): a detached KV entry must never start a load.
        state->PinLocked(*entry);
        return Pin(std::move(state), std::move(entry));
    }

    // Ownership transfers only on success, including when key operations throw.
    // Success does not pin the Value; eviction may remove it before Put returns.
    // Only the caller-supplied Value payload usage participates in admission.
    // Keys, index storage and Entry objects are not charged to the shared domain.
    // Zero usage is allowed; the byte budget does not impose an entry-count limit.
    // Copy usage: the caller may pass a member of a Value evicted after publication.
    [[nodiscard]] Result
    Put(const Key& key, std::unique_ptr<Value>& value, const ResourceUsage bytes) {
        if (!value || !bytes.AllGEZero()) {
            return folly::makeUnexpected(CacheMapError::InvalidArgument);
        }
        auto state = state_;
        const auto& dlist = state->dlist;
        std::unique_lock writer(state->put_mutex);
        if (state->closed) {
            return folly::makeUnexpected(CacheMapError::Closed);
        }
        auto old = state->entries.find(key);
        bool reserved = false;
        if (old) {
            std::unique_lock lock(old->mtx_);
            if (old->state_ == internal::ListNode::State::NOT_LOADED) {
                lock.unlock();
                old.reset();
            } else {
                if (old->pin_count_ != 0) {
                    return folly::makeUnexpected(CacheMapError::Pinned);
                }
                state->PinLocked(*old);
                old->replacing = true;
            }
        }
        auto finish = folly::makeGuard([&] {
            if (old) {
                std::unique_lock lock(old->mtx_);
                old->replacing = false;
                state->UnpinLocked(*old);
            }
            writer.unlock();
            if (reserved) {
                dlist->ReleaseLoadingResource(bytes);
            } else {
                dlist->notifyWaitingRequests();
            }
        });

        TEST_SYNC_POINT("CacheMap::Put:BeforeReserve");
        reserved = SemiInlineGet(dlist->ReserveLoadingResourceWithTimeout(bytes, std::chrono::milliseconds(0)));
        if (!reserved) {
            return folly::makeUnexpected(CacheMapError::InsufficientResource);
        }
        TEST_SYNC_POINT("CacheMap::Put:BeforePublish");

        {
            auto entry = old ? old : std::make_shared<Entry>(state.get(), key);
            std::unique_lock node_lock(entry->mtx_);
            std::lock_guard lock(dlist->list_mtx_);
            if (state->closed) {
                return folly::makeUnexpected(CacheMapError::Closed);
            }
            if (old && old->pin_count_ != 1) {
                return folly::makeUnexpected(CacheMapError::Pinned);
            }
            if (old) {
                // Keep the existing Key/index node. No key operation can throw
                // after the old Value is released.
                const auto previous = old->loaded_size_;
                old->unload();
                dlist->RefundLoadedResourceLocked(previous);
            } else {
                // put_mutex has excluded other writers since find(); concurrent
                // reads and removals cannot introduce a duplicate key.
                state->entries.prepend(entry);
            }
            entry->value = std::move(value);
            entry->loaded_size_ = bytes;
            entry->state_ = internal::ListNode::State::LOADED;
            dlist->ChargeLoadedResource(entry->loaded_size_);
            if (!old) {
                state->TouchLocked(*entry);
            }
            monitor::cache_kv_entry_count.Increment();
            monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Increment(bytes.memory_bytes);
            monitor::cache_kv_loaded_bytes(StorageType::DISK).Increment(bytes.file_bytes);
        }
        TEST_SYNC_POINT("CacheMap::Put:AfterPublish");
        return folly::unit;
    }

    [[nodiscard]] Result
    Erase(const Key& key) {
        auto state = state_;
        auto entry = state->entries.find(key);
        if (!entry) {
            return folly::makeUnexpected(CacheMapError::NotFound);
        }
        {
            std::unique_lock node_lock(entry->mtx_);
            if (entry->state_ == internal::ListNode::State::NOT_LOADED) {
                return folly::makeUnexpected(CacheMapError::NotFound);
            }
            if (entry->pin_count_ > (entry->replacing ? 1 : 0)) {
                return folly::makeUnexpected(CacheMapError::Pinned);
            }
            if (entry->replacing) {
                return folly::makeUnexpected(CacheMapError::Conflict);
            }
            std::lock_guard lock(state->dlist->list_mtx_);
            state->EraseLocked(*entry);
        }
        state->dlist->notifyWaitingRequests();
        return folly::unit;
    }

    [[nodiscard]] size_t
    Size() const {
        return state_->entries.size();
    }

 private:
    struct Entry final : internal::ListNode {
        Entry(CacheState* owner, const Key& key)
            : internal::ListNode(owner->dlist.get(), true),
              owner(owner),
              cache_key(key),
              hash_hook{nullptr, std::hash<Key>{}(key)} {
        }

        const Key&
        getKey() const {
            return cache_key;
        }

        void
        unload() override {
            value.reset();
            monitor::cache_kv_entry_count.Decrement();
            monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Decrement(loaded_size_.memory_bytes);
            monitor::cache_kv_loaded_bytes(StorageType::DISK).Decrement(loaded_size_.file_bytes);
            loaded_size_ = {};
            internal::ListNode::unload();
        }

        std::shared_ptr<internal::ListNode>
        detach_evicted() override {
            return owner->entries.remove(*this);
        }

        std::string
        key() const override {
            return fmt::format("keyed-entry:{}", fmt::ptr(this));
        }

        CacheState* const owner;
        const Key cache_key;
        internal::ChainedHashTable::Hook<Entry> hash_hook;
        std::unique_ptr<Value> value;
        bool replacing{false};
    };

    const std::shared_ptr<CacheState> state_;
};

template <CacheMapKey Key, typename Value>
struct CacheMap<Key, Value>::CacheState {
    using Map = internal::ChainedHashTable::Container<Key, Entry, &Entry::hash_hook>;

    explicit CacheState(std::shared_ptr<internal::DList> domain, const IndexConfig index_config)
        : dlist(std::move(domain)), entries(index_config) {
        if (!dlist || !dlist->event_base_thread_) {
            ThrowInfo(ErrorCode::InvalidParameter, "Keyed cache requires an initialized eviction-enabled DList");
        }
    }

    // Locked helpers require entry.mtx_. TouchLocked/EraseLocked additionally
    // require list_mtx_. Index methods never wait on either of these locks.
    void
    PinLocked(Entry& entry) {
        if (entry.pin_count_.fetch_add(1) == 0) {
            dlist->freezeItem(&entry, entry.loaded_size_);
        }
    }

    void
    TouchLocked(Entry& entry) {
        dlist->touchItemLocked(&entry, entry.state_ != internal::ListNode::State::CACHED, entry.loaded_size_);
        entry.state_ = internal::ListNode::State::CACHED;
    }

    void
    EraseLocked(Entry& entry) {
        const auto bytes = entry.loaded_size_;
        dlist->removeItemLocked(&entry, bytes);
        entry.unload();
        entries.remove(entry);
        dlist->RefundLoadedResourceLocked(bytes);
    }

    // Returns whether the entry became evictable or its payload was refunded.
    bool
    UnpinLocked(Entry& entry) {
        if (closed && entry.pin_count_ == 1) {
            // Remove while still pinned: its evictable bytes are already frozen.
            std::lock_guard lock(dlist->list_mtx_);
            EraseLocked(entry);
            entry.pin_count_.fetch_sub(1);
            return true;
        } else if (entry.pin_count_.fetch_sub(1) == 1) {
            // Reuse the Slot path, including its lock-free touch-window fast path.
            entry.touch_to_dlist(true);
            return true;
        }
        return false;
    }

    void
    Release(Entry& entry) {
        bool notify_waiters = false;
        {
            std::unique_lock lock(entry.mtx_);
            notify_waiters = UnpinLocked(entry);
        }
        if (notify_waiters) {
            dlist->notifyWaitingRequests();
        }
    }

    void
    Close() {
        {
            // Serialize shutdown against publication, but do not hold the list
            // lock while waiting for a node lock.
            std::lock_guard lock(dlist->list_mtx_);
            closed = true;
        }
        for (size_t bucket = 0; bucket < entries.getNumBuckets(); ++bucket) {
            while (auto entry = entries.getBucketHead(bucket)) {
                std::unique_lock node_lock(entry->mtx_);
                if (entry->state_ != internal::ListNode::State::NOT_LOADED) {
                    if (entry->pin_count_ == 0) {
                        std::lock_guard lock(dlist->list_mtx_);
                        EraseLocked(*entry);
                    } else {
                        entries.remove(*entry);
                    }
                }
                // Pins own their Entries independently of the index. Their last
                // release will unload them after shutdown.
            }
        }
        dlist->notifyWaitingRequests();
    }

    const std::shared_ptr<internal::DList> dlist;
    // Get, unpin, Close and eviction never acquire this mutex.
    std::mutex put_mutex;
    Map entries;
    std::atomic<bool> closed{false};
};

}  // namespace milvus::cachinglayer
