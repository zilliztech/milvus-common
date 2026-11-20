// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <folly/CancellationToken.h>
#include <folly/Expected.h>
#include <folly/OperationCancelled.h>
#include <folly/ScopeGuard.h>
#include <folly/Try.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/raw_logging.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "cachinglayer/lrucache/ChainedHashTable.h"
#include "cachinglayer/lrucache/DList.h"
#include "common/EasyAssert.h"
#include "common/OpContext.h"
#include "syncpoint/sync_point.h"

namespace milvus::cachinglayer {

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
    // Get returns a default-constructed (empty) Pin for a cache miss.
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

    using PinResult = folly::Expected<Pin, ErrorCode>;
    using Result = folly::Expected<folly::Unit, ErrorCode>;

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

    // Look up a resident value without loading it. A hit returns a read-only Pin
    // that keeps the value resident. A miss returns a successful result with an empty Pin.
    [[nodiscard]] PinResult
    Get(const Key& key) const noexcept {
        return TryOperation("Get", [&]() -> PinResult {
            auto state = state_;
            auto entry = state->entries.find(key);
            if (!entry) {
                if (state->closed) {
                    return folly::makeUnexpected(ErrorCode::FollyCancel);
                }
                return Pin{};
            }
            // The index lock is already released. This reference keeps the wrapper
            // alive even if eviction detaches it before we acquire the node lock.
            TEST_SYNC_POINT("CacheMap::Get:AfterLookup");
            std::shared_lock lock(entry->mtx_);
            if (state->closed) {
                return folly::makeUnexpected(ErrorCode::FollyCancel);
            }
            if (entry->state_ == internal::ListNode::State::NOT_LOADED) {
                return Pin{};
            }
            // Do not use ListNode::pin(): a detached KV entry must never start a load.
            state->PinLocked(*entry);
            return Pin(std::move(state), std::move(entry));
        });
    }

    // Reserve estimated memory and disk bytes before loading. A zero timeout skips
    // the queue. A positive timeout sets a queue deadline, and a negative timeout
    // allows an indefinite wait. Pair each success with UnReserve(estimate).
    [[nodiscard]] Result
    Reserve(const ResourceUsage estimate,
            const std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const noexcept {
        return TryOperation("Reserve", [&]() -> Result {
            if (!estimate.AllGEZero()) {
                return folly::makeUnexpected(ErrorCode::InvalidParameter);
            }
            const auto state = state_;
            if (estimate.AnyGTZero()) {
                OpContext ctx(state->close_source.getToken());
                auto admission = state->dlist->ReserveLoadingResourceWithTimeout(estimate, timeout, &ctx);
                if (!admission.isReady()) {
                    try {
                        TEST_SYNC_POINT("CacheMap::Reserve:BeforeWait");
                        admission.wait();
                    } catch (const std::bad_alloc&) {
                        // DList already owns this request. If Folly cannot allocate
                        // its waiter, still collect the grant so the caller can return it.
                        while (!admission.isReady()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                }
                if (!std::move(admission).value()) {
                    return folly::makeUnexpected(state->closed ? ErrorCode::FollyCancel
                                                               : ErrorCode::InsufficientResource);
                }
            }
            if (state->closed) {
                if (estimate.AnyGTZero()) {
                    // Release the previously reserved loading resource since the cache is closed.
                    state->dlist->ReleaseLoadingResource(estimate);
                }
                return folly::makeUnexpected(ErrorCode::FollyCancel);
            }
            return folly::unit;
        });
    }

    // Submit immediately. Keep the cache alive and consume the result even if loading
    // is canceled. Pair each success with UnReserve(estimate).
    [[nodiscard]] folly::SemiFuture<Result>
    ReserveAsync(const ResourceUsage estimate,
                 const std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const {
        if (!estimate.AllGEZero() || !estimate.AnyGTZero()) {
            return folly::makeSemiFuture(Reserve(estimate, timeout));
        }
        const auto state = state_;
        // DList can throw before returning a future. makeSemiFutureWith forwards the future
        // on success and captures synchronous exceptions so the continuation can map them to ErrorCode.
        auto admission = folly::makeSemiFutureWith([&] {
            OpContext ctx(state->close_source.getToken());
            return state->dlist->ReserveLoadingResourceWithTimeout(estimate, timeout, &ctx);
        });
        return std::move(admission).defer([state, estimate](folly::Try<bool>&& granted) {
            return TryOperation("ReserveAsync", [&]() -> Result {
                if (!granted.value()) {
                    return folly::makeUnexpected(state->closed ? ErrorCode::FollyCancel
                                                               : ErrorCode::InsufficientResource);
                }
                if (state->closed) {
                    state->dlist->ReleaseLoadingResource(estimate);
                    return folly::makeUnexpected(ErrorCode::FollyCancel);
                }
                return folly::unit;
            });
        });
    }

    // Return the original estimate once per successful Reserve, even if loading or Put fails.
    // This returns void because DList refunds the estimate before notifying waiters.
    // Retrying after a notification error would refund the same reservation twice.
    void
    UnReserve(const ResourceUsage& estimate) const noexcept {
        if (!estimate.AllGEZero()) {
            RAW_LOG(ERROR, "[MCL] CacheMap::UnReserve requires a non-negative estimate");
            return;
        }
        // A successful reservation is owed to the shared DList even during cache shutdown.
        const auto dlist = state_->dlist;
        static_cast<void>(TryOperation("UnReserve", [&]() -> Result {
            dlist->ReleaseLoadingResource(estimate);
            return folly::unit;
        }));
    }

    // Insert or replace a value and charge its resident payload size in bytes.
    // Success transfers ownership and returns a Pin. On failure the caller keeps
    // value. Replacing a pinned entry returns UnexpectedError. Call UnReserve
    // separately with the original loading estimate.
    [[nodiscard]] PinResult
    Put(const Key& key, std::unique_ptr<Value>& value, const ResourceUsage bytes) noexcept {
        Pin result;
        const auto status = TryOperation("Put", [&]() -> Result {
            const auto state = state_;
            if (!value || !bytes.AllGEZero()) {
                return folly::makeUnexpected(ErrorCode::InvalidParameter);
            }
            const auto& dlist = state->dlist;
            std::lock_guard writer(state->put_mutex);
            if (state->closed) {
                return folly::makeUnexpected(ErrorCode::FollyCancel);
            }
            auto old = state->entries.find(key);
            if (old) {
                std::unique_lock lock(old->mtx_);
                if (old->state_ == internal::ListNode::State::NOT_LOADED) {
                    lock.unlock();
                    old.reset();
                } else {
                    if (old->pin_count_ != 0) {
                        return folly::makeUnexpected(ErrorCode::UnexpectedError);
                    }
                    state->PinLocked(*old);
                }
            }
            auto unpin = folly::makeGuard([&] {
                if (old) {
                    std::unique_lock lock(old->mtx_);
                    state->UnpinLocked(*old);
                }
            });

            TEST_SYNC_POINT("CacheMap::Put:BeforePublish");

            {
                auto entry = old ? old : std::make_shared<Entry>(state.get(), key);
                std::unique_lock node_lock(entry->mtx_);
                std::lock_guard lock(dlist->list_mtx_);
                if (state->closed) {
                    return folly::makeUnexpected(ErrorCode::FollyCancel);
                }
                if (old && old->pin_count_ != 1) {
                    return folly::makeUnexpected(ErrorCode::UnexpectedError);
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
                // update metrics
                monitor::cache_kv_entry_count.Increment();
                monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Increment(bytes.memory_bytes);
                monitor::cache_kv_loaded_bytes(StorageType::DISK).Increment(bytes.file_bytes);
                state->PinLocked(*entry);
                result = Pin(state, std::move(entry));
            }
            TEST_SYNC_POINT("CacheMap::Put:AfterPublish");
            return folly::unit;
        });
        // After publication the cache owns the input; deliver its committed Pin.
        if (!status && !result) {
            return folly::makeUnexpected(status.error());
        }
        return PinResult(std::move(result));
    }

    // Remove an unpinned entry and refund its resident bytes. An absent key returns
    // OutOfRange, and a pinned entry returns UnexpectedError.
    [[nodiscard]] Result
    Erase(const Key& key) noexcept {
        return TryOperation("Erase", [&]() -> Result {
            auto state = state_;
            auto entry = state->entries.find(key);
            if (!entry) {
                return folly::makeUnexpected(state->closed ? ErrorCode::FollyCancel : ErrorCode::OutOfRange);
            }
            {
                std::unique_lock node_lock(entry->mtx_);
                if (state->closed) {
                    return folly::makeUnexpected(ErrorCode::FollyCancel);
                }
                if (entry->state_ == internal::ListNode::State::NOT_LOADED) {
                    return folly::makeUnexpected(ErrorCode::OutOfRange);
                }
                // The internal replacement pin also makes this entry busy.
                if (entry->pin_count_ != 0) {
                    return folly::makeUnexpected(ErrorCode::UnexpectedError);
                }
                std::lock_guard lock(state->dlist->list_mtx_);
                state->EraseLocked(*entry);
            }
            state->NotifyWaiters();
            return folly::unit;
        });
    }

    [[nodiscard]] size_t
    Size() const {
        return state_->entries.size();
    }

 private:
    // Locks and reservations unwind before error conversion. Expected cache
    // outcomes (miss, busy, capacity) are returned directly by the operation.
    template <typename Fn>
    static auto
    TryOperation(const char* operation, Fn&& fn) noexcept -> decltype(fn()) {
        const auto failure = [operation](ErrorCode code, const char* reason) noexcept {
            // Preserve exception diagnostics even when handling an allocation failure.
            RAW_LOG(ERROR, "[MCL] CacheMap::%s (%d): %s", operation, static_cast<int>(code), reason);
            return folly::makeUnexpected(code);
        };
        try {
            return fn();
        } catch (const SegcoreError& e) {
            return failure(e.get_error_code(), e.what());
        } catch (const std::bad_alloc& e) {
            return failure(ErrorCode::MemAllocateFailed, e.what());
        } catch (const std::invalid_argument& e) {
            return failure(ErrorCode::InvalidParameter, e.what());
        } catch (const std::out_of_range& e) {
            return failure(ErrorCode::OutOfRange, e.what());
        } catch (const std::length_error& e) {
            return failure(ErrorCode::OutOfRange, e.what());
        } catch (const folly::OperationCancelled& e) {
            return failure(ErrorCode::FollyCancel, e.what());
        } catch (const folly::FutureCancellation& e) {
            return failure(ErrorCode::FollyCancel, e.what());
        } catch (const folly::FutureException& e) {
            return failure(ErrorCode::FollyOtherException, e.what());
        } catch (const std::system_error& e) {
            if (e.code() == std::errc::not_enough_memory) {
                return failure(ErrorCode::MemAllocateFailed, e.what());
            }
            if (e.code() == std::errc::resource_unavailable_try_again) {
                return failure(ErrorCode::InsufficientResource, e.what());
            }
            return failure(ErrorCode::UnexpectedError, e.what());
        } catch (const std::exception& e) {
            return failure(ErrorCode::UnexpectedError, e.what());
        } catch (...) {
            // A user-provided key operation can also throw a non-standard exception.
            return failure(ErrorCode::UnexpectedError, "non-standard exception");
        }
    }

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

    // Erase, final unpin and close make capacity available under cache locks.
    // Run continuations after unlocking and keep exceptions out of RAII cleanup.
    void
    NotifyWaiters() const noexcept {
        static_cast<void>(TryOperation("NotifyWaiters", [&]() -> Result {
            dlist->notifyWaitingRequests();
            return folly::unit;
        }));
    }

    void
    Release(Entry& entry) {
        bool notify_waiters = false;
        {
            std::unique_lock lock(entry.mtx_);
            notify_waiters = UnpinLocked(entry);
        }
        if (notify_waiters) {
            NotifyWaiters();
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
        close_source.requestCancellation();
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
        NotifyWaiters();
    }

    const std::shared_ptr<internal::DList> dlist;
    // Get, unpin, Close and eviction never acquire this mutex.
    std::mutex put_mutex;
    Map entries;
    std::atomic<bool> closed{false};
    folly::CancellationSource close_source;
};

}  // namespace milvus::cachinglayer
