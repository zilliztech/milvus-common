// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <folly/CancellationToken.h>
#include <folly/Expected.h>
#include <folly/Function.h>
#include <folly/OperationCancelled.h>
#include <folly/Try.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/raw_logging.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

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
    struct Creation;
    // Index only active factories. Ordered hash groups keep the leader's iterator
    // valid across other requests and allow cleanup without calling user key operations.
    using Creations = std::multimap<size_t, std::shared_ptr<Creation>>;

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
                // Release the old entry after assignment so inline callbacks see the new pin.
                Pin old(std::move(*this));
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

    using PinResult = folly::Expected<Pin, SegcoreError>;
    using Result = folly::Expected<folly::Unit, SegcoreError>;
    using CreateResult = folly::Expected<std::pair<std::unique_ptr<Value>, ResourceUsage>, SegcoreError>;

    using IndexConfig = internal::ChainedHashTableConfig;

    explicit CacheMap(std::shared_ptr<internal::DList> dlist, const IndexConfig index_config = {})
        : state_(std::make_shared<CacheState>(std::move(dlist), index_config)) {
        // Initialize the error object before entering the noexcept API.
        static_cast<void>(&allocation_error_);
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
            // The index lock is already released. This reference keeps the wrapper
            // alive even if eviction detaches it before we acquire the node lock.
            if (entry) {
                TEST_SYNC_POINT("CacheMap::Get:AfterLookup");
            }
            return PinResident(std::move(state), std::move(entry));
        });
    }

    // Pin the resident value or run create() on a miss. Concurrent misses for this
    // key share one factory result. The factory runs outside cache locks and returns
    // the owned value and its resident usage, or a SegcoreError. A failed creation can be retried.
    [[nodiscard]] PinResult
    GetOrCreate(const Key& key, folly::FunctionRef<CreateResult()> create) noexcept {
        const auto state = state_;
        std::shared_ptr<Creation> flight;
        std::optional<typename Creations::iterator> registration;
        auto result = TryOperation("GetOrCreate", [&]() -> PinResult {
            const auto hash = std::hash<Key>{}(key);
            const auto bucket = state->entries.getBucket(hash);
            std::shared_ptr<Entry> resident;
            std::shared_ptr<Creation> joined;
            {
                std::lock_guard writer(bucket.mutex());
                // Lookup and registration share the stripe to avoid a third resident lookup.
                // Only try the node lock here. Eviction may already hold it while unlinking the key.
                resident = bucket.findLocked(key);
                auto hit = PinResident(state, resident, /*wait_for_node=*/false);
                if (!hit || hit.value()) {
                    return hit;
                }
                auto& creations = state->inflight_creations[bucket.stripe];
                const auto [first, last] = creations.equal_range(hash);
                for (auto it = first; it != last; ++it) {
                    if (it->second->entry->cache_key == key) {
                        joined = it->second;
                        break;
                    }
                }
                if (!joined) {
                    registration =
                        creations.emplace(hash, std::make_shared<Creation>(state.get(), key, hash, resident));
                    flight = (*registration)->second;
                }
            }
            // A joined caller must use the creation's outcome even if the old resident is erased and the cache closes.
            if (joined) {
                if (joined->creator == std::this_thread::get_id()) {
                    return folly::makeUnexpected(
                        SegcoreError(ErrorCode::InvalidParameter, "GetOrCreate recursively requested the same key"));
                }
                flight = std::move(joined);
                return Pin{};
            }
            if (resident) {
                TEST_SYNC_POINT("CacheMap::GetOrCreate:BeforePinResident");
                auto hit = PinResident(state, resident);
                if (!hit || hit.value()) {
                    return hit;
                }
                std::lock_guard writer(bucket.mutex());
                if (auto replacement = std::move(*flight->handoff)) {
                    return replacement;
                }
                flight->handoff.reset();
            }

            if (state->closed) {
                return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
            }
            // A detached wrapper may still be held by an earlier Get. Publish a fresh one in that case.
            auto candidate = resident ? std::make_shared<Entry>(state.get(), key, hash) : flight->entry;
            auto created = create();
            if (!created) {
                return folly::makeUnexpected(created.error());
            }
            auto& [value, bytes] = created.value();
            if (!value) {
                return folly::makeUnexpected(
                    SegcoreError(ErrorCode::InvalidParameter, "GetOrCreate factory returned a null value"));
            }
            if (!bytes.AllGEZero()) {
                return folly::makeUnexpected(
                    SegcoreError(ErrorCode::InvalidParameter, "GetOrCreate factory returned negative resident bytes"));
            }
            return Publish(state, std::move(candidate), value, bytes, flight.get());
        });

        if (!flight) {
            return result;
        }
        using Outcome = typename Creation::Outcome;
        if (registration) {
            flight->result = std::move(result);
            // Use the saved hash and iterator even if the factory made user key operations throw.
            const auto bucket = state->entries.getBucket(flight->entry->hash_hook.hash);
            std::lock_guard lock(bucket.mutex());
            state->inflight_creations[bucket.stripe].erase(*registration);
            flight->Finish(state->closed ? Outcome::Cancelled : Outcome::Completed);
        }
        return TryOperation("GetOrCreate", [&]() -> PinResult {
            if (!registration) {
                TEST_SYNC_POINT("CacheMap::GetOrCreate:Joined");
                flight->outcome.wait(Outcome::Pending, std::memory_order_acquire);
            }
            if (flight->outcome.load(std::memory_order_acquire) == Outcome::Cancelled) {
                return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
            }
            if (!flight->result) {
                return folly::makeUnexpected(flight->result.error());
            }
            // The creation's Pin stays valid even if the cache closes after completion.
            const auto& entry = flight->result.value().entry_;
            std::shared_lock lock(entry->mtx_);
            state->PinLocked(*entry);
            return Pin(state, entry);
        });
    }

    // Reserve estimated memory and disk bytes before loading. A zero timeout skips
    // the queue. A positive timeout sets a queue deadline, and a negative timeout
    // allows an indefinite wait. Pair each success with UnReserve(estimate).
    // Run potentially blocking calls off the DList event thread so it can deliver reservation timeouts.
    [[nodiscard]] Result
    Reserve(const ResourceUsage estimate,
            const std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const noexcept {
        return TryOperation("Reserve", [&]() -> Result {
            if (!estimate.AllGEZero()) {
                return folly::makeUnexpected(
                    SegcoreError(ErrorCode::InvalidParameter,
                                 fmt::format("CacheMap requires a non-negative loading estimate (memory={}, disk={})",
                                             estimate.memory_bytes, estimate.file_bytes)));
            }
            const auto state = state_;
            if (estimate.AnyGTZero()) {
                OpContext ctx(state->close_source.getToken());
                // The cache assumes its memory limit leaves enough headroom for DList to allocate the result future.
                // If that allocation fails after an immediate grant, the bytes remain reserved.
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
                    if (state->closed) {
                        return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
                    }
                    return folly::makeUnexpected(SegcoreError(
                        ErrorCode::InsufficientResource,
                        fmt::format("CacheMap loading reservation was denied (memory_bytes={}, file_bytes={})",
                                    estimate.memory_bytes, estimate.file_bytes)));
                }
            }
            if (state->closed) {
                if (estimate.AnyGTZero()) {
                    // Release the previously reserved loading resource since the cache is closed.
                    state->dlist->ReleaseLoadingResource(estimate);
                }
                return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
            }
            return folly::unit;
        });
    }

    // Submit immediately. Keep the cache alive and consume the result even if loading
    // is canceled. Pair each success with UnReserve(estimate).
    // Inline continuations may run on the DList event thread. Schedule continuations that reserve
    // or release resources on an executor that queues work on another thread.
    [[nodiscard]] folly::SemiFuture<Result>
    ReserveAsync(const ResourceUsage estimate,
                 const std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) const {
        if (!estimate.AllGEZero() || !estimate.AnyGTZero()) {
            return folly::makeSemiFuture(Reserve(estimate, timeout));
        }
        const auto state = state_;
        // DList can throw before returning a future. makeSemiFutureWith forwards the future
        // on success and captures synchronous exceptions so the continuation can return their code and message.
        auto admission = folly::makeSemiFutureWith([&] {
            OpContext ctx(state->close_source.getToken());
            return state->dlist->ReserveLoadingResourceWithTimeout(estimate, timeout, &ctx);
        });
        return std::move(admission).defer([state, estimate](folly::Try<bool>&& granted) {
            return TryOperation("ReserveAsync", [&]() -> Result {
                if (!granted.value()) {
                    if (state->closed) {
                        return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
                    }
                    return folly::makeUnexpected(SegcoreError(
                        ErrorCode::InsufficientResource,
                        fmt::format("CacheMap loading reservation was denied (memory_bytes={}, file_bytes={})",
                                    estimate.memory_bytes, estimate.file_bytes)));
                }
                if (state->closed) {
                    state->dlist->ReleaseLoadingResource(estimate);
                    return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
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

    // Insert an absent key and charge its resident payload size. A duplicate key
    // returns InvalidParameter. Success transfers ownership and returns a Pin.
    // On failure the caller keeps value. Call UnReserve with the original estimate.
    [[nodiscard]] PinResult
    Put(const Key& key, std::unique_ptr<Value>& value, const ResourceUsage bytes) noexcept {
        Pin result;
        const auto status = TryOperation("Put", [&]() -> Result {
            const auto state = state_;
            if (!value) {
                return folly::makeUnexpected(
                    SegcoreError(ErrorCode::InvalidParameter, "Put requires a non-null value"));
            }
            if (!bytes.AllGEZero()) {
                return folly::makeUnexpected(
                    SegcoreError(ErrorCode::InvalidParameter, "Put received negative resident bytes"));
            }
            if (state->closed) {
                return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
            }

            TEST_SYNC_POINT("CacheMap::Put:BeforePublish");
            auto entry = std::make_shared<Entry>(state.get(), key, std::hash<Key>{}(key));
            auto published = Publish(state, std::move(entry), value, bytes);
            if (!published) {
                return folly::makeUnexpected(published.error());
            }
            result = std::move(published).value();
            TEST_SYNC_POINT("CacheMap::Put:AfterPublish");
            return folly::unit;
        });
        // Once the input is owned by the cache, return its committed Pin.
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
                if (state->closed) {
                    return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
                }
                return folly::makeUnexpected(SegcoreError(ErrorCode::OutOfRange, "CacheMap key was not found"));
            }
            {
                std::unique_lock node_lock(entry->mtx_);
                if (state->closed) {
                    return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
                }
                if (entry->state_ == internal::ListNode::State::NOT_LOADED) {
                    return folly::makeUnexpected(
                        SegcoreError(ErrorCode::OutOfRange, "CacheMap entry is no longer resident"));
                }
                if (entry->pin_count_ != 0) {
                    return folly::makeUnexpected(
                        SegcoreError(ErrorCode::UnexpectedError, "Cannot erase a pinned cache entry"));
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
    static PinResult
    PinResident(std::shared_ptr<CacheState> state, std::shared_ptr<Entry> entry, const bool wait_for_node = true) {
        std::shared_lock<std::shared_mutex> lock;
        if (entry) {
            lock = wait_for_node ? std::shared_lock(entry->mtx_) : std::shared_lock(entry->mtx_, std::try_to_lock);
        }
        if (state->closed) {
            return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
        }
        if (!lock.owns_lock() || entry->state_ == internal::ListNode::State::NOT_LOADED) {
            return Pin{};
        }
        // Detached KV entries must never start a ListNode load.
        state->PinLocked(*entry);
        return Pin(std::move(state), std::move(entry));
    }

    // The key is absent and the caller holds its stripe and the private candidate's node lock.
    static Pin
    PublishLocked(const std::shared_ptr<CacheState>& state, const auto& bucket, const std::shared_ptr<Entry>& entry,
                  std::unique_ptr<Value>& value, const ResourceUsage bytes, const bool explicit_put) {
        Pin* handoff = nullptr;
        if (explicit_put) {
            const auto& creations = state->inflight_creations[bucket.stripe];
            const auto [first, last] = creations.equal_range(entry->hash_hook.hash);
            for (auto it = first; it != last; ++it) {
                auto& creation = *it->second;
                if (creation.handoff && !*creation.handoff && creation.entry->cache_key == entry->cache_key) {
                    handoff = &*creation.handoff;
                    break;
                }
            }
        }
        entry->value = std::move(value);
        entry->state_ = internal::ListNode::State::CACHED;
        entry->pin_count_.fetch_add(1);
        state->dlist->ChargeLoadedResource(bytes);
        monitor::cache_kv_entry_count.Increment();
        monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Increment(bytes.memory_bytes);
        monitor::cache_kv_loaded_bytes(StorageType::DISK).Increment(bytes.file_bytes);
        if (handoff) {
            state->PinLocked(*entry);
            *handoff = Pin(state, entry);
        }
        state->entries.prependLocked(bucket, entry);
        return Pin(state, entry);
    }

    static PinResult
    Publish(const std::shared_ptr<CacheState>& state, std::shared_ptr<Entry> entry, std::unique_ptr<Value>& value,
            const ResourceUsage bytes, Creation* creation = nullptr) {
        if (state->closed) {
            return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
        }
        const auto& dlist = state->dlist;
        const auto bucket = state->entries.getBucket(entry->hash_hook.hash);
        std::unique_lock node_lock(entry->mtx_);
        // The private candidate stays off the eviction budget until its key is published.
        entry->loaded_size_ = bytes;
        dlist->touchItem(entry.get(), true, std::nullopt);
        auto unlink = folly::makeGuard([&] {
            dlist->removeItem(entry.get(), {});
            entry->loaded_size_ = {};
        });
        TEST_SYNC_POINT("CacheMap::Publish:BeforeIndexInsert");
        std::shared_ptr<Entry> existing;
        {
            std::lock_guard writer(bucket.mutex());
            existing = bucket.findLocked(entry->cache_key);
            if (state->closed) {
                return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
            }
            if (!existing) {
                auto pin = PublishLocked(state, bucket, entry, value, bytes, !creation);
                unlink.dismiss();
                return pin;
            }
            if (!creation) {
                return folly::makeUnexpected(SegcoreError(ErrorCode::InvalidParameter, "CacheMap key already exists"));
            }
            creation->handoff.emplace();
        }

        // Pin the winner outside its stripe. Any Put that replaces it hands this creation a Pin
        // under that same stripe, so a detached winner never requires another bucket lookup.
        TEST_SYNC_POINT("CacheMap::GetOrCreate:BeforePinResident");
        auto hit = PinResident(state, std::move(existing));
        if (!hit || hit.value()) {
            return hit;
        }
        std::lock_guard writer(bucket.mutex());
        if (state->closed) {
            return folly::makeUnexpected(SegcoreError(ErrorCode::FollyCancel, "CacheMap is closed"));
        }
        if (*creation->handoff) {
            return std::move(*creation->handoff);
        }
        auto pin = PublishLocked(state, bucket, entry, value, bytes, false);
        unlink.dismiss();
        return pin;
    }

    // std::runtime_error copies share their diagnostic, including when allocation fails.
    inline static const SegcoreError allocation_error_{ErrorCode::MemAllocateFailed,
                                                       "CacheMap could not allocate an error message"};

    // Locks and reservations unwind before error conversion. Expected cache
    // outcomes (miss, busy, capacity) are returned directly by the operation.
    template <typename Fn>
    static auto
    TryOperation(const char* operation, Fn&& fn) noexcept -> decltype(fn()) {
        const auto log_error = [operation](ErrorCode code, const char* reason) noexcept {
            RAW_LOG(ERROR, "[MCL] CacheMap::%s (%d): %s", operation, static_cast<int>(code), reason);
        };
        const auto failure = [&](ErrorCode code, const char* reason) noexcept {
            log_error(code, reason);
            try {
                return folly::makeUnexpected(SegcoreError(code, reason));
            } catch (const std::bad_alloc&) {
                return folly::makeUnexpected(allocation_error_);
            }
        };
        try {
            return fn();
        } catch (const SegcoreError& e) {
            log_error(e.get_error_code(), e.what());
            return folly::makeUnexpected(e);
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
        Entry(CacheState* owner, const Key& key, const size_t hash)
            : internal::ListNode(owner->dlist.get(), true), owner(owner), cache_key(key), hash_hook{nullptr, hash} {
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
        internal::ChainedHashTableHook<Entry> hash_hook;
        std::unique_ptr<Value> value;
    };

    struct Creation {
        enum class Outcome : unsigned char { Pending, Completed, Cancelled };

        Creation(CacheState* state, const Key& key, const size_t hash, const std::shared_ptr<Entry>& resident)
            : entry(resident ? resident : std::make_shared<Entry>(state, key, hash)),
              on_close(state->close_source.getToken(), [this] { Finish(Outcome::Cancelled); }) {
            if (resident) {
                handoff.emplace();
            }
        }

        // Completion and Close race to wake every caller with the same outcome.
        void
        Finish(const Outcome terminal) noexcept {
            auto pending = Outcome::Pending;
            if (outcome.compare_exchange_strong(pending, terminal, std::memory_order_release)) {
                outcome.notify_all();
            }
        }

        const std::shared_ptr<Entry> entry;
        const std::thread::id creator = std::this_thread::get_id();
        // Engaged while resolving an observed resident outside its stripe. A replacing insertion
        // supplies one Pin before it becomes visible. Access is protected by the stripe.
        std::optional<Pin> handoff;
        // The creator writes the result. Callers read it only after observing Completed.
        PinResult result{Pin{}};
        std::atomic<Outcome> outcome{Outcome::Pending};
        folly::CancellationCallback on_close;
    };

    const std::shared_ptr<CacheState> state_;
};

template <CacheMapKey Key, typename Value>
struct CacheMap<Key, Value>::CacheState {
    using Map = internal::ChainedHashTable<Key, Entry, &Entry::hash_hook>;

    explicit CacheState(std::shared_ptr<internal::DList> domain, const IndexConfig index_config)
        : dlist(std::move(domain)), entries(index_config), inflight_creations(entries.getNumStripes()) {
        if (!dlist || !dlist->event_base_thread_) {
            ThrowInfo(ErrorCode::InvalidParameter, "Keyed cache requires an initialized eviction-enabled DList");
        }
    }

    // Locked helpers require entry.mtx_. EraseLocked additionally requires list_mtx_.
    void
    PinLocked(Entry& entry) {
        if (entry.pin_count_.fetch_add(1) == 0) {
            dlist->freezeItem(&entry, entry.loaded_size_);
        }
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
        // Each publisher checks closed under its stripe. An insertion that wins the race is
        // visible when this sweep takes that stripe.
        closed = true;
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
    Map entries;
    // Each registry is accessed only while the corresponding index stripe is locked.
    std::vector<Creations> inflight_creations;
    std::atomic<bool> closed{false};
    folly::CancellationSource close_source;
};

}  // namespace milvus::cachinglayer
