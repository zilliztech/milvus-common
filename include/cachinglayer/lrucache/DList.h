// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License
#pragma once

#include <folly/CancellationToken.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <folly/system/ThreadName.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>

#include "cachinglayer/LoadingOverhead.h"
#include "cachinglayer/Metrics.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/ListNode.h"
#include "common/OpContext.h"
#include "log/Log.h"

namespace milvus::cachinglayer::internal {

struct LoadingResourceReservationResult {
    // Explicit because a successful policy-derived reservation may be zero.
    bool success{false};
    // Unscaled request reservation corresponding to the applied transition.
    ResourceUsage reserved;
};

class DList : public std::enable_shared_from_this<DList> {
 public:
    DList(bool eviction_enabled, ResourceUsage max_memory, ResourceUsage low_watermark, ResourceUsage high_watermark,
          EvictionConfig eviction_config)
        : max_resource_limit_(max_memory),
          low_watermark_(low_watermark),
          high_watermark_(high_watermark),
          eviction_config_(eviction_config),
          next_request_id_(1) {
        // if eviction is disabled, we don't need to initialize the event base and thread
        if (!eviction_enabled) {
            return;
        }

        AssertInfo(low_watermark.AllGEZero(), "[MCL] low watermark must be greater than or equal to 0");
        AssertInfo((high_watermark - low_watermark).AllGEZero(),
                   "[MCL] high watermark must be greater than low watermark");
        AssertInfo((max_memory - high_watermark).AllGEZero(), "[MCL] max memory must be greater than high watermark");

        monitor::cache_capacity_bytes(StorageType::MEMORY).Set(max_memory.memory_bytes);
        monitor::cache_capacity_bytes(StorageType::DISK).Set(max_memory.file_bytes);
        monitor::cache_high_watermark_bytes(StorageType::MEMORY).Set(high_watermark.memory_bytes);
        monitor::cache_high_watermark_bytes(StorageType::DISK).Set(high_watermark.file_bytes);
        monitor::cache_low_watermark_bytes(StorageType::MEMORY).Set(low_watermark.memory_bytes);
        monitor::cache_low_watermark_bytes(StorageType::DISK).Set(low_watermark.file_bytes);

        // Initialize event base and thread
        event_base_thread_ = std::make_unique<folly::ScopedEventBaseThread>("cache-eb");

        if (eviction_config_.background_eviction_enabled) {
            LOG_INFO("[MCL] Starting periodic background eviction loop thread");
            bg_eviction_thread_ = std::thread(&DList::evictionLoop, this);
        }
    }

    void
    BindLoadingOverheadGroups(const LoadingOverheadConfig& config);

    void
    UnbindLoadingOverheadGroups(const LoadingOverheadConfig& config);

    // Creates one Group independently before Translator bindings reference it.
    std::shared_ptr<LoadingOverheadGroup>
    CreateLoadingOverheadGroup(LoadingOverheadDimension dimension, LoadingOverheadPolicy policy);

    // Replaces an existing Group policy. The next Reserve or Release reconciles
    // its reservation with the new target. The caller must follow Manager's
    // serialized owner and Budget/TP ordering contract.
    LoadingOverheadUpdateResult
    UpdateLoadingOverheadGroup(const std::shared_ptr<LoadingOverheadGroup>& group, LoadingOverheadPolicy policy);

    ~DList() {
        // waiting requests should be cleared before event base thread is stopped
        clearWaitingQueue();

        if (event_base_thread_) {
            auto* eb = event_base_thread_->getEventBase();
            if (eb->isInEventBaseThread()) {
                // Being destroyed from a callback on our own event base thread (e.g., a
                // weak_self.lock() callback held the last shared_ptr<DList>).  We cannot
                // drain or join the thread from itself — that causes EDEADLK.  Move the
                // ScopedEventBaseThread to a detached thread that will join it after the
                // current callback finishes and the event loop exits.
                auto eb_thread = std::move(event_base_thread_);
                eb_thread->getEventBase()->terminateLoopSoon();
                std::thread([eb_thread = std::move(eb_thread)]() mutable { eb_thread.reset(); }).detach();
            } else {
                // Normal path: drain pending callbacks, then destroy.
                eb->runInEventBaseThreadAndWait([]() {});
                event_base_thread_.reset();
            }
        }

        // stop eviction loop thread
        if (eviction_config_.background_eviction_enabled) {
            stop_bg_eviction_loop_ = true;
            bg_eviction_thread_cv_.notify_all();
            if (bg_eviction_thread_.joinable()) {
                bg_eviction_thread_.join();
            }
        }
    }

    // If after evicting all unpinned items, the used_resources_ is still larger than new_limit, false will be returned
    // and no eviction will be done.
    // Will throw if new_limit is negative.
    bool
    UpdateMaxLimit(const ResourceUsage& new_limit);

    // Updating a watermark retries queued reservations and may evict cache
    // entries while satisfying them. Validation failure is reported by exception.
    void
    UpdateLowWatermark(const ResourceUsage& new_low_watermark);

    void
    UpdateHighWatermark(const ResourceUsage& new_high_watermark);

    // True if no nodes in the list.
    bool
    IsEmpty() const;

    // Reserve loading resource with timeout, called before loading a cell.
    // When timeout > 0, the request will wait up to the specified duration.
    // When timeout == 0, the request will fail immediately without entering the
    //   waiting queue (best-effort mode, used for warmup scenarios).
    // When timeout < 0, the request will wait indefinitely (no timeout).
    //   Note: negative timeout requests use time_point::max() as their deadline,
    //   which means they sort after all finite-deadline requests in the waiting queue.
    folly::SemiFuture<bool>
    ReserveLoadingResourceWithTimeout(const ResourceUsage& size, std::chrono::milliseconds timeout,
                                      OpContext* ctx = nullptr);

    // Reserve with loading-overhead Group integration.
    // Group-managed overhead is capped across the bound Group before the
    // request total is scaled by loading_resource_factor.
    // A successful reservation may reserve zero bytes, so success is explicit.
    folly::SemiFuture<LoadingResourceReservationResult>
    ReserveLoadingResourceWithTimeout(const ResourceUsage& loaded, const ResourceUsage& overhead,
                                      const LoadingOverheadConfig* config, std::chrono::milliseconds timeout,
                                      OpContext* ctx = nullptr);

    // Release with loading-overhead Group integration.
    // Returns the actual unscaled size released (loaded + Group delta).
    ResourceUsage
    ReleaseLoadingResource(const ResourceUsage& loaded, const ResourceUsage& overhead,
                           const LoadingOverheadConfig* config);

    // Release resource used for loading, called after loading a cell.
    void
    ReleaseLoadingResource(const ResourceUsage& loading_size);

    // Called when a cell is loaded.
    void
    ChargeLoadedResource(const ResourceUsage& size);

    // Called only when a cell is manually released, eviction handle it internally, so it should not call this.
    void
    RefundLoadedResource(const ResourceUsage& size);

    // Caller must guarantee that the current thread holds the lock of list_node->mtx_.
    // touchItem is used in 2 places:
    // 1. when a loaded cell is pinned/unpinned, we need to touch it to refresh the LRU order.
    //    we don't update used_resources_ here.
    // 2. when a cell is loaded as a bonus, we need to touch it to insert into the LRU and update
    //    used_resources_ to track the memory usage(usage of such cell is not counted during reservation).
    // force_touch is used to force a DList touch event, mainly for testing.
    //
    // If the item is really touched by DList, its last_touch_ time will be updated.
    void
    touchItem(ListNode* list_node, bool force_touch = false, std::optional<ResourceUsage> size = std::nullopt);

    // Caller must guarantee that the current thread holds the lock of list_node->mtx_.
    // Removes the node from the list and updates used_resources_.
    void
    removeItem(ListNode* list_node, ResourceUsage size);

    // Called when a cell is frozen, i.e. the cell is pinned and will not be evicted.
    void
    freezeItem(ListNode* list_node [[maybe_unused]], ResourceUsage size);

    const EvictionConfig&
    eviction_config() const {
        return eviction_config_;
    }

 private:
    friend class DListTestFriend;

    // Waiting request for timeout-based memory reservation
    struct WaitingRequest {
        ResourceUsage required_size;  // initial policy-derived scaled requirement used for queue ordering
        ResourceUsage loaded;         // loaded portion (for Group-aware path)
        ResourceUsage overhead;       // overhead portion (for Group-aware path)
        // Queued requests own a copy so they never retain a pointer into Translator Meta.
        std::optional<LoadingOverheadConfig> loading_overhead_config;
        std::chrono::steady_clock::time_point deadline;
        folly::Promise<bool> bool_promise;
        folly::Promise<LoadingResourceReservationResult> resource_promise;
        bool use_resource_promise{false};
        uint64_t request_id;
        std::optional<folly::CancellationCallback> cancel_cb{std::nullopt};

        // Request-local constructor.
        WaitingRequest(ResourceUsage size, std::chrono::steady_clock::time_point dl, folly::Promise<bool> p,
                       uint64_t id)
            : required_size(size),
              deadline(dl),
              bool_promise(std::move(p)),
              resource_promise(folly::Promise<LoadingResourceReservationResult>::makeEmpty()),
              request_id(id) {
        }

        // Group-aware constructor.
        WaitingRequest(ResourceUsage required_size, ResourceUsage loaded, ResourceUsage overhead,
                       const LoadingOverheadConfig* config, std::chrono::steady_clock::time_point dl,
                       folly::Promise<LoadingResourceReservationResult> p, uint64_t id)
            : required_size(required_size),
              loaded(loaded),
              overhead(overhead),
              loading_overhead_config(config ? std::make_optional(*config) : std::nullopt),
              deadline(dl),
              bool_promise(folly::Promise<bool>::makeEmpty()),
              resource_promise(std::move(p)),
              use_resource_promise(true),
              request_id(id) {
        }

        const LoadingOverheadConfig*
        loadingOverheadConfig() const {
            return loading_overhead_config ? &loading_overhead_config.value() : nullptr;
        }

        void
        setValue(bool success, ResourceUsage actual = {}) {
            if (use_resource_promise) {
                resource_promise.setValue(
                    LoadingResourceReservationResult{success, success ? actual : ResourceUsage{}});
            } else {
                bool_promise.setValue(success);
            }
        }
    };

    struct LoadingResourceReservationAttempt {
        LoadingResourceReservationResult result;
        ResourceUsage required_size;
    };

    // Comparator for priority queue (smaller size and earlier deadline have higher priority)
    struct WaitingRequestComparator {
        bool
        operator()(const std::unique_ptr<WaitingRequest>& a, const std::unique_ptr<WaitingRequest>& b) {
            // First priority: deadline (earlier deadline has higher priority)
            if (a->deadline != b->deadline) {
                return a->deadline > b->deadline;
            }
            // Second priority: resource size (smaller size has higher priority)
            const auto total_a = static_cast<uint64_t>(std::max(a->required_size.memory_bytes, int64_t{0})) +
                                 static_cast<uint64_t>(std::max(a->required_size.file_bytes, int64_t{0}));
            const auto total_b = static_cast<uint64_t>(std::max(b->required_size.memory_bytes, int64_t{0})) +
                                 static_cast<uint64_t>(std::max(b->required_size.file_bytes, int64_t{0}));
            return total_a > total_b;
        }
    };

    // reserveResource without taking lock, must be called with lock held.
    bool
    reserveResourceInternal(const ResourceUsage& size);

    // Common implementation for resource reservation with eviction.
    // Returns {success, scaled_size_reserved}.
    std::pair<bool, ResourceUsage>
    reserveResourceInternalImpl(const ResourceUsage& size, std::function<void()> rollback);

    void
    validateLoadingOverheadBinding(const std::optional<LoadingOverheadGroupBinding>& binding,
                                   LoadingOverheadDimension dimension) const;

    ResourceUsage
    reserveLoadingOverhead(const LoadingOverheadConfig& config, const ResourceUsage& overhead);

    void
    rollbackLoadingOverhead(const LoadingOverheadConfig& config, const ResourceUsage& overhead,
                            const ResourceUsage& reserved) noexcept;

    ResourceUsage
    releaseLoadingOverhead(const LoadingOverheadConfig& config, const ResourceUsage& overhead);

    // Reserve with Groups under lock using the factor-adjusted target transition.
    // Returns the explicit result and the checked scaled requirement attempted.
    LoadingResourceReservationAttempt
    reserveResourceInternalWithOverhead(const ResourceUsage& loaded, const ResourceUsage& overhead,
                                        const LoadingOverheadConfig* config);

    void
    evictionLoop();

    // Try to evict some items so that the resources of evicted items are larger than expected_eviction.
    // If we cannot achieve the goal, but we can evict min_eviction, we will still perform eviction.
    // If we cannot even evict min_eviction, nothing will be evicted and false will be returned.
    // Must be called under the lock of list_mtx_.
    // Returns the logical amount of resources that are evicted. 0 means no eviction happened.
    ResourceUsage
    tryEvict(const ResourceUsage& expected_eviction, const ResourceUsage& min_eviction,
             const bool evict_expired_items = false);

    // Handle waiting requests when resources are available.
    // This method should be called with list_mtx_ already held.
    // Returns requests that need to be destroyed outside the lock to avoid deadlock.
    std::vector<std::unique_ptr<WaitingRequest>>
    handleWaitingRequests();

    // Fail one queued request and immediately retry requests behind it.
    // Must be called with list_mtx_ held; returned requests are destroyed by
    // the caller after releasing the lock.
    std::vector<std::unique_ptr<WaitingRequest>>
    failWaitingRequest(uint64_t request_id, const char* reason);

    // Clear all waiting requests (used in destructor)
    void
    clearWaitingQueue();

    // Must be called under the lock of list_mtx_ and list_node->mtx_.
    // ListNode is guaranteed to be not in the list.
    void
    pushHead(ListNode* list_node);

    // Must be called under the lock of list_mtx_ and list_node->mtx_.
    // If ListNode is not in the list, this function does nothing.
    // Returns true if ListNode is in the list and popped, false otherwise.
    bool
    popItem(ListNode* list_node);

    std::string
    usageInfo() const;

    // [NOTE]: this method is deprecated for now, we only use checkPhysicalMemoryLimit instead.
    // Physical resource protection methods
    // Returns the amount of memory/disk that needs to be evicted to satisfy physical resource limit
    // Returns 0 if no eviction needed, positive value if eviction needed.
    // For disk, it only checks whether the usage will exceed the disk capacity to avoid using up all disk space.
    // Does not obey the configured disk capacity limit. Reason is: we can't easily determine the amount of disk space
    // that is used by the cache(there may be other processes using the disk).
    [[maybe_unused]] ResourceUsage
    checkPhysicalResourceLimit(const ResourceUsage& size) const;

    // Physical memory limit check, used to check if the memory usage will exceed the physical memory limit.
    // Returns the amount of memory that needs to be evicted to satisfy physical memory limit.
    // Returns 0 if no eviction needed, positive value if eviction needed.
    ResourceUsage
    checkPhysicalMemoryLimit(const ResourceUsage& size) const;

    // not thread safe, use for debug only
    std::string
    chainString() const;

    // head_ is the most recently used item, tail_ is the least recently used item.
    // tail_ -> next -> ... -> head_
    // tail_ <- prev <- ... <- head_
    ListNode* head_ = nullptr;
    ListNode* tail_ = nullptr;

    // TODO(tiered storage 3): benchmark folly::DistributedMutex for this usecase.
    mutable std::mutex list_mtx_;
    std::atomic<ResourceUsage> max_resource_limit_;
    std::atomic<ResourceUsage> low_watermark_;
    std::atomic<ResourceUsage> high_watermark_;
    const EvictionConfig eviction_config_;

    std::thread bg_eviction_thread_;
    std::condition_variable bg_eviction_thread_cv_;
    std::atomic<bool> stop_bg_eviction_loop_{false};

    // Waiting queue for timeout-based memory reservation
    std::priority_queue<std::unique_ptr<WaitingRequest>, std::vector<std::unique_ptr<WaitingRequest>>,
                        WaitingRequestComparator>
        waiting_queue_;
    // using a separate variable to avoid locking just to check if the queue is empty.
    std::atomic<bool> waiting_queue_empty_{true};

    // Quick lookup map for waiting requests (for timeout handling)
    std::unordered_map<uint64_t, WaitingRequest*> waiting_requests_map_;

    // Counter for generating unique request IDs
    std::atomic<uint64_t> next_request_id_;

    // Total size of nodes that are loaded and unpinned, used for eviction.
    std::atomic<ResourceUsage> evictable_size_{};

    // Total size of nodes that are loading and loaded, used for memory reservation.
    std::atomic<ResourceUsage> total_loading_size_{};
    std::atomic<ResourceUsage> total_loaded_size_{};

    // EventBase and thread for handling timeout operations
    std::unique_ptr<folly::ScopedEventBaseThread> event_base_thread_;
};

}  // namespace milvus::cachinglayer::internal
