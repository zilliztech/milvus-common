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
#include "cachinglayer/lrucache/DList.h"

#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "cachinglayer/Metrics.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/ListNode.h"
#include "log/Log.h"

namespace milvus::cachinglayer::internal {

// Helper to clamp an atomic<ResourceUsage> to non-negative with custom logging.
template <typename LogFn>
inline void
ClampNonNegative(std::atomic<ResourceUsage>& counter, LogFn&& log_fn) {
    auto current = counter.load();
    if (!current.AllGEZero()) {
        log_fn(current);
        counter = ResourceUsage{0, 0};
    }
}

folly::SemiFuture<bool>
DList::ReserveLoadingResourceWithTimeout(const ResourceUsage& original_size, std::chrono::milliseconds timeout) {
    // NOTE: we can reserve more loading resources than the original request size by adjusting the
    // loading_resource_factor to avoid potential problems from bad resource estimation.
    auto size = original_size * eviction_config_.loading_resource_factor;

    // First try immediate reservation
    {
        std::unique_lock<std::mutex> list_lock(list_mtx_);
        if (!max_resource_limit_.load().CanHold(size)) {
            LOG_ERROR("[MCL] Failed to reserve size={} as it exceeds max_memory_={}.", size.ToString(),
                      max_resource_limit_.load().ToString());
            return folly::makeSemiFuture(false);
        }
        if (reserveResourceInternal(size)) {
            return folly::makeSemiFuture(true);
        }
    }

    // If immediate reservation fails, add to waiting queue
    std::unique_lock<std::mutex> lock(list_mtx_);

    auto deadline = std::chrono::steady_clock::now() + timeout;
    auto [promise, future] = folly::makePromiseContract<bool>();

    uint64_t request_id = next_request_id_.fetch_add(1);

    auto waiting_request =
        std::make_unique<WaitingRequest>(size, deadline, std::move(promise), event_base_.get(), request_id);
    waiting_requests_map_[request_id] = waiting_request.get();
    waiting_queue_.push(std::move(waiting_request));
    waiting_queue_empty_ = false;

    LOG_DEBUG(
        "[MCL] Request {} size {} added to waiting queue, scheduling timeout "
        "in {}ms",
        request_id, size.ToString(), timeout.count());

    event_base_->runInEventBaseThread([this, request_id, timeout]() {
        event_base_->runAfterDelay(
            [this, request_id]() {
                std::unique_lock<std::mutex> lock(list_mtx_);
                auto it = waiting_requests_map_.find(request_id);
                if (it != waiting_requests_map_.end()) {
                    LOG_WARN(
                        "[MCL] Reserve Request {} of size {} timed out, "
                        "notifying failure.",
                        request_id, it->second->required_size.ToString());
                    it->second->promise.setValue(false);
                    waiting_requests_map_.erase(it);
                }
            },
            static_cast<uint32_t>(timeout.count()));
    });

    return std::move(future);
}

bool
DList::reserveResourceInternal(const ResourceUsage& size) {
    auto using_resources = total_loaded_size_.load() + total_loading_size_.load();

    // Combined logical and physical memory limit check
    bool logical_limit_exceeded = !max_resource_limit_.load().CanHold(using_resources + size);
    auto physical_eviction_needed = checkPhysicalResourceLimit(size);

    // If either limit is exceeded, attempt unified eviction
    // we attempt eviction based on logical limit once, but multiple times on physical limit
    // because physical eviction may not be accurate.
    while (logical_limit_exceeded || physical_eviction_needed.AnyGTZero()) {
        ResourceUsage eviction_target;
        ResourceUsage min_eviction;

        if (logical_limit_exceeded) {
            // Calculate logical eviction requirements
            eviction_target = using_resources + size - low_watermark_;
            min_eviction = using_resources + size - max_resource_limit_.load();

            // Ensure non-negative values
            if (eviction_target.memory_bytes < 0) {
                eviction_target.memory_bytes = 0;
            }
            if (eviction_target.file_bytes < 0) {
                eviction_target.file_bytes = 0;
            }
            if (min_eviction.memory_bytes < 0) {
                min_eviction.memory_bytes = 0;
            }
            if (min_eviction.file_bytes < 0) {
                min_eviction.file_bytes = 0;
            }
        }

        if (physical_eviction_needed.AnyGTZero()) {
            // Combine with logical eviction target (take the maximum)
            eviction_target.memory_bytes =
                std::max(eviction_target.memory_bytes, physical_eviction_needed.memory_bytes);
            eviction_target.file_bytes = std::max(eviction_target.file_bytes, physical_eviction_needed.file_bytes);
            min_eviction.memory_bytes = std::max(min_eviction.memory_bytes, physical_eviction_needed.memory_bytes);
            min_eviction.file_bytes = std::max(min_eviction.file_bytes, physical_eviction_needed.file_bytes);
        }

        // Attempt unified eviction
        ResourceUsage evicted_size = tryEvict(eviction_target, min_eviction);
        if (!evicted_size.AnyGTZero()) {
            LOG_WARN(
                "[MCL] reserve resource with size={} failed due to all zero evicted_size, "
                "eviction_target={}, min_eviction={}",
                size.ToString(), eviction_target.ToString(), min_eviction.ToString());
            return false;
        }
        // logical limit is accurate, thus we can guarantee after one successful eviction, logical limit is satisfied.
        logical_limit_exceeded = false;

        if (!physical_eviction_needed.AnyGTZero()) {
            // we only need to evict for logical limit and we have succeeded.
            break;
        }

        if (physical_eviction_needed = checkPhysicalResourceLimit(size); !physical_eviction_needed.AnyGTZero()) {
            // if after eviction we no longer need to evict, we can break.
            break;
        }
        // else perform another round of eviction.
        LOG_TRACE(
            "[MCL] reserve resource with size={} failed due to insufficient resources, "
            "evicted_size={}, still need to evict {}",
            size.ToString(), evicted_size.ToString(), physical_eviction_needed.ToString());
    }

    total_loading_size_ += size;
    LOG_TRACE("[MCL] reserve resource with size={} success, total_loading_size={}, total_loaded_size={}",
              size.ToString(), total_loading_size_.load().ToString(), total_loaded_size_.load().ToString());

    return true;
}

void
DList::evictionLoop() {
    while (true) {
        std::unique_lock<std::mutex> lock(list_mtx_);
        if (eviction_thread_cv_.wait_for(lock, eviction_config_.eviction_interval,
                                         [this] { return stop_eviction_loop_.load(); })) {
            break;
        }
        auto using_resources = total_loaded_size_.load() + total_loading_size_.load();
        // if usage is above high watermark, evict until low watermark is reached.
        auto eviction_target = ResourceUsage{
            using_resources.memory_bytes >= high_watermark_.load().memory_bytes
                ? using_resources.memory_bytes - low_watermark_.load().memory_bytes
                : 0,
            using_resources.file_bytes >= high_watermark_.load().file_bytes
                ? using_resources.file_bytes - low_watermark_.load().file_bytes
                : 0,
        };
        const auto min_eviction = ResourceUsage{0, 0};
        if (eviction_target.AnyGTZero()) {
            tryEvict(eviction_target, min_eviction, eviction_config_.cache_cell_unaccessed_survival_time.count() > 0);
        }
        notifyWaitingRequests();
    }
}

std::string
DList::usageInfo() const {
    auto using_resources = total_loaded_size_.load() + total_loading_size_.load();
    auto curr_max_resource_limit = max_resource_limit_.load();
    auto curr_high_watermark = high_watermark_.load();
    auto curr_low_watermark = low_watermark_.load();
    constexpr double precision = 100.0;
    std::string info = fmt::format(
        "low_watermark_: {}; high_watermark_: {}; "
        "max_resource_limit_: {}; using_resources_: {} (",
        curr_low_watermark.ToString(), curr_high_watermark.ToString(), curr_max_resource_limit.ToString(),
        using_resources.ToString());

    if (using_resources.memory_bytes > 0) {
        info += fmt::format(
            ", {:.2}% of max, {:.2}% of high_watermark memory",
            static_cast<double>(using_resources.memory_bytes) / curr_max_resource_limit.memory_bytes * precision,
            static_cast<double>(using_resources.memory_bytes) / curr_high_watermark.memory_bytes * precision);
    }

    if (using_resources.file_bytes > 0) {
        info += fmt::format(
            ", {:.2}% of max, {:.2}% of high_watermark disk",
            static_cast<double>(using_resources.file_bytes) / curr_max_resource_limit.file_bytes * precision,
            static_cast<double>(using_resources.file_bytes) / curr_high_watermark.file_bytes * precision);
    }

    info += fmt::format("); evictable_size_: {}; total_loaded_size_: {}; total_loading_size_: {}; ",
                        evictable_size_.load().ToString(), total_loaded_size_.load().ToString(),
                        total_loading_size_.load().ToString());

    return info;
}

// this method is not thread safe, it does not attempt to lock each node, use for debug only.
std::string
DList::chainString() const {
    std::stringstream ss;
    ss << "[MCL] DList chain: ";
    size_t num_nodes = 0;
    for (auto it = tail_; it != nullptr; it = it->next_) {
        ss << "(" << it->key() << ", " << it->loaded_size().ToString() << ", pins=" << it->pin_count_ << ")";
        num_nodes++;
        if (it->next_ != nullptr) {
            ss << " -> ";
        }
    }
    ss << "Total nodes: " << num_nodes << std::endl;
    return ss.str();
}

ResourceUsage
DList::tryEvict(const ResourceUsage& expected_eviction, const ResourceUsage& min_eviction,
                const bool evict_expired_items) {
    // Fast path: check if we have enough evictable resources
    auto current_evictable = evictable_size_.load();
    if (!current_evictable.CanHold(min_eviction)) {
        LOG_INFO(
            "[MCL] evictable_size {} cannot satisfy min_eviction {}, giving up "
            "eviction without traversing list. Current usage: {}",
            current_evictable.ToString(), min_eviction.ToString(), usageInfo());
        return ResourceUsage{0, 0};
    }

    std::vector<ListNode*> to_evict;
    to_evict.reserve(32);
    // items are evicted because they are not used for a while, thus it should be ok to lock them
    // a little bit longer.
    std::vector<std::unique_lock<std::shared_mutex>> item_locks;
    item_locks.reserve(32);

    ResourceUsage size_to_evict;

    auto would_help = [&](const ResourceUsage& size) -> bool {
        auto need_memory = size_to_evict.memory_bytes < expected_eviction.memory_bytes;
        auto need_disk = size_to_evict.file_bytes < expected_eviction.file_bytes;
        return (need_memory && size.memory_bytes > 0) || (need_disk && size.file_bytes > 0);
    };

    auto time_threshold =
        evict_expired_items ? (std::chrono::steady_clock::now() - eviction_config_.cache_cell_unaccessed_survival_time)
                            : std::chrono::steady_clock::time_point::min();

    bool first_node_checked = false;
    // current_node_time is initialized when first_node_checked is set to true.
    std::chrono::steady_clock::time_point current_node_time;

    for (ListNode* it = tail_; it != nullptr; it = it->next_) {
        bool need_lock = (!first_node_checked) || (current_node_time < time_threshold) || would_help(it->loaded_size());

        if (!need_lock) {
            continue;
        }

        auto& lock = item_locks.emplace_back(it->mtx_, std::try_to_lock);

        if (!lock.owns_lock()) {
            // Failed to acquire lock, node is being used, skip it
            item_locks.pop_back();
            continue;
        }

        // If node is pinned, cannot evict
        if (it->pin_count_ > 0) {
            item_locks.pop_back();
            continue;
        }

        current_node_time = it->last_touch_;
        first_node_checked = true;

        if (current_node_time < time_threshold || would_help(it->loaded_size())) {
            to_evict.push_back(it);
            size_to_evict += it->loaded_size();
        } else {
            // Release lock if node is neither expired nor helpful
            item_locks.pop_back();
        }

        // Check if we should stop traversing:
        // Stop if we've collected enough eviction size,
        // and either we are not evicting expired items or the current node is expired.
        if (size_to_evict.CanHold(expected_eviction) && (!evict_expired_items || current_node_time >= time_threshold)) {
            break;
        }
    }
    if (!size_to_evict.AnyGTZero()) {
        // Do not spam log during eviction loop.
        if (!evict_expired_items) {
            LOG_DEBUG(
                "[MCL] No items can be evicted, expected_eviction {}, "
                "min_eviction {}, giving up eviction. Current usage: {}",
                expected_eviction.ToString(), min_eviction.ToString(), usageInfo());
        }
        return ResourceUsage{0, 0};
    }
    if (!size_to_evict.CanHold(expected_eviction)) {
        if (!size_to_evict.CanHold(min_eviction)) {
            LOG_ERROR(
                "[MCL] Cannot evict even min_eviction {}, max possible "
                "eviction {}, giving up eviction. Current usage: {}. This "
                "should have been rejected at entry of this function. "
                "Something must be wrong",
                min_eviction.ToString(), size_to_evict.ToString(), usageInfo());
            LOG_TRACE("[MCL] DList chain: {}", chainString());
            return ResourceUsage{0, 0};
        }
        LOG_DEBUG(
            "[MCL] cannot evict expected_eviction {} but can evict "
            "min_eviction {}, evicting as much({}) as possible. Current usage: "
            "{}",
            expected_eviction.ToString(), min_eviction.ToString(), size_to_evict.ToString(), usageInfo());
    }

    cachinglayer::monitor::cache_eviction_event_total(size_to_evict.storage_type()).Increment();
    for (auto* list_node : to_evict) {
        popItem(list_node);   // must succeed, otherwise the node is not in the list
        list_node->unload();  // NOTE: after unload(), the node's loaded_size() is reset to {0, 0} and state_ is reset
                              // to NOT_LOADED.
        // if this cell is evicted, loaded, pinned and unpinned within a single refresh window,
        // the cell should be inserted into the LRU list again.
        list_node->last_touch_ = std::chrono::steady_clock::now() - 2 * eviction_config_.cache_touch_window;
    }
    // Refund logically evicted resources manually, waiting requests notification is left to caller.
    total_loaded_size_ -= size_to_evict;
    evictable_size_ -= size_to_evict;
    ClampNonNegative(total_loaded_size_, [&](const ResourceUsage& curr) {
        LOG_ERROR(
            "[MCL] total_loaded_size_ became negative after eviction: evicted_size={}, current_total_loaded={}, "
            "usage_info={}",
            size_to_evict.ToString(), curr.ToString(), usageInfo());
    });
    ClampNonNegative(evictable_size_, [&](const ResourceUsage& curr) {
        LOG_ERROR(
            "[MCL] evictable_size_ became negative after eviction: evicted_size={}, current_evictable={}, "
            "usage_info={}",
            size_to_evict.ToString(), curr.ToString(), usageInfo());
    });

    LOG_TRACE("[MCL] Logically evicted size: {}", size_to_evict.ToString());

    switch (size_to_evict.storage_type()) {
        case StorageType::MEMORY:
            cachinglayer::monitor::cache_evicted_bytes_total(StorageType::MEMORY).Increment(size_to_evict.memory_bytes);
            break;
        case StorageType::DISK:
            cachinglayer::monitor::cache_evicted_bytes_total(StorageType::DISK).Increment(size_to_evict.file_bytes);
            break;
        case StorageType::MIXED:
            cachinglayer::monitor::cache_evicted_bytes_total(StorageType::MEMORY).Increment(size_to_evict.memory_bytes);
            cachinglayer::monitor::cache_evicted_bytes_total(StorageType::DISK).Increment(size_to_evict.file_bytes);
            break;
        default:
            ThrowInfo(ErrorCode::UnexpectedError, "Unknown StorageType");
    }
    return size_to_evict;
}

bool
DList::UpdateMaxLimit(const ResourceUsage& new_limit) {
    AssertInfo((new_limit - high_watermark_.load()).AllGEZero(),
               "[MCL] limit must be greater than high watermark. new_limit: "
               "{}, high_watermark: {}",
               new_limit.ToString(), high_watermark_.load().ToString());
    std::unique_lock<std::mutex> list_lock(list_mtx_);
    auto using_resources = total_loaded_size_.load() + total_loading_size_.load();
    if (!new_limit.CanHold(using_resources)) {
        // positive means amount owed
        auto deficit = using_resources - new_limit;
        // deficit is the hard limit of eviction, if we cannot evict deficit, we give
        // up the limit change.
        if (!tryEvict(deficit, deficit).AnyGTZero()) {
            return false;
        }
    }
    LOG_INFO("[MCL] UpdateMaxLimit: from {} to {}", max_resource_limit_.load().ToString(), new_limit.ToString());
    max_resource_limit_ = new_limit;
    cachinglayer::monitor::cache_capacity_bytes(StorageType::MEMORY).Set(max_resource_limit_.load().memory_bytes);
    cachinglayer::monitor::cache_capacity_bytes(StorageType::DISK).Set(max_resource_limit_.load().file_bytes);
    return true;
}

void
DList::UpdateLowWatermark(const ResourceUsage& new_low_watermark) {
    std::unique_lock<std::mutex> list_lock(list_mtx_);
    AssertInfo(new_low_watermark.AllGEZero(),
               "[MCL] low watermark must be greater than or "
               "equal to 0. new_low_watermark: {}",
               new_low_watermark.ToString());
    AssertInfo((high_watermark_.load() - new_low_watermark).AllGEZero(),
               "[MCL] low watermark must be less than or equal to high "
               "watermark. new_low_watermark: {}, high_watermark: {}",
               new_low_watermark.ToString(), high_watermark_.load().ToString());
    LOG_INFO("[MCL] UpdateLowWatermark: from {} to {}", low_watermark_.load().ToString(), new_low_watermark.ToString());
    low_watermark_ = new_low_watermark;
    cachinglayer::monitor::cache_low_watermark_bytes(StorageType::MEMORY).Set(low_watermark_.load().memory_bytes);
    cachinglayer::monitor::cache_low_watermark_bytes(StorageType::DISK).Set(low_watermark_.load().file_bytes);
}

void
DList::UpdateHighWatermark(const ResourceUsage& new_high_watermark) {
    std::unique_lock<std::mutex> list_lock(list_mtx_);
    AssertInfo((new_high_watermark - low_watermark_.load()).AllGEZero(),
               "[MCL] high watermark must be greater than or "
               "equal to low watermark. new_high_watermark: {}, low_watermark: {}",
               new_high_watermark.ToString(), low_watermark_.load().ToString());
    AssertInfo((max_resource_limit_.load() - new_high_watermark).AllGEZero(),
               "[MCL] high watermark must be less than or equal to max "
               "resource limit. new_high_watermark: {}, max_resource_limit: {}",
               new_high_watermark.ToString(), max_resource_limit_.load().ToString());
    LOG_INFO("[MCL] UpdateHighWatermark: from {} to {}", high_watermark_.load().ToString(),
             new_high_watermark.ToString());
    high_watermark_ = new_high_watermark;
    cachinglayer::monitor::cache_high_watermark_bytes(StorageType::MEMORY).Set(high_watermark_.load().memory_bytes);
    cachinglayer::monitor::cache_high_watermark_bytes(StorageType::DISK).Set(high_watermark_.load().file_bytes);
}

void
DList::ReleaseLoadingResource(const ResourceUsage& loading_size) {
    auto size = loading_size * eviction_config_.loading_resource_factor;
    total_loading_size_ -= size;
    ClampNonNegative(total_loading_size_, [&](const ResourceUsage& curr) {
        LOG_ERROR(
            "[MCL] total_loading_size_ became negative after release: release_scaled={}, original_release={}, "
            "loading_resource_factor={}, current_total_loading={}",
            size.ToString(), loading_size.ToString(), eviction_config_.loading_resource_factor, curr.ToString());
    });
    // Notify waiting requests that resources are available
    std::unique_lock<std::mutex> lock(list_mtx_);
    notifyWaitingRequests();
}

void
DList::touchItem(ListNode* list_node, bool force_touch, std::optional<ResourceUsage> size) {
    // update evictable_size_ if size is provided
    if (size.has_value()) {
        evictable_size_ += size.value();
    }
    // check if the node should be moved forward
    auto now = std::chrono::steady_clock::now();
    if (!force_touch && now - list_node->last_touch_ <= eviction_config_.cache_touch_window) {
        return;
    }
    // move the node to the head of the list
    std::lock_guard<std::mutex> list_lock(list_mtx_);
    popItem(list_node);
    pushHead(list_node);
    // If there are waiters, try to satisfy them
    if (size.has_value() && !waiting_queue_empty_) {
        notifyWaitingRequests();
    }
    list_node->last_touch_ = now;
}

void
DList::removeItem(ListNode* list_node, ResourceUsage size) {
    std::lock_guard<std::mutex> list_lock(list_mtx_);
    if (popItem(list_node) && list_node->pin_count_ == 0) {
        evictable_size_ -= size;
        ClampNonNegative(evictable_size_, [&](const ResourceUsage& curr) {
            LOG_ERROR(
                "[MCL] evictable_size_ became negative after remove: removed_size={}, current_evictable={}, "
                "usage_info={}",
                size.ToString(), curr.ToString(), usageInfo());
        });
        // if the cell is evicted, loaded, pinned and unpinned within a single refresh window,
        // the cell should be inserted into the LRU list again.
        list_node->last_touch_ = std::chrono::steady_clock::now() - 2 * eviction_config_.cache_touch_window;
    }
}

void
DList::freezeItem(ListNode* list_node [[maybe_unused]], ResourceUsage size) {
    AssertInfo(list_node->pin_count_ > 0, "[MCL] freezeItem should be called on a cell with pin_count_ > 0, but got {}",
               list_node->pin_count_);
    evictable_size_ -= size;
    ClampNonNegative(evictable_size_, [&](const ResourceUsage& curr) {
        LOG_ERROR(
            "[MCL] evictable_size_ became negative after freeze: frozen_size={}, current_evictable={}, usage_info={}",
            size.ToString(), curr.ToString(), usageInfo());
    });
}

void
DList::pushHead(ListNode* list_node) {
    if (head_ == nullptr) {
        head_ = list_node;
        tail_ = list_node;
    } else {
        list_node->prev_ = head_;
        head_->next_ = list_node;
        head_ = list_node;
    }
}

bool
DList::popItem(ListNode* list_node) {
    if (list_node->prev_ == nullptr && list_node->next_ == nullptr && list_node != head_) {
        // list_node is not in the list
        return false;
    }
    if (head_ == tail_) {
        head_ = tail_ = nullptr;
        list_node->prev_ = list_node->next_ = nullptr;
    } else if (head_ == list_node) {
        head_ = list_node->prev_;
        head_->next_ = nullptr;
        list_node->prev_ = nullptr;
    } else if (tail_ == list_node) {
        tail_ = list_node->next_;
        tail_->prev_ = nullptr;
        list_node->next_ = nullptr;
    } else {
        list_node->prev_->next_ = list_node->next_;
        list_node->next_->prev_ = list_node->prev_;
        list_node->prev_ = list_node->next_ = nullptr;
    }
    return true;
}

bool
DList::IsEmpty() const {
    std::lock_guard<std::mutex> list_lock(list_mtx_);
    return head_ == nullptr;
}

void
DList::ChargeLoadedResource(const ResourceUsage& size) {
    total_loaded_size_ += size;
}

void
DList::RefundLoadedResource(const ResourceUsage& size) {
    total_loaded_size_ -= size;
    ClampNonNegative(total_loaded_size_, [&](const ResourceUsage& curr) {
        LOG_ERROR(
            "[MCL] total_loaded_size_ became negative after refund loaded resource: refund_size={}, "
            "current_total_loaded={}, "
            "usage_info={}",
            size.ToString(), curr.ToString(), usageInfo());
    });
    // Notify waiting requests that resources are available
    std::unique_lock<std::mutex> lock(list_mtx_);
    notifyWaitingRequests();
}

void
DList::notifyWaitingRequests() {
    while (!waiting_queue_.empty()) {
        auto& request_ptr_ref = const_cast<std::unique_ptr<WaitingRequest>&>(waiting_queue_.top());

        // Check if request has expired
        if (std::chrono::steady_clock::now() > request_ptr_ref->deadline) {
            // This request is expired. We will handle its cleanup here to avoid
            // a race with the timeout handler. We "claim" the request by
            // erasing it from the map.
            auto request = std::move(request_ptr_ref);
            waiting_queue_.pop();

            if (waiting_requests_map_.erase(request->request_id) > 0) {
                // If we successfully erased it, it means the timeout handler hasn't
                // run yet. We are now responsible for fulfilling the promise.
                LOG_DEBUG(
                    "[MCL] Request {} expired, cleaned up by "
                    "notifyWaitingRequests.",
                    request->request_id);
                request->event_base->runInEventBaseThread(
                    [promise = std::move(request->promise)]() mutable { promise.setValue(false); });
            }
            // If erase returned 0, the timeout handler ran first and claimed the
            // request. We don't need to do anything with the promise.
            continue;
        }

        if (reserveResourceInternal(request_ptr_ref->required_size)) {
            auto request = std::move(request_ptr_ref);
            waiting_queue_.pop();
            waiting_requests_map_.erase(request->request_id);

            // Success - notify the request
            request->event_base->runInEventBaseThread(
                [promise = std::move(request->promise), request_id = request->request_id]() mutable {
                    LOG_DEBUG("[MCL] Executing success notification for request {}", request_id);
                    promise.setValue(true);
                });
        } else {
            LOG_DEBUG("[MCL] Request {} of size {} cannot be satisfied, breaking.", request_ptr_ref->request_id,
                      request_ptr_ref->required_size.ToString());
            // Cannot satisfy even with eviction.
            // The largest/oldest obstacle is at the top of the queue.
            // No point trying for smaller requests.
            break;
        }
    }
    waiting_queue_empty_ = waiting_queue_.empty();
}

void
DList::clearWaitingQueue() {
    std::unique_lock<std::mutex> lock(list_mtx_);

    // Notify all waiting requests that they failed
    while (!waiting_queue_.empty()) {
        auto& request = waiting_queue_.top();
        try {
            request->promise.setValue(false);
        } catch (const std::exception& e) {
            LOG_WARN("[MCL] Failed to set value for request {}, error: {}", request->request_id, e.what());
        }
        waiting_queue_.pop();
    }

    waiting_queue_empty_ = true;
    waiting_requests_map_.clear();
}

ResourceUsage
DList::checkPhysicalResourceLimit(const ResourceUsage& size) const {
    static SystemResourceInfo infinity = {std::numeric_limits<int64_t>::max(), 0};
    auto sys_mem = size.memory_bytes > 0 ? getSystemMemoryInfo() : infinity;
    auto sys_disk = size.file_bytes > 0 ? getSystemDiskInfo(eviction_config_.disk_path) : infinity;

    auto used = ResourceUsage{sys_mem.used_bytes, sys_disk.used_bytes};
    auto current_loading = total_loading_size_.load();
    auto projected_usage = current_loading + size + used;

    auto limit = ResourceUsage{
        static_cast<int64_t>(sys_mem.total_bytes * eviction_config_.overloaded_memory_threshold_percentage),
        static_cast<int64_t>(sys_disk.total_bytes * eviction_config_.max_disk_usage_percentage)};

    auto eviction_needed = projected_usage - limit;
    if (eviction_needed.memory_bytes < 0) {
        eviction_needed.memory_bytes = 0;
    }
    if (eviction_needed.file_bytes < 0) {
        eviction_needed.file_bytes = 0;
    }

    LOG_TRACE(
        "[MCL] Physical resource check: "
        "projected_usage={}(used={}, loading={}, requesting={}), limit={} "
        "(mem {}% disk {}% of total {}), eviction_needed={}",
        projected_usage.ToString(), used.ToString(), current_loading.ToString(), size.ToString(), limit.ToString(),
        eviction_config_.overloaded_memory_threshold_percentage * 100, eviction_config_.max_disk_usage_percentage * 100,
        ResourceUsage{sys_mem.total_bytes, sys_disk.total_bytes}.ToString(), eviction_needed.ToString());

    return eviction_needed;
}

}  // namespace milvus::cachinglayer::internal
