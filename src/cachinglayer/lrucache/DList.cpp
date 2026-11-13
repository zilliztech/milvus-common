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

#include <folly/ScopeGuard.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <vector>

#include "cachinglayer/Metrics.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/ListNode.h"
#include "log/Log.h"
#include "syncpoint/sync_point.h"

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

namespace {

ResourceUsage
RequestLocalLoading(const LoadingResourceUsage& usage, const LoadingOverheadConfig* config) {
    auto local = usage.resident + usage.private_overhead;
    if (config == nullptr || !config->memory.has_value()) {
        local.memory_bytes += usage.shared_overhead.memory_bytes;
    }
    if (config == nullptr || !config->file.has_value()) {
        local.file_bytes += usage.shared_overhead.file_bytes;
    }
    return local;
}

}  // namespace

void
DList::completeWaitingRequests(std::vector<WaitingRequestCompletion>& completions) {
    for (auto& completion : completions) {
        if (completion.resource_promise.valid()) {
            completion.resource_promise.setValue(completion.result);
        } else if (completion.bool_promise.valid()) {
            completion.bool_promise.setValue(completion.result.success);
        }
    }
}

LoadingOverheadGroupHandle
DList::CreateLoadingOverheadGroup(LoadingOverheadDimension dimension, LoadingOverheadPolicy policy) {
    if (dimension != LoadingOverheadDimension::kMemory && dimension != LoadingOverheadDimension::kFile) {
        return nullptr;
    }
    return LoadingOverheadGroupHandle(new LoadingOverheadGroup(dimension, std::move(policy)));
}

void
DList::BindLoadingOverheadGroups(const LoadingOverheadConfig& config) {
    std::lock_guard<std::mutex> lock(list_mtx_);
    validateLoadingOverheadBinding(config.memory, LoadingOverheadDimension::kMemory);
    validateLoadingOverheadBinding(config.file, LoadingOverheadDimension::kFile);
    bool memory_bound = false;
    try {
        if (config.memory.has_value()) {
            config.memory->group->bind(config.memory->max_runtime_unit);
            memory_bound = true;
        }
        if (config.file.has_value()) {
            config.file->group->bind(config.file->max_runtime_unit);
        }
    } catch (...) {
        if (memory_bound) {
            config.memory->group->unbind(config.memory->max_runtime_unit);
        }
        throw;
    }
}

void
DList::UnbindLoadingOverheadGroups(const LoadingOverheadConfig& config) {
    {
        std::lock_guard<std::mutex> lock(list_mtx_);
        if (config.memory.has_value()) {
            if (config.memory->group) {
                config.memory->group->unbind(config.memory->max_runtime_unit);
            } else {
                LOG_ERROR("[MCL] LoadingOverheadGroup cannot unbind invalid memory Group");
            }
        }
        if (config.file.has_value()) {
            if (config.file->group) {
                config.file->group->unbind(config.file->max_runtime_unit);
            } else {
                LOG_ERROR("[MCL] LoadingOverheadGroup cannot unbind invalid file Group");
            }
        }
        if (reservation_notifications_needed_) {
            notifyWaitingRequests();
        }
    }
}

LoadingOverheadUpdateResult
DList::UpdateLoadingOverheadGroup(const LoadingOverheadGroupHandle& group, LoadingOverheadPolicy policy) {
    LoadingOverheadUpdateResult result;
    {
        std::unique_lock<std::mutex> lock(list_mtx_);
        result = group ? group->updatePolicy(std::move(policy)) : LoadingOverheadUpdateResult::kInvalidArgument;
        if (result == LoadingOverheadUpdateResult::kApplied) {
            notifyWaitingRequests();
        }
    }
    return result;
}

folly::SemiFuture<LoadingResourceReservationResult>
DList::ReserveLoadingResourceWithTimeout(const LoadingResourceUsage& usage, const LoadingOverheadConfig* config,
                                         std::chrono::milliseconds timeout, OpContext* ctx) {
    // 1. Hold the lock until we have reserved resources or added this request to the queue.
    // This prevents the background worker from checking the queue before our request is in it.
    std::unique_lock<std::mutex> lock(list_mtx_);
    // Resource releases do not always take this lock. Enable notifications before checking
    // resources, so a concurrent release wakes the worker even if we have not queued the request yet.
    reservation_notifications_needed_ = true;
    // Before returning and unlocking, disable notifications if no requests are left waiting.
    auto restore_notifications =
        folly::makeGuard([&] { reservation_notifications_needed_ = !waiting_requests_map_.empty(); });
    TEST_SYNC_POINT("DList::Reserve:BeforeAttempt");
    // 2. Try to reserve now, evicting cached data if needed. The helper accounts for overhead
    // shared with other requests in the same Group. On failure, it undoes that overhead reservation.
    auto attempt = reserveResourceInternalWithOverhead(usage, config);
    if (attempt.result.success) {
        return folly::makeSemiFuture(attempt.result);
    }

    // Fail without waiting if this request alone exceeds capacity. Do not include the extra
    // shared overhead here: another request may reserve it for the Group before we retry.
    const auto minimum_required_size = RequestLocalLoading(usage, config) * eviction_config_.loading_resource_factor;
    if (!max_resource_limit_.load().CanHold(minimum_required_size)) {
        LOG_ERROR(
            "[MCL] Failed to reserve Group-aware loading resource because the minimum requirement={} "
            "exceeds capacity={}",
            minimum_required_size.ToString(), max_resource_limit_.load().ToString());
        return folly::makeSemiFuture(LoadingResourceReservationResult{});
    }

    // timeout == 0 means the caller does not want to wait after the attempt above fails.
    if (timeout.count() == 0) {
        return folly::makeSemiFuture(LoadingResourceReservationResult{});
    }

    // 3. Resources are unavailable, so add the request to the waiting queue before unlocking.
    // The queue decides retry order. The map records which requests still need a result;
    // once a request is removed from the map, later timeout/cancel callbacks must ignore it.
    // A negative timeout means wait indefinitely, behind requests with a finite deadline.
    TEST_SYNC_POINT("DList::Reserve:BeforeEnqueue");
    auto deadline =
        timeout.count() > 0 ? std::chrono::steady_clock::now() + timeout : std::chrono::steady_clock::time_point::max();
    auto [promise, future] = folly::makePromiseContract<LoadingResourceReservationResult>();
    uint64_t request_id = next_request_id_.fetch_add(1);

    auto waiting_request = std::make_unique<WaitingRequest>(attempt.required_size, usage, config, deadline,
                                                            std::move(promise), request_id);
    WaitingRequest* request_ptr = waiting_request.get();
    waiting_requests_map_[request_id] = request_ptr;
    waiting_queue_.push(std::move(waiting_request));
    // A pending callback must not keep DList alive; it exits if DList has been destroyed.
    std::weak_ptr<DList> weak_self = shared_from_this();

    // 4. For a positive timeout, ask the EventBase thread to fail the request after the delay.
    // If the request has already succeeded or been cancelled, failWaitingRequest does nothing.
    // Record failure under the lock, then deliver the result after unlocking: delivering a result
    // can run caller code that needs this same lock.
    if (timeout.count() > 0) {
        LOG_DEBUG("[MCL] Request {} resident={} private={} shared={} added to waiting queue, timeout={}ms", request_id,
                  usage.resident.ToString(), usage.private_overhead.ToString(), usage.shared_overhead.ToString(),
                  timeout.count());

        event_base_thread_->getEventBase()->runInEventBaseThread([weak_self, request_id, timeout]() {
            auto self = weak_self.lock();
            if (!self) {
                return;
            }
            self->event_base_thread_->getEventBase()->runAfterDelay(
                [weak_self, request_id]() {
                    auto self = weak_self.lock();
                    if (!self) {
                        return;
                    }
                    std::vector<WaitingRequestCompletion> completions;
                    auto complete = folly::makeGuard([&] { completeWaitingRequests(completions); });
                    {
                        std::unique_lock<std::mutex> lock(self->list_mtx_);
                        self->failWaitingRequest(request_id, "timed out", completions);
                    }
                },
                static_cast<uint32_t>(timeout.count()));
        });
    }

    // 5. Let the caller cancel while waiting. If the token is already cancelled, emplace()
    // may run this callback immediately while we still hold list_mtx_. Send the work to EventBase
    // so the callback does not try to take the same lock again and deadlock.
    if (ctx && ctx->cancellation_token.canBeCancelled()) {
        request_ptr->cancel_cb.emplace(ctx->cancellation_token, [weak_self, request_id]() {
            auto self = weak_self.lock();
            if (!self) {
                return;
            }
            self->event_base_thread_->getEventBase()->runInEventBaseThread([weak_self, request_id]() {
                auto self = weak_self.lock();
                if (!self) {
                    return;
                }
                std::vector<WaitingRequestCompletion> completions;
                auto complete = folly::makeGuard([&] { completeWaitingRequests(completions); });
                {
                    std::unique_lock<std::mutex> lock(self->list_mtx_);
                    self->failWaitingRequest(request_id, "cancelled", completions);
                }
            });
        });
    }

    // 6. Return a future now; this function does not wait for resources.
    // The background worker retries the queued request. The caller receives success when resources
    // are reserved, or failure if the request times out, is cancelled, or DList shuts down.
    return std::move(future);
}

folly::SemiFuture<bool>
DList::ReserveLoadingResourceWithTimeout(const ResourceUsage& original_size, std::chrono::milliseconds timeout,
                                         OpContext* ctx) {
    // NOTE: we can reserve more loading resources than the original request size by adjusting the
    // loading_resource_factor to avoid potential problems from bad resource estimation.
    const auto size = original_size * eviction_config_.loading_resource_factor;

    // 1. Hold the lock until we have reserved resources or added this request to the queue.
    // This prevents the background worker from checking the queue before our request is in it.
    std::unique_lock<std::mutex> lock(list_mtx_);
    // Resource releases do not always take this lock. Enable notifications before checking
    // resources, so a concurrent release wakes the worker even if we have not queued the request yet.
    reservation_notifications_needed_ = true;
    // Before returning and unlocking, disable notifications if no requests are left waiting.
    auto restore_notifications =
        folly::makeGuard([&] { reservation_notifications_needed_ = !waiting_requests_map_.empty(); });
    TEST_SYNC_POINT("DList::Reserve:BeforeAttempt");
    // 2. Fail without waiting if the request exceeds capacity. Otherwise try to reserve now,
    // evicting cached data if needed. On success, return a future that already contains true.
    if (!max_resource_limit_.load().CanHold(size)) {
        LOG_ERROR("[MCL] Failed to reserve size={} as it exceeds max_memory_={}.", size.ToString(),
                  max_resource_limit_.load().ToString());
        return folly::makeSemiFuture(false);
    }
    if (reserveResourceInternal(size)) {
        return folly::makeSemiFuture(true);
    }

    // timeout == 0 means the caller does not want to wait after the attempt above fails.
    if (timeout.count() == 0) {
        return folly::makeSemiFuture(false);
    }

    // 3. Resources are unavailable, so add the request to the waiting queue before unlocking.
    // The queue decides retry order. The map records which requests still need a result;
    // once a request is removed from the map, later timeout/cancel callbacks must ignore it.
    // A negative timeout means wait indefinitely, behind requests with a finite deadline.
    TEST_SYNC_POINT("DList::Reserve:BeforeEnqueue");
    auto deadline =
        timeout.count() > 0 ? std::chrono::steady_clock::now() + timeout : std::chrono::steady_clock::time_point::max();
    auto [promise, future] = folly::makePromiseContract<bool>();

    uint64_t request_id = next_request_id_.fetch_add(1);

    auto waiting_request = std::make_unique<WaitingRequest>(size, deadline, std::move(promise), request_id);
    WaitingRequest* request_ptr = waiting_request.get();
    waiting_requests_map_[request_id] = request_ptr;
    waiting_queue_.push(std::move(waiting_request));
    // A pending callback must not keep DList alive; it exits if DList has been destroyed.
    std::weak_ptr<DList> weak_self = shared_from_this();

    // 4. For a positive timeout, ask the EventBase thread to fail the request after the delay.
    // If the request has already succeeded or been cancelled, failWaitingRequest does nothing.
    // Record failure under the lock, then deliver the result after unlocking: delivering a result
    // can run caller code that needs this same lock.
    if (timeout.count() > 0) {
        LOG_DEBUG(
            "[MCL] Request {} size {} added to waiting queue, scheduling timeout "
            "in {}ms",
            request_id, size.ToString(), timeout.count());

        event_base_thread_->getEventBase()->runInEventBaseThread([weak_self, request_id, timeout]() {
            auto self = weak_self.lock();
            if (!self) {
                return;  // DList already destroyed
            }
            self->event_base_thread_->getEventBase()->runAfterDelay(
                [weak_self, request_id]() {
                    auto self = weak_self.lock();
                    if (!self) {
                        return;  // DList already destroyed
                    }
                    std::vector<WaitingRequestCompletion> completions;
                    auto complete = folly::makeGuard([&] { completeWaitingRequests(completions); });
                    {
                        std::unique_lock<std::mutex> lock(self->list_mtx_);
                        self->failWaitingRequest(request_id, "timed out", completions);
                    }
                },
                static_cast<uint32_t>(timeout.count()));
        });
    }

    // 5. Let the caller cancel while waiting. If the token is already cancelled, emplace()
    // may run this callback immediately while we still hold list_mtx_. Send the work to EventBase
    // so the callback does not try to take the same lock again and deadlock.
    if (ctx && ctx->cancellation_token.canBeCancelled()) {
        request_ptr->cancel_cb.emplace(ctx->cancellation_token, [weak_self, request_id]() {
            auto self = weak_self.lock();
            if (!self) {
                return;  // DList already destroyed
            }
            self->event_base_thread_->getEventBase()->runInEventBaseThread([weak_self, request_id]() {
                auto self = weak_self.lock();
                if (!self) {
                    return;  // DList already destroyed
                }
                std::vector<WaitingRequestCompletion> completions;
                auto complete = folly::makeGuard([&] { completeWaitingRequests(completions); });
                {
                    std::unique_lock<std::mutex> lock(self->list_mtx_);
                    self->failWaitingRequest(request_id, "cancelled", completions);
                }
            });
        });
    }

    // 6. Return a future now; this function does not wait for resources.
    // The background worker retries the queued request. The caller receives success when resources
    // are reserved, or failure if the request times out, is cancelled, or DList shuts down.
    return std::move(future);
}

bool
DList::reserveResourceInternal(const ResourceUsage& size) {
    auto rollback = []() {};
    auto [success, _] = reserveResourceInternalImpl(size, rollback);
    return success;
}

void
DList::validateLoadingOverheadBinding(const std::optional<LoadingOverheadGroupBinding>& binding,
                                      LoadingOverheadDimension dimension) const {
    if (!binding.has_value()) {
        return;
    }
    if (!binding->group) {
        ThrowInfo(milvus::ErrorCode::InvalidParameter,
                  "loading-overhead binding requires a Group from the matching dimension");
    }
    binding->group->validateBinding(dimension, binding->max_runtime_unit);
}

DList::LoadingOverheadDelta
DList::reserveLoadingOverhead(const LoadingOverheadConfig& config, const ResourceUsage& overhead) {
    if (!overhead.AllGEZero()) {
        ThrowInfo(milvus::ErrorCode::InvalidParameter, "loading overhead must be non-negative");
    }

    validateLoadingOverheadBinding(config.memory, LoadingOverheadDimension::kMemory);
    validateLoadingOverheadBinding(config.file, LoadingOverheadDimension::kFile);

    if (config.memory.has_value()) {
        config.memory->group->validateReserve(overhead.memory_bytes);
    }
    if (config.file.has_value()) {
        config.file->group->validateReserve(overhead.file_bytes);
    }

    LoadingOverheadDelta delta{.unscaled_delta = overhead};
    ResourceUsage previous;
    ResourceUsage current;
    if (config.memory.has_value()) {
        previous.memory_bytes = config.memory->group->overhead_reserved_;
        delta.unscaled_delta.memory_bytes = config.memory->group->reserve(overhead.memory_bytes);
        current.memory_bytes = config.memory->group->overhead_reserved_;
    }
    if (config.file.has_value()) {
        previous.file_bytes = config.file->group->overhead_reserved_;
        delta.unscaled_delta.file_bytes = config.file->group->reserve(overhead.file_bytes);
        current.file_bytes = config.file->group->overhead_reserved_;
    }
    delta.scaled_group_delta =
        current * eviction_config_.loading_resource_factor - previous * eviction_config_.loading_resource_factor;
    return delta;
}

void
DList::rollbackLoadingOverhead(const LoadingOverheadConfig& config, const ResourceUsage& overhead,
                               const ResourceUsage& reserved) noexcept {
    if (config.memory.has_value() && config.memory->group) {
        config.memory->group->rollbackReserve(overhead.memory_bytes, reserved.memory_bytes);
    }
    if (config.file.has_value() && config.file->group) {
        config.file->group->rollbackReserve(overhead.file_bytes, reserved.file_bytes);
    }
}

DList::LoadingOverheadDelta
DList::releaseLoadingOverhead(const LoadingOverheadConfig& config, const ResourceUsage& overhead) {
    validateLoadingOverheadBinding(config.memory, LoadingOverheadDimension::kMemory);
    validateLoadingOverheadBinding(config.file, LoadingOverheadDimension::kFile);

    LoadingOverheadDelta delta{.unscaled_delta = overhead};
    ResourceUsage previous;
    ResourceUsage current;
    if (config.memory.has_value()) {
        previous.memory_bytes = config.memory->group->overhead_reserved_;
        delta.unscaled_delta.memory_bytes = config.memory->group->release(overhead.memory_bytes);
        current.memory_bytes = config.memory->group->overhead_reserved_;
    }
    if (config.file.has_value()) {
        previous.file_bytes = config.file->group->overhead_reserved_;
        delta.unscaled_delta.file_bytes = config.file->group->release(overhead.file_bytes);
        current.file_bytes = config.file->group->overhead_reserved_;
    }
    delta.scaled_group_delta =
        previous * eviction_config_.loading_resource_factor - current * eviction_config_.loading_resource_factor;
    return delta;
}

DList::LoadingResourceReservationAttempt
DList::reserveResourceInternalWithOverhead(const LoadingResourceUsage& usage, const LoadingOverheadConfig* config) {
    const auto delta = config != nullptr ? reserveLoadingOverhead(*config, usage.shared_overhead)
                                         : LoadingOverheadDelta{.unscaled_delta = usage.shared_overhead};

    auto rollback = [this, config, shared = usage.shared_overhead, delta]() {
        if (config != nullptr) {
            rollbackLoadingOverhead(*config, shared, delta.unscaled_delta);
        }
    };

    const auto unscaled = usage.resident + usage.private_overhead + delta.unscaled_delta;
    const auto scaled =
        RequestLocalLoading(usage, config) * eviction_config_.loading_resource_factor + delta.scaled_group_delta;

    if (!max_resource_limit_.load().CanHold(scaled)) {
        rollback();
        return {.result = LoadingResourceReservationResult{}, .required_size = scaled};
    }

    auto [success, _] = reserveResourceInternalImpl(scaled, rollback);
    if (!success) {
        return {.result = LoadingResourceReservationResult{}, .required_size = scaled};
    }

    LOG_TRACE(
        "[MCL] reserve with loading-overhead Groups: resident={}, private={}, shared={}, delta={}, unscaled={}, "
        "scaled={}, "
        "total_loading={}",
        usage.resident.ToString(), usage.private_overhead.ToString(), usage.shared_overhead.ToString(),
        delta.unscaled_delta.ToString(), unscaled.ToString(), scaled.ToString(), total_loading_size_.load().ToString());
    return {.result = LoadingResourceReservationResult{.success = true, .reserved = unscaled}, .required_size = scaled};
}

std::pair<bool, ResourceUsage>
DList::reserveResourceInternalImpl(const ResourceUsage& size, std::function<void()> rollback) {
    TEST_SYNC_POINT("DList::ReserveResourceInternal");
    const auto using_resources = total_loaded_size_.load() + total_loading_size_.load();
    const auto required_resources = using_resources + size;

    // Combined logical and physical memory limit check
    bool logical_limit_exceeded = !max_resource_limit_.load().CanHold(required_resources);
    auto physical_eviction_needed = checkPhysicalMemoryLimit(size);
    if (physical_eviction_needed.AnyGTZero() && !logical_limit_exceeded) {
        waiting_for_physical_memory_ = true;
    }

    // Try once, then retry at most twice if actual memory usage is still too high.
    // A fixed bound prevents this request from holding list_mtx_ through unlimited eviction rounds.
    constexpr int kMaxEvictionRetries = 2;
    for (int attempt = 0;
         attempt <= kMaxEvictionRetries && (logical_limit_exceeded || physical_eviction_needed.AnyGTZero());
         ++attempt) {
        ResourceUsage eviction_target;
        ResourceUsage min_eviction;

        if (logical_limit_exceeded) {
            // Calculate logical eviction requirements
            eviction_target = required_resources - low_watermark_;
            min_eviction = required_resources - max_resource_limit_.load();

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
            rollback();
            return {false, {}};
        }

        // Successful eviction freed at least min_eviction, so the logical quota is now sufficient.
        // Further rounds must use the newly read physical deficit, not the original logical deficit.
        logical_limit_exceeded = false;
        if (physical_eviction_needed.AnyGTZero()) {
            physical_eviction_needed = checkPhysicalMemoryLimit(size);
            waiting_for_physical_memory_ |= physical_eviction_needed.AnyGTZero();
        }
    }

    if (physical_eviction_needed.AnyGTZero()) {
        rollback();
        return {false, {}};
    }

    total_loading_size_ += size;
    LOG_TRACE("[MCL] reserve resource success, size={}, total_loading_size={}, total_loaded_size={}", size.ToString(),
              total_loading_size_.load().ToString(), total_loaded_size_.load().ToString());

    return {true, size};
}

void
DList::evictionLoop() {
    // Use 500ms for a nonpositive interval so physical-memory waiters still get periodic retries.
    const auto eviction_interval = eviction_config_.eviction_interval.count() > 0 ? eviction_config_.eviction_interval
                                                                                  : std::chrono::milliseconds(500);
    auto next_eviction = std::chrono::steady_clock::now() + eviction_interval;
    bool scanning = false;
    bool restoring = false;
    bool retry_requested = false;
    // Keep requests that could not reserve resources out of the queue until this scan finishes,
    // so the next batch can try other requests with the same deadline instead of retrying these.
    std::vector<std::unique_ptr<WaitingRequest>> deferred_requests;
    std::optional<std::chrono::steady_clock::time_point> blocked_deadline;
    while (true) {
        // 1. When no batch is in progress, wait for a notification or the next periodic check.
        // Between batches, continue immediately but still check whether shutdown was requested.
        {
            std::unique_lock lock(bg_eviction_mtx_);
            if (!scanning && !restoring) {
                bg_eviction_thread_cv_.wait_until(lock, next_eviction, [this, &retry_requested] {
                    return stop_bg_eviction_loop_ || waiting_requests_pending_ || retry_requested;
                });
                // Several notifications need only one scan. Notifications received during a scan
                // stay in waiting_requests_pending_ and cause another scan after the current one.
                retry_requested |= waiting_requests_pending_;
                waiting_requests_pending_ = false;
            }
            if (stop_bg_eviction_loop_) {
                break;
            }
        }

        std::vector<WaitingRequestCompletion> completions;
        {
            std::unique_lock lock(list_mtx_);
            // 2. On each periodic check, retry requests blocked by physical memory: other parts of
            // the process may have freed memory without notifying DList. Do this even if periodic
            // eviction is disabled. Requests blocked only by DList's quota wait for a notification.
            if (std::chrono::steady_clock::now() >= next_eviction) {
                // shortcut: Physical retries scan the shared queue; split it if mixed-queue retries become costly.
                retry_requested |= waiting_for_physical_memory_;
                if (eviction_config_.background_eviction_enabled) {
                    // At or above the high watermark, try to evict down to the low watermark.
                    // If eviction frees resources, also retry the waiting requests.
                    const auto using_resources = total_loaded_size_.load() + total_loading_size_.load();
                    const auto eviction_target = ResourceUsage{
                        using_resources.memory_bytes >= high_watermark_.load().memory_bytes
                            ? using_resources.memory_bytes - low_watermark_.load().memory_bytes
                            : 0,
                        using_resources.file_bytes >= high_watermark_.load().file_bytes
                            ? using_resources.file_bytes - low_watermark_.load().file_bytes
                            : 0,
                    };
                    retry_requested |=
                        tryEvict(eviction_target, {}, eviction_config_.cache_cell_unaccessed_survival_time.count() > 0)
                            .AnyGTZero();
                }
                next_eviction = std::chrono::steady_clock::now() + eviction_interval;
            }
            // 3. Start a requested scan once the previous scan and reinsertion are finished.
            // Clear the physical-memory flag; reservation attempts set it again if still blocked.
            if (!scanning && !restoring && retry_requested) {
                retry_requested = false;
                waiting_for_physical_memory_ = false;
                scanning = true;
            }
            // 4. Process one batch under list_mtx_, then unlock before the next batch so other
            // threads can use the list. Both reservation attempts and reinsertion are limited to 64.
            if (restoring) {
                // Put unsuccessful requests back only after the scan has finished.
                for (size_t i = 0; i < 64 && !deferred_requests.empty(); ++i) {
                    waiting_queue_.push(std::move(deferred_requests.back()));
                    deferred_requests.pop_back();
                }
                restoring = !deferred_requests.empty();
            } else if (scanning) {
                scanning = handleWaitingRequests(completions, deferred_requests, blocked_deadline);
                if (!scanning) {
                    restoring = !deferred_requests.empty();
                }
            }
            if (!scanning && !restoring) {
                blocked_deadline.reset();
            }
            // Resource releases need to wake this worker only while requests are still waiting.
            reservation_notifications_needed_ = !waiting_requests_map_.empty();
        }
        // 5. Deliver results after unlocking, on the EventBase thread. Delivering a result can run
        // caller code that destroys DList. Its destructor waits for this worker to exit, so doing
        // that here on the worker itself would make it wait for its own exit.
        if (!completions.empty()) {
            event_base_thread_->getEventBase()->runInEventBaseThread(
                [completions = std::move(completions)]() mutable { completeWaitingRequests(completions); });
        }
        bool more_batches = scanning || restoring;
        TEST_SYNC_POINT_CALLBACK("DList::WaitingBatchComplete", &more_batches);
    }
    // 6. Before exiting, return any temporarily removed requests to the queue. Shutdown waits for
    // this worker to exit, then clears the queue and reports failure to all remaining callers.
    std::lock_guard lock(list_mtx_);
    for (auto& request : deferred_requests) {
        waiting_queue_.push(std::move(request));
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
        "max_resource_limit_: {}; total_loaded_size_: {}; total_loading_size_: {}; using_resources_: {} (",
        curr_low_watermark.ToString(), curr_high_watermark.ToString(), curr_max_resource_limit.ToString(),
        total_loaded_size_.load().ToString(), total_loading_size_.load().ToString(), using_resources.ToString());

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
        if (!evict_expired_items && expected_eviction.AnyGTZero()) {
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
    cachinglayer::monitor::cache_evicted_bytes_total(StorageType::MEMORY).Increment(size_to_evict.memory_bytes);
    cachinglayer::monitor::cache_evicted_bytes_total(StorageType::DISK).Increment(size_to_evict.file_bytes);

    return size_to_evict;
}

bool
DList::UpdateMaxLimit(const ResourceUsage& new_limit) {
    AssertInfo((new_limit - high_watermark_.load()).AllGEZero(),
               "[MCL] limit must be greater than high watermark. new_limit: "
               "{}, high_watermark: {}",
               new_limit.ToString(), high_watermark_.load().ToString());
    {
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
        notifyWaitingRequests();
    }
    return true;
}

void
DList::UpdateLowWatermark(const ResourceUsage& new_low_watermark) {
    {
        std::unique_lock<std::mutex> list_lock(list_mtx_);
        AssertInfo(new_low_watermark.AllGEZero(),
                   "[MCL] low watermark must be greater than or "
                   "equal to 0. new_low_watermark: {}",
                   new_low_watermark.ToString());
        AssertInfo((high_watermark_.load() - new_low_watermark).AllGEZero(),
                   "[MCL] low watermark must be less than or equal to high "
                   "watermark. new_low_watermark: {}, high_watermark: {}",
                   new_low_watermark.ToString(), high_watermark_.load().ToString());
        LOG_INFO("[MCL] UpdateLowWatermark: from {} to {}", low_watermark_.load().ToString(),
                 new_low_watermark.ToString());
        low_watermark_ = new_low_watermark;
        cachinglayer::monitor::cache_low_watermark_bytes(StorageType::MEMORY).Set(low_watermark_.load().memory_bytes);
        cachinglayer::monitor::cache_low_watermark_bytes(StorageType::DISK).Set(low_watermark_.load().file_bytes);
        notifyWaitingRequests();
    }
}

void
DList::UpdateHighWatermark(const ResourceUsage& new_high_watermark) {
    {
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
        notifyWaitingRequests();
    }
}

ResourceUsage
DList::ReleaseLoadingResource(const LoadingResourceUsage& usage, const LoadingOverheadConfig* config) {
    LoadingOverheadDelta delta{.unscaled_delta = usage.shared_overhead};
    ResourceUsage unscaled;
    {
        std::unique_lock<std::mutex> lock(list_mtx_);
        if (config != nullptr) {
            delta = releaseLoadingOverhead(*config, usage.shared_overhead);
        }
        unscaled = usage.resident + usage.private_overhead + delta.unscaled_delta;
        const auto scaled =
            RequestLocalLoading(usage, config) * eviction_config_.loading_resource_factor + delta.scaled_group_delta;
        total_loading_size_ -= scaled;
        ClampNonNegative(total_loading_size_, [&](const ResourceUsage& curr) {
            LOG_ERROR(
                "[MCL] total_loading_size_ negative after Group release: resident={}, private={}, shared={}, "
                "current={}",
                usage.resident.ToString(), usage.private_overhead.ToString(), usage.shared_overhead.ToString(),
                curr.ToString());
        });
        notifyWaitingRequests();
    }
    return unscaled;
}

void
DList::ReleaseLoadingResource(const ResourceUsage& loading_size) {
    const auto scaled_size = loading_size * eviction_config_.loading_resource_factor;
    total_loading_size_ -= scaled_size;
    ClampNonNegative(total_loading_size_, [&](const ResourceUsage& curr) {
        LOG_ERROR(
            "[MCL] total_loading_size_ became negative after release: release_scaled={}, original_release={}, "
            "loading_resource_factor={}, current_total_loading={}",
            scaled_size.ToString(), loading_size.ToString(), eviction_config_.loading_resource_factor, curr.ToString());
    });
    // Notify waiting requests that resources are available
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
               list_node->pin_count_.load());
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
    notifyWaitingRequests();
}

void
DList::notifyWaitingRequests() {
    if (!reservation_notifications_needed_) {
        return;
    }
    {
        std::lock_guard lock(bg_eviction_mtx_);
        waiting_requests_pending_ = true;
    }
    bg_eviction_thread_cv_.notify_one();
}

bool
DList::handleWaitingRequests(std::vector<WaitingRequestCompletion>& completions,
                             std::vector<std::unique_ptr<WaitingRequest>>& deferred_requests,
                             std::optional<std::chrono::steady_clock::time_point>& blocked_deadline) {
    constexpr size_t kBatchSize = 64;
    completions.reserve(std::min(kBatchSize, waiting_queue_.size()));
    size_t processed = 0;
    while (!waiting_queue_.empty()) {
        if (blocked_deadline.has_value() && waiting_queue_.top()->deadline != blocked_deadline.value()) {
            break;
        }
        if (processed++ == kBatchSize) {
            return true;
        }
        // Allocate completion storage before claiming a request or reserving its resources.
        auto& completion = completions.emplace_back();
        auto& request_ptr_ref = const_cast<std::unique_ptr<WaitingRequest>&>(waiting_queue_.top());

        // Check if request was already handled by timeout/cancel (not in map anymore)
        if (waiting_requests_map_.find(request_ptr_ref->request_id) == waiting_requests_map_.end()) {
            LOG_DEBUG("[MCL] Request {} has already been handled, destroying and continuing.",
                      request_ptr_ref->request_id);
            completion.retired = std::move(request_ptr_ref);
            waiting_queue_.pop();
            continue;
        }

        // Check if request has expired
        if (std::chrono::steady_clock::now() > request_ptr_ref->deadline) {
            // This request is expired. We will handle its cleanup here to avoid
            // a race with the timeout handler. We "claim" the request by
            // erasing it from the map.
            auto& request = completion.retired;
            request = std::move(request_ptr_ref);
            waiting_queue_.pop();

            if (waiting_requests_map_.erase(request->request_id) > 0) {
                // If we successfully erased it, it means the timeout/cancel handler hasn't
                // run yet. We are now responsible for fulfilling the promise.
                completion.SetResult(*request, false);
                LOG_DEBUG(
                    "[MCL] Request {} expired, cleaned up by "
                    "handleWaitingRequests.",
                    request->request_id);
            }
            // If erase returned 0, the timeout/cancel handler ran first and claimed the
            // request. We don't need to do anything with the promise.

            continue;
        }

        // Try to fulfill: Group-aware path computes delta under lock;
        // request-local path uses required_size directly.
        //
        // NOTE: A Group-aware request's required_size is a cached, best-effort heap key. Dynamic policy or aggregate
        // state changes may make it stale while queued. If the stale heap top still fits, we intentionally admit it
        // without rebuilding all equal-deadline peers; strict size ordering would require repeated O(N) refreshes as
        // each admission changes shared Group state. A failed attempt refreshes the key and scans equal-deadline peers.
        bool fulfilled = false;
        ResourceUsage actual{};
        auto attempted_requirement = request_ptr_ref->required_size;
        if (request_ptr_ref->use_resource_promise) {
            try {
                auto attempt = reserveResourceInternalWithOverhead(request_ptr_ref->usage,
                                                                   request_ptr_ref->loadingOverheadConfig());
                fulfilled = attempt.result.success;
                actual = attempt.result.reserved;
                attempted_requirement = attempt.required_size;
            } catch (const milvus::SegcoreError& error) {
                if (error.get_error_code() != milvus::ErrorCode::InvalidParameter) {
                    throw;
                }
                auto& request = completion.retired;
                request = std::move(request_ptr_ref);
                waiting_queue_.pop();
                if (waiting_requests_map_.erase(request->request_id) > 0) {
                    completion.SetResult(*request, false);
                    LOG_WARN("[MCL] Request {} failed while retrying loading-overhead reservation: {}",
                             request->request_id, error.what());
                }
                continue;
            }
        } else {
            if (reserveResourceInternal(request_ptr_ref->required_size)) {
                fulfilled = true;
                actual = request_ptr_ref->required_size;
            }
        }
        if (fulfilled) {
            auto& request = completion.retired;
            request = std::move(request_ptr_ref);
            waiting_queue_.pop();

            if (waiting_requests_map_.erase(request->request_id) > 0) {
                completion.SetResult(*request, true, actual);
                LOG_DEBUG("[MCL] Reserved resources for request {}", request->request_id);
            } else {
                // Request was already handled by timeout/cancel, rollback.
                LOG_WARN("[MCL] Request {} was already handled by timeout/cancel, rolling back.", request->request_id);
                if (request->use_resource_promise) {
                    const auto* config = request->loadingOverheadConfig();
                    auto delta = LoadingOverheadDelta{.unscaled_delta = request->usage.shared_overhead};
                    if (config != nullptr) {
                        delta = releaseLoadingOverhead(*config, request->usage.shared_overhead);
                    }
                    const auto scaled =
                        RequestLocalLoading(request->usage, config) * eviction_config_.loading_resource_factor +
                        delta.scaled_group_delta;
                    total_loading_size_ -= scaled;
                } else {
                    total_loading_size_ -= actual;
                }
            }
        } else {
            const auto required_size =
                request_ptr_ref->use_resource_promise ? attempted_requirement : request_ptr_ref->required_size;
            const auto minimum_required_size =
                request_ptr_ref->use_resource_promise
                    ? RequestLocalLoading(request_ptr_ref->usage, request_ptr_ref->loadingOverheadConfig()) *
                          eviction_config_.loading_resource_factor
                    : required_size;
            if (!max_resource_limit_.load().CanHold(minimum_required_size)) {
                auto& request = completion.retired;
                request = std::move(request_ptr_ref);
                waiting_queue_.pop();
                if (waiting_requests_map_.erase(request->request_id) > 0) {
                    completion.SetResult(*request, false);
                    LOG_WARN(
                        "[MCL] Request {} is permanently impossible (minimum_required_size={} capacity={}), "
                        "failing immediately.",
                        request->request_id, minimum_required_size.ToString(), max_resource_limit_.load().ToString());
                }
                continue;
            }
            auto& request = deferred_requests.emplace_back();
            request = std::move(request_ptr_ref);
            waiting_queue_.pop();
            request->required_size = required_size;
            if (!blocked_deadline.has_value()) {
                blocked_deadline = request->deadline;
            }
            LOG_DEBUG("[MCL] Request {} of size {} cannot be satisfied, checking peers with the same deadline.",
                      request->request_id, required_size.ToString());
            completions.pop_back();
        }
    }
    return false;
}

void
DList::failWaitingRequest(uint64_t request_id, const char* reason, std::vector<WaitingRequestCompletion>& completions) {
    auto it = waiting_requests_map_.find(request_id);
    if (it == waiting_requests_map_.end()) {
        return;
    }
    auto& completion = completions.emplace_back();
    LOG_WARN("[MCL] Waiting request {} {}.", request_id, reason);
    completion.SetResult(*it->second, false);
    waiting_requests_map_.erase(it);
    notifyWaitingRequests();
}

void
DList::clearWaitingQueue() {
    std::vector<WaitingRequestCompletion> completions;
    auto complete = folly::makeGuard([&] { completeWaitingRequests(completions); });
    {
        std::unique_lock<std::mutex> lock(list_mtx_);
        while (!waiting_queue_.empty()) {
            auto& completion = completions.emplace_back();
            completion.retired = std::move(const_cast<std::unique_ptr<WaitingRequest>&>(waiting_queue_.top()));
            waiting_queue_.pop();
            auto& request = completion.retired;
            if (waiting_requests_map_.erase(request->request_id) > 0) {
                completion.SetResult(*request, false);
            }
        }
        reservation_notifications_needed_ = false;
    }
}

ResourceUsage
DList::checkPhysicalMemoryLimit(const ResourceUsage& size) const {
    if (size.memory_bytes <= 0) {
        return ResourceUsage{0, 0};
    }
    auto current_loading_mem = total_loading_size_.load().memory_bytes;
    auto sys_mem = getSystemMemoryInfo();
    TEST_SYNC_POINT_CALLBACK("DList::PhysicalMemoryInfo", &sys_mem);
    auto projected_mem_usage = current_loading_mem + size.memory_bytes + sys_mem.used_bytes;

    auto eviction_mem_needed =
        projected_mem_usage -
        static_cast<int64_t>(sys_mem.total_bytes * eviction_config_.overloaded_memory_threshold_percentage);
    if (eviction_mem_needed < 0) {
        eviction_mem_needed = 0;
    }

    LOG_TRACE(
        "[MCL] Physical memory check: "
        "projected_mem_usage={}(used_mem={}, loading_mem={}, requesting_mem={}), limit_mem={} (mem {}% of total {}), "
        "eviction_mem_needed={}",
        FormatBytes(projected_mem_usage), FormatBytes(sys_mem.used_bytes), FormatBytes(current_loading_mem),
        FormatBytes(size.memory_bytes),
        FormatBytes(
            static_cast<int64_t>(sys_mem.total_bytes * eviction_config_.overloaded_memory_threshold_percentage)),
        eviction_config_.overloaded_memory_threshold_percentage * 100, FormatBytes(sys_mem.total_bytes),
        FormatBytes(eviction_mem_needed));

    return ResourceUsage{eviction_mem_needed, 0};
}

}  // namespace milvus::cachinglayer::internal
