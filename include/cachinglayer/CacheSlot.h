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

#include <fmt/ranges.h>
#include <folly/Synchronized.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <flat_hash_map/flat_hash_map.hpp>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "cachinglayer/Metrics.h"
#include "cachinglayer/SharedLoad.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"
#include "cachinglayer/lrucache/ListNode.h"
#include "common/OpContext.h"
#include "log/Log.h"

namespace milvus::cachinglayer {

template <typename CellT>
class CellAccessor;

// - The action of pinning cells is not started until the returned SemiFuture is scheduled on an executor.
// - Once the future is scheduled, CacheSlot must live until the future is ready.
// - The returned CellAccessor stores a shared_ptr of CacheSlot, thus will keep CacheSlot alive.
// - DList must be managed by shared_ptr; the slot retains it while background work drains.
template <typename CellT>
class CacheSlot final : public std::enable_shared_from_this<CacheSlot<CellT>> {
 public:
    // TODO(tiered storage 1): the CellT should return its actual usage, once loaded. And we use this to report metrics.
    static_assert(std::is_same_v<ResourceUsage, decltype(std::declval<CellT>().CellByteSize())>,
                  "CellT must have a CellByteSize() method that returns a ResourceUsage "
                  "representing the memory consumption of the cell");

    CacheSlot(std::unique_ptr<Translator<CellT>> translator, internal::DList* dlist, bool evictable, bool self_reserve,
              bool storage_usage_tracking_enabled, std::chrono::milliseconds loading_timeout,
              std::chrono::milliseconds warmup_loading_timeout = std::chrono::milliseconds(0))
        : translator_(std::move(translator)),
          cell_id_mapping_mode_(translator_->meta()->cell_id_mapping_mode),
          cell_data_type_(translator_->meta()->cell_data_type),
          storage_type_(translator_->meta()->storage_type),
          dlist_(dlist),
          dlist_owner_(dlist->shared_from_this()),
          evictable_(evictable),
          self_reserve_(self_reserve),
          storage_usage_tracking_enabled_(storage_usage_tracking_enabled),
          loading_timeout_(loading_timeout),
          warmup_loading_timeout_(warmup_loading_timeout) {
        active_loads_.resize(translator_->num_cells());
        cells_.reserve(translator_->num_cells());
        for (cid_t i = 0; i < static_cast<cid_t>(translator_->num_cells()); ++i) {
            cells_.push_back(std::make_unique<CacheCell>(this, i));
        }
        monitor::cache_slot_count(cell_data_type_, storage_type_).Increment();
        monitor::cache_cell_count(cell_data_type_, storage_type_).Increment(translator_->num_cells());
    }

    CacheSlot(const CacheSlot&) = delete;
    CacheSlot&
    operator=(const CacheSlot&) = delete;
    CacheSlot(CacheSlot&&) = delete;
    CacheSlot&
    operator=(CacheSlot&&) = delete;

    // Warmup should only be called once before any Pin operation.
    // For async warmup, this method returns immediately and warmup happens in background.
    // Access to cells before warmup completes will block until the cell is loaded.
    void
    Warmup(OpContext* ctx, const std::shared_ptr<folly::CPUThreadPoolExecutor>& prefetch_pool = nullptr) {
        auto warmup_policy = translator_->meta()->cache_warmup_policy;

        switch (warmup_policy) {
            case CacheWarmupPolicy::CacheWarmupPolicy_Disable:
                return;

            case CacheWarmupPolicy::CacheWarmupPolicy_Async: {
                auto cids = AllCellIds();

                if (prefetch_pool) {
                    std::weak_ptr<CacheSlot<CellT>> weak_self = this->shared_from_this();
                    prefetch_pool->add([weak_self, cids = std::move(cids)]() {
                        try {
                            auto self = weak_self.lock();
                            if (!self) {
                                // CacheSlot was destroyed, skip warmup
                                return;
                            }
                            // Note: ctx is not captured - async warmup doesn't track storage usage
                            auto drain = folly::makeGuard([self] { SemiInlineGet(self->WaitForLoads()); });
                            self->PinCellsDirect(nullptr, cids, self->warmup_loading_timeout_);
                            // If the slot is not evictable, we don't need to pin the cells anymore after warmup.
                            self->skip_pin_.store(!self->evictable_, std::memory_order_release);
                        } catch (const std::exception& e) {
                            LOG_ERROR("[MCL] Async warmup failed: {}", e.what());
                        }
                    });
                    return;
                }
                // Fallback to sync if no pool provided
                LOG_WARN("[MCL] Async warmup requested but no prefetch pool provided, falling back to sync");
                // TODO: Warmup is not tracked for now
                PinCellsDirect(ctx, cids, warmup_loading_timeout_);
                skip_pin_.store(!evictable_, std::memory_order_release);
                return;
            }

            case CacheWarmupPolicy::CacheWarmupPolicy_Sync: {
                auto cids = AllCellIds();
                // Sync warmup (original behavior)
                // TODO: Warmup is not tracked for now
                PinCellsDirect(ctx, cids, warmup_loading_timeout_);
                skip_pin_.store(!evictable_, std::memory_order_release);
                return;
            }
        }
    }

    folly::SemiFuture<std::shared_ptr<CellAccessor<CellT>>>
    PinAllCells(OpContext* ctx) {
        if (skip_pin_.load(std::memory_order_acquire)) {
            return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(),
                                                         std::vector<internal::ListNode::NodePin>());
        }
        return folly::makeSemiFuture().deferValue([this, ctx](auto&&) {
            std::vector<cid_t> cids;
            cids.resize(cells_.size());
            std::iota(cids.begin(), cids.end(), 0);
            return PinInternal(ctx, cids, loading_timeout_);
        });
    }

    folly::SemiFuture<std::shared_ptr<CellAccessor<CellT>>>
    PinCells(OpContext* ctx, const std::vector<uid_t>& uids) {
        if (skip_pin_.load(std::memory_order_acquire)) {
            return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(),
                                                         std::vector<internal::ListNode::NodePin>());
        }
        monitor::cache_access_event_total(cell_data_type_, storage_type_).Increment();
        return folly::makeSemiFuture().deferValue(
            [this, uids = std::vector<uid_t>(uids), ctx](auto&&) -> std::shared_ptr<CellAccessor<CellT>> {
                auto count = std::min(uids.size(), cells_.size());
                ska::flat_hash_set<cid_t> involved_cids_set;
                involved_cids_set.reserve(count);
                switch (cell_id_mapping_mode_) {
                    case CellIdMappingMode::IDENTICAL: {
                        for (auto& uid : uids) {
                            involved_cids_set.insert(uid);
                        }
                        break;
                    }
                    case CellIdMappingMode::ALWAYS_ZERO: {
                        if (uids.size() > 0) {
                            involved_cids_set.insert(0);
                        }
                        break;
                    }
                    default: {
                        for (auto& uid : uids) {
                            auto cid = cell_id_of(uid);
                            involved_cids_set.insert(cid);
                        }
                    }
                }
                std::vector<cid_t> involved_cids_vec;
                involved_cids_vec.reserve(involved_cids_set.size());
                std::copy(involved_cids_set.begin(), involved_cids_set.end(), std::back_inserter(involved_cids_vec));
                return PinInternal(ctx, involved_cids_vec, loading_timeout_);
            });
    }

    std::shared_ptr<CellAccessor<CellT>>
    PinOneCellDirect(OpContext* ctx, const uid_t& uid) {
        if (skip_pin_.load(std::memory_order_acquire)) {
            return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(),
                                                         std::vector<internal::ListNode::NodePin>());
        }
        auto cid = 0;
        switch (cell_id_mapping_mode_) {
            case CellIdMappingMode::IDENTICAL: {
                cid = uid;
                break;
            }
            case CellIdMappingMode::ALWAYS_ZERO: {
                cid = 0;
                break;
            }
            default: {
                cid = cell_id_of(uid);
            }
        }
        return PinInternal(ctx, {cid}, loading_timeout_, true);
    }

    std::shared_ptr<CellAccessor<CellT>>
    PinCellsDirect(OpContext* ctx, const std::vector<uid_t>& uids,
                   std::optional<std::chrono::milliseconds> timeout_override = std::nullopt) {
        if (skip_pin_.load(std::memory_order_acquire)) {
            return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(),
                                                         std::vector<internal::ListNode::NodePin>());
        }
        auto count = std::min(uids.size(), cells_.size());
        ska::flat_hash_set<cid_t> involved_cids;
        involved_cids.reserve(count);
        switch (cell_id_mapping_mode_) {
            case CellIdMappingMode::IDENTICAL: {
                for (auto& uid : uids) {
                    involved_cids.insert(uid);
                }
                break;
            }
            case CellIdMappingMode::ALWAYS_ZERO: {
                if (uids.size() > 0) {
                    involved_cids.insert(0);
                }
                break;
            }
            default: {
                for (auto& uid : uids) {
                    auto cid = cell_id_of(uid);
                    involved_cids.insert(cid);
                }
            }
        }
        std::vector<cid_t> involved_cids_vec;
        involved_cids_vec.reserve(involved_cids.size());
        std::copy(involved_cids.begin(), involved_cids.end(), std::back_inserter(involved_cids_vec));
        return PinInternal(ctx, involved_cids_vec, timeout_override.value_or(loading_timeout_));
    }

    // Wait for currently issued loads to release their resources, including
    // background work still draining after a request has been cancelled.
    folly::SemiFuture<folly::Unit>
    WaitForLoads() {
        std::vector<folly::SemiFuture<folly::Unit>> drains;
        {
            std::lock_guard lock(loads_mutex_);
            for (const auto& load : active_loads_)
                if (load)
                    drains.push_back(load->drained.getSemiFuture());
        }
        return folly::collectAll(drains).deferValue(
            [self = this->shared_from_this()](auto&&) { return folly::Unit{}; });
    }

    // Manually evicts the cell if it is LOADED and not pinned.
    // Returns true if the eviction happened.
    bool
    ManualEvict(cid_t cid) {
        return cells_[cid]->manual_evict();
    }

    // Manually evicts all cells that are LOADED and not pinned.
    // Returns true if eviction happened on any cell.
    bool
    ManualEvictAll() {
        bool evicted = false;
        for (cid_t cid = 0; cid < cells_.size(); ++cid) {
            if (cells_[cid]->manual_evict()) {
                evicted = true;
            }
        }
        return evicted;
    }

    [[nodiscard]] size_t
    num_cells() const {
        return translator_->num_cells();
    }

    [[nodiscard]] ResourceUsage
    size_of_cell(cid_t cid) const {
        return cells_[cid]->loaded_size();
    }

    Meta*
    meta() {
        return translator_->meta();
    }

    ~CacheSlot() {
        cells_.clear();
        monitor::cache_slot_count(cell_data_type_, storage_type_).Decrement();
        monitor::cache_cell_count(cell_data_type_, storage_type_).Decrement(translator_->num_cells());
    }

 private:
    friend class CellAccessor<CellT>;
    friend class SharedLoadTestFriend;

    [[nodiscard]] std::vector<cid_t>
    AllCellIds() const {
        std::vector<cid_t> cids(translator_->num_cells());
        std::iota(cids.begin(), cids.end(), 0);
        return cids;
    }

    std::shared_ptr<CellAccessor<CellT>>
    PinInternal(OpContext* ctx, const std::vector<cid_t>& cids, std::chrono::milliseconds timeout,
                bool single = false) {
        if (auto resident = TryPinResident(ctx, cids))
            return resident;
        auto pin_result = SemiInlineGet(PinShared(ctx, cids, timeout, single));
        auto all_pins = std::move(pin_result.first);
        if (ctx && storage_usage_tracking_enabled_) {
            ctx->storage_usage.scanned_cold_bytes.fetch_add(pin_result.second);
            ctx->storage_usage.scanned_total_bytes.fetch_add(translator_->cells_storage_bytes(cids));
        }

        return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(), std::move(all_pins));
    }

    std::shared_ptr<CellAccessor<CellT>>
    TryPinResident(OpContext* ctx, const std::vector<cid_t>& cids) {
        if (ctx && ctx->cancellation_token.isCancellationRequested())
            ThrowInfo(ErrorCode::FollyCancel, "Cache wait cancelled");
        // An external reservation owner must still wait for any producer to drain.
        if (!self_reserve_)
            return nullptr;
        for (auto cid : cids) {
            if (cid < 0 || cid >= static_cast<cid_t>(cells_.size()))
                ThrowInfo(ErrorCode::OutOfRange, "cid {} out of range, slot has {} cells", cid, cells_.size());
        }
        std::vector<internal::ListNode::NodePin> pins;
        pins.reserve(cids.size());
        for (auto cid : cids) {
            auto pin = cells_[cid]->try_pin();
            if (!pin)
                return nullptr;  // Release partial pins before entering the coordinated miss path.
            pins.push_back(std::move(*pin));
        }
        for (auto cid : cids)
            monitor::cache_cell_access_hit_bytes_total(cell_data_type_, storage_type_)
                .Increment(cells_[cid]->local_storage_bytes());
        if (ctx && storage_usage_tracking_enabled_)
            ctx->storage_usage.scanned_total_bytes.fetch_add(translator_->cells_storage_bytes(cids));
        return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(), std::move(pins));
    }

    [[nodiscard]] cid_t
    cell_id_of(uid_t uid) const {
        switch (cell_id_mapping_mode_) {
            case CellIdMappingMode::IDENTICAL:
                return uid;
            case CellIdMappingMode::ALWAYS_ZERO:
                return 0;
            default:
                return translator_->cell_id_of(uid);
        }
    }

    void
    RunLoad(OpContext* ctx, std::unordered_set<cid_t>&& cids, std::chrono::milliseconds timeout) {
        std::vector<cid_t> loading_cids;
        auto start = std::chrono::steady_clock::now();
        bool reservation_success = false;

        loading_cids = std::vector<cid_t>(cids.begin(), cids.end());

        auto run_load_internal = [&]() {
            if (ctx && ctx->cancellation_token.isCancellationRequested()) {
                ThrowInfo(ErrorCode::FollyCancel, "Operation cancelled, stop loading cache cells");
            }
            start = std::chrono::steady_clock::now();
            auto results = translator_->get_cells(ctx, loading_cids);
            if (ctx && ctx->cancellation_token.isCancellationRequested()) {
                ThrowInfo(ErrorCode::FollyCancel, "Shared cache load cancelled");
            }
            auto latency =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
            for (auto& result : results) {
                cells_[result.first]->set_cell(std::move(result.second), cids.count(result.first) > 0);
            }
            monitor::cache_load_latency_microseconds(cell_data_type_, storage_type_).Observe(latency.count());
        };

        // The caller owns loading-resource admission when self reservation is disabled.
        // Skip DList reservation here to avoid accounting the same load twice.
        if (!self_reserve_) {
            run_load_internal();
            return;
        }

        auto bonus_cids = translator_->bonus_cells_to_be_loaded(loading_cids);
        loading_cids.insert(loading_cids.end(), bonus_cids.begin(), bonus_cids.end());

        ResourceUsage estimated_loading_usage;
        ResourceUsage estimated_loaded_usage;
        auto reserve_loading_cids = [&](const std::vector<cid_t>& cids_to_load) {
            auto estimate = translator_->estimated_loading_usage(cids_to_load);
            estimated_loaded_usage = estimate.first;
            estimated_loading_usage = estimate.second;
            return SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout(estimated_loading_usage,
                                                                           estimated_loaded_usage, timeout, ctx));
        };
        reservation_success = reserve_loading_cids(loading_cids);

        if (!bonus_cids.empty()) {
            // if the reservation failed, try to reserve only the essential loading resource
            if (!reservation_success) {
                LOG_WARN(
                    "[MCL] CacheSlot reserve loading resource with bonus cells failed, try to reserve only "
                    "essential "
                    "loading resource");
                loading_cids.resize(loading_cids.size() - bonus_cids.size());
                reservation_success = reserve_loading_cids(loading_cids);
            }
        }

        if (!reservation_success) {
            LOG_ERROR(
                "[MCL] CacheSlot failed to reserve resource for "
                "cells: key={}, cell_ids=[{}], total "
                "resource_needed_for_loading={}",
                translator_->key(), fmt::join(loading_cids, ","), estimated_loading_usage.ToString());
            ThrowInfo(ErrorCode::InsufficientResource,
                      "[MCL] CacheSlot failed to reserve resource for "
                      "cells: key={}, cell_ids=[{}], total "
                      "resource_needed_for_loading={}",
                      translator_->key(), fmt::join(loading_cids, ","), estimated_loading_usage.ToString());
        }

        monitor::cache_loading_bytes(cell_data_type_, StorageType::MEMORY)
            .Increment(estimated_loading_usage.memory_bytes);
        monitor::cache_loading_bytes(cell_data_type_, StorageType::DISK).Increment(estimated_loading_usage.file_bytes);
        monitor::cache_cell_loading_count(cell_data_type_, storage_type_).Increment(loading_cids.size());

        // Defer releasing the estimated loading usage.
        auto defer_release =
            folly::makeGuard([this, &estimated_loading_usage, &estimated_loaded_usage, &loading_cids]() {
                try {
                    dlist_->ReleaseLoadingResource(estimated_loading_usage, estimated_loaded_usage);
                    monitor::cache_cell_loading_count(cell_data_type_, storage_type_).Decrement(loading_cids.size());
                    monitor::cache_loading_bytes(cell_data_type_, StorageType::MEMORY)
                        .Decrement(estimated_loading_usage.memory_bytes);
                    monitor::cache_loading_bytes(cell_data_type_, StorageType::DISK)
                        .Decrement(estimated_loading_usage.file_bytes);
                } catch (...) {
                    auto exception = std::current_exception();
                    auto ew = folly::exception_wrapper(exception);
                    LOG_ERROR(
                        "[MCL] CacheSlot failed to release loading resource for cells with exception, something must "
                        "be wrong: "
                        "key={}, "
                        "loading_cids=[{}], error={}",
                        translator_->key(), fmt::join(loading_cids, ","), ew.what());
                }
            });

        LOG_TRACE(
            "[MCL] CacheSlot reserveLoadingResourceWithTimeout {} sec "
            "result: {} time: {} sec, resource_needed: {}, key: {}",
            timeout.count() / 1000.0, reservation_success ? "success" : "failed",
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() *
                1.0 / 1000,
            estimated_loading_usage.ToString(), translator_->key());

        run_load_internal();
    }

    using SharedLoad = internal::SharedLoad;
    using Pins = std::vector<internal::ListNode::NodePin>;
    using PinResult = std::pair<Pins, int64_t>;

    void
    ReleaseLoadWaiters(const std::vector<std::shared_ptr<SharedLoad>>& loads, bool cancelled) {
        for (const auto& load : loads) {
            bool cancel = false;
            {
                std::lock_guard lock(loads_mutex_);
                --load->waiters;
                if (cancelled && load->waiters == 0 && !load->finished && !load->closing) {
                    load->closing = true;
                    cancel = true;
                }
            }
            // Callbacks may finish a load inline; never cancel under loads_mutex_.
            if (cancel)
                load->cancellation.requestCancellation();
        }
    }

    void
    FinishSharedLoad(const std::shared_ptr<SharedLoad>& load, folly::exception_wrapper error) {
        {
            std::lock_guard lock(loads_mutex_);
            // Keep the registry entry until errors have been published outside the lock.
            load->closing = true;
        }
        if (error) {
            monitor::cache_load_event_fail_total(cell_data_type_, storage_type_).Increment();
            for (auto cid : load->cids) cells_[cid]->set_error(error);
        }
        {
            std::lock_guard lock(loads_mutex_);
            load->finished = true;
            for (auto cid : load->cids) active_loads_[cid].reset();
        }
        // RunLoad's resource guards have already run. Only now may a new generation start.
        load->drained.setValue(folly::Unit{});
    }

    void
    StartSharedLoad(std::shared_ptr<SharedLoad> load, std::chrono::milliseconds timeout) {
        auto self = this->shared_from_this();
        try {
            // timeout applies only to resource admission, as before shared loading.
            // The last departing waiter cancels the producer; no new I/O deadline.
            internal::SharedLoadExecutor().add([self, load, timeout] {
                folly::exception_wrapper error;
                try {
                    self->RunLoad(&load->context, std::unordered_set<cid_t>(load->cids), timeout);
                } catch (...) {
                    error = folly::exception_wrapper(std::current_exception());
                }
                self->FinishSharedLoad(load, std::move(error));
            });
        } catch (...) {
            FinishSharedLoad(load, folly::exception_wrapper(std::current_exception()));
        }
    }

    folly::SemiFuture<PinResult>
    PinShared(OpContext* ctx, const std::vector<cid_t>& cids, std::chrono::milliseconds timeout, bool single) {
        auto self = this->shared_from_this();
        auto token = ctx ? ctx->cancellation_token : folly::CancellationToken{};
        if (token.isCancellationRequested())
            ThrowInfo(ErrorCode::FollyCancel, "Cache wait cancelled");
        std::vector<folly::SemiFuture<internal::ListNode::NodePin>> futures;
        std::vector<folly::SemiFuture<folly::Unit>> drains;
        std::vector<std::shared_ptr<SharedLoad>> loads;
        std::vector<cid_t> claimed;
        std::vector<cid_t> cold;
        cold.reserve(cids.size());
        auto released = std::make_shared<std::atomic<bool>>(false);
        int64_t cold_bytes = 0;
        futures.reserve(cids.size());
        drains.reserve(cids.size());
        loads.reserve(cids.size());
        claimed.reserve(cids.size());
        std::shared_ptr<SharedLoad> created;
        std::unique_lock lock(loads_mutex_);
        for (auto cid : cids) {
            if (cid < 0 || cid >= static_cast<cid_t>(cells_.size())) {
                ThrowInfo(ErrorCode::OutOfRange, "cid {} out of range, slot has {} cells", cid, cells_.size());
            }
        }
        for (auto cid : cids) {
            auto load = active_loads_[cid];
            if (load && load->closing) {
                auto drained = load->drained.getSemiFuture();
                lock.unlock();
                // A cancelled generation is draining, not available for joining. Do
                // not start another download until it has released all its resources.
                return internal::WaitForSharedLoad(std::move(drained), token, self)
                    .deferValue([self, ctx, cids, timeout, single](auto&&) {
                        return self->PinShared(ctx, cids, timeout, single);
                    });
            }
        }
        try {
            for (auto cid : cids) {
                auto [need_load, pin] = cells_[cid]->pin();
                if (need_load)
                    claimed.push_back(cid);
                if (need_load ||
                    (single && std::holds_alternative<folly::SemiFuture<internal::ListNode::NodePin>>(pin)))
                    cold.push_back(cid);
                if (std::holds_alternative<internal::ListNode::NodePin>(pin)) {
                    futures.push_back(folly::makeSemiFuture(std::get<internal::ListNode::NodePin>(std::move(pin))));
                } else {
                    futures.push_back(std::get<folly::SemiFuture<internal::ListNode::NodePin>>(std::move(pin)));
                }
                auto load = active_loads_[cid];
                if (load && std::find(loads.begin(), loads.end(), load) == loads.end())
                    loads.push_back(load);
                if (need_load) {
                    monitor::cache_cell_access_miss_bytes_total(cell_data_type_, storage_type_)
                        .Increment(cells_[cid]->local_storage_bytes());
                } else {
                    monitor::cache_cell_access_hit_bytes_total(cell_data_type_, storage_type_)
                        .Increment(cells_[cid]->local_storage_bytes());
                }
            }
            if (!claimed.empty()) {
                created = std::make_shared<SharedLoad>(ctx);
                created->cids.insert(claimed.begin(), claimed.end());
                loads.push_back(created);
            }
            for (const auto& load : loads) drains.push_back(load->drained.getSemiFuture());
            if (ctx && storage_usage_tracking_enabled_ && !cold.empty())
                cold_bytes = translator_->cells_storage_bytes(cold);
        } catch (...) {
            auto error = folly::exception_wrapper(std::current_exception());
            lock.unlock();
            for (auto cid : claimed) cells_[cid]->set_error(error);
            throw;
        }
        for (auto cid : claimed) active_loads_[cid] = created;
        for (const auto& load : loads) ++load->waiters;
        lock.unlock();

        // Aggregation retains NodePins and the slot until every producer is drained.
        // An early-cancelled waiter receives a separate result without destroying them.
        auto unsubscribe = folly::makeGuard([&] {
            if (!released->exchange(true))
                ReleaseLoadWaiters(loads, true);
        });
        bool launched = false;
        auto fail_unlaunched = folly::makeGuard([&] {
            if (created && !launched)
                FinishSharedLoad(created, folly::make_exception_wrapper<std::bad_alloc>());
        });
        auto pins = folly::collectAll(futures).deferValue(
            [drains = folly::collectAll(drains),
             cold_bytes](std::vector<folly::Try<internal::ListNode::NodePin>>&& results) mutable {
                return std::move(drains).deferValue([results = std::move(results), cold_bytes](auto&&) mutable {
                    Pins pins;
                    pins.reserve(results.size());
                    for (auto& result : results) pins.push_back(std::move(result).value());
                    return PinResult{std::move(pins), cold_bytes};
                });
            });
        auto result =
            internal::WaitForSharedLoad(std::move(pins), token, self, [self, loads, released](bool cancelled) {
                if (!released->exchange(true))
                    self->ReleaseLoadWaiters(loads, cancelled);
            });
        if (!self_reserve_) {
            // The external owner may release its reservation when this call returns.
            // End our subscription immediately, but retain that owner's call until drain.
            result = std::move(result).defer([self](folly::Try<PinResult>&& value) {
                return self->WaitForLoads().deferValue(
                    [value = std::move(value)](auto&&) mutable { return std::move(value).value(); });
            });
        }
        unsubscribe.dismiss();
        if (created) {
            launched = true;
            StartSharedLoad(created, timeout);
        }
        return result;
    }

    struct CacheCell : internal::ListNode {
     public:
        CacheCell() = delete;

        CacheCell(CacheSlot<CellT>* slot, cid_t cid)
            : internal::ListNode(slot->dlist_, slot->evictable_),
              slot_(slot),
              cid_(cid),
              local_storage_bytes_(slot->translator_->cells_storage_bytes({cid})) {
        }

        // use the default destructor
        ~CacheCell() override {
            mark_unload([this]() {
                if (cell_) {
                    auto saved_loaded_size = loaded_size_;
                    clear_data();
                    slot_->dlist_->RefundLoadedResource(saved_loaded_size);
                }
            });
        }

        CellT*
        cell() {
            return cell_.get();
        }

        // Be careful that even though only a single thread can request loading a cell,
        // it is still possible that multiple threads call set_cell() concurrently.
        // For example, 2 RunLoad() calls tries to download cell 4 and 6, and both decided
        // to also download cell 5, if they finished at the same time, they will call set_cell()
        // of cell 5 concurrently.
        void
        set_cell(std::unique_ptr<CellT> cell, bool requesting_thread) {
            mark_loaded(
                [this, cell = std::move(cell)]() mutable {
                    cell_ = std::move(cell);
                    loaded_size_ = cell_->CellByteSize();
                    if (!loaded_size_.AnyGTZero()) {
                        LOG_WARN(
                            "[MCL] CacheSlot Cell {} has zero size, use "
                            "estimated size from translator",
                            key());
                        loaded_size_ = slot_->translator_->estimated_loading_usage(std::vector<cid_t>{cid_}).first;
                    }
                    slot_->dlist_->ChargeLoadedResource(loaded_size_);
                    life_start_ = std::chrono::steady_clock::now();
                    monitor::cache_loaded_bytes(slot_->cell_data_type_, StorageType::MEMORY)
                        .Increment(loaded_size_.memory_bytes);
                    monitor::cache_loaded_bytes(slot_->cell_data_type_, StorageType::DISK)
                        .Increment(loaded_size_.file_bytes);
                    monitor::cache_cell_loaded_count(slot_->cell_data_type_, slot_->storage_type_).Increment();
                },
                requesting_thread);
            LOG_TRACE("[MCL] CacheSlot Cell loaded: key={}, size={}", key(), loaded_size_.ToString());
        }

        void
        set_error(folly::exception_wrapper error) {
            internal::ListNode::set_error(std::move(error));
        }

        // Note: must be called under the lock of mtx_ and should only be called by eviction.
        void
        unload() override {
            clear_data();
            internal::ListNode::unload();
        }

        int64_t
        local_storage_bytes() const {
            return local_storage_bytes_;
        }

     protected:
        void
        clear_data() {
            if (cell_) {
                auto life_time = std::chrono::steady_clock::now() - life_start_;
                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(life_time).count();
                monitor::cache_cell_lifetime_seconds(slot_->cell_data_type_, slot_->storage_type_).Observe(seconds);
                cell_ = nullptr;
                monitor::cache_cell_loaded_count(slot_->cell_data_type_, slot_->storage_type_).Decrement();
                monitor::cache_loaded_bytes(slot_->cell_data_type_, StorageType::MEMORY)
                    .Decrement(loaded_size_.memory_bytes);
                monitor::cache_loaded_bytes(slot_->cell_data_type_, StorageType::DISK)
                    .Decrement(loaded_size_.file_bytes);
                LOG_TRACE("[MCL] CacheSlot Cell unloaded: key={}, size={}", key(), loaded_size_.ToString());
                loaded_size_ = {0, 0};  // reset loaded_size_ to 0,0 to avoid double refund from dlist_
            }
        }

        std::string
        key() const override {
            return fmt::format("{}:{}", slot_->translator_->key(), cid_);
        }

     private:
        CacheSlot<CellT>* slot_{nullptr};
        cid_t cid_{0};
        std::unique_ptr<CellT> cell_{nullptr};
        std::chrono::steady_clock::time_point life_start_{};
        int64_t local_storage_bytes_{0};
    };

    const std::unique_ptr<Translator<CellT>> translator_;
    // Each CacheCell's cid_t is its index in vector.
    // Using unique_ptr because CacheCell is non-movable (inherits from ListNode).
    // Once initialized, cells_ should never be resized.
    std::vector<std::unique_ptr<CacheCell>> cells_;
    CellIdMappingMode cell_id_mapping_mode_;
    CellDataType cell_data_type_;
    StorageType storage_type_;
    internal::DList* dlist_;
    std::shared_ptr<internal::DList> dlist_owner_;
    std::mutex loads_mutex_;
    std::vector<std::shared_ptr<SharedLoad>> active_loads_;
    const bool evictable_;
    const bool self_reserve_;
    const bool storage_usage_tracking_enabled_;
    std::chrono::milliseconds loading_timeout_{100000};
    std::chrono::milliseconds warmup_loading_timeout_{0};
    std::atomic<bool> skip_pin_{false};
};

// - A thin wrapper for accessing cells in a CacheSlot.
// - When this class is created, the cells are loaded and pinned.
// - Accessing cells through this class does not incur any lock overhead.
// - Accessing cells that are not pinned by this CellAccessor is undefined behavior.
template <typename CellT>
class CellAccessor {
 public:
    CellAccessor(std::shared_ptr<CacheSlot<CellT>> slot, std::vector<internal::ListNode::NodePin> pins)
        : slot_(std::move(slot)), pins_(std::move(pins)) {
    }

    CellT*
    get_cell_of(uid_t uid) {
        auto cid = slot_->cell_id_of(uid);
        return slot_->cells_[cid]->cell();
    }

    CellT*
    get_ith_cell(cid_t cid) {
        return slot_->cells_[cid]->cell();
    }

 private:
    // pins must be destroyed before slot_ is destroyed, thus
    // pins_ should be a member after slot_.
    std::shared_ptr<CacheSlot<CellT>> slot_;
    std::vector<internal::ListNode::NodePin> pins_;
};

// TODO(tiered storage 4): this class is a temp solution. Later we should modify all usage of this class
// to use folly::SemiFuture instead: all data access should happen within deferValue().
// Current impl requires the T type to be movable/copyable.
template <typename T>
class PinWrapper {
 public:
    PinWrapper() = default;
    PinWrapper(std::any raii, T&& content) : raii_(std::move(raii)), content_(std::move(content)) {
    }

    PinWrapper(std::any raii, const T& content) : raii_(std::move(raii)), content_(content) {
    }

    // For those that does not need a pin. eg: growing segment, views that actually copies the data, etc.
    PinWrapper(T&& content) : raii_(nullptr), content_(std::move(content)) {
    }
    PinWrapper(const T& content) : raii_(nullptr), content_(content) {
    }

    PinWrapper(PinWrapper&& other) noexcept : raii_(std::move(other.raii_)), content_(std::move(other.content_)) {
    }

    PinWrapper(const PinWrapper& other) : raii_(other.raii_), content_(other.content_) {
    }

    PinWrapper&
    operator=(PinWrapper&& other) noexcept {
        if (this != &other) {
            std::swap(raii_, other.raii_);
            std::swap(content_, other.content_);
        }
        return *this;
    }

    PinWrapper&
    operator=(const PinWrapper& other) {
        if (this != &other) {
            raii_ = other.raii_;
            content_ = other.content_;
        }
        return *this;
    }

    T&
    get() {
        return content_;
    }

    const T&
    get() const {
        return content_;
    }

    template <typename T2, typename Fn>
    PinWrapper<T2>
    transform(Fn&& transformer) && {
        T2 transformed = transformer(std::move(content_));
        return PinWrapper<T2>(std::move(raii_), std::move(transformed));
    }

 private:
    // CellAccessor is templated on CellT, we don't want to enforce that in this class.
    std::any raii_{nullptr};
    T content_;
};

}  // namespace milvus::cachinglayer
