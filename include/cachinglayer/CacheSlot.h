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

#include <folly/Synchronized.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include <algorithm>
#include <any>
#include <chrono>
#include <cstddef>
#include <exception>
#include <flat_hash_map/flat_hash_map.hpp>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "cachinglayer/Metrics.h"
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
template <typename CellT>
class CacheSlot final : public std::enable_shared_from_this<CacheSlot<CellT>> {
 public:
    // TODO(tiered storage 1): the CellT should return its actual usage, once loaded. And we use this to report metrics.
    static_assert(std::is_same_v<ResourceUsage, decltype(std::declval<CellT>().CellByteSize())>,
                  "CellT must have a CellByteSize() method that returns a ResourceUsage "
                  "representing the memory consumption of the cell");

    CacheSlot(std::unique_ptr<Translator<CellT>> translator, internal::DList* dlist, bool evictable, bool self_reserve,
              bool storage_usage_tracking_enabled, std::chrono::milliseconds loading_timeout)
        : translator_(std::move(translator)),
          cell_id_mapping_mode_(translator_->meta()->cell_id_mapping_mode),
          cell_data_type_(translator_->meta()->cell_data_type),
          storage_type_(translator_->meta()->storage_type),
          dlist_(dlist),
          evictable_(evictable),
          self_reserve_(self_reserve),
          storage_usage_tracking_enabled_(storage_usage_tracking_enabled),
          loading_timeout_(loading_timeout) {
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
    void
    Warmup() {
        auto warmup_policy = translator_->meta()->cache_warmup_policy;

        if (warmup_policy == CacheWarmupPolicy::CacheWarmupPolicy_Disable) {
            return;
        }

        std::vector<cid_t> cids;
        cids.reserve(translator_->num_cells());
        for (cid_t i = 0; i < translator_->num_cells(); ++i) {
            cids.push_back(i);
        }
        // TODO: Warmup is not tracked for now
        PinCellsDirect(nullptr, cids);

        // If the slot is not evictable, we don't need to pin the cells anymore after warmup.
        skip_pin_ = !evictable_;
    }

    folly::SemiFuture<std::shared_ptr<CellAccessor<CellT>>>
    PinAllCells(OpContext* ctx) {
        if (skip_pin_) {
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
        if (skip_pin_) {
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
        if (skip_pin_) {
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
        auto [need_load, result] = cells_[cid]->pin();
        auto cell_storage_bytes = translator_->cells_storage_bytes({cid});
        if (need_load) {
            monitor::cache_cell_access_miss_bytes_total(cell_data_type_, storage_type_).Increment(cell_storage_bytes);
        } else {
            monitor::cache_cell_access_hit_bytes_total(cell_data_type_, storage_type_).Increment(cell_storage_bytes);
        }
        if (std::holds_alternative<internal::ListNode::NodePin>(result)) {
            std::vector<internal::ListNode::NodePin> pins;
            pins.push_back(std::get<internal::ListNode::NodePin>(std::move(result)));
            if (ctx && storage_usage_tracking_enabled_) {
                ctx->storage_usage.scanned_total_bytes.fetch_add(cell_storage_bytes);
            }
            return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(), std::move(pins));
        } else {
            auto pin_future = std::get<folly::SemiFuture<internal::ListNode::NodePin>>(std::move(result));
            if (need_load) {
                RunLoad(ctx, {cid}, loading_timeout_);
            }
            std::vector<internal::ListNode::NodePin> pins;
            pins.push_back(SemiInlineGet(std::move(pin_future)));
            if (ctx && storage_usage_tracking_enabled_) {
                ctx->storage_usage.scanned_cold_bytes.fetch_add(cell_storage_bytes);
                ctx->storage_usage.scanned_total_bytes.fetch_add(cell_storage_bytes);
            }
            return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(), std::move(pins));
        }
    }

    std::shared_ptr<CellAccessor<CellT>>
    PinCellsDirect(OpContext* ctx, const std::vector<uid_t>& uids) {
        if (skip_pin_) {
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
        return PinInternal(ctx, involved_cids_vec, loading_timeout_);
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
        monitor::cache_slot_count(cell_data_type_, storage_type_).Decrement();
        monitor::cache_cell_count(cell_data_type_, storage_type_).Decrement(translator_->num_cells());
    }

 private:
    friend class CellAccessor<CellT>;

    std::shared_ptr<CellAccessor<CellT>>
    PinInternal(OpContext* ctx, const std::vector<cid_t>& cids, std::chrono::milliseconds timeout) {
        std::vector<folly::SemiFuture<internal::ListNode::NodePin>> futures;
        std::vector<internal::ListNode::NodePin> ready_pins;
        std::unordered_set<cid_t> need_load_cids;
        futures.reserve(cids.size());
        ready_pins.reserve(cids.size());
        need_load_cids.reserve(cids.size());
        for (const auto& cid : cids) {
            if (cid >= static_cast<cid_t>(cells_.size())) {
                ThrowInfo(ErrorCode::OutOfRange, "cid {} out of range, slot has {} cells. key={}", cid, cells_.size(),
                          translator_->key());
            }
        }

        for (const auto& cid : cids) {
            auto [need_load, result] = cells_[cid]->pin();
            if (std::holds_alternative<internal::ListNode::NodePin>(result)) {
                ready_pins.push_back(std::get<internal::ListNode::NodePin>(std::move(result)));
            } else {
                futures.push_back(std::get<folly::SemiFuture<internal::ListNode::NodePin>>(std::move(result)));
            }
            if (need_load) {
                need_load_cids.insert(cid);
                monitor::cache_cell_access_miss_bytes_total(cell_data_type_, storage_type_)
                    .Increment(cells_[cid]->local_storage_bytes());
            } else {
                monitor::cache_cell_access_hit_bytes_total(cell_data_type_, storage_type_)
                    .Increment(cells_[cid]->local_storage_bytes());
            }
        }

        if (!need_load_cids.empty()) {
            RunLoad(ctx, std::move(need_load_cids), timeout);
        }

        std::vector<internal::ListNode::NodePin> all_pins;
        all_pins.reserve(cids.size());

        for (auto& pin : ready_pins) {
            all_pins.push_back(std::move(pin));
        }

        if (!futures.empty()) {
            auto future_pins = SemiInlineGet(folly::collect(futures));
            for (auto& pin : future_pins) {
                all_pins.push_back(std::move(pin));
            }
        }

        if (ctx && storage_usage_tracking_enabled_) {
            if (!need_load_cids.empty()) {
                std::vector<cid_t> need_load_cids_vec(need_load_cids.begin(), need_load_cids.end());
                ctx->storage_usage.scanned_cold_bytes.fetch_add(translator_->cells_storage_bytes(need_load_cids_vec));
            }
            ctx->storage_usage.scanned_total_bytes.fetch_add(translator_->cells_storage_bytes(cids));
        }

        return std::make_shared<CellAccessor<CellT>>(this->shared_from_this(), std::move(all_pins));
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
        ResourceUsage essential_loading_resource{};
        ResourceUsage bonus_loading_resource{};
        std::vector<cid_t> loading_cids;
        try {
            auto start = std::chrono::steady_clock::now();
            bool reservation_success = false;

            loading_cids = std::vector<cid_t>(cids.begin(), cids.end());

            auto run_load_internal = [&]() {
                if (ctx && ctx->cancellation_token.isCancellationRequested()) {
                    throw std::runtime_error("Operation cancelled, stop loading cache cells");
                }
                start = std::chrono::steady_clock::now();
                auto results = translator_->get_cells(ctx, loading_cids);
                auto latency =
                    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
                for (auto& result : results) {
                    cells_[result.first]->set_cell(std::move(result.second), cids.count(result.first) > 0);
                }
                monitor::cache_load_latency_microseconds(cell_data_type_, storage_type_).Observe(latency.count());
            };

            if (!self_reserve_) {
                run_load_internal();
                return;
            }

            // bonus cells should be empty if self_reserve_ is false.
            auto bonus_cids = translator_->bonus_cells_to_be_loaded(loading_cids);

            for (auto& cid : loading_cids) {
                essential_loading_resource += translator_->estimated_byte_size_of_cell(cid).second;
            }

            for (auto& cid : bonus_cids) {
                bonus_loading_resource += translator_->estimated_byte_size_of_cell(cid).second;
            }

            auto resource_needed_for_loading = essential_loading_resource + bonus_loading_resource;
            reservation_success =
                SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout(resource_needed_for_loading, timeout, ctx));

            if (!bonus_cids.empty()) {
                // if the reservation failed, try to reserve only the essential loading resource
                if (!reservation_success) {
                    LOG_WARN(
                        "[MCL] CacheSlot reserve loading resource with bonus cells failed, try to reserve only "
                        "essential "
                        "loading resource");
                    resource_needed_for_loading = essential_loading_resource;
                    reservation_success = SemiInlineGet(
                        dlist_->ReserveLoadingResourceWithTimeout(resource_needed_for_loading, timeout, ctx));
                } else {
                    // if the reservation succeeded, we can load the bonus cells
                    loading_cids.insert(loading_cids.end(), bonus_cids.begin(), bonus_cids.end());
                }
            }

            if (!reservation_success) {
                LOG_ERROR(
                    "[MCL] CacheSlot failed to reserve resource for "
                    "cells: key={}, cell_ids=[{}], total "
                    "resource_needed_for_loading={}",
                    translator_->key(), fmt::join(loading_cids, ","), resource_needed_for_loading.ToString());
                ThrowInfo(ErrorCode::InsufficientResource,
                          "[MCL] CacheSlot failed to reserve resource for "
                          "cells: key={}, cell_ids=[{}], total "
                          "resource_needed_for_loading={}",
                          translator_->key(), fmt::join(loading_cids, ","), resource_needed_for_loading.ToString());
            }

            monitor::cache_loading_bytes(cell_data_type_, StorageType::MEMORY)
                .Increment(resource_needed_for_loading.memory_bytes);
            monitor::cache_loading_bytes(cell_data_type_, StorageType::DISK)
                .Increment(resource_needed_for_loading.file_bytes);
            monitor::cache_cell_loading_count(cell_data_type_, storage_type_).Increment(loading_cids.size());

            // defer release resource_needed_for_loading
            auto defer_release = folly::makeGuard([this, &resource_needed_for_loading, &loading_cids]() {
                try {
                    dlist_->ReleaseLoadingResource(resource_needed_for_loading);
                    monitor::cache_cell_loading_count(cell_data_type_, storage_type_).Decrement(loading_cids.size());
                    monitor::cache_loading_bytes(cell_data_type_, StorageType::MEMORY)
                        .Decrement(resource_needed_for_loading.memory_bytes);
                    monitor::cache_loading_bytes(cell_data_type_, StorageType::DISK)
                        .Decrement(resource_needed_for_loading.file_bytes);
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
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                        .count() *
                    1.0 / 1000,
                resource_needed_for_loading.ToString(), translator_->key());

            run_load_internal();
        } catch (...) {
            auto exception = std::current_exception();
            auto ew = folly::exception_wrapper(exception);
            monitor::cache_load_event_fail_total(cell_data_type_, storage_type_).Increment();
            // set_error should only be called on the cells that are actually loaded, bonus cells are not considered.
            for (auto cid : cids) {
                cells_[cid]->set_error(ew);
            }
        }
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
                        loaded_size_ = slot_->translator_->estimated_byte_size_of_cell(cid_).first;
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
    const bool evictable_;
    const bool self_reserve_;
    const bool storage_usage_tracking_enabled_;
    std::chrono::milliseconds loading_timeout_{100000};
    bool skip_pin_{false};
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
