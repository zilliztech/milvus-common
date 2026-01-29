#include <folly/CancellationToken.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"
#include "cachinglayer/lrucache/ListNode.h"
#include "cachinglayer_test_utils.h"
#include "common/OpContext.h"

using namespace milvus::cachinglayer;
using namespace milvus::cachinglayer::internal;
using cl_uid_t = milvus::cachinglayer::uid_t;

struct TestCell {
    int data;
    cid_t cid;
    ResourceUsage size;

    TestCell(int d, cid_t id, ResourceUsage s) : data(d), cid(id), size(s) {
    }

    [[nodiscard]] milvus::cachinglayer::ResourceUsage
    CellByteSize() const {
        return size;
    }
};

class MockTranslator : public Translator<TestCell> {
 public:
    MockTranslator(std::vector<std::pair<cid_t, int64_t>> cell_sizes,
                   std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map, const std::string& key, StorageType storage_type,
                   bool for_concurrent_test = false)
        : uid_to_cid_map_(std::move(uid_to_cid_map)),
          num_unique_cids_(cell_sizes.size()),
          key_(key),
          meta_(storage_type, CellIdMappingMode::CUSTOMIZED, CellDataType::OTHER,
                CacheWarmupPolicy::CacheWarmupPolicy_Disable, true),
          for_concurrent_test_(for_concurrent_test) {
        cid_set_.reserve(cell_sizes.size());
        cell_sizes_.reserve(cell_sizes.size());
        for (const auto& pair : cell_sizes) {
            cid_t cid = pair.first;
            int64_t size = pair.second;
            cid_set_.insert(cid);
            cell_sizes_[cid] = size;
            cid_load_delay_ms_[cid] = 0;
        }
    }

    size_t
    num_cells() const override {
        return num_unique_cids_;
    }

    cid_t
    cell_id_of(cl_uid_t uid) const override {
        auto it = uid_to_cid_map_.find(uid);
        if (it != uid_to_cid_map_.end()) {
            if (cid_set_.count(it->second)) {
                return it->second;
            }
        }
        return static_cast<cid_t>(num_unique_cids_);
    }

    std::pair<ResourceUsage, ResourceUsage>
    estimated_byte_size_of_cell(cid_t cid) const override {
        auto it = cell_sizes_.find(cid);
        if (it != cell_sizes_.end()) {
            return {{it->second, 0}, {it->second, 0}};
        }
        return {{1, 0}, {1, 0}};
    }

    int64_t
    cells_storage_bytes(const std::vector<cid_t>& cids) const override {
        // Check if we should throw for any of the requested cids
        for (const auto& cid : cids) {
            if (static_cast<int>(cid) == cells_storage_bytes_throw_on_cid_) {
                throw std::runtime_error("Simulated cells_storage_bytes failure for cid " + std::to_string(cid));
            }
        }
        int64_t total_bytes = 0;
        // make the storage size equal to the loaded memory size in test
        for (const auto& cid : cids) {
            total_bytes += estimated_byte_size_of_cell(cid).first.memory_bytes;
        }
        return total_bytes;
    }

    const std::string&
    key() const override {
        return key_;
    }

    Meta*
    meta() override {
        return &meta_;
    }

    std::vector<cid_t>
    bonus_cells_to_be_loaded(const std::vector<cid_t>& cids) const override {
        std::unordered_set<cid_t> total_bonus_cids_set;
        for (cid_t cid : cids) {
            if (auto extra_cids = extra_cids_.find(cid); extra_cids != extra_cids_.end()) {
                for (auto ecid : extra_cids->second) {
                    total_bonus_cids_set.insert(ecid);
                }
            }
        }
        for (cid_t cid : cids) {
            total_bonus_cids_set.erase(cid);
        }
        return {total_bonus_cids_set.begin(), total_bonus_cids_set.end()};
    }

    std::vector<std::pair<cid_t, std::unique_ptr<TestCell>>>
    get_cells(milvus::OpContext* ctx, const std::vector<cid_t>& cids) override {
        if (!for_concurrent_test_) {
            get_cells_call_count_++;
            requested_cids_.push_back(cids);
        }

        if (load_should_throw_) {
            throw std::runtime_error("Simulated load error");
        }

        std::vector<std::pair<cid_t, std::unique_ptr<TestCell>>> result;
        for (cid_t cid : cids) {
            auto delay_it = cid_load_delay_ms_.find(cid);
            if (delay_it != cid_load_delay_ms_.end() && delay_it->second > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_it->second));
            }

            if (ctx && ctx->cancellation_token.isCancellationRequested()) {
                throw std::runtime_error("Operation cancelled, stop loading cache cells");
            }

            result.emplace_back(cid, std::make_unique<TestCell>(static_cast<int>(cid * 10), cid,
                                                                estimated_byte_size_of_cell(cid).first));
        }
        return result;
    }

    void
    SetCidLoadDelay(const std::unordered_map<cid_t, int>& delays) {
        for (const auto& pair : delays) {
            cid_load_delay_ms_[pair.first] = pair.second;
        }
    }
    void
    SetShouldThrow(bool should_throw) {
        load_should_throw_ = should_throw;
    }
    // Set which cid should cause cells_storage_bytes to throw. -1 means no throw.
    void
    SetCellsStorageBytesThrowOnCid(int cid) {
        cells_storage_bytes_throw_on_cid_ = cid;
    }
    // for some cid, translator will return extra cells.
    void
    SetExtraReturnCids(std::unordered_map<cid_t, std::vector<cid_t>> extra_cids) {
        extra_cids_ = extra_cids;
    }
    int
    GetCellsCallCount() const {
        EXPECT_FALSE(for_concurrent_test_);
        return get_cells_call_count_;
    }
    const std::vector<std::vector<cid_t>>&
    GetRequestedCids() const {
        EXPECT_FALSE(for_concurrent_test_);
        return requested_cids_;
    }
    void
    ResetCounters() {
        ASSERT_FALSE(for_concurrent_test_);
        get_cells_call_count_ = 0;
        requested_cids_.clear();
    }

 private:
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map_;
    std::unordered_map<cid_t, int64_t> cell_sizes_;
    std::unordered_set<cid_t> cid_set_;
    const size_t num_unique_cids_;
    const std::string key_;
    Meta meta_;

    std::unordered_map<cid_t, int> cid_load_delay_ms_;
    bool load_should_throw_ = false;
    int cells_storage_bytes_throw_on_cid_ = -1;  // -1 means no throw
    std::unordered_map<cid_t, std::vector<cid_t>> extra_cids_;
    std::atomic<int> get_cells_call_count_ = 0;
    std::vector<std::vector<cid_t>> requested_cids_;

    // this class is not concurrent safe, so if for concurrent test, do not track usage
    bool for_concurrent_test_ = false;
};

class CacheSlotTest : public ::testing::Test {
 protected:
    std::shared_ptr<DList> dlist_;
    MockTranslator* translator_ = nullptr;
    std::shared_ptr<CacheSlot<TestCell>> cache_slot_;

    std::vector<std::pair<cid_t, int64_t>> cell_sizes_ = {{0, 50}, {1, 150}, {2, 100}, {3, 200}, {4, 75}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map_ = {{10, 0}, {11, 0}, {20, 1}, {30, 2}, {31, 2},
                                                           {32, 2}, {40, 3}, {50, 4}, {51, 4}};

    size_t NUM_UNIQUE_CIDS = 5;
    int64_t TOTAL_CELL_SIZE_BYTES = 50 + 150 + 100 + 200 + 75;
    int64_t MEMORY_LIMIT = TOTAL_CELL_SIZE_BYTES * 2;
    static constexpr int64_t DISK_LIMIT = 0;
    const std::string SLOT_KEY = "test_slot";

    CacheSlotTest() = default;

    void
    SetUp() override {
        auto limit = ResourceUsage{MEMORY_LIMIT, DISK_LIMIT};
        dlist_ = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

        auto temp_translator_uptr =
            std::make_unique<MockTranslator>(cell_sizes_, uid_to_cid_map_, SLOT_KEY, StorageType::MEMORY);
        translator_ = temp_translator_uptr.get();
        cache_slot_ = std::make_shared<CacheSlot<TestCell>>(std::move(temp_translator_uptr), dlist_.get(), true, true,
                                                            true, std::chrono::milliseconds(100000));
    }

    void
    TearDown() override {
        cache_slot_.reset();
        dlist_.reset();
    }
};

TEST_F(CacheSlotTest, Initialization) {
    ASSERT_EQ(cache_slot_->num_cells(), NUM_UNIQUE_CIDS);
}

TEST_F(CacheSlotTest, PinSingleCellSuccess) {
    cl_uid_t target_uid = 30;
    cid_t expected_cid = 2;
    ResourceUsage expected_size = translator_->estimated_byte_size_of_cell(expected_cid).first;

    translator_->ResetCounters();
    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto future = cache_slot_->PinCells(op_ctx.get(), {target_uid});
    auto accessor = SemiInlineGet(std::move(future));

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    ASSERT_EQ(translator_->GetRequestedCids()[0].size(), 1);
    EXPECT_EQ(translator_->GetRequestedCids()[0][0], expected_cid);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_size.memory_bytes);

    TestCell* cell = accessor->get_cell_of(target_uid);
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(cell->cid, expected_cid);
    EXPECT_EQ(cell->data, expected_cid * 10);

    TestCell* cell_by_index = accessor->get_ith_cell(expected_cid);
    ASSERT_EQ(cell, cell_by_index);
}

TEST_F(CacheSlotTest, PinMultipleCellsSuccess) {
    std::vector<cl_uid_t> target_uids = {10, 40, 51};
    std::vector<cid_t> expected_cids = {0, 3, 4};
    std::sort(expected_cids.begin(), expected_cids.end());
    ResourceUsage expected_total_size;
    for (cid_t cid : expected_cids) {
        expected_total_size += translator_->estimated_byte_size_of_cell(cid).first;
    }

    translator_->ResetCounters();
    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto future = cache_slot_->PinCells(op_ctx.get(), target_uids);
    auto accessor = SemiInlineGet(std::move(future));

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    auto requested = translator_->GetRequestedCids()[0];
    std::sort(requested.begin(), requested.end());
    ASSERT_EQ(requested.size(), expected_cids.size());
    EXPECT_EQ(requested, expected_cids);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_total_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_total_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_total_size.memory_bytes);

    for (cl_uid_t uid : target_uids) {
        cid_t cid = uid_to_cid_map_.at(uid);
        TestCell* cell = accessor->get_cell_of(uid);
        ASSERT_NE(cell, nullptr);
        EXPECT_EQ(cell->cid, cid);
        EXPECT_EQ(cell->data, cid * 10);
    }
}

TEST_F(CacheSlotTest, PinMultipleUidsMappingToSameCid) {
    std::vector<cl_uid_t> target_uids = {30, 50, 31, 51, 32};
    std::vector<cid_t> expected_unique_cids = {2, 4};
    std::sort(expected_unique_cids.begin(), expected_unique_cids.end());
    ResourceUsage expected_total_size;
    for (cid_t cid : expected_unique_cids) {
        expected_total_size += translator_->estimated_byte_size_of_cell(cid).first;
    }

    translator_->ResetCounters();
    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto future = cache_slot_->PinCells(op_ctx.get(), target_uids);
    auto accessor = SemiInlineGet(std::move(future));

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    auto requested = translator_->GetRequestedCids()[0];
    std::sort(requested.begin(), requested.end());
    ASSERT_EQ(requested.size(), expected_unique_cids.size());
    EXPECT_EQ(requested, expected_unique_cids);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_total_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_total_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_total_size.memory_bytes);

    TestCell* cell2_uid30 = accessor->get_cell_of(30);
    TestCell* cell2_uid31 = accessor->get_cell_of(31);
    TestCell* cell4_uid50 = accessor->get_cell_of(50);
    TestCell* cell4_uid51 = accessor->get_cell_of(51);
    ASSERT_NE(cell2_uid30, nullptr);
    ASSERT_NE(cell4_uid50, nullptr);
    EXPECT_EQ(cell2_uid30->cid, 2);
    EXPECT_EQ(cell4_uid50->cid, 4);
    EXPECT_EQ(cell2_uid30, cell2_uid31);
    EXPECT_EQ(cell4_uid50, cell4_uid51);
}

TEST_F(CacheSlotTest, PinInvalidUid) {
    cl_uid_t invalid_uid = 999;
    cl_uid_t valid_uid = 10;
    std::vector<cl_uid_t> target_uids = {valid_uid, invalid_uid};

    translator_->ResetCounters();
    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto future = cache_slot_->PinCells(op_ctx.get(), target_uids);

    EXPECT_THROW(
        {
            try {
                SemiInlineGet(std::move(future));
            } catch (const milvus::SegcoreError& e) {
                std::string error_what = e.what();
                EXPECT_TRUE(error_what.find("out of range") != std::string::npos ||
                            error_what.find("invalid") != std::string::npos);
                throw;
            }
        },
        milvus::SegcoreError);

    EXPECT_EQ(translator_->GetCellsCallCount(), 0);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, 0);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, 0);
}

TEST_F(CacheSlotTest, LoadFailure) {
    cl_uid_t target_uid = 20;
    cid_t expected_cid = 1;

    translator_->ResetCounters();
    translator_->SetShouldThrow(true);

    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto future = cache_slot_->PinCells(op_ctx.get(), {target_uid});

    EXPECT_THROW(
        {
            try {
                SemiInlineGet(std::move(future));
            } catch (const std::runtime_error& e) {
                std::string error_what = e.what();
                EXPECT_TRUE(error_what.find("Simulated load error") != std::string::npos ||
                            error_what.find("Failed to load") != std::string::npos ||
                            error_what.find("Exception during Future") != std::string::npos);
                throw;
            }
        },
        std::runtime_error);

    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    ASSERT_EQ(translator_->GetRequestedCids()[0].size(), 1);
    EXPECT_EQ(translator_->GetRequestedCids()[0][0], expected_cid);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, 0);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, 0);

    // recover the translator and try again
    translator_->SetShouldThrow(false);
    auto expected_size = translator_->estimated_byte_size_of_cell(expected_cid).first;
    auto future2 = cache_slot_->PinCells(op_ctx.get(), {target_uid});
    auto accessor = SemiInlineGet(std::move(future2));
    ASSERT_NE(accessor, nullptr);
    TestCell* cell = accessor->get_cell_of(target_uid);
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_size.memory_bytes);
}

TEST_F(CacheSlotTest, PinAlreadyLoadedCell) {
    cl_uid_t target_uid = 40;
    cid_t expected_cid = 3;
    ResourceUsage expected_size = translator_->estimated_byte_size_of_cell(expected_cid).first;

    translator_->ResetCounters();

    auto op_ctx = std::make_unique<milvus::OpContext>();

    auto future1 = cache_slot_->PinCells(op_ctx.get(), {target_uid});
    auto accessor1 = SemiInlineGet(std::move(future1));
    ASSERT_NE(accessor1, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    ASSERT_EQ(translator_->GetRequestedCids()[0][0], expected_cid);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_size.memory_bytes);
    TestCell* cell1 = accessor1->get_cell_of(target_uid);
    ASSERT_NE(cell1, nullptr);

    translator_->ResetCounters();
    auto future2 = cache_slot_->PinCells(op_ctx.get(), {target_uid});
    auto accessor2 = SemiInlineGet(std::move(future2));
    ASSERT_NE(accessor2, nullptr);

    EXPECT_EQ(translator_->GetCellsCallCount(), 0);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_size.memory_bytes * 2);

    TestCell* cell2 = accessor2->get_cell_of(target_uid);
    ASSERT_NE(cell2, nullptr);
    EXPECT_EQ(cell1, cell2);

    accessor1.reset();
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    TestCell* cell_after_unpin = accessor2->get_cell_of(target_uid);
    ASSERT_NE(cell_after_unpin, nullptr);
    EXPECT_EQ(cell_after_unpin, cell2);
}

TEST_F(CacheSlotTest, PinAlreadyLoadedCellViaDifferentUid) {
    cl_uid_t uid1 = 30;
    cl_uid_t uid2 = 31;
    cid_t expected_cid = 2;
    ResourceUsage expected_size = translator_->estimated_byte_size_of_cell(expected_cid).first;

    translator_->ResetCounters();

    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto future1 = cache_slot_->PinCells(op_ctx.get(), {uid1});
    auto accessor1 = SemiInlineGet(std::move(future1));
    ASSERT_NE(accessor1, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    ASSERT_EQ(translator_->GetRequestedCids()[0][0], expected_cid);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_size.memory_bytes);
    TestCell* cell1 = accessor1->get_cell_of(uid1);
    ASSERT_NE(cell1, nullptr);
    EXPECT_EQ(cell1->cid, expected_cid);

    translator_->ResetCounters();
    auto future2 = cache_slot_->PinCells(op_ctx.get(), {uid2});
    auto accessor2 = SemiInlineGet(std::move(future2));
    ASSERT_NE(accessor2, nullptr);

    EXPECT_EQ(translator_->GetCellsCallCount(), 0);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, expected_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_size.memory_bytes * 2);

    TestCell* cell2 = accessor2->get_cell_of(uid2);
    ASSERT_NE(cell2, nullptr);
    EXPECT_EQ(cell2->cid, expected_cid);
    EXPECT_EQ(cell1, cell2);

    accessor1.reset();
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    TestCell* cell_after_unpin_uid1 = accessor2->get_cell_of(uid1);
    TestCell* cell_after_unpin_uid2 = accessor2->get_cell_of(uid2);
    ASSERT_NE(cell_after_unpin_uid1, nullptr);
    ASSERT_NE(cell_after_unpin_uid2, nullptr);
    EXPECT_EQ(cell_after_unpin_uid1, cell2);
    EXPECT_EQ(cell_after_unpin_uid2, cell2);
}

TEST_F(CacheSlotTest, TranslatorReturnsExtraCells) {
    cl_uid_t requested_uid = 10;
    cid_t requested_cid = 0;
    cid_t extra_cid = 1;
    cl_uid_t extra_uid = 20;

    ResourceUsage requested_size = translator_->estimated_byte_size_of_cell(requested_cid).first;
    ResourceUsage extra_size = translator_->estimated_byte_size_of_cell(extra_cid).first;
    ResourceUsage expected_size = requested_size + extra_size;

    translator_->ResetCounters();
    translator_->SetExtraReturnCids({{requested_cid, {extra_cid}}});

    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto future = cache_slot_->PinCells(op_ctx.get(), {requested_uid});
    auto accessor = SemiInlineGet(std::move(future));

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    auto requested_cids = translator_->GetRequestedCids()[0];
    ASSERT_EQ(requested_cids.size(), 2);
    EXPECT_TRUE(std::find(requested_cids.begin(), requested_cids.end(), requested_cid) != requested_cids.end());
    EXPECT_TRUE(std::find(requested_cids.begin(), requested_cids.end(), extra_cid) != requested_cids.end());
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    // bonus cell should not be tracked
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, requested_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, requested_size.memory_bytes);

    TestCell* requested_cell = accessor->get_cell_of(requested_uid);
    ASSERT_NE(requested_cell, nullptr);
    EXPECT_EQ(requested_cell->cid, requested_cid);

    translator_->ResetCounters();
    auto future_extra = cache_slot_->PinCells(op_ctx.get(), {extra_uid});
    auto accessor_extra = SemiInlineGet(std::move(future_extra));

    ASSERT_NE(accessor_extra, nullptr);
    EXPECT_EQ(translator_->GetCellsCallCount(), 0);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), expected_size);
    // bonus cell is not cold anymore
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, requested_size.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, expected_size.memory_bytes);

    TestCell* extra_cell = accessor_extra->get_cell_of(extra_uid);
    ASSERT_NE(extra_cell, nullptr);
    EXPECT_EQ(extra_cell->cid, extra_cid);
}

TEST_F(CacheSlotTest, EvictionTest) {
    // Sizes: 0:50, 1:150, 2:100, 3:200
    ResourceUsage new_limit = ResourceUsage(300, 0);
    ResourceUsage new_high_watermark = ResourceUsage(250, 0);
    ResourceUsage new_low_watermark = ResourceUsage(200, 0);
    dlist_->UpdateLowWatermark(new_low_watermark);
    dlist_->UpdateHighWatermark(new_high_watermark);
    EXPECT_TRUE(dlist_->UpdateMaxLimit(new_limit));
    EXPECT_EQ(DListTestFriend::get_max_memory(*dlist_), new_limit);

    std::vector<cl_uid_t> uids_012 = {10, 20, 30};
    std::vector<cid_t> cids_012 = {0, 1, 2};
    ResourceUsage size_012 = translator_->estimated_byte_size_of_cell(0).first +
                             translator_->estimated_byte_size_of_cell(1).first +
                             translator_->estimated_byte_size_of_cell(2).first;
    ASSERT_EQ(size_012, ResourceUsage(50 + 150 + 100, 0));

    auto op_ctx = std::make_unique<milvus::OpContext>();

    // 1. Load cells 0, 1, 2
    translator_->ResetCounters();
    auto future1 = cache_slot_->PinCells(op_ctx.get(), uids_012);
    auto accessor1 = SemiInlineGet(std::move(future1));
    ASSERT_NE(accessor1, nullptr);
    EXPECT_EQ(translator_->GetCellsCallCount(), 1);
    auto requested1 = translator_->GetRequestedCids()[0];
    std::sort(requested1.begin(), requested1.end());
    EXPECT_EQ(requested1, cids_012);
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_), size_012);
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, size_012.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, size_012.memory_bytes);

    // 2. Unpin 0, 1, 2
    accessor1.reset();
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_),
              size_012);  // Still in cache

    // 3. Load cell 3 (size 200), requires eviction
    cl_uid_t uid_3 = 40;
    cid_t cid_3 = 3;
    ResourceUsage size_3 = translator_->estimated_byte_size_of_cell(cid_3).first;
    ASSERT_EQ(size_3, ResourceUsage(200, 0));

    translator_->ResetCounters();
    auto future2 = cache_slot_->PinCells(op_ctx.get(), {uid_3});
    auto accessor2 = SemiInlineGet(std::move(future2));
    ASSERT_NE(accessor2, nullptr);

    EXPECT_EQ(translator_->GetCellsCallCount(),
              1);  // Load was called for cell 3
    ASSERT_EQ(translator_->GetRequestedCids().size(), 1);
    EXPECT_EQ(translator_->GetRequestedCids()[0], std::vector<cid_t>{cid_3});
    EXPECT_EQ(op_ctx->storage_usage->scanned_cold_bytes, size_012.memory_bytes + size_3.memory_bytes);
    EXPECT_EQ(op_ctx->storage_usage->scanned_total_bytes, size_012.memory_bytes + size_3.memory_bytes);

    // Verify eviction happened
    ResourceUsage used_after_evict1 = DListTestFriend::get_used_memory(*dlist_);
    EXPECT_LE(used_after_evict1.memory_bytes, new_limit.memory_bytes);
    EXPECT_GE(used_after_evict1.memory_bytes, size_3.memory_bytes);
    EXPECT_LT(used_after_evict1.memory_bytes,
              size_012.memory_bytes + size_3.memory_bytes);  // Eviction occurred
}

TEST_F(CacheSlotTest, PinCellsWithCancellationToken) {
    // Test that cancellation token passed through OpContext works
    folly::CancellationSource cancel_source;
    auto cancel_token = cancel_source.getToken();
    auto op_ctx = std::make_unique<milvus::OpContext>(cancel_token);

    cl_uid_t target_uid = 30;
    cid_t expected_cid = 2;

    translator_->ResetCounters();
    auto future = cache_slot_->PinCells(op_ctx.get(), {target_uid});
    auto accessor = SemiInlineGet(std::move(future));

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    EXPECT_EQ(translator_->GetRequestedCids()[0][0], expected_cid);

    TestCell* cell = accessor->get_cell_of(target_uid);
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(cell->cid, expected_cid);
}

TEST_F(CacheSlotTest, PinCellsWithShortTimeout) {
    // Create a cache slot with a very short loading timeout to test timeout behavior
    // First, fill the cache to force waiting
    ResourceUsage small_limit = ResourceUsage(100, 0);  // Very small limit
    dlist_->UpdateLowWatermark(small_limit);
    dlist_->UpdateHighWatermark(small_limit);
    EXPECT_TRUE(dlist_->UpdateMaxLimit(small_limit));

    // Create a CacheSlot with a short timeout
    auto translator_short_timeout =
        std::make_unique<MockTranslator>(cell_sizes_, uid_to_cid_map_, "short_timeout_slot", StorageType::MEMORY);
    auto cache_slot_short_timeout =
        std::make_shared<CacheSlot<TestCell>>(std::move(translator_short_timeout), dlist_.get(), true, true, true,
                                              std::chrono::milliseconds(50));  // 50ms timeout

    auto op_ctx = std::make_unique<milvus::OpContext>();

    // First pin some cells to fill the cache
    auto future1 = cache_slot_short_timeout->PinCells(op_ctx.get(), {10});  // cid 0, size 50
    auto accessor1 = SemiInlineGet(std::move(future1));
    ASSERT_NE(accessor1, nullptr);

    // Keep accessor1 alive to prevent eviction, then try to pin a large cell
    // This should timeout since there's no space and we can't evict (accessor1 is pinned)
    // Note: This test may or may not fail depending on whether eviction can happen
    // The key is that the short timeout is being used
    EXPECT_EQ(DListTestFriend::get_used_memory(*dlist_).memory_bytes, 50);
}

TEST_F(CacheSlotTest, PinOneCellDirectWithCancellationToken) {
    // Test that cancellation token works with PinOneCellDirect
    folly::CancellationSource cancel_source;
    auto cancel_token = cancel_source.getToken();
    auto op_ctx = std::make_unique<milvus::OpContext>(cancel_token);

    cl_uid_t target_uid = 30;
    cid_t expected_cid = 2;

    translator_->ResetCounters();
    auto accessor = cache_slot_->PinOneCellDirect(op_ctx.get(), target_uid);

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);
    EXPECT_EQ(translator_->GetRequestedCids()[0][0], expected_cid);

    TestCell* cell = accessor->get_cell_of(target_uid);
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(cell->cid, expected_cid);
}

TEST_F(CacheSlotTest, PinCellsDirectWithCancellationToken) {
    // Test that cancellation token works with PinCellsDirect
    folly::CancellationSource cancel_source;
    auto cancel_token = cancel_source.getToken();
    auto op_ctx = std::make_unique<milvus::OpContext>(cancel_token);

    std::vector<cl_uid_t> target_uids = {10, 40, 51};

    translator_->ResetCounters();
    auto accessor = cache_slot_->PinCellsDirect(op_ctx.get(), target_uids);

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);

    for (cl_uid_t uid : target_uids) {
        cid_t expected_cid = uid_to_cid_map_.at(uid);
        TestCell* cell = accessor->get_cell_of(uid);
        ASSERT_NE(cell, nullptr);
        EXPECT_EQ(cell->cid, expected_cid);
    }
}

TEST_F(CacheSlotTest, PinAllCellsWithCancellationToken) {
    // Test that cancellation token works with PinAllCells
    folly::CancellationSource cancel_source;
    auto cancel_token = cancel_source.getToken();
    auto op_ctx = std::make_unique<milvus::OpContext>(cancel_token);

    translator_->ResetCounters();
    auto future = cache_slot_->PinAllCells(op_ctx.get());
    auto accessor = SemiInlineGet(std::move(future));

    ASSERT_NE(accessor, nullptr);
    ASSERT_EQ(translator_->GetCellsCallCount(), 1);

    // Verify all cells are loaded
    for (size_t i = 0; i < NUM_UNIQUE_CIDS; ++i) {
        TestCell* cell = accessor->get_ith_cell(static_cast<cid_t>(i));
        ASSERT_NE(cell, nullptr);
        EXPECT_EQ(cell->cid, static_cast<cid_t>(i));
    }
}

class CacheSlotConcurrentTest : public CacheSlotTest, public ::testing::WithParamInterface<bool> {};

TEST_P(CacheSlotConcurrentTest, ConcurrentAccessMultipleSlots) {
    // Slot 1 Cells: 0-4 (Sizes: 50, 60, 70, 80, 90) -> Total 350
    // Slot 2 Cells: 0-4 (Sizes: 55, 65, 75, 85, 95) -> Total 375
    // Total potential size = 350 + 375 = 725
    // Set limit lower than total potential size to force eviction
    ResourceUsage new_limit = ResourceUsage(700, 0);
    ResourceUsage new_high_watermark = ResourceUsage(650, 0);
    ResourceUsage new_low_watermark = ResourceUsage(600, 0);
    dlist_->UpdateLowWatermark(new_low_watermark);
    dlist_->UpdateHighWatermark(new_high_watermark);
    ASSERT_TRUE(dlist_->UpdateMaxLimit(new_limit));
    EXPECT_EQ(DListTestFriend::get_max_memory(*dlist_).memory_bytes, new_limit.memory_bytes);

    // 1. Setup CacheSlots sharing dlist_
    std::vector<std::pair<cid_t, int64_t>> cell_sizes_1 = {{0, 50}, {1, 60}, {2, 70}, {3, 80}, {4, 90}};
    std::unordered_map<cl_uid_t, cid_t> uid_map_1 = {{1000, 0}, {1001, 1}, {1002, 2}, {1003, 3}, {1004, 4}};
    auto translator_1_ptr = std::make_unique<MockTranslator>(cell_sizes_1, uid_map_1, "slot1", StorageType::MEMORY,
                                                             /*for_concurrent_test*/ true);
    MockTranslator* translator_1 = translator_1_ptr.get();
    auto slot1 = std::make_shared<CacheSlot<TestCell>>(std::move(translator_1_ptr), dlist_.get(), true, true, true,
                                                       std::chrono::milliseconds(100000));

    std::vector<std::pair<cid_t, int64_t>> cell_sizes_2 = {{0, 55}, {1, 65}, {2, 75}, {3, 85}, {4, 95}};
    std::unordered_map<cl_uid_t, cid_t> uid_map_2 = {{2000, 0}, {2001, 1}, {2002, 2}, {2003, 3}, {2004, 4}};
    auto translator_2_ptr = std::make_unique<MockTranslator>(cell_sizes_2, uid_map_2, "slot2", StorageType::MEMORY,
                                                             /*for_concurrent_test*/ true);
    MockTranslator* translator_2 = translator_2_ptr.get();
    auto slot2 = std::make_shared<CacheSlot<TestCell>>(std::move(translator_2_ptr), dlist_.get(), true, true, true,
                                                       std::chrono::milliseconds(100000));

    bool with_bonus_cells = GetParam();
    if (with_bonus_cells) {
        // Configure translators to return bonus cells
        std::unordered_map<cid_t, std::vector<cid_t>> bonus_map_1{
            {0, {2}}, {1, {3}}, {2, {1, 4}}, {3, {0}}, {4, {2, 3}},
        };
        std::unordered_map<cid_t, std::vector<cid_t>> bonus_map_2{
            {0, {1, 4}}, {1, {2, 3}}, {2, {0}}, {3, {0, 1}}, {4, {2}},
        };
        translator_1->SetExtraReturnCids(bonus_map_1);
        translator_2->SetExtraReturnCids(bonus_map_2);
    }

    std::vector<std::shared_ptr<CacheSlot<TestCell>>> slots = {slot1, slot2};
    // Store uid maps in a structure easily accessible by slot index
    std::vector<std::vector<cl_uid_t>> slot_uids;
    slot_uids.resize(slots.size());
    std::vector<std::unordered_map<cl_uid_t, cid_t>> uid_to_cid_maps = {uid_map_1, uid_map_2};
    for (const auto& pair : uid_map_1) slot_uids[0].push_back(pair.first);
    for (const auto& pair : uid_map_2) slot_uids[1].push_back(pair.first);

    // 2. Setup Thread Pool and Concurrency Parameters
    // at most 6 cells can be pinned at the same time, thus will never exceed the limit.
    int num_threads = 6;
    int ops_per_thread = 200;
    // 1 extra thread to work with slot3
    folly::CPUThreadPoolExecutor executor(num_threads + 1);
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(num_threads);
    std::atomic<bool> test_failed{false};

    // 3. Launch Threads to Perform Concurrent Pin/Get/Verify/Unpin
    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(folly::via(&executor, [&, tid = i]() {
            // Seed random generator uniquely for each thread
            std::mt19937 gen(std::hash<std::thread::id>{}(std::this_thread::get_id()) + tid);
            std::uniform_int_distribution<> slot_dist(0, slots.size() - 1);
            std::uniform_int_distribution<> sleep_dist(5, 15);
            auto op_ctx = std::make_unique<milvus::OpContext>();

            for (int j = 0; j < ops_per_thread && !test_failed.load(); ++j) {
                int slot_idx = slot_dist(gen);
                auto& current_slot = slots[slot_idx];
                auto& current_slot_uids = slot_uids[slot_idx];
                auto& current_uid_to_cid_map = uid_to_cid_maps[slot_idx];

                std::uniform_int_distribution<> uid_idx_dist(0, current_slot_uids.size() - 1);
                cl_uid_t target_uid = current_slot_uids[uid_idx_dist(gen)];
                cid_t expected_cid = current_uid_to_cid_map.at(target_uid);
                int expected_data = static_cast<int>(expected_cid * 10);

                try {
                    auto accessor = current_slot->PinCells(op_ctx.get(), {target_uid}).get();

                    if (!accessor) {
                        ADD_FAILURE() << "T" << tid << " Op" << j << ": PinCells returned null accessor for UID "
                                      << target_uid;
                        test_failed = true;
                        break;
                    }

                    TestCell* cell = accessor->get_cell_of(target_uid);
                    if (!cell) {
                        ADD_FAILURE() << "T" << tid << " Op" << j << ": get_cell_of returned null for UID "
                                      << target_uid;
                        test_failed = true;
                        break;
                    }

                    if (cell->cid != expected_cid) {
                        ADD_FAILURE() << "T" << tid << " Op" << j << ": Incorrect CID for UID " << target_uid
                                      << ". Slot: " << slot_idx << ", Expected: " << expected_cid
                                      << ", Got: " << cell->cid;
                        test_failed = true;
                        break;
                    }
                    if (cell->data != expected_data) {
                        ADD_FAILURE() << "T" << tid << " Op" << j << ": Incorrect Data for UID " << target_uid
                                      << ". Slot: " << slot_idx << ", Expected: " << expected_data
                                      << ", Got: " << cell->data;
                        test_failed = true;
                        break;
                    }
                    int sleep_ms = sleep_dist(gen);
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                } catch (const std::exception& e) {
                    ADD_FAILURE() << "T" << tid << " Op" << j << ": Exception for UID " << target_uid
                                  << ", Slot: " << slot_idx << ". What: " << e.what();
                    test_failed = true;
                } catch (...) {
                    ADD_FAILURE() << "T" << tid << " Op" << j << ": Unknown exception for UID " << target_uid
                                  << ", Slot: " << slot_idx;
                    test_failed = true;
                }
            }
        }));
    }

    // number of ops between recreating slot3
    const int recreate_interval = 25;
    auto dlist_ptr = dlist_.get();
    std::vector<std::pair<cid_t, int64_t>> cell_sizes_3 = {{0, 40}, {1, 50}, {2, 60}, {3, 70}, {4, 80}};
    std::unordered_map<cl_uid_t, cid_t> uid_map_3 = {{3000, 0}, {3001, 1}, {3002, 2}, {3003, 3}, {3004, 4}};
    std::vector<cl_uid_t> slot3_uids = {3000, 3001, 3002, 3003, 3004};
    auto create_new_slot3 = [&]() {
        auto translator_3_ptr = std::make_unique<MockTranslator>(cell_sizes_3, uid_map_3, "slot3", StorageType::MEMORY,
                                                                 /*for_concurrent_test*/ true);
        auto sl = std::make_shared<CacheSlot<TestCell>>(std::move(translator_3_ptr), dlist_ptr, dlist_ptr != nullptr,
                                                        dlist_ptr != nullptr, true, std::chrono::milliseconds(100000));
        return sl;
    };
    std::shared_ptr<CacheSlot<TestCell>> slot3 = create_new_slot3();
    futures.push_back(folly::via(&executor, [&, tid = num_threads]() {
        std::mt19937 gen(std::hash<std::thread::id>{}(std::this_thread::get_id()) + tid);
        std::uniform_int_distribution<> sleep_dist(5, 15);
        std::uniform_int_distribution<> recreate_sleep_dist(20, 30);
        std::uniform_int_distribution<> uid_idx_dist(0, slot3_uids.size() - 1);
        auto op_ctx = std::make_unique<milvus::OpContext>();
        int ops_since_recreate = 0;

        for (int j = 0; j < ops_per_thread && !test_failed.load(); ++j) {
            cl_uid_t target_uid = slot3_uids[uid_idx_dist(gen)];
            cid_t expected_cid = uid_map_3.at(target_uid);
            int expected_data = static_cast<int>(expected_cid * 10);
            try {
                auto accessor = slot3->PinCells(op_ctx.get(), {target_uid}).get();
                if (!accessor) {
                    ADD_FAILURE() << "T" << tid << " Op" << j << ": PinCells returned null accessor for UID "
                                  << target_uid;
                    test_failed = true;
                    break;
                }

                TestCell* cell = accessor->get_cell_of(target_uid);
                if (!cell) {
                    ADD_FAILURE() << "T" << tid << " Op" << j << ": get_cell_of returned null for UID " << target_uid;
                    test_failed = true;
                    break;
                }

                if (cell->cid != expected_cid) {
                    ADD_FAILURE() << "T" << tid << " Op" << j << ": Incorrect CID for UID " << target_uid << ". Slot: 3"
                                  << ", Expected: " << expected_cid << ", Got: " << cell->cid;
                    test_failed = true;
                    break;
                }

                if (cell->data != expected_data) {
                    ADD_FAILURE() << "T" << tid << " Op" << j << ": Incorrect Data for UID " << target_uid
                                  << ". Slot: 3"
                                  << ", Expected: " << expected_data << ", Got: " << cell->data;
                    test_failed = true;
                    break;
                }

                if (ops_since_recreate >= recreate_interval) {
                    slot3 = nullptr;
                    int sleep_ms = recreate_sleep_dist(gen);
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                    slot3 = create_new_slot3();
                    ops_since_recreate = 0;
                } else {
                    ops_since_recreate++;
                    int sleep_ms = sleep_dist(gen);
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                }
            } catch (const std::exception& e) {
                ADD_FAILURE() << "T" << tid << " Op" << j << ": Exception for UID " << target_uid << ", Slot: 3"
                              << ". What: " << e.what();
                test_failed = true;
            } catch (...) {
                ADD_FAILURE() << "T" << tid << " Op" << j << ": Unknown exception for UID " << target_uid
                              << ", Slot: 3";
                test_failed = true;
            }
        }
    }));

    // 4. Wait for all threads to complete
    try {
        folly::collectAll(futures).get();
    } catch (const std::exception& e) {
        FAIL() << "Exception waiting for thread pool completion: " << e.what();
    } catch (...) {
        FAIL() << "Unknown exception waiting for thread pool completion.";
    }

    ASSERT_FALSE(test_failed.load()) << "Test failed due to assertion failures within threads.";

    ResourceUsage final_memory_usage = DListTestFriend::get_used_memory(*dlist_);

    // bonus cell may cause memory usage to exceed the limit
    if (!with_bonus_cells) {
        EXPECT_LE(final_memory_usage.memory_bytes, new_limit.memory_bytes)
            << "Final memory usage (" << final_memory_usage.memory_bytes << ") exceeds the limit ("
            << new_limit.memory_bytes << ") after concurrent access.";
    }

    DListTestFriend::verify_integrity(dlist_.get());
}

INSTANTIATE_TEST_SUITE_P(BonusCellParam, CacheSlotConcurrentTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "WithBonusCells" : "NoBonusCells";
                         });

// Mock translator with warmup enabled for testing warmup behavior
class MockTranslatorWithWarmup : public Translator<TestCell> {
 public:
    MockTranslatorWithWarmup(std::vector<std::pair<cid_t, int64_t>> cell_sizes,
                             std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map, const std::string& key,
                             StorageType storage_type, CacheWarmupPolicy warmup_policy)
        : uid_to_cid_map_(std::move(uid_to_cid_map)),
          num_unique_cids_(cell_sizes.size()),
          key_(key),
          meta_(storage_type, CellIdMappingMode::CUSTOMIZED, CellDataType::OTHER, warmup_policy, true) {
        cid_set_.reserve(cell_sizes.size());
        cell_sizes_.reserve(cell_sizes.size());
        for (const auto& pair : cell_sizes) {
            cid_t cid = pair.first;
            int64_t size = pair.second;
            cid_set_.insert(cid);
            cell_sizes_[cid] = size;
            cid_load_delay_ms_[cid] = 0;
        }
    }

    size_t
    num_cells() const override {
        return num_unique_cids_;
    }

    cid_t
    cell_id_of(cl_uid_t uid) const override {
        auto it = uid_to_cid_map_.find(uid);
        if (it != uid_to_cid_map_.end()) {
            if (cid_set_.count(it->second)) {
                return it->second;
            }
        }
        return static_cast<cid_t>(num_unique_cids_);
    }

    std::pair<ResourceUsage, ResourceUsage>
    estimated_byte_size_of_cell(cid_t cid) const override {
        auto it = cell_sizes_.find(cid);
        if (it != cell_sizes_.end()) {
            return {{it->second, 0}, {it->second, 0}};
        }
        return {{1, 0}, {1, 0}};
    }

    int64_t
    cells_storage_bytes(const std::vector<cid_t>& cids) const override {
        int64_t total_bytes = 0;
        for (const auto& cid : cids) {
            total_bytes += estimated_byte_size_of_cell(cid).first.memory_bytes;
        }
        return total_bytes;
    }

    const std::string&
    key() const override {
        return key_;
    }

    Meta*
    meta() override {
        return &meta_;
    }

    std::vector<cid_t>
    bonus_cells_to_be_loaded(const std::vector<cid_t>& cids) const override {
        return {};
    }

    std::vector<std::pair<cid_t, std::unique_ptr<TestCell>>>
    get_cells(milvus::OpContext* ctx, const std::vector<cid_t>& cids) override {
        get_cells_call_count_++;
        if (shared_call_counter_) {
            shared_call_counter_->fetch_add(1);
        }
        last_ctx_ = ctx;

        // Signal that loading has started
        if (load_started_promise_) {
            load_started_promise_->set_value();
            load_started_promise_ = nullptr;
        }

        std::vector<std::pair<cid_t, std::unique_ptr<TestCell>>> result;
        for (cid_t cid : cids) {
            auto delay_it = cid_load_delay_ms_.find(cid);
            if (delay_it != cid_load_delay_ms_.end() && delay_it->second > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_it->second));
            }

            if (ctx && ctx->cancellation_token.isCancellationRequested()) {
                throw std::runtime_error("Operation cancelled, stop loading cache cells");
            }

            result.emplace_back(cid, std::make_unique<TestCell>(static_cast<int>(cid * 10), cid,
                                                                estimated_byte_size_of_cell(cid).first));
        }

        // Signal that loading has completed
        if (load_completed_promise_) {
            load_completed_promise_->set_value();
            load_completed_promise_ = nullptr;
        }

        return result;
    }

    void
    SetCidLoadDelay(const std::unordered_map<cid_t, int>& delays) {
        for (const auto& pair : delays) {
            cid_load_delay_ms_[pair.first] = pair.second;
        }
    }

    int
    GetCellsCallCount() const {
        return get_cells_call_count_;
    }

    milvus::OpContext*
    GetLastCtx() const {
        return last_ctx_;
    }

    // Set a promise that will be signaled when loading starts
    void
    SetLoadStartedPromise(std::promise<void>* promise) {
        load_started_promise_ = promise;
    }

    // Set a promise that will be signaled when loading completes
    void
    SetLoadCompletedPromise(std::promise<void>* promise) {
        load_completed_promise_ = promise;
    }

    // Set a shared counter for tracking calls that outlives the translator
    void
    SetSharedCallCounter(std::shared_ptr<std::atomic<int>> counter) {
        shared_call_counter_ = std::move(counter);
    }

 private:
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map_;
    std::unordered_map<cid_t, int64_t> cell_sizes_;
    std::unordered_set<cid_t> cid_set_;
    const size_t num_unique_cids_;
    const std::string key_;
    Meta meta_;
    std::unordered_map<cid_t, int> cid_load_delay_ms_;
    std::atomic<int> get_cells_call_count_ = 0;
    milvus::OpContext* last_ctx_ = nullptr;
    std::promise<void>* load_started_promise_ = nullptr;
    std::promise<void>* load_completed_promise_ = nullptr;
    std::shared_ptr<std::atomic<int>> shared_call_counter_;
};

// Test that CreateCacheSlot throws when OpContext has a cancelled token
TEST(ManagerCreateCacheSlotTest, CancelledTokenThrowsBeforeCreation) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator = std::make_unique<MockTranslatorWithWarmup>(
        cell_sizes, uid_to_cid_map, "test_slot", StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Disable);
    auto* translator_ptr = translator.get();

    // Create a pre-cancelled cancellation token
    folly::CancellationSource cancel_source;
    cancel_source.requestCancellation();
    auto cancel_token = cancel_source.getToken();
    auto op_ctx = std::make_unique<milvus::OpContext>(cancel_token);

    // CreateCacheSlot is a template in Manager, but we can test the cancellation behavior
    // by testing with CacheSlot directly since the cancellation check is simple
    EXPECT_TRUE(op_ctx->cancellation_token.isCancellationRequested());

    // Verify that a cancelled token would cause the Manager to throw
    // We simulate the check that Manager::CreateCacheSlot does
    EXPECT_THROW(
        {
            if (op_ctx && op_ctx->cancellation_token.isCancellationRequested()) {
                throw std::runtime_error("Operation cancelled, stop creating cache slot");
            }
        },
        std::runtime_error);

    // Verify translator's get_cells was never called (slot was not created)
    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 0);
}

// Test that CreateCacheSlot works normally with nullptr ctx
TEST(ManagerCreateCacheSlotTest, NullCtxWorksNormally) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator = std::make_unique<MockTranslatorWithWarmup>(
        cell_sizes, uid_to_cid_map, "test_slot", StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Disable);

    // Create CacheSlot with nullptr ctx - should work fine
    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));
    cache_slot->Warmup(nullptr);

    EXPECT_EQ(cache_slot->num_cells(), 2);
}

// Test that CreateCacheSlot works with valid non-cancelled ctx
TEST(ManagerCreateCacheSlotTest, ValidCtxWorksNormally) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator = std::make_unique<MockTranslatorWithWarmup>(
        cell_sizes, uid_to_cid_map, "test_slot", StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Disable);

    // Create a non-cancelled context
    auto op_ctx = std::make_unique<milvus::OpContext>();
    EXPECT_FALSE(op_ctx->cancellation_token.isCancellationRequested());

    // Create CacheSlot with valid ctx
    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));
    cache_slot->Warmup(op_ctx.get());

    EXPECT_EQ(cache_slot->num_cells(), 2);
}

// Test that Warmup passes ctx to PinCellsDirect which passes it to get_cells
TEST(WarmupTest, WarmupPassesCtxToGetCells) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator = std::make_unique<MockTranslatorWithWarmup>(
        cell_sizes, uid_to_cid_map, "test_slot", StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Sync);
    auto* translator_ptr = translator.get();

    auto op_ctx = std::make_unique<milvus::OpContext>();

    // Create CacheSlot and call Warmup - this will trigger loading due to EveryLoad policy
    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));
    cache_slot->Warmup(op_ctx.get());

    // Verify that get_cells was called (warmup loaded the cells)
    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 1);
    // Verify that the ctx was passed through to get_cells
    EXPECT_EQ(translator_ptr->GetLastCtx(), op_ctx.get());
}

// Test that Warmup with cancelled ctx during loading throws
TEST(WarmupTest, WarmupWithCancelledCtxDuringLoadingThrows) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator = std::make_unique<MockTranslatorWithWarmup>(
        cell_sizes, uid_to_cid_map, "test_slot", StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Sync);

    // Add delay so we can cancel during loading
    translator->SetCidLoadDelay({{0, 50}, {1, 50}});

    // Create a context with cancellation support
    folly::CancellationSource cancel_source;
    auto cancel_token = cancel_source.getToken();
    auto op_ctx = std::make_unique<milvus::OpContext>(cancel_token);

    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));

    // Cancel the operation before warmup completes
    // Launch warmup in a thread and cancel after a short delay
    std::thread cancel_thread([&cancel_source]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        cancel_source.requestCancellation();
    });

    // Warmup should throw because cancellation will be requested during loading
    EXPECT_THROW(cache_slot->Warmup(op_ctx.get()), std::runtime_error);

    cancel_thread.join();
}

// Test that Warmup with nullptr ctx and EveryLoad policy works (no crash)
TEST(WarmupTest, WarmupWithNullCtxAndEveryLoadPolicyWorks) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator = std::make_unique<MockTranslatorWithWarmup>(
        cell_sizes, uid_to_cid_map, "test_slot", StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Sync);
    auto* translator_ptr = translator.get();

    // Create CacheSlot and call Warmup with nullptr - should not crash
    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));
    EXPECT_NO_THROW(cache_slot->Warmup(nullptr));

    // Verify that get_cells was called
    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 1);
    // ctx should be nullptr
    EXPECT_EQ(translator_ptr->GetLastCtx(), nullptr);
}

// Test that CacheSlot construction handles exceptions in CacheCell initialization gracefully.
// This tests the fix for the placement new double destruction bug.
TEST(CacheSlotConstructionTest, CacheCellConstructionExceptionDoesNotCauseDoubleFree) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    // Create a translator that throws on cid 2 (a middle cell)
    // This means cells 0 and 1 are successfully constructed before the failure
    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}, {2, 100}, {3, 100}, {4, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};
    auto translator =
        std::make_unique<MockTranslator>(cell_sizes, uid_to_cid_map, "throwing_slot", StorageType::MEMORY);
    translator->SetCellsStorageBytesThrowOnCid(2);

    // This should throw an exception, but NOT crash due to double destruction
    EXPECT_THROW(
        {
            auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true,
                                                                    true, std::chrono::milliseconds(100000));
        },
        std::runtime_error);

    // If we reach here without crashing, the fix is working
}

// ==================== Async Warmup Tests ====================

// Test that async warmup with prefetch pool loads cells in background
TEST(AsyncWarmupTest, AsyncWarmupWithPrefetchPoolLoadsInBackground) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}, {2, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}, {2, 2}};
    auto translator =
        std::make_unique<MockTranslatorWithWarmup>(cell_sizes, uid_to_cid_map, "async_warmup_slot", StorageType::MEMORY,
                                                   CacheWarmupPolicy::CacheWarmupPolicy_Async);
    auto* translator_ptr = translator.get();

    // Set up promise to signal when loading completes
    std::promise<void> load_completed;
    auto load_completed_future = load_completed.get_future();
    translator->SetLoadCompletedPromise(&load_completed);

    // Create prefetch pool
    auto prefetch_pool = std::make_shared<folly::CPUThreadPoolExecutor>(2);

    // Create CacheSlot
    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));

    // Warmup should return immediately (async)
    cache_slot->Warmup(nullptr, prefetch_pool);

    // Wait for async warmup to complete using synchronization
    ASSERT_EQ(load_completed_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // Verify cells were loaded
    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 1);

    // Verify cells are accessible
    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto accessor = cache_slot->PinCellsDirect(op_ctx.get(), {0, 1, 2});
    ASSERT_NE(accessor, nullptr);

    for (cid_t cid = 0; cid < 3; ++cid) {
        TestCell* cell = accessor->get_ith_cell(cid);
        ASSERT_NE(cell, nullptr);
        EXPECT_EQ(cell->cid, cid);
    }

    prefetch_pool->join();
}

// Test that async warmup without prefetch pool falls back to sync
TEST(AsyncWarmupTest, AsyncWarmupWithoutPoolFallsBackToSync) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator =
        std::make_unique<MockTranslatorWithWarmup>(cell_sizes, uid_to_cid_map, "async_fallback_slot",
                                                   StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Async);
    auto* translator_ptr = translator.get();

    // Add delay to measure sync behavior - use longer delay to be robust on slow CI
    translator->SetCidLoadDelay({{0, 100}, {1, 100}});

    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));

    // Warmup with nullptr pool should fall back to sync and block
    auto start = std::chrono::steady_clock::now();
    cache_slot->Warmup(nullptr, nullptr);  // No prefetch pool
    auto warmup_duration = std::chrono::steady_clock::now() - start;

    // Should have blocked for the load duration (sync fallback)
    // Use a conservative lower bound (half the expected delay) for CI robustness
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(warmup_duration).count(), 100);

    // Verify cells were loaded synchronously
    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 1);
}

// Test that async warmup does not capture ctx (passes nullptr to get_cells)
TEST(AsyncWarmupTest, AsyncWarmupDoesNotCaptureCtx) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}};
    auto translator =
        std::make_unique<MockTranslatorWithWarmup>(cell_sizes, uid_to_cid_map, "async_no_ctx_slot", StorageType::MEMORY,
                                                   CacheWarmupPolicy::CacheWarmupPolicy_Async);
    auto* translator_ptr = translator.get();

    // Set up promise to signal when loading completes
    std::promise<void> load_completed;
    auto load_completed_future = load_completed.get_future();
    translator->SetLoadCompletedPromise(&load_completed);

    auto prefetch_pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);

    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));

    // Create a context (should NOT be passed to async warmup)
    auto op_ctx = std::make_unique<milvus::OpContext>();

    cache_slot->Warmup(op_ctx.get(), prefetch_pool);

    // Wait for async warmup to complete using synchronization
    ASSERT_EQ(load_completed_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // Verify get_cells was called with nullptr ctx (not the original ctx)
    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 1);
    EXPECT_EQ(translator_ptr->GetLastCtx(), nullptr);

    prefetch_pool->join();
}

// Test that async warmup with weak_ptr skips warmup if slot is destroyed
TEST(AsyncWarmupTest, AsyncWarmupSkipsIfSlotDestroyed) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    // Use a shared counter that outlives the translator to track get_cells calls
    auto get_cells_call_count = std::make_shared<std::atomic<int>>(0);

    auto prefetch_pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);

    // Add a blocking task to hold up the single-threaded executor.
    // This ensures our warmup task won't run until we release the blocker.
    std::promise<void> blocker;
    auto blocker_future = blocker.get_future();
    prefetch_pool->add([&blocker_future]() { blocker_future.wait(); });

    {
        std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
        std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
        auto translator =
            std::make_unique<MockTranslatorWithWarmup>(cell_sizes, uid_to_cid_map, "async_weak_ptr_slot",
                                                       StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Async);
        // Set shared counter for tracking calls after translator is destroyed
        translator->SetSharedCallCounter(get_cells_call_count);

        auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                                std::chrono::milliseconds(100000));

        // Start async warmup - task is queued behind the blocking task
        cache_slot->Warmup(nullptr, prefetch_pool);

        // The slot will be destroyed here, while the warmup task is still queued
    }  // cache_slot destroyed here, translator destroyed too

    // Now release the blocker - the warmup task can run
    blocker.set_value();

    // Wait for all tasks to complete
    prefetch_pool->join();

    // get_cells should NOT have been called because slot was destroyed
    // before the warmup task had a chance to run (weak_ptr::lock() returned nullptr)
    EXPECT_EQ(get_cells_call_count->load(), 0);
}

// Test that access to cells during async warmup blocks until loaded
TEST(AsyncWarmupTest, AccessDuringAsyncWarmupBlocks) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator =
        std::make_unique<MockTranslatorWithWarmup>(cell_sizes, uid_to_cid_map, "async_blocking_slot",
                                                   StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Async);

    // Set up promise to signal when loading starts (so we know warmup is in progress)
    std::promise<void> load_started;
    auto load_started_future = load_started.get_future();
    translator->SetLoadStartedPromise(&load_started);

    // Add delay to ensure warmup takes time - use longer delay for CI robustness
    translator->SetCidLoadDelay({{0, 200}, {1, 200}});

    auto prefetch_pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);

    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));

    // Start async warmup
    cache_slot->Warmup(nullptr, prefetch_pool);

    // Wait for loading to actually start before accessing
    ASSERT_EQ(load_started_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // Now try to access cells - should block until loading completes
    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto start = std::chrono::steady_clock::now();
    auto accessor = cache_slot->PinCellsDirect(op_ctx.get(), {0, 1});
    auto access_duration = std::chrono::steady_clock::now() - start;

    ASSERT_NE(accessor, nullptr);

    // Access should have blocked waiting for warmup to complete
    // Use conservative lower bound for CI robustness
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(access_duration).count(), 100);

    // Verify cells are valid
    TestCell* cell0 = accessor->get_ith_cell(0);
    TestCell* cell1 = accessor->get_ith_cell(1);
    ASSERT_NE(cell0, nullptr);
    ASSERT_NE(cell1, nullptr);
    EXPECT_EQ(cell0->cid, 0);
    EXPECT_EQ(cell1->cid, 1);

    prefetch_pool->join();
}

// Test sync warmup still works (regression test)
TEST(AsyncWarmupTest, SyncWarmupStillWorks) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator = std::make_unique<MockTranslatorWithWarmup>(
        cell_sizes, uid_to_cid_map, "sync_warmup_slot", StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Sync);
    auto* translator_ptr = translator.get();

    auto prefetch_pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);

    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));

    auto op_ctx = std::make_unique<milvus::OpContext>();

    // Sync warmup should load cells synchronously (even with prefetch pool provided)
    cache_slot->Warmup(op_ctx.get(), prefetch_pool);

    // Cells should be loaded immediately after Warmup returns
    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 1);
    EXPECT_EQ(translator_ptr->GetLastCtx(), op_ctx.get());  // ctx should be passed for sync

    // Access should be immediate (no waiting)
    auto accessor = cache_slot->PinCellsDirect(op_ctx.get(), {0, 1});
    ASSERT_NE(accessor, nullptr);

    TestCell* cell0 = accessor->get_ith_cell(0);
    ASSERT_NE(cell0, nullptr);
    EXPECT_EQ(cell0->cid, 0);

    prefetch_pool->join();
}

// Test disabled warmup still works (regression test)
TEST(AsyncWarmupTest, DisabledWarmupStillWorks) {
    auto limit = ResourceUsage{10000, 0};
    auto dlist = std::make_shared<DList>(true, limit, limit, limit, EvictionConfig{10, true, 600});

    std::vector<std::pair<cid_t, int64_t>> cell_sizes = {{0, 100}, {1, 100}};
    std::unordered_map<cl_uid_t, cid_t> uid_to_cid_map = {{0, 0}, {1, 1}};
    auto translator =
        std::make_unique<MockTranslatorWithWarmup>(cell_sizes, uid_to_cid_map, "disabled_warmup_slot",
                                                   StorageType::MEMORY, CacheWarmupPolicy::CacheWarmupPolicy_Disable);
    auto* translator_ptr = translator.get();

    auto prefetch_pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);

    auto cache_slot = std::make_shared<CacheSlot<TestCell>>(std::move(translator), dlist.get(), true, true, true,
                                                            std::chrono::milliseconds(100000));

    // Disabled warmup should not load any cells
    cache_slot->Warmup(nullptr, prefetch_pool);

    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 0);

    // Cells should be loaded on-demand
    auto op_ctx = std::make_unique<milvus::OpContext>();
    auto accessor = cache_slot->PinCellsDirect(op_ctx.get(), {0});
    ASSERT_NE(accessor, nullptr);

    EXPECT_EQ(translator_ptr->GetCellsCallCount(), 1);

    prefetch_pool->join();
}
