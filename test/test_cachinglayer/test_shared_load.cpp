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

#include <folly/CancellationToken.h>
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/Manager.h"
#include "cachinglayer_test_utils.h"

namespace milvus::cachinglayer {
class SharedLoadTestFriend {
 public:
    template <typename T>
    static std::unique_lock<std::mutex>
    LockLoads(CacheSlot<T>& slot) {
        return std::unique_lock(slot.loads_mutex_);
    }

    template <typename T>
    static int
    Pins(const CacheSlot<T>& slot, cid_t cid) {
        return slot.cells_[cid]->pin_count_.load();
    }
};
}  // namespace milvus::cachinglayer

namespace {
using namespace milvus;
using namespace milvus::cachinglayer;
using namespace std::chrono_literals;

struct SharedCell {
    cid_t id;
    ResourceUsage
    CellByteSize() const {
        return {100, 0};
    }
};
struct LoadGate {
    std::mutex mutex;
    std::condition_variable cv;
    int calls = 0;
    bool started = false;
    bool cancelled = false;
    bool finish = false;
    bool drain_on_cancel = true;
    std::function<void(OpContext*)> before_load;

    void
    Release() {
        std::lock_guard lock(mutex);
        finish = true;
        cv.notify_all();
    }
    bool
    Wait(bool LoadGate::*field) {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, 5s, [&] { return this->*field; });
    }
};
class SharedTranslator : public Translator<SharedCell> {
 public:
    explicit SharedTranslator(std::shared_ptr<LoadGate> gate) : gate_(std::move(gate)) {
    }
    size_t
    num_cells() const override {
        return 2;
    }
    cid_t
    cell_id_of(milvus::cachinglayer::uid_t uid) const override {
        return uid;
    }
    const std::string&
    key() const override {
        return key_;
    }
    Meta*
    meta() override {
        return &meta_;
    }
    int64_t
    cells_storage_bytes(const std::vector<cid_t>& ids) const override {
        return ids.size() * 100;
    }
    LoadingResourceUsage
    estimated_loading_usage(const std::vector<cid_t>& ids) const override {
        return {{static_cast<int64_t>(ids.size() * 100), 0}, {}, {}};
    }
    std::vector<std::pair<cid_t, std::unique_ptr<SharedCell>>>
    get_cells(OpContext* ctx, const std::vector<cid_t>& ids) override {
        auto gate = gate_;
        if (gate->before_load)
            gate->before_load(ctx);
        int call;
        {
            std::lock_guard lock(gate->mutex);
            call = ++gate->calls;
        }
        if (call == 1) {
            folly::CancellationCallback cb(ctx->cancellation_token, [gate] {
                std::lock_guard lock(gate->mutex);
                gate->cancelled = true;
                gate->cv.notify_all();
            });
            std::unique_lock lock(gate->mutex);
            gate->started = true;
            gate->cv.notify_all();
            gate->cv.wait(lock, [&] { return gate->finish || (gate->cancelled && gate->drain_on_cancel); });
        }
        if (ctx->cancellation_token.isCancellationRequested()) {
            ThrowInfo(ErrorCode::FollyCancel, "shared test load cancelled");
        }
        std::vector<std::pair<cid_t, std::unique_ptr<SharedCell>>> result;
        for (auto id : ids) result.emplace_back(id, std::make_unique<SharedCell>(SharedCell{id}));
        return result;
    }

 private:
    std::shared_ptr<LoadGate> gate_;
    std::string key_ = "shared-cancellation-test";
    Meta meta_{StorageType::MEMORY, CellIdMappingMode::IDENTICAL, CellDataType::OTHER,
               CacheWarmupPolicy::CacheWarmupPolicy_Disable, true};
};

class SharedLoadCancellationTest : public ::testing::TestWithParam<int> {
 protected:
    using Accessor = std::shared_ptr<CellAccessor<SharedCell>>;
    using Pending = std::future<Accessor>;
    std::shared_ptr<internal::DList> dlist = std::make_shared<internal::DList>(
        true, ResourceUsage{1000, 0}, ResourceUsage{1000, 0}, ResourceUsage{1000, 0}, EvictionConfig{10, true, 600});
    std::shared_ptr<LoadGate> gate = std::make_shared<LoadGate>();
    std::shared_ptr<CacheSlot<SharedCell>> slot = std::make_shared<CacheSlot<SharedCell>>(
        std::make_unique<SharedTranslator>(gate), dlist.get(), true, true, true, 5s, 5s);

    void
    TearDown() override {
        gate->Release();
        SemiInlineGet(slot->WaitForLoads());
    }
    Pending
    Pin(OpContext* ctx, std::vector<milvus::cachinglayer::uid_t> ids = {0}) {
        return std::async(std::launch::async, [this, ctx, ids = std::move(ids)] {
            if (GetParam() == 0)
                return slot->PinOneCellDirect(ctx, ids[0]);
            if (GetParam() == 1)
                return slot->PinCellsDirect(ctx, ids);
            return SemiInlineGet(slot->PinCells(ctx, ids));
        });
    }
    bool
    WaitPins(int count) {
        auto deadline = std::chrono::steady_clock::now() + 5s;
        while (SharedLoadTestFriend::Pins(*slot, 0) != count && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        return SharedLoadTestFriend::Pins(*slot, 0) == count;
    }
    void
    ExpectCancelled(Pending& pending) {
        ASSERT_EQ(pending.wait_for(5s), std::future_status::ready);
        try {
            pending.get();
            FAIL() << "expected cancellation";
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), ErrorCode::FollyCancel);
        }
    }
};

TEST_P(SharedLoadCancellationTest, InitiatorCancellationDoesNotCancelFollower) {
    folly::CancellationSource a_cancel;
    OpContext a(a_cancel.getToken()), b;
    Pending a_result, b_result;
    auto release = folly::makeGuard([&] { gate->Release(); });
    a_result = Pin(&a);
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    b_result = Pin(&b);
    ASSERT_TRUE(WaitPins(2));
    a_cancel.requestCancellation();
    EXPECT_EQ(a_result.wait_for(1s), std::future_status::ready);
    {
        std::lock_guard lock(gate->mutex);
        EXPECT_FALSE(gate->cancelled);
    }
    EXPECT_EQ(b_result.wait_for(0ms), std::future_status::timeout);
    gate->Release();
    ExpectCancelled(a_result);
    ASSERT_EQ(b_result.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(b_result.get()->get_cell_of(0)->id, 0);
    EXPECT_EQ(gate->calls, 1);
}

TEST_P(SharedLoadCancellationTest, FollowerCancellationDoesNotCancelInitiator) {
    folly::CancellationSource b_cancel;
    OpContext a, b(b_cancel.getToken());
    Pending a_result, b_result;
    auto release = folly::makeGuard([&] { gate->Release(); });
    a_result = Pin(&a);
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    b_result = Pin(&b);
    ASSERT_TRUE(WaitPins(2));
    b_cancel.requestCancellation();
    EXPECT_EQ(b_result.wait_for(1s), std::future_status::ready);
    {
        std::lock_guard lock(gate->mutex);
        EXPECT_FALSE(gate->cancelled);
    }
    gate->Release();
    ExpectCancelled(b_result);
    EXPECT_EQ(a_result.get()->get_cell_of(0)->id, 0);
    EXPECT_EQ(gate->calls, 1);
}

TEST_P(SharedLoadCancellationTest, LastWaiterCancelsLoadButRetainsResourcesUntilDrain) {
    gate->drain_on_cancel = false;
    folly::CancellationSource a_cancel;
    auto a = std::make_unique<OpContext>(a_cancel.getToken());
    OpContext newcomer;
    Pending a_result, newcomer_result;
    auto release = folly::makeGuard([&] { gate->Release(); });
    a_result = Pin(a.get());
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    a_cancel.requestCancellation();
    ASSERT_TRUE(gate->Wait(&LoadGate::cancelled));
    EXPECT_EQ(a_result.wait_for(1s), std::future_status::ready);
    if (a_result.wait_for(0ms) == std::future_status::ready) {
        ExpectCancelled(a_result);
        a.reset();
    }
    EXPECT_GT(internal::DListTestFriend::get_loading_memory(*dlist).memory_bytes, 0);
    newcomer_result = Pin(&newcomer);
    EXPECT_EQ(newcomer_result.wait_for(50ms), std::future_status::timeout);
    gate->Release();
    if (a_result.valid())
        ExpectCancelled(a_result);
    EXPECT_EQ(newcomer_result.get()->get_cell_of(0)->id, 0);
    EXPECT_EQ(gate->calls, 2);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), ResourceUsage{});
}

TEST_P(SharedLoadCancellationTest, AdmissionWaitSurvivesInitiatorCancellation) {
    slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true, true,
                                                   true, -1ms, 0ms);
    const ResourceUsage held{1000, 0};
    ASSERT_TRUE(SemiInlineGet(dlist->ReserveLoadingResourceWithTimeout(held, 0ms)));
    folly::CancellationSource a_cancel;
    OpContext a(a_cancel.getToken()), b;
    Pending a_result, b_result;
    auto release = folly::makeGuard([&] {
        gate->Release();
        dlist->ReleaseLoadingResource(held);
    });
    a_result = Pin(&a);
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (internal::DListTestFriend::get_waiting_request_count(*dlist) == 0 &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_EQ(internal::DListTestFriend::get_waiting_request_count(*dlist), 1);
    b_result = Pin(&b);
    ASSERT_TRUE(WaitPins(2));
    a_cancel.requestCancellation();
    ExpectCancelled(a_result);
    EXPECT_EQ(internal::DListTestFriend::get_waiting_request_count(*dlist), 1);
    EXPECT_EQ(gate->calls, 0);
    dlist->ReleaseLoadingResource(held);
    release.dismiss();
    auto finish = folly::makeGuard([&] { gate->Release(); });
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    {
        std::lock_guard lock(gate->mutex);
        EXPECT_FALSE(gate->cancelled);
    }
    gate->Release();
    EXPECT_EQ(b_result.get()->get_cell_of(0)->id, 0);
    EXPECT_EQ(gate->calls, 1);
}

TEST_P(SharedLoadCancellationTest, LastAdmissionWaiterCancelsBeforeDownload) {
    slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true, true,
                                                   true, -1ms, 0ms);
    const ResourceUsage held{1000, 0};
    ASSERT_TRUE(SemiInlineGet(dlist->ReserveLoadingResourceWithTimeout(held, 0ms)));
    folly::CancellationSource a_cancel;
    OpContext a(a_cancel.getToken());
    Pending result;
    auto release = folly::makeGuard([&] {
        gate->Release();
        dlist->ReleaseLoadingResource(held);
    });
    result = Pin(&a);
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (internal::DListTestFriend::get_waiting_request_count(*dlist) == 0 &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_EQ(internal::DListTestFriend::get_waiting_request_count(*dlist), 1);
    a_cancel.requestCancellation();
    ExpectCancelled(result);
    SemiInlineGet(slot->WaitForLoads());
    EXPECT_EQ(internal::DListTestFriend::get_waiting_request_count(*dlist), 0);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), held);
    EXPECT_EQ(gate->calls, 0);
}

TEST_P(SharedLoadCancellationTest, AllWaitersLeavingCancelsExactlyOneDownload) {
    folly::CancellationSource a_cancel, b_cancel;
    OpContext a(a_cancel.getToken()), b(b_cancel.getToken());
    Pending a_result, b_result;
    auto release = folly::makeGuard([&] { gate->Release(); });
    a_result = Pin(&a);
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    b_result = Pin(&b);
    ASSERT_TRUE(WaitPins(2));
    a_cancel.requestCancellation();
    ExpectCancelled(a_result);
    {
        std::lock_guard lock(gate->mutex);
        EXPECT_FALSE(gate->cancelled);
    }
    b_cancel.requestCancellation();
    ExpectCancelled(b_result);
    ASSERT_TRUE(gate->Wait(&LoadGate::cancelled));
    SemiInlineGet(slot->WaitForLoads());
    EXPECT_EQ(gate->calls, 1);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), ResourceUsage{});
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), ResourceUsage{});
}

TEST_P(SharedLoadCancellationTest, BatchRemainsNeededByFollowerOfAnotherCell) {
    if (GetParam() == 0)
        GTEST_SKIP() << "requires the batch entry point";
    folly::CancellationSource a_cancel;
    OpContext a(a_cancel.getToken()), b;
    Pending a_result, b_result;
    auto release = folly::makeGuard([&] { gate->Release(); });
    a_result = Pin(&a, {0, 1});
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    b_result = Pin(&b, {1});
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (SharedLoadTestFriend::Pins(*slot, 1) != 2 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_EQ(SharedLoadTestFriend::Pins(*slot, 1), 2);
    a_cancel.requestCancellation();
    ExpectCancelled(a_result);
    {
        std::lock_guard lock(gate->mutex);
        EXPECT_FALSE(gate->cancelled);
    }
    gate->Release();
    EXPECT_EQ(b_result.get()->get_cell_of(1)->id, 1);
    EXPECT_EQ(gate->calls, 1);
}

TEST_P(SharedLoadCancellationTest, AdmissionTimeoutDoesNotLimitTranslation) {
    // Cover default foreground/warmup settings and a short positive admission
    // budget. Translation may outlive all of them while a caller still waits.
    for (auto timeout : {-1ms, 0ms, 20ms}) {
        SCOPED_TRACE(timeout.count());
        gate = std::make_shared<LoadGate>();
        slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true,
                                                       true, true, timeout, 0ms);
        OpContext caller;
        Pending result;
        auto release = folly::makeGuard([&] { gate->Release(); });
        result = Pin(&caller);
        ASSERT_TRUE(gate->Wait(&LoadGate::started));
        {
            std::unique_lock lock(gate->mutex);
            EXPECT_FALSE(gate->cv.wait_for(lock, 100ms, [&] { return gate->cancelled; }));
        }
        EXPECT_EQ(result.wait_for(0ms), std::future_status::timeout);
        gate->Release();
        ASSERT_EQ(result.wait_for(5s), std::future_status::ready);
        EXPECT_EQ(result.get()->get_cell_of(0)->id, 0);
        SemiInlineGet(slot->WaitForLoads());
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), ResourceUsage{});
    }
}

TEST_P(SharedLoadCancellationTest, AdmissionTimeoutStillRejectsWhenResourcesAreUnavailable) {
    for (auto timeout : {0ms, 20ms}) {
        SCOPED_TRACE(timeout.count());
        slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true,
                                                       true, true, timeout, 0ms);
        const ResourceUsage held{1000, 0};
        ASSERT_TRUE(SemiInlineGet(dlist->ReserveLoadingResourceWithTimeout(held, 0ms)));
        Pending result;
        auto release = folly::makeGuard([&] { dlist->ReleaseLoadingResource(held); });
        result = Pin(nullptr);
        ASSERT_EQ(result.wait_for(5s), std::future_status::ready);
        try {
            result.get();
            FAIL() << "expected admission failure";
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), ErrorCode::InsufficientResource);
        }
        EXPECT_EQ(gate->calls, 0);
    }
}

TEST_P(SharedLoadCancellationTest, ResidentHitsDoNotWaitForSharedLoadMutex) {
    gate->Release();
    auto initial = Pin(nullptr).get();
    initial.reset();  // Exercise CACHED -> pinned, not just an already-pinned cell.
    OpContext caller;
    auto lock = SharedLoadTestFriend::LockLoads(*slot);
    auto result = Pin(&caller);
    auto status = result.wait_for(1s);
    lock.unlock();  // A failed fast path must not leave the test's future blocked.
    ASSERT_EQ(status, std::future_status::ready);
    auto accessor = result.get();
    EXPECT_EQ(accessor->get_cell_of(0)->id, 0);
    EXPECT_EQ(caller.storage_usage.scanned_cold_bytes.load(), 0);
    EXPECT_EQ(caller.storage_usage.scanned_total_bytes.load(), 100);
    EXPECT_FALSE(slot->ManualEvict(0));
    accessor.reset();
    EXPECT_TRUE(slot->ManualEvict(0));
    EXPECT_EQ(gate->calls, 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), ResourceUsage{});
}

TEST_P(SharedLoadCancellationTest, PreCancelledResidentHitStillReturnsCancellation) {
    gate->Release();
    auto resident = Pin(nullptr).get();
    folly::CancellationSource cancel;
    cancel.requestCancellation();
    OpContext caller(cancel.getToken());
    auto result = Pin(&caller);
    ExpectCancelled(result);
    EXPECT_EQ(caller.storage_usage.scanned_total_bytes.load(), 0);
    EXPECT_EQ(SharedLoadTestFriend::Pins(*slot, 0), 1);
    EXPECT_EQ(gate->calls, 1);
}

TEST_P(SharedLoadCancellationTest, PartialResidentBatchFallsBackWithoutLeakingPins) {
    if (GetParam() == 0)
        GTEST_SKIP() << "requires the batch entry point";
    gate->Release();
    Pin(nullptr, {0}).get().reset();
    OpContext caller;
    auto accessor = Pin(&caller, {0, 1}).get();
    EXPECT_EQ(accessor->get_cell_of(0)->id, 0);
    EXPECT_EQ(accessor->get_cell_of(1)->id, 1);
    EXPECT_EQ(SharedLoadTestFriend::Pins(*slot, 0), 1);
    EXPECT_EQ(SharedLoadTestFriend::Pins(*slot, 1), 1);
    EXPECT_EQ(caller.storage_usage.scanned_cold_bytes.load(), 100);
    EXPECT_EQ(caller.storage_usage.scanned_total_bytes.load(), 200);
    EXPECT_EQ(gate->calls, 2);
    accessor.reset();
    EXPECT_TRUE(slot->ManualEvict(0));
    EXPECT_TRUE(slot->ManualEvict(1));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), ResourceUsage{});
}

TEST_P(SharedLoadCancellationTest, CompletionRacingCancellationLeavesFollowerSuccessful) {
    for (int i = 0; i < 30; ++i) {
        SCOPED_TRACE(i);
        gate = std::make_shared<LoadGate>();
        slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true,
                                                       true, true, 5s, 5s);
        folly::CancellationSource a_cancel;
        OpContext a(a_cancel.getToken()), b;
        Pending a_result, b_result;
        auto release = folly::makeGuard([&] { gate->Release(); });
        a_result = Pin(&a);
        ASSERT_TRUE(gate->Wait(&LoadGate::started));
        b_result = Pin(&b);
        ASSERT_TRUE(WaitPins(2));
        auto cancel = std::async(std::launch::async, [&] { a_cancel.requestCancellation(); });
        gate->Release();
        cancel.get();
        try {
            EXPECT_NE(a_result.get(), nullptr);
        } catch (const SegcoreError& e) {
            EXPECT_EQ(e.get_error_code(), ErrorCode::FollyCancel);
        }
        EXPECT_EQ(b_result.get()->get_cell_of(0)->id, 0);
        SemiInlineGet(slot->WaitForLoads());
        EXPECT_EQ(gate->calls, 1);
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), ResourceUsage{});
    }
}

TEST_P(SharedLoadCancellationTest, CallerOwnedReservationIsRetainedUntilLoadDrains) {
    slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true, false,
                                                   true, 5s, 5s);
    gate->drain_on_cancel = false;
    folly::CancellationSource cancel;
    OpContext caller(cancel.getToken());
    Pending result;
    auto release = folly::makeGuard([&] { gate->Release(); });
    result = Pin(&caller);
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    cancel.requestCancellation();
    ASSERT_TRUE(gate->Wait(&LoadGate::cancelled));
    // The caller may release its external reservation as soon as this method returns.
    EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
    gate->Release();
    ExpectCancelled(result);
}

TEST_P(SharedLoadCancellationTest, CancelledNewcomerDoesNotRestartDrainingLoad) {
    gate->drain_on_cancel = false;
    folly::CancellationSource a_cancel, c_cancel;
    OpContext a(a_cancel.getToken()), c(c_cancel.getToken());
    Pending first, newcomer;
    auto release = folly::makeGuard([&] { gate->Release(); });
    first = Pin(&a);
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    a_cancel.requestCancellation();
    ExpectCancelled(first);
    newcomer = Pin(&c);
    EXPECT_EQ(newcomer.wait_for(50ms), std::future_status::timeout);
    c_cancel.requestCancellation();
    ExpectCancelled(newcomer);
    gate->Release();
    SemiInlineGet(slot->WaitForLoads());
    EXPECT_EQ(gate->calls, 1);
}

class ScopedLoadThreads {
 public:
    explicit ScopedLoadThreads(size_t count) : previous_(internal::SharedLoadThreadPool().numThreads()) {
        Manager::SetLoadPoolThreads(count);
    }
    ~ScopedLoadThreads() {
        Manager::SetLoadPoolThreads(previous_);
    }

 private:
    size_t previous_;
};

TEST(SharedLoadExecutorTest, ConfiguredConcurrencyAllowsMoreThanEightLoads) {
    ScopedLoadThreads threads(12);
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{10000, 0}, ResourceUsage{10000, 0},
                                                   ResourceUsage{10000, 0}, EvictionConfig{10, false, 600});
    std::vector<std::shared_ptr<LoadGate>> gates;
    std::vector<std::shared_ptr<CacheSlot<SharedCell>>> slots;
    std::vector<std::future<std::shared_ptr<CellAccessor<SharedCell>>>> results;
    auto release = folly::makeGuard([&] {
        for (const auto& gate : gates) gate->Release();
    });
    for (int i = 0; i < 12; ++i) {
        auto gate = std::make_shared<LoadGate>();
        gates.push_back(gate);
        auto slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true,
                                                            true, true, -1ms, 0ms);
        slots.push_back(slot);
        results.push_back(std::async(std::launch::async, [slot] { return slot->PinOneCellDirect(nullptr, 0); }));
        ASSERT_TRUE(gate->Wait(&LoadGate::started)) << "load " << i;
    }
    // All twelve are inside synchronous translation concurrently, not just queued.
    for (const auto& gate : gates) gate->Release();
    for (auto& result : results) EXPECT_EQ(result.get()->get_cell_of(0)->id, 0);
}

TEST(SharedLoadExecutorTest, QueueTimeDoesNotConsumeAdmissionTimeout) {
    ScopedLoadThreads threads(1);
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{1000, 0}, ResourceUsage{1000, 0},
                                                   ResourceUsage{1000, 0}, EvictionConfig{10, false, 600});
    auto busy = std::make_shared<LoadGate>();
    auto queued = std::make_shared<LoadGate>();
    queued->finish = true;
    auto first = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(busy), dlist.get(), true,
                                                         true, true, -1ms, 0ms);
    auto second = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(queued), dlist.get(), true,
                                                          true, true, 20ms, 0ms);
    std::future<std::shared_ptr<CellAccessor<SharedCell>>> a, b;
    auto release = folly::makeGuard([&] { busy->Release(); });
    a = std::async(std::launch::async, [first] { return first->PinOneCellDirect(nullptr, 0); });
    ASSERT_TRUE(busy->Wait(&LoadGate::started));
    b = std::async(std::launch::async, [second] { return second->PinOneCellDirect(nullptr, 0); });
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (SharedLoadTestFriend::Pins(*second, 0) != 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    ASSERT_EQ(SharedLoadTestFriend::Pins(*second, 0), 1);
    EXPECT_EQ(b.wait_for(100ms), std::future_status::timeout);
    busy->Release();
    EXPECT_EQ(a.get()->get_cell_of(0)->id, 0);
    EXPECT_EQ(b.get()->get_cell_of(0)->id, 0);
    EXPECT_EQ(queued->calls, 1);
}

TEST_P(SharedLoadCancellationTest, LastWaiterCanCancelWhileProducerIsQueued) {
    ScopedLoadThreads threads(1);
    auto busy = std::make_shared<LoadGate>();
    auto first = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(busy), dlist.get(), true,
                                                         true, true, -1ms, 0ms);
    folly::CancellationSource cancel;
    OpContext caller(cancel.getToken());
    Pending a, queued;
    auto release = folly::makeGuard([&] { busy->Release(); });
    a = std::async(std::launch::async, [first] { return first->PinOneCellDirect(nullptr, 0); });
    ASSERT_TRUE(busy->Wait(&LoadGate::started));
    queued = Pin(&caller);
    ASSERT_TRUE(WaitPins(1));
    cancel.requestCancellation();
    EXPECT_EQ(queued.wait_for(1s), std::future_status::ready);
    // Release the worker only after checking that the queued request has returned.
    busy->Release();
    ExpectCancelled(queued);
    EXPECT_EQ(a.get()->get_cell_of(0)->id, 0);
    SemiInlineGet(slot->WaitForLoads());
    EXPECT_EQ(gate->calls, 0);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), ResourceUsage{});
}

TEST_P(SharedLoadCancellationTest, ConcurrentHitsAndEvictionKeepAccessorsValid) {
    gate->Release();
    Pin(nullptr).get().reset();
    std::atomic<bool> stop{false};
    auto evictor = std::async(std::launch::async, [&] {
        while (!stop.load()) {
            if (SharedLoadTestFriend::Pins(*slot, 0) == 0)
                slot->ManualEvict(0);
            std::this_thread::yield();
        }
    });
    auto finish = folly::makeGuard([&] { stop = true; });
    for (int i = 0; i < 100; ++i) {
        auto accessor = Pin(nullptr).get();
        EXPECT_EQ(accessor->get_cell_of(0)->id, 0);
    }
    stop = true;
    evictor.get();
    SemiInlineGet(slot->WaitForLoads());
    slot->ManualEvict(0);
    EXPECT_EQ(SharedLoadTestFriend::Pins(*slot, 0), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), ResourceUsage{});
}

TEST(SharedLoadLifecycleTest, SynchronousNestedLoadsDoNotExhaustExecutor) {
    ScopedLoadThreads threads(1);
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{10000, 0}, ResourceUsage{10000, 0},
                                                   ResourceUsage{10000, 0}, EvictionConfig{10, true, 600});
    std::shared_ptr<CacheSlot<SharedCell>> slot;
    // More nested synchronous loads than the executor's worker count.
    for (int i = 0; i < 12; ++i) {
        auto gate = std::make_shared<LoadGate>();
        gate->finish = true;
        if (slot)
            gate->before_load = [child = slot](OpContext* ctx) { child->PinOneCellDirect(ctx, 0); };
        slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true,
                                                       true, true, 500ms, 500ms);
    }
    auto result = std::async(std::launch::async, [slot] { return slot->PinOneCellDirect(nullptr, 0); });
    ASSERT_EQ(result.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(result.get()->get_cell_of(0)->id, 0);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), ResourceUsage{});
}

TEST(SharedLoadLifecycleTest, CancelledRequestCanReleaseItsContextSlotAndDList) {
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{1000, 0}, ResourceUsage{1000, 0},
                                                   ResourceUsage{1000, 0}, EvictionConfig{10, true, 600});
    auto gate = std::make_shared<LoadGate>();
    gate->drain_on_cancel = false;
    auto slot = std::make_shared<CacheSlot<SharedCell>>(std::make_unique<SharedTranslator>(gate), dlist.get(), true,
                                                        true, true, 5s, 5s);
    std::weak_ptr<internal::DList> weak_dlist = dlist;
    std::weak_ptr<CacheSlot<SharedCell>> weak_slot = slot;
    folly::CancellationSource cancel;
    auto caller = std::make_unique<OpContext>(cancel.getToken());
    std::future<std::shared_ptr<CellAccessor<SharedCell>>> result;
    auto release = folly::makeGuard([&] { gate->Release(); });
    result = std::async(std::launch::async, [slot, ctx = caller.get()] { return slot->PinOneCellDirect(ctx, 0); });
    ASSERT_TRUE(gate->Wait(&LoadGate::started));
    cancel.requestCancellation();
    ASSERT_EQ(result.wait_for(5s), std::future_status::ready);
    try {
        result.get();
        FAIL() << "expected cancellation";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
    }
    caller.reset();
    slot.reset();
    dlist.reset();
    EXPECT_FALSE(weak_slot.expired());
    EXPECT_FALSE(weak_dlist.expired());
    gate->Release();
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while ((!weak_slot.expired() || !weak_dlist.expired()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    EXPECT_TRUE(weak_slot.expired());
    EXPECT_TRUE(weak_dlist.expired());
}

INSTANTIATE_TEST_SUITE_P(EntryPoints, SharedLoadCancellationTest, ::testing::Values(0, 1, 2));
}  // namespace
