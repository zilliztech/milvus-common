#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cachinglayer/LRUCache.h"
#include "cachinglayer_test_utils.h"
#include "syncpoint/sync_point.h"

namespace milvus::cachinglayer {

class LRUCacheTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        EvictionConfig config;
        config.cache_touch_window = std::chrono::milliseconds(0);
        dlist_ = std::make_shared<internal::DList>(true, ResourceUsage{8192, 0}, ResourceUsage{8192, 0},
                                                   ResourceUsage{8192, 0}, config);
        cache_ = std::make_unique<LRUCache<std::string, int>>(dlist_);
    }

    std::shared_ptr<internal::DList> dlist_;
    std::unique_ptr<LRUCache<std::string, int>> cache_;
};

TEST_F(LRUCacheTest, GetPinsTheInsertedValueAndMissDoesNotInsert) {
    EXPECT_FALSE(cache_->Get("missing"));
    EXPECT_EQ(cache_->Size(), 0);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(cache_->Put("a", value, {sizeof(int), 0}));
    EXPECT_EQ(value, nullptr);
    auto handle = cache_->Get("a");
    ASSERT_TRUE(handle);
    EXPECT_EQ(*handle, 7);
    EXPECT_EQ(cache_->Size(), 1);
}

TEST_F(LRUCacheTest, PinnedReplacementAndEraseFailWithoutConsumingInput) {
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(cache_->Put("a", value, {32, 0}));
    auto handle = cache_->Get("a");
    auto replacement = std::make_unique<int>(9);
    auto result = cache_->Put("a", replacement, {64, 0});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), LRUCacheError::Pinned);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(*replacement, 9);
    EXPECT_EQ(cache_->Erase("a").error(), LRUCacheError::Pinned);
    EXPECT_EQ(*handle, 7);
    handle = {};
    ASSERT_TRUE(cache_->Put("a", replacement, {64, 0}));
    EXPECT_EQ(*cache_->Get("a"), 9);
    ASSERT_TRUE(cache_->Erase("a"));
    EXPECT_FALSE(cache_->Get("a"));
    EXPECT_EQ(cache_->Erase("a").error(), LRUCacheError::NotFound);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, EvictsLeastRecentlyUnpinnedEntryFromBothListAndMap) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    auto c = std::make_unique<int>(3);
    ASSERT_TRUE(cache_->Put("a", a, {3000, 0}));
    ASSERT_TRUE(cache_->Put("b", b, {3000, 0}));
    { auto access = cache_->Get("a"); }
    ASSERT_TRUE(cache_->Put("c", c, {3000, 0}));
    EXPECT_FALSE(cache_->Get("b"));
    EXPECT_EQ(*cache_->Get("a"), 1);
    EXPECT_EQ(*cache_->Get("c"), 3);
    EXPECT_EQ(cache_->Size(), 2);
}

TEST_F(LRUCacheTest, AllPinnedCapacityFailurePreservesEntriesAndInput) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    auto c = std::make_unique<int>(3);
    ASSERT_TRUE(cache_->Put("a", a, {3000, 0}));
    ASSERT_TRUE(cache_->Put("b", b, {3000, 0}));
    auto first = cache_->Get("a");
    auto second = cache_->Get("b");
    auto result = cache_->Put("c", c, {3000, 0});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), LRUCacheError::InsufficientResource);
    EXPECT_NE(c, nullptr);
    EXPECT_EQ(*first, 1);
    EXPECT_EQ(*second, 2);
    EXPECT_EQ(cache_->Size(), 2);
}

TEST_F(LRUCacheTest, FailedReplacementKeepsOldValue) {
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(cache_->Put("a", value, {3000, 0}));
    auto replacement = std::make_unique<int>(9);
    auto result = cache_->Put("a", replacement, {7000, 0});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), LRUCacheError::InsufficientResource);
    EXPECT_EQ(*cache_->Get("a"), 7);
    EXPECT_NE(replacement, nullptr);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, HandleKeepsValueAliveAfterCacheDestruction) {
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(cache_->Put("a", value, {32, 0}));
    auto handle = cache_->Get("a");
    cache_.reset();
    EXPECT_EQ(*handle, 7);
    EXPECT_GT(internal::DListTestFriend::get_used_memory(*dlist_).memory_bytes, 0);
    handle = {};
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, GlobalReservationEvictsEntriesAndReclaimsEmptyIndex) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    ASSERT_TRUE(cache_->Put("a", a, {3000, 0}));
    ASSERT_TRUE(cache_->Put("b", b, {3000, 0}));
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({7000, 0}, std::chrono::milliseconds(0))));
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_FALSE(cache_->Get("a"));
    EXPECT_FALSE(cache_->Get("b"));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
    dlist_->ReleaseLoadingResource({7000, 0});
}

TEST_F(LRUCacheTest, EmptyPayloadsStillConsumeBoundedMetadata) {
    for (int i = 0; i < 200; ++i) {
        auto value = std::make_unique<int>(i);
        ASSERT_TRUE(cache_->Put(std::to_string(i), value, {0, 0}));
    }
    EXPECT_LT(cache_->Size(), 200);
    EXPECT_LE(internal::DListTestFriend::get_used_memory(*dlist_).memory_bytes, 8192);
    cache_.reset();
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, FullCapacityReservationAlsoReclaimsIndexBuckets) {
    for (int i = 0; i < 3; ++i) {
        auto value = std::make_unique<int>(i);
        ASSERT_TRUE(cache_->Put(std::to_string(i), value, {32, 0}));
    }
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(0))));
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
    dlist_->ReleaseLoadingResource({8192, 0});
}

TEST_F(LRUCacheTest, BucketCreditCannotEvictAnInsufficientSubset) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    ASSERT_TRUE(cache_->Put("a", a, {32, 0}));
    ASSERT_TRUE(cache_->Put("b", b, {32, 0}));
    auto pinned = cache_->Get("a");
    EXPECT_FALSE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(0))));
    EXPECT_EQ(cache_->Size(), 2);
    EXPECT_EQ(*cache_->Get("b"), 2);
    pinned = {};
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(0))));
    dlist_->ReleaseLoadingResource({8192, 0});
}

TEST_F(LRUCacheTest, IndexGrowthPreservesPinnedValuesAndKeyAddresses) {
    ASSERT_TRUE(dlist_->UpdateMaxLimit({16384, 0}));
    std::vector<Pin<int>> handles;
    for (int i = 0; i < 25; ++i) {
        auto value = std::make_unique<int>(i);
        ASSERT_TRUE(cache_->Put(std::to_string(i), value, {sizeof(int), 0}));
        handles.push_back(cache_->Get(std::to_string(i)));
    }
    EXPECT_EQ(cache_->Size(), 25);
    for (int i = 0; i < 25; ++i) {
        EXPECT_EQ(*handles[i], i);
    }
    handles.clear();
    for (int i = 0; i < 25; ++i) {
        EXPECT_TRUE(cache_->Erase(std::to_string(i)));
    }
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, MovingHandleReleasesItsOldCacheAndKeepsTheNewOne) {
    auto second = std::make_unique<LRUCache<std::string, int>>(dlist_);
    auto integer_keys = std::make_unique<LRUCache<int, int>>(dlist_);
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    auto c = std::make_unique<int>(3);
    ASSERT_TRUE(cache_->Put("same", a, {32, 0}));
    ASSERT_TRUE(second->Put("same", b, {32, 0}));
    ASSERT_TRUE(integer_keys->Put(1, c, {32, 0}));
    Pin<int> first_handle = cache_->Get("same");
    Pin<int> second_handle = second->Get("same");
    Pin<int> integer_handle = integer_keys->Get(1);
    cache_.reset();
    second.reset();
    integer_keys.reset();
    first_handle = std::move(second_handle);
    EXPECT_FALSE(second_handle);
    EXPECT_EQ(*first_handle, 2);
    first_handle = std::move(integer_handle);
    EXPECT_FALSE(integer_handle);
    EXPECT_EQ(*first_handle, 3);
    first_handle = {};
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, RejectsInvalidValuesWithoutChangingExistingData) {
    auto value = std::make_unique<int>(7);
    EXPECT_EQ(cache_->Put("a", value, {-1, 0}).error(), LRUCacheError::InvalidArgument);
    EXPECT_NE(value, nullptr);
    EXPECT_EQ(cache_->Size(), 0);
    std::unique_ptr<int> empty;
    EXPECT_EQ(cache_->Put("a", empty, {}).error(), LRUCacheError::InvalidArgument);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST(LRUCacheAccountingTest, IntegerKeysAndDiskLimitsShareTheSameAdmission) {
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{8192, 100}, ResourceUsage{8192, 100},
                                                   ResourceUsage{8192, 100}, EvictionConfig{});
    LRUCache<int, int> cache(dlist);
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    ASSERT_TRUE(cache.Put(10, a, {32, 70}));
    auto pin = cache.Get(10);
    EXPECT_EQ(cache.Put(20, b, {32, 70}).error(), LRUCacheError::InsufficientResource);
    pin = {};
    ASSERT_TRUE(cache.Put(20, b, {32, 70}));
    EXPECT_FALSE(cache.Get(10));
    EXPECT_EQ(*cache.Get(20), 2);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist).file_bytes, 70);
    ASSERT_TRUE(cache.Erase(20));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, TypeErasedCorePreservesSignedAndUnsignedIntegerKeys) {
    LRUCache<int64_t, int> signed_cache(dlist_);
    LRUCache<uint64_t, int> unsigned_cache(dlist_);
    for (const int64_t key : {std::numeric_limits<int64_t>::min(), int64_t{-1}, int64_t{0}}) {
        auto value = std::make_unique<int>(key == 0 ? 3 : key == -1 ? 2 : 1);
        ASSERT_TRUE(signed_cache.Put(key, value, {sizeof(int), 0}));
    }
    auto largest = std::make_unique<int>(4);
    ASSERT_TRUE(unsigned_cache.Put(std::numeric_limits<uint64_t>::max(), largest, {sizeof(int), 0}));
    EXPECT_EQ(*signed_cache.Get(std::numeric_limits<int64_t>::min()), 1);
    EXPECT_EQ(*signed_cache.Get(-1), 2);
    EXPECT_EQ(*signed_cache.Get(0), 3);
    EXPECT_EQ(*unsigned_cache.Get(std::numeric_limits<uint64_t>::max()), 4);
    EXPECT_FALSE(unsigned_cache.Get(0));
}

TEST_F(LRUCacheTest, TypeErasedCorePreservesEmbeddedNullsInStringKeys) {
    const std::string key("a\0b", 3);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(cache_->Put(key, value, {sizeof(int), 0}));
    EXPECT_FALSE(cache_->Get("a"));
    EXPECT_EQ(*cache_->Get(key), 7);
    EXPECT_EQ(cache_->Erase("a").error(), LRUCacheError::NotFound);
    EXPECT_TRUE(cache_->Erase(key));
}

TEST_F(LRUCacheTest, TypeErasedCorePreservesConstValueOwnershipOnFailure) {
    LRUCache<int, const int> cache(dlist_);
    auto value = std::make_unique<const int>(7);
    ASSERT_TRUE(cache.Put(1, value, {sizeof(int), 0}));
    auto handle = cache.Get(1);
    ASSERT_TRUE(handle);
    auto replacement = std::make_unique<const int>(9);
    EXPECT_EQ(cache.Put(1, replacement, {sizeof(int), 0}).error(), LRUCacheError::Pinned);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(*replacement, 9);
    EXPECT_EQ(*handle, 7);
}

TEST_F(LRUCacheTest, TypeErasedCoreUsesTheOriginalDefaultDeleter) {
    class Value {
     public:
        explicit Value(int& destroyed) : destroyed_(destroyed) {
        }

     private:
        friend struct std::default_delete<Value>;
        ~Value() {
            ++destroyed_;
        }
        int& destroyed_;
    };
    int destroyed = 0;
    LRUCache<int, Value> cache(dlist_);
    auto value = std::make_unique<Value>(destroyed);
    ASSERT_TRUE(cache.Put(1, value, {sizeof(Value), 0}));
    EXPECT_EQ(destroyed, 0);
    ASSERT_TRUE(cache.Erase(1));
    EXPECT_EQ(destroyed, 1);
}

TEST_F(LRUCacheTest, ConcurrentGetsKeepTheirValueStableAcrossMutationsAndEviction) {
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 4; ++worker) {
        threads.emplace_back([&, worker] {
            for (int i = 0; i < 100; ++i) {
                const auto key = std::to_string(i % 12);
                auto value = std::make_unique<int>(worker * 100 + i);
                const auto result = cache_->Put(key, value, {128, 0});
                if (!result) {
                    EXPECT_TRUE(result.error() == LRUCacheError::Pinned ||
                                result.error() == LRUCacheError::InsufficientResource);
                    EXPECT_NE(value, nullptr);
                }
                auto handle = cache_->Get(key);
                if (handle) {
                    const auto snapshot = *handle;
                    std::this_thread::yield();
                    EXPECT_EQ(*handle, snapshot);
                    EXPECT_EQ(cache_->Erase(key).error(), LRUCacheError::Pinned);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
    EXPECT_LE(internal::DListTestFriend::get_used_memory(*dlist_).memory_bytes, 8192);
    cache_.reset();
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(LRUCacheTest, LongKeysAreChargedAndMetricsReturnToBaseline) {
    const auto entries = monitor::cache_kv_entry_count.Value();
    const auto metadata = monitor::cache_kv_metadata_bytes.Value();
    const auto payload = monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Value();
    auto value = std::make_unique<int>(7);
    EXPECT_EQ(cache_->Put(std::string(10000, 'k'), value, {}).error(), LRUCacheError::InsufficientResource);
    EXPECT_NE(value, nullptr);
    EXPECT_EQ(cache_->Size(), 0);
    ASSERT_TRUE(cache_->Put("a", value, {32, 0}));
    EXPECT_EQ(monitor::cache_kv_entry_count.Value(), entries + 1);
    EXPECT_GT(monitor::cache_kv_metadata_bytes.Value(), metadata);
    EXPECT_EQ(monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Value(), payload + 32);
    cache_.reset();
    EXPECT_EQ(monitor::cache_kv_entry_count.Value(), entries);
    EXPECT_EQ(monitor::cache_kv_metadata_bytes.Value(), metadata);
    EXPECT_EQ(monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Value(), payload);
}

TEST(LRUCacheAccountingTest, BackgroundEvictionPreservesPinnedEntryAndRemovesOtherKeys) {
    struct Signal {
        std::mutex mutex;
        std::condition_variable changed;
        int destroyed{0};
    };
    struct Value {
        std::shared_ptr<Signal> signal;
        ~Value() {
            std::lock_guard lock(signal->mutex);
            ++signal->destroyed;
            signal->changed.notify_all();
        }
    };
    auto signal = std::make_shared<Signal>();
    EvictionConfig config{0, true, 1};
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{8192, 0}, ResourceUsage{1024, 0},
                                                   ResourceUsage{8192, 0}, config);
    LRUCache<int, Value> cache(dlist);
    auto first = std::make_unique<Value>();
    first->signal = signal;
    ASSERT_TRUE(cache.Put(0, first, {1000, 0}));
    auto pin = cache.Get(0);
    ASSERT_TRUE(pin);
    dlist->UpdateHighWatermark({2048, 0});
    for (int key = 1; key < 3; ++key) {
        auto value = std::make_unique<Value>();
        value->signal = signal;
        ASSERT_TRUE(cache.Put(key, value, {1000, 0}));
    }
    {
        std::unique_lock lock(signal->mutex);
        ASSERT_TRUE(signal->changed.wait_for(lock, std::chrono::seconds(5), [&] { return signal->destroyed == 2; }));
    }
    EXPECT_EQ(cache.Size(), 1);
    EXPECT_TRUE(cache.Get(0));
    EXPECT_FALSE(cache.Get(1));
    EXPECT_FALSE(cache.Get(2));
}

#ifdef ENABLE_SYNCPOINT
TEST_F(LRUCacheTest, PreparationAndPostReservationFailurePreserveTheOldEntry) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(cache_->Put("a", value, {32, 0}));
    const auto before = internal::DListTestFriend::get_used_memory(*dlist_);
    for (const auto* point : {"LRUCache::Put:BeforeReserve", "LRUCache::Put:BeforePublish"}) {
        SCOPED_TRACE(point);
        sync->SetCallBack(point, [](void*) { throw std::bad_alloc(); });
        sync->EnableProcessing();
        auto replacement = std::make_unique<int>(9);
        EXPECT_THROW(static_cast<void>(cache_->Put("a", replacement, {64, 0})), std::bad_alloc);
        sync->Reset();
        EXPECT_NE(replacement, nullptr);
        EXPECT_EQ(*cache_->Get("a"), 7);
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), before);
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
    }
    EXPECT_TRUE(cache_->Erase("a"));
}

TEST_F(LRUCacheTest, GetDuringReplacementForcesPinnedErrorAtPublication) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(cache_->Put("a", value, {32, 0}));
    folly::Baton<> prepared;
    folly::Baton<> resume;
    sync->SetCallBack("LRUCache::Put:BeforeReserve", [&](void*) {
        prepared.post();
        resume.wait();
    });
    sync->EnableProcessing();
    auto replacement = std::make_unique<int>(9);
    auto result = std::async(std::launch::async, [&] { return cache_->Put("a", replacement, {64, 0}); });
    auto unblock = folly::makeGuard([&] { resume.post(); });
    ASSERT_TRUE(prepared.try_wait_for(std::chrono::seconds(5)));
    EXPECT_EQ(cache_->Erase("a").error(), LRUCacheError::Conflict);
    auto pin = cache_->Get("a");
    ASSERT_TRUE(pin);
    resume.post();
    EXPECT_EQ(result.get().error(), LRUCacheError::Pinned);
    EXPECT_EQ(*pin, 7);
    EXPECT_NE(replacement, nullptr);
}
#endif

}  // namespace milvus::cachinglayer
