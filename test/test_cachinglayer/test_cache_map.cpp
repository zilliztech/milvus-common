#include <folly/OperationCancelled.h>
#include <folly/ScopeGuard.h>
#include <folly/coro/BlockingWait.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/executors/ManualExecutor.h>
#include <folly/futures/Future.h>
#include <folly/synchronization/Baton.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "cachinglayer/CacheMap.h"
#include "cachinglayer_test_utils.h"
#include "mock_list_node.h"
#include "syncpoint/sync_point.h"

struct CacheMapTestKey {
    int group;
    int item;
    // Equal keys can own different amounts of metadata.
    std::vector<char> metadata;
    inline static std::atomic<int64_t> live_key_bytes{0};
    inline static std::atomic<size_t> copies{0};
    inline static thread_local size_t comparisons = 0;
    inline static bool fail_copy = false;
    inline static bool fail_hash = false;
    inline static bool fail_equal = false;

    CacheMapTestKey(const int group, const int item, const size_t key_bytes = 0)
        : group(group), item(item), metadata(key_bytes, 'x') {
        live_key_bytes += metadata.size();
    }
    CacheMapTestKey(const CacheMapTestKey& other) : group(other.group), item(other.item), metadata(other.metadata) {
        if (fail_copy) {
            throw std::bad_alloc{};
        }
        ++copies;
        live_key_bytes += metadata.size();
    }
    ~CacheMapTestKey() {
        live_key_bytes -= metadata.size();
    }
    bool
    operator==(const CacheMapTestKey& other) const {
        ++comparisons;
        if (fail_equal) {
            throw std::invalid_argument("key equality failed");
        }
        return group == other.group && item == other.item;
    }
};

template <>
struct std::hash<CacheMapTestKey> {
    size_t
    operator()(const CacheMapTestKey&) const {
        if (CacheMapTestKey::fail_hash) {
            throw milvus::SegcoreError(milvus::ErrorCode::FileReadFailed, "key hash failed");
        }
        return 0;  // Deliberately collide distinct keys to exercise equality.
    }
};

struct CacheMapNoexceptHashKey : CacheMapTestKey {
    using CacheMapTestKey::CacheMapTestKey;
};

template <>
struct std::hash<CacheMapNoexceptHashKey> {
    size_t
    operator()(const CacheMapNoexceptHashKey&) const noexcept {
        return 0;
    }
};

struct CacheMapNonComparableKey {};

struct CacheMapBlockingKey {
    uint64_t value;
    inline static thread_local std::function<void()> on_equal;
    bool
    operator==(const CacheMapBlockingKey& other) const {
        if (on_equal) {
            on_equal();
        }
        return value == other.value;
    }
};

template <>
struct std::hash<CacheMapBlockingKey> {
    size_t
    operator()(const CacheMapBlockingKey& key) const noexcept {
        return key.value;
    }
};

template <>
struct std::hash<CacheMapNonComparableKey> {
    size_t
    operator()(const CacheMapNonComparableKey&) const noexcept {
        return 0;
    }
};

namespace milvus::cachinglayer {

template <typename Key>
constexpr bool accepts_cache_key = requires { typename CacheMap<Key, int>; };

static_assert(accepts_cache_key<int>);
static_assert(accepts_cache_key<std::string>);
static_assert(accepts_cache_key<CacheMapTestKey>);
static_assert(!accepts_cache_key<std::vector<int>>);  // Equality but no std::hash.
static_assert(!accepts_cache_key<CacheMapNonComparableKey>);
static_assert(!accepts_cache_key<std::unique_ptr<int>>);  // Hash/equality but not copyable.

TEST(ChainedHashTableTest, AlignedHashesDistributeAcrossBucketsAndRemainRemovable) {
    struct Node {
        explicit Node(const uint64_t value) : key{value}, hook{nullptr, std::hash<CacheMapBlockingKey>{}(key)} {
        }
        const CacheMapBlockingKey&
        getKey() const {
            return key;
        }
        const CacheMapBlockingKey key;
        internal::ChainedHashTableHook<Node> hook;
    };
    using Table = internal::ChainedHashTable<CacheMapBlockingKey, Node, &Node::hook>;
    // The explicit identity hasher makes this regression independent of the
    // standard library's integer hash. Exercise both aligned offsets and high bits.
    for (const int shift : {12, std::numeric_limits<size_t>::digits - 12}) {
        SCOPED_TRACE(shift);
        Table table(internal::ChainedHashTableConfig{});
        std::vector<std::shared_ptr<Node>> nodes;
        for (uint64_t i = 0; i < 1024; ++i) {
            auto node = std::make_shared<Node>(i << shift);
            ASSERT_EQ(table.insertIfAbsent(node), node);
            nodes.push_back(std::move(node));
        }
        EXPECT_EQ(table.size(), nodes.size());
        auto duplicate = std::make_shared<Node>(nodes.front()->getKey().value);
        EXPECT_EQ(table.insertIfAbsent(duplicate), nodes.front());
        EXPECT_EQ(table.size(), nodes.size());
        EXPECT_FALSE(duplicate->hook.next);
        size_t occupied_buckets = 0;
        size_t longest_chain = 0;
        for (size_t bucket = 0; bucket < table.getNumBuckets(); ++bucket) {
            size_t length = 0;
            for (auto node = table.getBucketHead(bucket); node; node = node->hook.next) {
                ++length;
            }
            occupied_buckets += length != 0;
            longest_chain = std::max(longest_chain, length);
        }
        // Guard against collapsed buckets without requiring an exact hash output.
        EXPECT_GT(occupied_buckets, 256);
        EXPECT_LT(longest_chain, 16);
        for (const auto& node : nodes) {
            EXPECT_EQ(table.find(node->key), node);
        }
        // Remove oldest first, the order that degenerates with a single long chain.
        for (const auto& node : nodes) {
            EXPECT_EQ(table.remove(*node), node);
            EXPECT_FALSE(node->hook.next);
            EXPECT_FALSE(table.find(node->key));
        }
        EXPECT_EQ(table.size(), 0);
    }
}

template <CacheMapKey Key, typename Value>
static typename CacheMap<Key, Value>::PinResult
ReserveAndPut(CacheMap<Key, Value>& cache, const std::type_identity_t<Key>& key, std::unique_ptr<Value>& value,
              const ResourceUsage bytes) {
    auto reserved = cache.Reserve(bytes);
    if (!reserved) {
        return folly::makeUnexpected(reserved.error());
    }
    SCOPE_EXIT {
        cache.UnReserve(bytes);
    };
    return cache.Put(key, value, bytes);
}

class CacheMapTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        EvictionConfig config;
        config.cache_touch_window = std::chrono::milliseconds(0);
        dlist_ = std::make_shared<internal::DList>(true, ResourceUsage{8192, 0}, ResourceUsage{8192, 0},
                                                   ResourceUsage{8192, 0}, config);
        cache_ = std::make_unique<CacheMap<std::string, int>>(dlist_);
    }

    std::shared_ptr<internal::DList> dlist_;
    std::unique_ptr<CacheMap<std::string, int>> cache_;
};

TEST_F(CacheMapTest, CustomKeysPreserveIdentityAcrossCollisionsAndEviction) {
    CacheMap<CacheMapTestKey, int> cache(dlist_);
    CacheMapTestKey key{1, 1};
    auto first = std::make_unique<int>(7);
    auto second = std::make_unique<int>(9);
    ASSERT_TRUE(ReserveAndPut(cache, key, first, {3000, 0}));
    key.item = 2;
    ASSERT_TRUE(ReserveAndPut(cache, key, second, {3000, 0}));
    EXPECT_EQ(*cache.Get({1, 1}).value(), 7);
    EXPECT_EQ(*cache.Get({1, 2}).value(), 9);
    EXPECT_FALSE(cache.Get({2, 1}).value());
    auto pinned = cache.Get({1, 1}).value();
    auto replacement = std::make_unique<int>(11);
    EXPECT_EQ(ReserveAndPut(cache, {1, 1}, replacement, {3000, 0}).error().get_error_code(),
              ErrorCode::InvalidParameter);
    EXPECT_NE(replacement, nullptr);
    EXPECT_EQ(cache.Erase({1, 1}).error().get_error_code(), ErrorCode::UnexpectedError);
    ASSERT_TRUE(ReserveAndPut(cache, {2, 1}, replacement, {3000, 0}));
    EXPECT_FALSE(cache.Get({1, 2}).value());
    EXPECT_EQ(*pinned, 7);
    EXPECT_EQ(*cache.Get({2, 1}).value(), 11);
    pinned = {};
    EXPECT_TRUE(cache.Erase({1, 1}));
    EXPECT_TRUE(cache.Erase({2, 1}));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, CustomKeyStorageDoesNotChangePayloadCharge) {
    CacheMap<CacheMapTestKey, int> cache(dlist_);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1, 4096}, value, {sizeof(int), 0}));
    ASSERT_EQ(CacheMapTestKey::live_key_bytes.load(), 4096);
    const auto before = internal::DListTestFriend::get_used_memory(*dlist_);
    EXPECT_EQ(before, (ResourceUsage{sizeof(int), 0}));
    auto duplicate = std::make_unique<int>(9);
    auto result = ReserveAndPut(cache, {1, 1}, duplicate, {sizeof(int), 0});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().get_error_code(), ErrorCode::InvalidParameter);
    EXPECT_NE(duplicate, nullptr);
    EXPECT_EQ(*cache.Get({1, 1}).value(), 7);
    EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 4096);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), before);
    ASSERT_TRUE(cache.Erase({1, 1}));
    EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, CustomKeyExceptionsPreserveInputAndResidentEntries) {
    auto reset = folly::makeGuard([] {
        CacheMapTestKey::fail_copy = false;
        CacheMapTestKey::fail_hash = false;
        CacheMapTestKey::fail_equal = false;
    });
    CacheMap<CacheMapTestKey, int> cache(dlist_);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1}, value, {32, 0}));
    const auto before = internal::DListTestFriend::get_used_memory(*dlist_);
    for (const auto& [failure, code] : {std::pair{&CacheMapTestKey::fail_hash, ErrorCode::FileReadFailed},
                                        std::pair{&CacheMapTestKey::fail_equal, ErrorCode::InvalidParameter},
                                        std::pair{&CacheMapTestKey::fail_copy, ErrorCode::MemAllocateFailed}}) {
        auto input = std::make_unique<int>(9);
        *failure = true;
        EXPECT_EQ(ReserveAndPut(cache, {1, 2}, input, {32, 0}).error().get_error_code(), code);
        if (failure != &CacheMapTestKey::fail_copy) {
            EXPECT_EQ(cache.Get({1, 2}).error().get_error_code(), code);
            EXPECT_EQ(cache.Erase({1, 2}).error().get_error_code(), code);
        }
        *failure = false;
        ASSERT_NE(input, nullptr);
        EXPECT_EQ(*input, 9);
        EXPECT_EQ(cache.Size(), 1);
        EXPECT_EQ(*cache.Get({1, 1}).value(), 7);
        EXPECT_FALSE(cache.Get({1, 2}).value());
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), before);
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
    }
}

TEST_F(CacheMapTest, CustomKeyCleanupDoesNotInvokeHashOrEquality) {
    auto reset = folly::makeGuard([] {
        CacheMapTestKey::fail_hash = false;
        CacheMapTestKey::fail_equal = false;
    });
    auto cache = std::make_unique<CacheMap<CacheMapTestKey, int>>(dlist_);
    auto first = std::make_unique<int>(7);
    auto second = std::make_unique<int>(9);
    auto third = std::make_unique<int>(11);
    ASSERT_TRUE(ReserveAndPut(*cache, {1, 1}, first, {32, 0}));
    ASSERT_TRUE(ReserveAndPut(*cache, {1, 2}, second, {32, 0}));
    ASSERT_TRUE(ReserveAndPut(*cache, {1, 3}, third, {32, 0}));
    auto pin = cache->Get({1, 1}).value();
    ASSERT_TRUE(pin);
    // Exceed the current free capacity regardless of platform-specific node sizes.
    const ResourceUsage reservation{internal::DListTestFriend::get_max_memory(*dlist_).memory_bytes -
                                        internal::DListTestFriend::get_used_memory(*dlist_).memory_bytes + 1,
                                    0};
    auto failed = cache->GetOrCreate({1, 4}, []() -> CacheMap<CacheMapTestKey, int>::CreateResult {
        CacheMapTestKey::fail_hash = true;
        CacheMapTestKey::fail_equal = true;
        return folly::makeUnexpected(SegcoreError(ErrorCode::FileReadFailed, "metadata read failed"));
    });
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().get_error_code(), ErrorCode::FileReadFailed);
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout(reservation, std::chrono::milliseconds(0))));
    EXPECT_LT(cache->Size(), 3);
    dlist_->ReleaseLoadingResource(reservation);
    EXPECT_NO_THROW(cache.reset());
    ASSERT_TRUE(pin);
    EXPECT_EQ(*pin, 7);
    EXPECT_NO_THROW(pin = {});
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, GetPinsTheInsertedValueAndMissDoesNotInsert) {
    auto miss = cache_->Get("a");
    ASSERT_TRUE(miss);
    EXPECT_FALSE(miss.value());
    EXPECT_EQ(cache_->Size(), 0);
    const ResourceUsage estimate{8192, 0};
    ASSERT_TRUE(cache_->Reserve(estimate));
    auto value = std::make_unique<int>(7);
    auto result = cache_->Put("a", value, {sizeof(int), 0});
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), estimate);
    cache_->UnReserve(estimate);
    ASSERT_TRUE(result);
    EXPECT_EQ(value, nullptr);
    CacheMap<std::string, int>::Pin inserted = std::move(result).value();
    ASSERT_TRUE(inserted);
    EXPECT_EQ(*inserted, 7);
    auto found = cache_->Get("a");
    static_assert(std::is_same_v<decltype(found), decltype(result)>);
    ASSERT_TRUE(found);
    EXPECT_EQ(*found.value(), 7);
    EXPECT_EQ(&*found.value(), &*inserted);
    EXPECT_EQ(cache_->Size(), 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{sizeof(int), 0}));
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, GetOrCreateCallsFactoryOnlyOnMiss) {
    int calls = 0;
    auto create = [&] { return std::pair{std::make_unique<int>(++calls), ResourceUsage{32, 0}}; };
    auto first = cache_->GetOrCreate("a", create);
    ASSERT_TRUE(first);
    ASSERT_TRUE(first.value());
    EXPECT_EQ(*first.value(), 1);
    {
        auto second = cache_->GetOrCreate("a", create);
        ASSERT_TRUE(second);
        EXPECT_EQ(&*second.value(), &*first.value());
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(cache_->Size(), 1);
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{32, 0}));
        first.value() = {};
        EXPECT_EQ(cache_->Erase("a").error().get_error_code(), ErrorCode::UnexpectedError);
    }
    ASSERT_TRUE(cache_->Erase("a"));
    auto next = cache_->GetOrCreate("a", create);
    ASSERT_TRUE(next);
    EXPECT_EQ(*next.value(), 2);
}

TEST_F(CacheMapTest, GetOrCreateUsesValuePublishedByPutWhileFactoryRuns) {
    auto result = cache_->GetOrCreate("a", [&] {
        auto first = std::make_unique<int>(7);
        EXPECT_TRUE(ReserveAndPut(*cache_, "a", first, {16, 0}));
        return std::pair{std::make_unique<int>(9), ResourceUsage{64, 0}};
    });
    ASSERT_TRUE(result);
    EXPECT_EQ(*result.value(), 7);
    EXPECT_EQ(cache_->Size(), 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{16, 0}));
}

TEST_F(CacheMapTest, GetOrCreateReportsCloseDuringFailingFactory) {
    auto result = cache_->GetOrCreate("a", [&]() -> CacheMap<std::string, int>::CreateResult {
        cache_.reset();
        throw SegcoreError(ErrorCode::FileReadFailed, "metadata read failed");
    });
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().get_error_code(), ErrorCode::FollyCancel);
}

TEST_F(CacheMapTest, GetOrCreateScansTheResidentBucketTwiceOnCreation) {
    CacheMap<CacheMapTestKey, int> cache(dlist_);
    auto sentinel = std::make_unique<int>(1);
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1}, sentinel, {32, 0}));
    int calls = 0;
    auto create = [&] { return std::pair{std::make_unique<int>(++calls), ResourceUsage{32, 0}}; };
    // Every key hashes to the same bucket. Each miss compares with the one resident sentinel.
    CacheMapTestKey::comparisons = 0;
    auto created = cache.GetOrCreate({1, 2}, create);
    ASSERT_TRUE(created && created.value());
    EXPECT_EQ(CacheMapTestKey::comparisons, 2);
    CacheMapTestKey::comparisons = 0;
    auto hit = cache.GetOrCreate({1, 2}, create);
    ASSERT_TRUE(hit && hit.value());
    EXPECT_EQ(CacheMapTestKey::comparisons, 1);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(&*created.value(), &*hit.value());
}

TEST_F(CacheMapTest, UnReserveAfterAbandonedLoadReturnsCapacityWithoutPublishing) {
    const ResourceUsage capacity{8192, 0};
    auto reserved = cache_->Reserve(capacity);
    static_assert(std::is_same_v<decltype(reserved), CacheMap<std::string, int>::Result>);
    ASSERT_TRUE(reserved);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_FALSE(cache_->Get("a").value());
    EXPECT_EQ(cache_->Reserve({1, 0}).error().get_error_code(), ErrorCode::InsufficientResource);
    EXPECT_EQ(cache_->Reserve({1, 0}, std::chrono::milliseconds(10)).error().get_error_code(),
              ErrorCode::InsufficientResource);
    cache_->UnReserve(capacity);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
    ASSERT_TRUE(cache_->ReserveAsync(capacity).get());
    cache_->UnReserve(capacity);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, ReserveAsyncReturnsTheAdmissionResult) {
    using Cache = CacheMap<std::string, int>;
    const ResourceUsage capacity{8192, 0};
    ASSERT_TRUE(cache_->ReserveAsync({}).get());
    cache_->UnReserve({});
    auto first = cache_->ReserveAsync(capacity);
    static_assert(std::is_same_v<decltype(first), folly::SemiFuture<Cache::Result>>);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
    ASSERT_TRUE(folly::coro::blockingWait(std::move(first)));
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
    EXPECT_EQ(cache_->ReserveAsync({-1, 0}).get().error().get_error_code(), ErrorCode::InvalidParameter);
    auto denied = cache_->ReserveAsync({1, 0}).get();
    ASSERT_FALSE(denied);
    EXPECT_EQ(denied.error().get_error_code(), ErrorCode::InsufficientResource);
    EXPECT_THAT(denied.error().what(), ::testing::HasSubstr("memory_bytes=1, file_bytes=0"));
    EXPECT_EQ(cache_->ReserveAsync({1, 0}, std::chrono::milliseconds(10)).get().error().get_error_code(),
              ErrorCode::InsufficientResource);

    auto pending = cache_->ReserveAsync(capacity, std::chrono::seconds(2));
    EXPECT_FALSE(pending.isReady());
    EXPECT_EQ(internal::DListTestFriend::get_waiting_request_count(*dlist_), 1);
    cache_->UnReserve(capacity);
    ASSERT_TRUE(std::move(pending).get(std::chrono::seconds(2)));
    auto value = std::make_unique<int>(42);
    auto published = cache_->Put("answer", value, {sizeof(*value), 0});
    ASSERT_TRUE(published);
    cache_->UnReserve(capacity);
    EXPECT_EQ(*published.value(), 42);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, ReserveAsyncCancelsOnCloseAndRefundsAGrantAwaitingDelivery) {
    const ResourceUsage capacity{8192, 0};
    for (const bool grant_before_close : {false, true}) {
        ASSERT_TRUE(cache_->Reserve(capacity));
        auto closing = std::make_unique<CacheMap<int, int>>(dlist_);
        folly::ManualExecutor executor;
        auto pending = closing->ReserveAsync(capacity, std::chrono::milliseconds(-1)).via(&executor);
        if (grant_before_close) {
            cache_->UnReserve(capacity);
            // Wait for the grant to reach the executor. getVia() delivers it after close.
            executor.wait();
            EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
        }
        closing.reset();
        auto result = std::move(pending).getVia(&executor);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().get_error_code(), ErrorCode::FollyCancel);
        if (!grant_before_close) {
            cache_->UnReserve(capacity);
        }
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
    }
}

TEST_F(CacheMapTest, GetDoesNotWaitForListLock) {
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", value, {32, 0}));
    auto lock = internal::DListTestFriend::test_lock_list(dlist_.get());
    auto lookup = std::async(std::launch::async, [&] { return cache_->Get("a"); });
    EXPECT_EQ(lookup.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    lock.unlock();
    auto pin = lookup.get().value();
    ASSERT_TRUE(pin);
    EXPECT_EQ(*pin, 7);
}

TEST_F(CacheMapTest, GetOnAnotherStripeDoesNotWaitForBlockedKeyComparison) {
    CacheMap<CacheMapBlockingKey, int> cache(dlist_, {.buckets_power = 1, .locks_power = 1});
    auto first = std::make_unique<int>(7);
    auto second = std::make_unique<int>(9);
    // These identity hashes select different stripes after mixing.
    ASSERT_TRUE(ReserveAndPut(cache, {0}, first, {32, 0}));
    ASSERT_TRUE(ReserveAndPut(cache, {3}, second, {32, 0}));
    folly::Baton<> comparing;
    folly::Baton<> resume;
    auto blocked = std::async(std::launch::async, [&] {
        CacheMapBlockingKey::on_equal = [&] {
            comparing.post();
            resume.wait();
        };
        return cache.Get({0});
    });
    auto unblock = folly::makeGuard([&] { resume.post(); });
    ASSERT_TRUE(comparing.try_wait_for(std::chrono::seconds(5)));
    auto independent = std::async(std::launch::async, [&] { return cache.Get({3}); });
    EXPECT_EQ(independent.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    resume.post();
    auto first_pin = blocked.get().value();
    auto second_pin = independent.get().value();
    ASSERT_TRUE(first_pin);
    ASSERT_TRUE(second_pin);
    EXPECT_EQ(*first_pin, 7);
    EXPECT_EQ(*second_pin, 9);
}

TEST_F(CacheMapTest, BlockedKeyComparisonAllowsPutAndGetOrCreateOnAnotherStripe) {
    for (const bool blocked_put : {false, true}) {
        SCOPED_TRACE(blocked_put);
        CacheMap<CacheMapBlockingKey, int> cache(dlist_, {.buckets_power = 1, .locks_power = 1});
        auto first = std::make_unique<int>(7);
        ASSERT_TRUE(ReserveAndPut(cache, {0}, first, {32, 0}));
        auto unused_factory = [] {
            ADD_FAILURE() << "A resident key must skip the factory";
            return std::pair{std::make_unique<int>(0), ResourceUsage{}};
        };
        folly::Baton<> comparing, resume;
        auto blocked = std::async(std::launch::async, [&] {
            CacheMapBlockingKey::on_equal = [&] {
                comparing.post();
                resume.wait();
            };
            if (blocked_put) {
                auto duplicate = std::make_unique<int>(0);
                return ReserveAndPut(cache, {0}, duplicate, {32, 0});
            }
            return cache.GetOrCreate({0}, unused_factory);
        });
        auto unblock = folly::makeGuard([&] { resume.post(); });
        ASSERT_TRUE(comparing.try_wait_for(std::chrono::seconds(5)));
        auto independent = std::async(std::launch::async, [&] {
            auto second = std::make_unique<int>(9);
            auto published = ReserveAndPut(cache, {3}, second, {32, 0});
            if (!published) {
                return published;
            }
            return cache.GetOrCreate({3}, unused_factory);
        });
        EXPECT_EQ(independent.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        resume.post();
        unblock.dismiss();
        auto a = blocked.get();
        auto b = independent.get();
        ASSERT_TRUE(b && b.value());
        if (blocked_put) {
            ASSERT_FALSE(a);
            EXPECT_EQ(a.error().get_error_code(), ErrorCode::InvalidParameter);
        } else {
            ASSERT_TRUE(a && a.value());
            EXPECT_EQ(*a.value(), 7);
        }
        EXPECT_EQ(*b.value(), 9);
    }
}

TEST_F(CacheMapTest, InvalidIndexConfigurationIsRejected) {
    using Cache = CacheMap<int, int>;
    EXPECT_THROW((Cache(dlist_, {.buckets_power = 1, .locks_power = 2})), std::invalid_argument);
    EXPECT_THROW((Cache(dlist_, {.buckets_power = std::numeric_limits<size_t>::digits, .locks_power = 0})),
                 std::invalid_argument);
}

TEST(CacheMapLockingTest, UnpinWithinTouchWindowDoesNotWaitForListLock) {
    EvictionConfig config;
    config.cache_touch_window = std::chrono::hours(1);
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{100, 0}, ResourceUsage{100, 0},
                                                   ResourceUsage{100, 0}, config);
    CacheMap<int, int> cache(dlist);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, 1, value, {32, 0}));
    auto pin = cache.Get(1).value();
    ASSERT_TRUE(pin);
    auto lock = internal::DListTestFriend::test_lock_list(dlist.get());
    auto unpin = std::async(std::launch::async, [pin = std::move(pin)]() mutable { pin = {}; });
    EXPECT_EQ(unpin.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    lock.unlock();
    unpin.get();
    EXPECT_TRUE(cache.Erase(1));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), ResourceUsage{});
}

TEST(CacheMapLockingTest, NonFinalUnpinDoesNotRetryPendingReservations) {
    for (const bool close_cache : {false, true}) {
        SCOPED_TRACE(close_cache);
        EvictionConfig config;
        config.cache_touch_window = std::chrono::hours(1);
        const ResourceUsage capacity{100, 0};
        auto dlist = std::make_shared<internal::DList>(true, capacity, capacity, capacity, config);
        auto cache = std::make_unique<CacheMap<int, int>>(dlist);
        auto value = std::make_unique<int>(7);
        ASSERT_TRUE(ReserveAndPut(*cache, 1, value, capacity));
        auto retained = cache->Get(1).value();
        auto temporary = cache->Get(1).value();
        ASSERT_TRUE(retained);
        ASSERT_TRUE(temporary);
        if (close_cache) {
            cache.reset();
        }
        auto pending = dlist->ReserveLoadingResourceWithTimeout(capacity, std::chrono::milliseconds(-1))
                           .via(&folly::InlineExecutor::instance())
                           .thenValue([&](const bool success) {
                               // Final unpin must notify outside both cache locks.
                               EXPECT_TRUE(dlist->IsEmpty());
                               EXPECT_FALSE(retained);
                               return success;
                           });
        ASSERT_FALSE(pending.isReady());
        ASSERT_EQ(internal::DListTestFriend::get_waiting_request_count(*dlist), 1);

        auto lock = internal::DListTestFriend::test_lock_list(dlist.get());
        auto unpin = std::async(std::launch::async, [pin = std::move(temporary)]() mutable { pin = {}; });
        EXPECT_EQ(unpin.wait_for(std::chrono::seconds(1)), std::future_status::ready);
        lock.unlock();
        unpin.get();
        EXPECT_FALSE(pending.isReady());
        EXPECT_EQ(*retained, 7);
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), capacity);

        // The last pin either makes the entry evictable or refunds it after Close.
        retained = {};
        EXPECT_TRUE(std::move(pending).get(std::chrono::seconds(2)));
        dlist->ReleaseLoadingResource(capacity);
        EXPECT_TRUE(dlist->IsEmpty());
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), ResourceUsage{});
    }
}

TEST_F(CacheMapTest, DuplicatePutPreservesPinnedAndUnpinnedValues) {
    auto value = std::make_unique<int>(7);
    const auto* original = value.get();
    auto inserted = ReserveAndPut(*cache_, "a", value, {32, 0});
    ASSERT_TRUE(inserted);
    auto handle = std::move(inserted).value();
    auto input = std::make_unique<int>(9);
    const auto* candidate = input.get();
    auto result = ReserveAndPut(*cache_, "a", input, {64, 0});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().get_error_code(), ErrorCode::InvalidParameter);
    EXPECT_THAT(result.error().what(), ::testing::HasSubstr("key already exists"));
    ASSERT_EQ(input.get(), candidate);
    EXPECT_EQ(cache_->Erase("a").error().get_error_code(), ErrorCode::UnexpectedError);
    EXPECT_EQ(*handle, 7);
    handle = {};
    auto unpinned_duplicate = ReserveAndPut(*cache_, "a", input, {64, 0});
    ASSERT_FALSE(unpinned_duplicate);
    EXPECT_EQ(unpinned_duplicate.error().get_error_code(), ErrorCode::InvalidParameter);
    EXPECT_EQ(input.get(), candidate);
    EXPECT_EQ(&*cache_->Get("a").value(), original);
    EXPECT_EQ(cache_->Size(), 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{32, 0}));
    ASSERT_TRUE(cache_->Erase("a"));
    EXPECT_FALSE(cache_->Get("a").value());
    EXPECT_EQ(cache_->Erase("a").error().get_error_code(), ErrorCode::OutOfRange);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
    auto reinserted = ReserveAndPut(*cache_, "a", input, {64, 0});
    ASSERT_TRUE(reinserted);
    EXPECT_EQ(&*reinserted.value(), candidate);
    EXPECT_EQ(input, nullptr);
}

TEST_F(CacheMapTest, ConcurrentPutsForTheSameKeyAcceptExactlyOneValue) {
    std::latch writers(2);
    auto put = [&](std::unique_ptr<int>& value) {
        writers.arrive_and_wait();
        auto result = ReserveAndPut(*cache_, "a", value, {32, 0});
        if (result) {
            EXPECT_EQ(value, nullptr);
            return true;
        }
        EXPECT_EQ(result.error().get_error_code(), ErrorCode::InvalidParameter);
        EXPECT_NE(value, nullptr);
        return false;
    };
    auto first = std::make_unique<int>(1);
    auto second = std::make_unique<int>(2);
    auto writer = std::async(std::launch::async, [&] { return put(first); });
    const bool second_won = put(second);
    ASSERT_NE(writer.get(), second_won);
    EXPECT_EQ(*cache_->Get("a").value(), second_won ? 2 : 1);
    EXPECT_EQ(cache_->Size(), 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{32, 0}));
}

TEST_F(CacheMapTest, EvictsLeastRecentlyUnpinnedEntryFromBothListAndMap) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    auto c = std::make_unique<int>(3);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", a, {3000, 0}));
    ASSERT_TRUE(ReserveAndPut(*cache_, "b", b, {3000, 0}));
    { auto access = cache_->Get("a"); }
    ASSERT_TRUE(ReserveAndPut(*cache_, "c", c, {3000, 0}));
    EXPECT_FALSE(cache_->Get("b").value());
    EXPECT_EQ(*cache_->Get("a").value(), 1);
    EXPECT_EQ(*cache_->Get("c").value(), 3);
    EXPECT_EQ(cache_->Size(), 2);
}

TEST_F(CacheMapTest, AllPinnedCapacityFailurePreservesEntriesAndInput) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    auto c = std::make_unique<int>(3);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", a, {3000, 0}));
    ASSERT_TRUE(ReserveAndPut(*cache_, "b", b, {3000, 0}));
    auto first = cache_->Get("a").value();
    auto second = cache_->Get("b").value();
    auto result = ReserveAndPut(*cache_, "c", c, {3000, 0});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().get_error_code(), ErrorCode::InsufficientResource);
    EXPECT_NE(c, nullptr);
    EXPECT_EQ(*first, 1);
    EXPECT_EQ(*second, 2);
    EXPECT_EQ(cache_->Size(), 2);
}

TEST_F(CacheMapTest, EvictedKeyCanBeInsertedAgain) {
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", value, {3000, 0}));
    EXPECT_EQ(cache_->Reserve({8193, 0}).error().get_error_code(), ErrorCode::InsufficientResource);
    EXPECT_EQ(*cache_->Get("a").value(), 7);
    const ResourceUsage estimate{7000, 0};
    ASSERT_TRUE(cache_->Reserve(estimate));
    EXPECT_FALSE(cache_->Get("a").value());
    auto replacement = std::make_unique<int>(9);
    ASSERT_TRUE(cache_->Put("a", replacement, estimate));
    cache_->UnReserve(estimate);
    EXPECT_EQ(*cache_->Get("a").value(), 9);
}

TEST_F(CacheMapTest, HandleKeepsValueAliveAfterCacheDestruction) {
    CacheMap<int, int> other(dlist_);
    auto other_value = std::make_unique<int>(9);
    ASSERT_TRUE(ReserveAndPut(other, 1, other_value, {32, 0}));
    auto value = std::make_unique<int>(7);
    auto result = ReserveAndPut(*cache_, "a", value, {32, 0});
    ASSERT_TRUE(result);
    auto handle = std::move(result).value();
    cache_.reset();
    EXPECT_EQ(*handle, 7);
    EXPECT_GT(internal::DListTestFriend::get_used_memory(*dlist_).memory_bytes, 0);
    const auto capacity = internal::DListTestFriend::get_max_memory(*dlist_);
    auto pending = dlist_->ReserveLoadingResourceWithTimeout(capacity, std::chrono::milliseconds(-1))
                       .via(&folly::InlineExecutor::instance())
                       .thenValue([&](const bool success) {
                           // Both operations take the domain lock: completion must run after unpin releases it.
                           EXPECT_TRUE(dlist_->IsEmpty());
                           EXPECT_EQ(other.Size(), 0);
                           return success;
                       });
    EXPECT_FALSE(pending.isReady());
    handle = {};
    // Removing the closed cache's frozen node must leave the other entry evictable.
    EXPECT_TRUE(std::move(pending).get(std::chrono::seconds(2)));
    dlist_->ReleaseLoadingResource(capacity);
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, GlobalReservationEvictsEntriesFromListAndMap) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", a, {3000, 0}));
    ASSERT_TRUE(ReserveAndPut(*cache_, "b", b, {3000, 0}));
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({7000, 0}, std::chrono::milliseconds(0))));
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_FALSE(cache_->Get("a").value());
    EXPECT_FALSE(cache_->Get("b").value());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
    dlist_->ReleaseLoadingResource({7000, 0});
}

TEST_F(CacheMapTest, BatchEvictionReleasesKeyedOwnersWhileKeepingSlotLocked) {
    // A Slot-like node remains externally owned after unloading.
    ::testing::NiceMock<internal::MockListNode> slot(dlist_.get());
    slot.test_set_loaded_size({32, 0});
    slot.test_set_state(internal::ListNode::State::CACHED);
    dlist_->ChargeLoadedResource({32, 0});
    dlist_->touchItem(&slot, true, ResourceUsage{32, 0});

    struct Value {
        std::function<void()> on_destroy;
        ~Value() {
            if (on_destroy) {
                on_destroy();
            }
        }
    };
    CacheMap<CacheMapTestKey, Value> cache(dlist_);
    auto value = std::make_unique<Value>();
    ASSERT_TRUE(ReserveAndPut(cache, {1, 0}, value, {32, 0}));
    auto pinned = cache.Get({1, 0}).value();
    ASSERT_TRUE(pinned);

    value = std::make_unique<Value>();
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1, 4096}, value, {32, 0}));
    value = std::make_unique<Value>();
    value->on_destroy = [&] {
        // The preceding KV wrapper (including its Key storage) should already be
        // destroyed. Slot locks must stay held until the batch finishes accounting.
        EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 0);
        auto slot_unlocked = std::async(std::launch::async, [&] {
            std::unique_lock lock(slot.test_get_mutex(), std::try_to_lock);
            return lock.owns_lock();
        });
        EXPECT_FALSE(slot_unlocked.get());
    };
    ASSERT_TRUE(ReserveAndPut(cache, {1, 2}, value, {32, 0}));
    ASSERT_EQ(CacheMapTestKey::live_key_bytes.load(), 4096);

    // Evict the Slot and both unpinned KV entries, skipping the pinned entry in
    // between. This also exercises victim/lock alignment after a skipped node.
    const ResourceUsage reservation{8160, 0};
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout(reservation, std::chrono::milliseconds(0))));
    dlist_->ReleaseLoadingResource(reservation);
    EXPECT_EQ(slot.test_get_state(), internal::ListNode::State::NOT_LOADED);
    EXPECT_EQ(cache.Size(), 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{32, 0}));
    EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 0);
    pinned = {};
    EXPECT_TRUE(cache.Erase({1, 0}));
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, EmptyPayloadsDoNotConsumeCapacity) {
    for (int i = 0; i < 200; ++i) {
        auto value = std::make_unique<int>(i);
        ASSERT_TRUE(ReserveAndPut(*cache_, std::to_string(i), value, {0, 0}));
    }
    EXPECT_EQ(cache_->Size(), 200);
    EXPECT_EQ(*cache_->Get("199").value(), 199);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
    cache_.reset();
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, FullCapacityReservationEvictsAllPayloads) {
    for (int i = 0; i < 3; ++i) {
        auto value = std::make_unique<int>(i);
        ASSERT_TRUE(ReserveAndPut(*cache_, std::to_string(i), value, {32, 0}));
    }
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{96, 0}));
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(0))));
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
    dlist_->ReleaseLoadingResource({8192, 0});
}

TEST_F(CacheMapTest, PinnedPayloadPreventsFullCapacityReservation) {
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", a, {32, 0}));
    ASSERT_TRUE(ReserveAndPut(*cache_, "b", b, {32, 0}));
    auto pinned = cache_->Get("a").value();
    EXPECT_FALSE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(0))));
    EXPECT_EQ(cache_->Size(), 2);
    EXPECT_EQ(*cache_->Get("b").value(), 2);
    pinned = {};
    ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(0))));
    dlist_->ReleaseLoadingResource({8192, 0});
}

TEST_F(CacheMapTest, ChainedInsertsPreservePinsWithoutCopyingResidentKeys) {
    ASSERT_TRUE(dlist_->UpdateMaxLimit({16384, 0}));
    // This hasher is deliberately not noexcept; inserts must still avoid index copies.
    CacheMap<CacheMapTestKey, int> cache(dlist_, {.buckets_power = 0, .locks_power = 0});
    const auto copies_before = CacheMapTestKey::copies.load();
    std::vector<CacheMap<CacheMapTestKey, int>::Pin> handles;
    for (int i = 0; i < 64; ++i) {
        auto value = std::make_unique<int>(i);
        ASSERT_TRUE(ReserveAndPut(cache, {1, i}, value, {sizeof(int), 0}));
        handles.push_back(cache.Get({1, i}).value());
    }
    EXPECT_EQ(CacheMapTestKey::copies.load() - copies_before, 64);
    EXPECT_EQ(cache.Size(), 64);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{64 * sizeof(int), 0}));
    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(*handles[i], i);
    }
    handles.clear();
    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(cache.Erase({1, i}));
    }
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, MovingPinReleasesItsOldCacheAndKeepsTheNewOne) {
    auto second = std::make_unique<CacheMap<std::string, int>>(dlist_);
    auto integer_keys = std::make_unique<CacheMap<int, int>>(dlist_);
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    auto c = std::make_unique<int>(3);
    ASSERT_TRUE(ReserveAndPut(*cache_, "same", a, {32, 0}));
    ASSERT_TRUE(ReserveAndPut(*second, "same", b, {32, 0}));
    ASSERT_TRUE(ReserveAndPut(*integer_keys, 1, c, {32, 0}));
    CacheMap<std::string, int>::Pin first_handle = cache_->Get("same").value();
    CacheMap<std::string, int>::Pin second_handle = second->Get("same").value();
    CacheMap<int, int>::Pin integer_handle = integer_keys->Get(1).value();
    cache_.reset();
    second.reset();
    integer_keys.reset();
    first_handle = std::move(second_handle);
    EXPECT_FALSE(second_handle);
    EXPECT_EQ(*first_handle, 2);
    static_assert(!std::is_assignable_v<decltype(first_handle)&, decltype(integer_handle)&&>);
    first_handle = {};
    EXPECT_EQ(*integer_handle, 3);
    EXPECT_FALSE(dlist_->IsEmpty());
    integer_handle = {};
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, RejectsInvalidValuesWithoutChangingExistingData) {
    auto value = std::make_unique<int>(7);
    EXPECT_EQ(cache_->Reserve({-1, 0}).error().get_error_code(), ErrorCode::InvalidParameter);
    const ResourceUsage estimate{32, 0};
    ASSERT_TRUE(cache_->Reserve(estimate));
    auto negative = cache_->Put("a", value, {0, -1});
    ASSERT_FALSE(negative);
    EXPECT_EQ(negative.error().get_error_code(), ErrorCode::InvalidParameter);
    EXPECT_THAT(negative.error().what(), ::testing::HasSubstr("negative resident bytes"));
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), estimate);
    cache_->UnReserve(estimate);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
    EXPECT_NE(value, nullptr);
    EXPECT_EQ(cache_->Size(), 0);
    std::unique_ptr<int> empty;
    auto missing = ReserveAndPut(*cache_, "a", empty, {});
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().get_error_code(), ErrorCode::InvalidParameter);
    EXPECT_THAT(missing.error().what(), ::testing::HasSubstr("non-null value"));
    EXPECT_EQ(cache_->Reserve({8193, 0}).error().get_error_code(), ErrorCode::InsufficientResource);
    EXPECT_NE(value, nullptr);
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
    EXPECT_EQ(cache_->GetOrCreate("a", [] { return std::pair{std::unique_ptr<int>{}, ResourceUsage{32, 0}}; })
                  .error()
                  .get_error_code(),
              ErrorCode::InvalidParameter);
    auto failed = cache_->GetOrCreate("a", []() -> CacheMap<std::string, int>::CreateResult {
        throw SegcoreError(ErrorCode::FileReadFailed, "metadata load failed");
    });
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().get_error_code(), ErrorCode::FileReadFailed);
    EXPECT_STREQ(failed.error().what(), "metadata load failed");
    EXPECT_EQ(cache_->Size(), 0);
}

TEST(CacheMapAccountingTest, IntegerKeysAndDiskLimitsShareTheSameAdmission) {
    auto dlist = std::make_shared<internal::DList>(true, ResourceUsage{8192, 100}, ResourceUsage{8192, 100},
                                                   ResourceUsage{8192, 100}, EvictionConfig{});
    CacheMap<int, int> cache(dlist);
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    ASSERT_TRUE(ReserveAndPut(cache, 10, a, {32, 70}));
    auto pin = cache.Get(10).value();
    EXPECT_EQ(ReserveAndPut(cache, 20, b, {32, 70}).error().get_error_code(), ErrorCode::InsufficientResource);
    pin = {};
    ASSERT_TRUE(ReserveAndPut(cache, 20, b, {32, 70}));
    EXPECT_FALSE(cache.Get(10).value());
    EXPECT_EQ(*cache.Get(20).value(), 2);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist).file_bytes, 70);
    ASSERT_TRUE(cache.Erase(20));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), (ResourceUsage{0, 0}));
}

TEST(CacheMapAccountingTest, PutChargesActualBytesAndUnReserveRefundsTheScaledEstimate) {
    EvictionConfig config;
    config.loading_resource_factor = 2;
    const ResourceUsage limit{100, 100};
    auto dlist = std::make_shared<internal::DList>(true, limit, limit, limit, config);
    CacheMap<int, int> cache(dlist);
    const ResourceUsage estimate{50, 50};
    for (const auto actual : {ResourceUsage{70, 30}, ResourceUsage{0, 10}, ResourceUsage{}}) {
        ASSERT_TRUE(cache.Reserve(estimate));
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), limit);
        auto value = std::make_unique<int>(1);
        ASSERT_TRUE(cache.Put(1, value, actual));
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), limit);
        cache.UnReserve(estimate);
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist), ResourceUsage{});
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), actual);
        ASSERT_TRUE(cache.Erase(1));
    }
}

TEST_F(CacheMapTest, PreservesSignedAndUnsignedIntegerKeys) {
    CacheMap<int64_t, int> signed_cache(dlist_);
    CacheMap<uint64_t, int> unsigned_cache(dlist_);
    for (const int64_t key : {std::numeric_limits<int64_t>::min(), int64_t{-1}, int64_t{0}}) {
        auto value = std::make_unique<int>(key == 0 ? 3 : key == -1 ? 2 : 1);
        ASSERT_TRUE(ReserveAndPut(signed_cache, key, value, {sizeof(int), 0}));
    }
    auto largest = std::make_unique<int>(4);
    ASSERT_TRUE(ReserveAndPut(unsigned_cache, std::numeric_limits<uint64_t>::max(), largest, {sizeof(int), 0}));
    EXPECT_EQ(*signed_cache.Get(std::numeric_limits<int64_t>::min()).value(), 1);
    EXPECT_EQ(*signed_cache.Get(-1).value(), 2);
    EXPECT_EQ(*signed_cache.Get(0).value(), 3);
    EXPECT_EQ(*unsigned_cache.Get(std::numeric_limits<uint64_t>::max()).value(), 4);
    EXPECT_FALSE(unsigned_cache.Get(0).value());
}

TEST_F(CacheMapTest, PreservesEmbeddedNullsInStringKeys) {
    const std::string key("a\0b", 3);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(*cache_, key, value, {sizeof(int), 0}));
    EXPECT_FALSE(cache_->Get("a").value());
    EXPECT_EQ(*cache_->Get(key).value(), 7);
    EXPECT_EQ(cache_->Erase("a").error().get_error_code(), ErrorCode::OutOfRange);
    EXPECT_TRUE(cache_->Erase(key));
}

TEST_F(CacheMapTest, PreservesConstValueOwnershipOnFailure) {
    CacheMap<int, const int> cache(dlist_);
    auto value = std::make_unique<const int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, 1, value, {sizeof(int), 0}));
    auto handle = cache.Get(1).value();
    ASSERT_TRUE(handle);
    auto replacement = std::make_unique<const int>(9);
    EXPECT_EQ(ReserveAndPut(cache, 1, replacement, {sizeof(int), 0}).error().get_error_code(),
              ErrorCode::InvalidParameter);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(*replacement, 9);
    EXPECT_EQ(*handle, 7);
}

TEST_F(CacheMapTest, UsesTheOriginalDefaultDeleter) {
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
    CacheMap<int, Value> cache(dlist_);
    auto value = std::make_unique<Value>(destroyed);
    ASSERT_TRUE(ReserveAndPut(cache, 1, value, {sizeof(Value), 0}));
    EXPECT_EQ(destroyed, 0);
    ASSERT_TRUE(cache.Erase(1));
    EXPECT_EQ(destroyed, 1);
}

TEST_F(CacheMapTest, ConcurrentGetsKeepTheirValueStableAcrossMutationsAndEviction) {
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 4; ++worker) {
        threads.emplace_back([&, worker] {
            for (int i = 0; i < 100; ++i) {
                const auto key = std::to_string(i % 12);
                auto value = std::make_unique<int>(worker * 100 + i);
                const auto result = ReserveAndPut(*cache_, key, value, {128, 0});
                if (!result) {
                    EXPECT_TRUE(result.error().get_error_code() == ErrorCode::InvalidParameter ||
                                result.error().get_error_code() == ErrorCode::InsufficientResource);
                    EXPECT_NE(value, nullptr);
                } else {
                    ASSERT_TRUE(result.value());
                    EXPECT_EQ(*result.value(), worker * 100 + i);
                }
                auto found = cache_->Get(key);
                ASSERT_TRUE(found);
                if (const auto& handle = found.value()) {
                    const auto snapshot = *handle;
                    std::this_thread::yield();
                    EXPECT_EQ(*handle, snapshot);
                    EXPECT_EQ(cache_->Erase(key).error().get_error_code(), ErrorCode::UnexpectedError);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    int64_t resident_keys = 0;
    for (int key = 0; key < 12; ++key) {
        if (cache_->Get(std::to_string(key)).value()) {
            ++resident_keys;
        }
    }
    // Concurrent writers must publish at most one Entry for each key.
    EXPECT_EQ(cache_->Size(), resident_keys);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{resident_keys * 128, 0}));
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
    EXPECT_LE(internal::DListTestFriend::get_used_memory(*dlist_).memory_bytes, 8192);
    cache_.reset();
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, LongKeysDoNotConsumePayloadCapacity) {
    const auto entries = monitor::cache_kv_entry_count.Value();
    const auto payload = monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Value();
    auto value = std::make_unique<int>(7);
    const auto capacity = internal::DListTestFriend::get_max_memory(*dlist_);
    ASSERT_TRUE(ReserveAndPut(*cache_, std::string(10000, 'k'), value, capacity));
    EXPECT_EQ(monitor::cache_kv_entry_count.Value(), entries + 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), capacity);
    EXPECT_EQ(monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Value(), payload + capacity.memory_bytes);
    cache_.reset();
    EXPECT_EQ(monitor::cache_kv_entry_count.Value(), entries);
    EXPECT_EQ(monitor::cache_kv_loaded_bytes(StorageType::MEMORY).Value(), payload);
}

TEST(CacheMapAccountingTest, BackgroundEvictionPreservesPinnedEntryAndRemovesOtherKeys) {
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
    CacheMap<int, Value> cache(dlist);
    auto first = std::make_unique<Value>();
    first->signal = signal;
    ASSERT_TRUE(ReserveAndPut(cache, 0, first, {1000, 0}));
    auto pin = cache.Get(0).value();
    ASSERT_TRUE(pin);
    dlist->UpdateHighWatermark({2048, 0});
    for (int key = 1; key < 3; ++key) {
        auto value = std::make_unique<Value>();
        value->signal = signal;
        ASSERT_TRUE(ReserveAndPut(cache, key, value, {1000, 0}));
    }
    {
        std::unique_lock lock(signal->mutex);
        ASSERT_TRUE(signal->changed.wait_for(lock, std::chrono::seconds(5), [&] { return signal->destroyed == 2; }));
    }
    EXPECT_TRUE(cache.Get(0).value());
    // Value destruction precedes index removal. These lookups wait for any
    // eviction holding the node locks before checking the index count.
    EXPECT_FALSE(cache.Get(1).value());
    EXPECT_FALSE(cache.Get(2).value());
    EXPECT_EQ(cache.Size(), 1);
}

TEST_F(CacheMapTest, BackgroundEvictionExpiresZeroBytePayloads) {
    std::promise<void> destroyed;
    auto completion = destroyed.get_future();
    struct Value {
        std::promise<void>* destroyed;
        ~Value() {
            destroyed->set_value();
        }
    };
    EvictionConfig config{0, true, 10};
    config.cache_cell_unaccessed_survival_time = std::chrono::seconds(1);
    const ResourceUsage capacity{8192, 0};
    auto dlist = std::make_shared<internal::DList>(true, capacity, capacity, capacity, config);
    CacheMap<int, Value> cache(dlist);
    auto value = std::make_unique<Value>();
    value->destroyed = &destroyed;
    ASSERT_TRUE(ReserveAndPut(cache, 1, value, {}));
    ASSERT_EQ(completion.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    // The destructor signal can arrive before detach_evicted(). IsEmpty takes
    // the list lock, so it also waits for the eviction batch to finish.
    EXPECT_TRUE(dlist->IsEmpty());
    EXPECT_EQ(cache.Size(), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist), (ResourceUsage{0, 0}));
}

#ifdef ENABLE_SYNCPOINT
TEST_F(CacheMapTest, PutWaitingForDListLeavesItsIndexStripeAvailable) {
    CacheMap<int, int> cache(dlist_, {.buckets_power = 0, .locks_power = 0});
    auto resident = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, 1, resident, {32, 0}));
    const ResourceUsage bytes{32, 0};
    ASSERT_TRUE(cache.Reserve(bytes));
    auto refund = folly::makeGuard([&] { cache.UnReserve(bytes); });
    folly::Baton<> publishing;
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([&] { sync->Reset(); });
    sync->SetCallBack("CacheMap::Put:BeforePublish", [&](void*) { publishing.post(); });
    sync->EnableProcessing();

    auto list_lock = internal::DListTestFriend::test_lock_list(dlist_.get());
    auto writer = std::async(std::launch::async, [&] {
        auto value = std::make_unique<int>(9);
        return cache.Put(2, value, bytes);
    });
    auto unblock = folly::makeGuard([&] { list_lock.unlock(); });
    ASSERT_TRUE(publishing.try_wait_for(std::chrono::seconds(5)));
    auto reader = std::async(std::launch::async, [&] {
        return cache.GetOrCreate(1, [] {
            ADD_FAILURE() << "The key is already resident";
            return std::pair{std::make_unique<int>(0), ResourceUsage{}};
        });
    });
    EXPECT_EQ(reader.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    list_lock.unlock();
    unblock.dismiss();
    ASSERT_TRUE(writer.get());
    auto hit = reader.get();
    ASSERT_TRUE(hit && hit.value());
    EXPECT_EQ(*hit.value(), 7);
}

TEST_F(CacheMapTest, GetOrCreateFinishesAContendedLookupWithoutRescanning) {
    struct Value {
        int number;
        folly::Baton<>* erasing = nullptr;
        folly::Baton<>* resume = nullptr;
        ~Value() {
            if (erasing) {
                erasing->post();
                EXPECT_TRUE(resume->try_wait_for(std::chrono::seconds(5)));
            }
        }
    };
    using Cache = CacheMap<int, Value>;
    for (const bool replace : {false, true}) {
        SCOPED_TRACE(replace);
        folly::Baton<> erasing, resume, waiter_ready, deliver;
        auto cache = std::make_unique<Cache>(dlist_);
        auto first = std::make_unique<Value>();
        first->number = 7;
        first->erasing = &erasing;
        first->resume = &resume;
        ASSERT_TRUE(ReserveAndPut(*cache, 1, first, {32, 0}));
        auto erased = std::async(std::launch::async, [&] { return cache->Erase(1); });
        auto unblock = folly::makeGuard([&] { resume.post(); });
        ASSERT_TRUE(erasing.try_wait_for(std::chrono::seconds(5)));

        std::atomic<int> calls{0};
        auto create = [&] {
            ++calls;
            auto value = std::make_unique<Value>();
            value->number = 9;
            return std::pair{std::move(value), ResourceUsage{64, 0}};
        };
        auto* sync = milvus::SyncPoint::GetInstance();
        auto reset = folly::makeGuard([&] { sync->Reset(); });
        std::future<Cache::PinResult> waiter;
        int waits = 0;
        auto on_lookup = [&, leader = std::this_thread::get_id()](void*) {
            if (std::this_thread::get_id() != leader) {
                waiter_ready.post();
                EXPECT_TRUE(deliver.try_wait_for(std::chrono::seconds(5)));
                return;
            }
            if (++waits == 1) {
                // Both callers see the old entry while Erase holds its node lock.
                waiter = std::async(std::launch::async, [&] { return cache->GetOrCreate(1, create); });
                EXPECT_TRUE(waiter_ready.try_wait_for(std::chrono::seconds(5)));
                resume.post();
                unblock.dismiss();
                ASSERT_TRUE(erased.get());
                if (replace) {
                    auto replacement = std::make_unique<Value>();
                    replacement->number = 42;
                    ASSERT_TRUE(ReserveAndPut(*cache, 1, replacement, {32, 0}));
                }
            }
        };
        sync->SetCallBack("CacheMap::GetOrCreate:BeforePinResident", on_lookup);
        sync->SetCallBack("CacheMap::GetOrCreate:Joined", on_lookup);
        sync->EnableProcessing();
        auto result = cache->GetOrCreate(1, create);
        EXPECT_EQ(cache->Size(), 1);
        cache.reset();
        deliver.post();
        ASSERT_TRUE(waiter.valid());
        auto second = waiter.get();
        ASSERT_TRUE(result && result.value());
        ASSERT_TRUE(second) << second.error().what();
        ASSERT_TRUE(second.value());
        EXPECT_EQ(&*second.value(), &*result.value());
        EXPECT_EQ(result.value()->number, replace ? 42 : 9);
        EXPECT_EQ(calls.load(), replace ? 0 : 1);
        EXPECT_EQ(waits, 1);
    }
}

TEST_F(CacheMapTest, GetOrCreatePublishesOnceWhenTheWinningPutIsErased) {
    enum class Race { Erase, Replace, ReplaceAndClose };
    const ResourceUsage bytes{32, 0};
    for (const auto race : {Race::Erase, Race::Replace, Race::ReplaceAndClose}) {
        SCOPED_TRACE(static_cast<int>(race));
        using Cache = CacheMap<CacheMapTestKey, int>;
        auto cache = std::make_unique<Cache>(dlist_);
        const CacheMapTestKey key{1, 1};
        auto sentinel = std::make_unique<int>(0);
        ASSERT_TRUE(ReserveAndPut(*cache, {1, 0}, sentinel, bytes));
        auto* sync = milvus::SyncPoint::GetInstance();
        auto reset = folly::makeGuard([&] { sync->Reset(); });
        int waits = 0;
        sync->SetCallBack("CacheMap::GetOrCreate:BeforePinResident", [&](void*) {
            if (++waits == 1) {
                std::async(std::launch::async, [&] {
                    ASSERT_TRUE(cache->Erase(key));
                    if (race != Race::Erase) {
                        auto replacement = std::make_unique<int>(42);
                        ASSERT_TRUE(ReserveAndPut(*cache, key, replacement, bytes));
                        // The waiting creation now holds its own Pin on the replacement.
                        EXPECT_EQ(cache->Erase(key).error().get_error_code(), ErrorCode::UnexpectedError);
                    }
                    if (race == Race::ReplaceAndClose) {
                        cache.reset();
                    }
                }).get();
            }
        });
        sync->EnableProcessing();
        int calls = 0;
        // All keys collide. The first scan sees the sentinel, the second the competing Put.
        // Writer-thread key comparisons are independent of the caller's count.
        CacheMapTestKey::comparisons = 0;
        auto result = cache->GetOrCreate(key, [&] {
            ++calls;
            std::async(std::launch::async, [&] {
                auto first = std::make_unique<int>(7);
                ASSERT_TRUE(ReserveAndPut(*cache, key, first, bytes));
            }).get();
            return std::pair{std::make_unique<int>(9), ResourceUsage{64, 0}};
        });
        EXPECT_EQ(CacheMapTestKey::comparisons, 2);
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(waits, 1);
        if (race == Race::ReplaceAndClose) {
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().get_error_code(), ErrorCode::FollyCancel);
            EXPECT_TRUE(dlist_->IsEmpty());
            EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
        } else {
            ASSERT_TRUE(result && result.value());
            EXPECT_EQ(*result.value(), race == Race::Replace ? 42 : 9);
            EXPECT_EQ(cache->Size(), 2);
            EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_),
                      (ResourceUsage{race == Race::Replace ? 64 : 96, 0}));
        }
    }
}

TEST_F(CacheMapTest, GetOrCreateSharesOneFactoryAndItsOutcome) {
    using Cache = CacheMap<CacheMapTestKey, int>;
    enum class CloseAt { Never, DuringFactory, AfterCompletion };
    auto* sync = milvus::SyncPoint::GetInstance();
    const auto run = [&](ErrorCode factory_code, CloseAt close) {
        SCOPED_TRACE(::testing::Message()
                     << "factory_code=" << static_cast<int>(factory_code) << ", close=" << static_cast<int>(close));
        auto cache = std::make_unique<Cache>(dlist_);
        auto* map = cache.get();
        folly::Baton<> started, joined, resume, deliver;
        std::atomic<int> calls{0};
        auto create = [&]() -> Cache::CreateResult {
            if (++calls == 1) {
                started.post();
                resume.wait();
            }
            if (factory_code == ErrorCode::FileReadFailed) {
                return folly::makeUnexpected(SegcoreError(factory_code, "metadata read failed in factory"));
            }
            if (factory_code == ErrorCode::UnexpectedError) {
                throw std::runtime_error("metadata read failed in factory");
            }
            return std::pair{std::make_unique<int>(7), ResourceUsage{32, 0}};
        };
        auto reset = folly::makeGuard([&] { sync->Reset(); });
        sync->SetCallBack("CacheMap::GetOrCreate:Joined", [&](void*) {
            joined.post();
            if (close == CloseAt::AfterCompletion) {
                deliver.wait();
            }
        });
        sync->EnableProcessing();
        std::future<Cache::PinResult> waiter, unrelated;
        auto leader = std::async(std::launch::async, [&] { return map->GetOrCreate({1, 1}, create); });
        auto unblock = folly::makeGuard([&] {
            resume.post();
            deliver.post();
        });
        ASSERT_TRUE(started.try_wait_for(std::chrono::seconds(5)));
        EXPECT_FALSE(map->Get({1, 1}).value());
        EXPECT_EQ(map->Size(), 0);
        waiter = std::async(std::launch::async, [&] { return map->GetOrCreate({1, 1}, create); });
        ASSERT_TRUE(joined.try_wait_for(std::chrono::seconds(2)));
        // The key has the same hash. Its factory can still finish while the first one is blocked.
        unrelated = std::async(std::launch::async, [&] {
            return map->GetOrCreate({1, 2}, [] { return std::pair{std::make_unique<int>(9), ResourceUsage{8, 0}}; });
        });
        ASSERT_EQ(unrelated.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        if (close == CloseAt::DuringFactory) {
            cache.reset();
            EXPECT_EQ(waiter.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        }
        resume.post();
        auto first = leader.get();
        if (close == CloseAt::AfterCompletion) {
            cache.reset();
        }
        deliver.post();
        unblock.dismiss();
        auto second = waiter.get();
        ASSERT_TRUE(unrelated.get());
        EXPECT_EQ(calls.load(), 1);
        const auto expected = close == CloseAt::DuringFactory ? ErrorCode::FollyCancel : factory_code;
        if (expected == ErrorCode::Success) {
            ASSERT_TRUE(first);
            ASSERT_TRUE(second);
            EXPECT_EQ(&*first.value(), &*second.value());
            first.value() = {};
            EXPECT_EQ(*second.value(), 7);
            if (cache) {
                EXPECT_EQ(map->Erase({1, 1}).error().get_error_code(), ErrorCode::UnexpectedError);
            }
        } else {
            ASSERT_FALSE(first);
            ASSERT_FALSE(second);
            EXPECT_EQ(first.error().get_error_code(), expected);
            EXPECT_EQ(second.error().get_error_code(), expected);
            EXPECT_STREQ(first.error().what(), second.error().what());
            if (cache) {
                EXPECT_STREQ(first.error().what(), "metadata read failed in factory");
                EXPECT_STREQ(second.error().what(), "metadata read failed in factory");
                EXPECT_FALSE(map->Get({1, 1}).value());
                auto retried =
                    map->GetOrCreate({1, 1}, [] { return std::pair{std::make_unique<int>(11), ResourceUsage{16, 0}}; });
                ASSERT_TRUE(retried);
                EXPECT_EQ(*retried.value(), 11);
            }
        }
    };
    for (const auto code : {ErrorCode::Success, ErrorCode::FileReadFailed, ErrorCode::UnexpectedError}) {
        run(code, CloseAt::Never);
        run(code, CloseAt::DuringFactory);
        run(code, CloseAt::AfterCompletion);
    }
}

TEST_F(CacheMapTest, ReserveReportsAllocationFailureBeforeEnqueue) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    const ResourceUsage capacity{8192, 0};
    ASSERT_TRUE(cache_->Reserve(capacity));
    sync->SetCallBack("DList::Reserve:BeforeEnqueue", [](void*) { throw std::bad_alloc{}; });
    sync->EnableProcessing();
    const auto sync_result = cache_->Reserve({1, 0}, std::chrono::milliseconds(100));
    const auto async_result = cache_->ReserveAsync({1, 0}, std::chrono::milliseconds(100)).get();
    ASSERT_FALSE(sync_result);
    ASSERT_FALSE(async_result);
    EXPECT_EQ(sync_result.error().get_error_code(), ErrorCode::MemAllocateFailed);
    EXPECT_EQ(async_result.error().get_error_code(), ErrorCode::MemAllocateFailed);
    EXPECT_STREQ(sync_result.error().what(), std::bad_alloc{}.what());
    EXPECT_STREQ(async_result.error().what(), sync_result.error().what());
    EXPECT_EQ(internal::DListTestFriend::get_waiting_request_count(*dlist_), 0);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
    cache_->UnReserve(capacity);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, WaitingReserveWakesOnRefundOrCacheClose) {
    auto* sync = milvus::SyncPoint::GetInstance();
    const ResourceUsage capacity{8192, 0};
    for (const bool close : {false, true}) {
        auto cache = std::make_unique<CacheMap<int, int>>(dlist_);
        ASSERT_TRUE(cache_->Reserve(capacity));
        folly::Baton<> waiting;
        auto reset = folly::makeGuard([sync] { sync->Reset(); });
        sync->SetCallBack("CacheMap::Reserve:BeforeWait", [&](void*) {
            waiting.post();
            if (!close) {
                throw std::bad_alloc{};
            }
        });
        sync->EnableProcessing();
        auto pending = std::async(std::launch::async, [&, map = cache.get()] {
            return map->Reserve(capacity, close ? std::chrono::milliseconds(-1) : std::chrono::seconds(5));
        });
        auto unblock = folly::makeGuard([&] { cache_->UnReserve(capacity); });
        ASSERT_TRUE(waiting.try_wait_for(std::chrono::seconds(5)));
        if (close) {
            cache.reset();
            EXPECT_EQ(pending.wait_for(std::chrono::seconds(2)), std::future_status::ready);
            EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
        } else {
            EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
        }
        cache_->UnReserve(capacity);
        unblock.dismiss();
        if (!close && pending.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            ADD_FAILURE() << "UnReserve did not wake the waiting caller";
            cache.reset();
        }
        auto result = pending.get();
        if (close) {
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().get_error_code(), ErrorCode::FollyCancel);
        } else {
            ASSERT_TRUE(result);
            EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
            cache->UnReserve(capacity);
        }
    }
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, WaitingForNodeDoesNotHoldIndexStripe) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    struct Value {
        int number;
        folly::Baton<>* deleting{nullptr};
        folly::Baton<>* resume{nullptr};
        ~Value() {
            if (deleting) {
                deleting->post();
                resume->wait();
            }
        }
    };
    folly::Baton<> deleting;
    folly::Baton<> resume;
    folly::Baton<> looked_up;
    CacheMap<CacheMapTestKey, Value> cache(dlist_);
    auto first = std::make_unique<Value>();
    first->number = 7;
    first->deleting = &deleting;
    first->resume = &resume;
    auto second = std::make_unique<Value>();
    second->number = 9;
    auto unblock_setup = folly::makeGuard([&] { resume.post(); });
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1}, first, {32, 0}));
    ASSERT_TRUE(ReserveAndPut(cache, {1, 2}, second, {32, 0}));
    auto eraser = std::async(std::launch::async, [&] { return cache.Erase({1, 1}); });
    auto unblock = folly::makeGuard([&] { resume.post(); });
    ASSERT_TRUE(deleting.try_wait_for(std::chrono::seconds(5)));
    // Erase holds the first node and DList locks while destroying its
    // payload. A lookup waiting for that node must release the shared index stripe.
    sync->SetCallBack("CacheMap::Get:AfterLookup", [&](void*) { looked_up.post(); });
    sync->EnableProcessing();
    auto waiting = std::async(std::launch::async, [&] { return cache.Get({1, 1}); });
    EXPECT_TRUE(looked_up.try_wait_for(std::chrono::seconds(5)));
    auto independent = std::async(std::launch::async, [&] { return cache.Get({1, 2}); });
    EXPECT_EQ(independent.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    resume.post();
    EXPECT_TRUE(eraser.get());
    auto waiting_pin = waiting.get().value();
    auto independent_pin = independent.get().value();
    EXPECT_FALSE(waiting_pin);
    ASSERT_TRUE(independent_pin);
    EXPECT_EQ(independent_pin->number, 9);
}

TEST_F(CacheMapTest, EvictionAfterLookupReturnsMissWithoutReloading) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", value, {32, 0}));
    bool reached = false;
    sync->SetCallBack("CacheMap::Get:AfterLookup", [&](void*) {
        reached = true;
        ASSERT_TRUE(SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(0))));
        dlist_->ReleaseLoadingResource({8192, 0});
    });
    sync->EnableProcessing();
    EXPECT_FALSE(cache_->Get("a").value());
    EXPECT_TRUE(reached);
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, LookupExceptionReleasesDetachedEntry) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    CacheMap<CacheMapTestKey, int> cache(dlist_);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1, 4096}, value, {32, 0}));
    using Failure = std::pair<std::exception_ptr, ErrorCode>;
    for (const auto& [error, code] :
         {Failure{std::make_exception_ptr(std::out_of_range("lookup")), ErrorCode::OutOfRange},
          Failure{std::make_exception_ptr(std::system_error(std::make_error_code(std::errc::not_enough_memory))),
                  ErrorCode::MemAllocateFailed},
          Failure{std::make_exception_ptr(
                      std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again))),
                  ErrorCode::InsufficientResource},
          Failure{std::make_exception_ptr(
                      std::system_error(std::make_error_code(std::errc::resource_deadlock_would_occur))),
                  ErrorCode::UnexpectedError},
          Failure{std::make_exception_ptr(std::runtime_error("lookup")), ErrorCode::UnexpectedError},
          Failure{std::make_exception_ptr(42), ErrorCode::UnexpectedError}}) {
        sync->SetCallBack("CacheMap::Get:AfterLookup", [&](void*) { std::rethrow_exception(error); });
        sync->EnableProcessing();
        EXPECT_EQ(cache.Get({1, 1}).error().get_error_code(), code);
        sync->Reset();
    }
    sync->SetCallBack("CacheMap::Get:AfterLookup", [&](void*) {
        EXPECT_TRUE(cache.Erase({1, 1}));
        throw std::bad_alloc();
    });
    sync->EnableProcessing();
    EXPECT_EQ(cache.Get({1, 1}).error().get_error_code(), ErrorCode::MemAllocateFailed);
    EXPECT_EQ(cache.Size(), 0);
    EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 0);
    EXPECT_TRUE(dlist_->IsEmpty());
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, InFlightOperationsReportClose) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    for (const std::string point :
         {"CacheMap::Get:AfterLookup", "CacheMap::Put:BeforePublish", "CacheMap::Publish:BeforeIndexInsert"}) {
        SCOPED_TRACE(point);
        auto cache = std::make_unique<CacheMap<int, int>>(dlist_);
        auto first = std::make_unique<int>(7);
        ASSERT_TRUE(ReserveAndPut(*cache, 1, first, {32, 0}));
        sync->SetCallBack(point, [&](void*) { cache.reset(); });
        sync->EnableProcessing();
        auto input = std::make_unique<int>(9);
        const auto result = point == "CacheMap::Get:AfterLookup" ? cache->Get(1) : cache->Put(2, input, {64, 0});
        sync->Reset();
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().get_error_code(), ErrorCode::FollyCancel);
        EXPECT_NE(input, nullptr);
        EXPECT_TRUE(dlist_->IsEmpty());
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
    }

    // Grant the queued request, then close its cache before Reserve observes the grant.
    const ResourceUsage capacity{8192, 0};
    ASSERT_TRUE(cache_->Reserve(capacity));
    auto closing = std::make_unique<CacheMap<int, int>>(dlist_);
    sync->SetCallBack("CacheMap::Reserve:BeforeWait", [&](void*) {
        cache_->UnReserve(capacity);
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), capacity);
        closing.reset();
    });
    sync->EnableProcessing();
    const auto admission = closing->Reserve(capacity, std::chrono::milliseconds(-1));
    ASSERT_FALSE(admission);
    EXPECT_EQ(admission.error().get_error_code(), ErrorCode::FollyCancel);
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, EraseAndReinsertAfterLookupDoesNotPinTheDetachedEntry) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", value, {32, 0}));
    sync->SetCallBack("CacheMap::Get:AfterLookup", [&](void*) {
        ASSERT_TRUE(cache_->Erase("a"));
        auto replacement = std::make_unique<int>(9);
        ASSERT_TRUE(ReserveAndPut(*cache_, "a", replacement, {64, 0}));
    });
    sync->EnableProcessing();
    EXPECT_FALSE(cache_->Get("a").value());
    sync->Reset();
    EXPECT_EQ(*cache_->Get("a").value(), 9);
    EXPECT_EQ(cache_->Size(), 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{64, 0}));
}

TEST_F(CacheMapTest, PutPinsValueBeforeReturningFromPublication) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    struct Value {
        ResourceUsage usage{32, 0};
    };
    CacheMap<int, Value> cache(dlist_);
    bool observed_publication = false;
    sync->SetCallBack("CacheMap::Put:AfterPublish", [&](void*) {
        observed_publication = true;
        const auto erased = cache.Erase(1);
        EXPECT_FALSE(erased);
        if (!erased) {
            EXPECT_EQ(erased.error().get_error_code(), ErrorCode::UnexpectedError);
        }
        // Put still holds its 32-byte loading reservation. This request can
        // succeed only by evicting the newly published 32-byte payload.
        const ResourceUsage reservation{8160, 0};
        const auto admitted =
            SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout(reservation, std::chrono::milliseconds(0)));
        EXPECT_FALSE(admitted);
        if (admitted) {
            dlist_->ReleaseLoadingResource(reservation);
        }
    });
    sync->EnableProcessing();
    auto value = std::make_unique<Value>();
    auto result = ReserveAndPut(cache, 1, value, value->usage);
    sync->Reset();
    ASSERT_TRUE(result);
    EXPECT_TRUE(observed_publication);
    EXPECT_EQ(value, nullptr);
    EXPECT_EQ(cache.Size(), 1);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{32, 0}));
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
    const auto erased = cache.Erase(1);
    EXPECT_FALSE(erased);
    if (!erased) {
        EXPECT_EQ(erased.error().get_error_code(), ErrorCode::UnexpectedError);
    }
    result.value() = {};
    EXPECT_TRUE(cache.Erase(1));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, PostPublicationExceptionPreservesPinAndNotifiesAfterUnpin) {
    auto* sync = milvus::SyncPoint::GetInstance();
    std::optional<folly::Future<bool>> pending;
    bool callback_ran = false;
    auto reset = folly::makeGuard([&] {
        sync->Reset();
        internal::DListTestFriend::test_clear_waiting_queue(dlist_.get());
    });
    sync->SetCallBack("CacheMap::Put:AfterPublish", [&](void*) {
        pending.emplace(dlist_->ReserveLoadingResourceWithTimeout({8192, 0}, std::chrono::milliseconds(-1))
                            .via(&folly::InlineExecutor::instance())
                            .thenValue([&](const bool success) {
                                if (!success) {
                                    return false;
                                }
                                callback_ran = true;
                                EXPECT_TRUE(dlist_->IsEmpty());
                                // The last pin can notify a continuation that reenters Put.
                                auto input = std::make_unique<int>(9);
                                const auto result = ReserveAndPut(*cache_, "b", input, {1, 0});
                                EXPECT_FALSE(result);
                                if (!result) {
                                    EXPECT_EQ(result.error().get_error_code(), ErrorCode::InsufficientResource);
                                }
                                EXPECT_NE(input, nullptr);
                                return success;
                            }));
        EXPECT_FALSE(pending->isReady());
        // The input is already owned by the cache. Put must return the committed Pin.
        throw std::bad_alloc();
    });
    sync->EnableProcessing();
    auto value = std::make_unique<int>(7);
    auto result = ReserveAndPut(*cache_, "a", value, {32, 0});
    sync->Reset();
    ASSERT_TRUE(result);
    EXPECT_EQ(*result.value(), 7);
    EXPECT_EQ(value, nullptr);
    ASSERT_TRUE(pending);
    EXPECT_FALSE(pending->isReady());
    result.value() = {};
    EXPECT_TRUE(std::move(*pending).get(std::chrono::seconds(2)));
    EXPECT_TRUE(callback_ran);
    EXPECT_EQ(cache_->Size(), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), ResourceUsage{});
    dlist_->ReleaseLoadingResource({8192, 0});
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), ResourceUsage{});
}

TEST_F(CacheMapTest, PublicationDoesNotRepeatKeyComparison) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    CacheMap<CacheMapNoexceptHashKey, int> cache(dlist_);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1}, value, {32, 0}));
    const auto before = internal::DListTestFriend::get_used_memory(*dlist_);
    sync->SetCallBack("CacheMap::Put:BeforePublish", [&](void*) {
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{32, 0}));
    });
    sync->EnableProcessing();
    auto input = std::make_unique<int>(9);
    CacheMapTestKey::comparisons = 0;
    ASSERT_TRUE(ReserveAndPut(cache, {1, 2, 1024}, input, {32, 0}));
    EXPECT_EQ(CacheMapTestKey::comparisons, 1);
    sync->Reset();
    EXPECT_EQ(input, nullptr);
    EXPECT_EQ(cache.Size(), 2);
    EXPECT_EQ(*cache.Get({1, 1}).value(), 7);
    EXPECT_EQ(*cache.Get({1, 2}).value(), 9);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), before + (ResourceUsage{32, 0}));
    EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
    EXPECT_TRUE(cache.Erase({1, 2}));
    EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 0);
}

TEST_F(CacheMapTest, CustomKeyPublicationFailurePreservesInputAndResidentEntries) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] {
        sync->Reset();
        CacheMapTestKey::fail_hash = false;
        CacheMapTestKey::fail_copy = false;
    });
    CacheMap<CacheMapTestKey, int> cache(dlist_);
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(cache, {1, 1, 2048}, value, {32, 0}));
    const auto before = internal::DListTestFriend::get_used_memory(*dlist_);
    for (const auto& [failure, code] : {std::pair{&CacheMapTestKey::fail_hash, ErrorCode::FileReadFailed},
                                        std::pair{&CacheMapTestKey::fail_copy, ErrorCode::MemAllocateFailed}}) {
        sync->SetCallBack("CacheMap::Put:BeforePublish", [&, failure](void*) {
            // Only the Value payload participates in admission.
            EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{32, 0}));
            *failure = true;
        });
        sync->EnableProcessing();
        auto input = std::make_unique<int>(9);
        EXPECT_EQ(ReserveAndPut(cache, {1, 2}, input, {32, 0}).error().get_error_code(), code);
        *failure = false;
        sync->Reset();
        ASSERT_NE(input, nullptr);
        EXPECT_EQ(*input, 9);
        EXPECT_EQ(cache.Size(), 1);
        EXPECT_EQ(*cache.Get({1, 1}).value(), 7);
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), before);
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
        EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 2048);
    }
    auto input = std::make_unique<int>(9);
    ASSERT_TRUE(ReserveAndPut(cache, {1, 2}, input, {32, 0}));
    ASSERT_TRUE(cache.Erase({1, 2}));
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), before);
    ASSERT_TRUE(cache.Erase({1, 1}));
    EXPECT_EQ(CacheMapTestKey::live_key_bytes.load(), 0);
    EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), (ResourceUsage{0, 0}));
}

TEST_F(CacheMapTest, ReservationAndPublicationErrorsPreserveResidentEntries) {
    auto* sync = milvus::SyncPoint::GetInstance();
    auto reset = folly::makeGuard([sync] { sync->Reset(); });
    auto value = std::make_unique<int>(7);
    ASSERT_TRUE(ReserveAndPut(*cache_, "a", value, {32, 0}));
    const auto before = internal::DListTestFriend::get_used_memory(*dlist_);
    struct Failure {
        const char* point;
        std::exception_ptr error;
        ErrorCode code;
    };
    for (const auto& [point, error, code] : {
             Failure{"DList::Reserve:BeforeAttempt", std::make_exception_ptr(std::bad_alloc{}),
                     ErrorCode::MemAllocateFailed},
             Failure{"DList::Reserve:BeforeAttempt", std::make_exception_ptr(folly::FutureCancellation{}),
                     ErrorCode::FollyCancel},
             Failure{"DList::Reserve:BeforeAttempt", std::make_exception_ptr(folly::OperationCancelled{}),
                     ErrorCode::FollyCancel},
             Failure{"DList::Reserve:BeforeAttempt", std::make_exception_ptr(folly::FutureNotReady{}),
                     ErrorCode::FollyOtherException},
             Failure{"CacheMap::Put:BeforePublish", std::make_exception_ptr(std::bad_alloc{}),
                     ErrorCode::MemAllocateFailed},
             Failure{"CacheMap::Put:BeforePublish", std::make_exception_ptr(std::length_error("key too long")),
                     ErrorCode::OutOfRange},
             Failure{"CacheMap::Publish:BeforeIndexInsert", std::make_exception_ptr(std::bad_alloc{}),
                     ErrorCode::MemAllocateFailed},
         }) {
        SCOPED_TRACE(point);
        sync->SetCallBack(point, [&](void*) { std::rethrow_exception(error); });
        sync->EnableProcessing();
        auto input = std::make_unique<int>(9);
        EXPECT_EQ(ReserveAndPut(*cache_, "b", input, {64, 0}).error().get_error_code(), code);
        sync->Reset();
        EXPECT_NE(input, nullptr);
        EXPECT_FALSE(cache_->Get("b").value());
        EXPECT_EQ(*cache_->Get("a").value(), 7);
        EXPECT_EQ(internal::DListTestFriend::get_used_memory(*dlist_), before);
        EXPECT_EQ(internal::DListTestFriend::get_loading_memory(*dlist_), (ResourceUsage{0, 0}));
    }
    EXPECT_TRUE(cache_->Erase("a"));
}

#endif

}  // namespace milvus::cachinglayer
