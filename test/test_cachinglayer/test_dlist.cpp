#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"
#include "cachinglayer_test_utils.h"
#include "common/EasyAssert.h"
#include "mock_list_node.h"

using milvus::cachinglayer::cid_t;
using milvus::cachinglayer::EvictionConfig;
using milvus::cachinglayer::ResourceUsage;
using milvus::cachinglayer::internal::DList;
using milvus::cachinglayer::internal::DListTestFriend;
using milvus::cachinglayer::internal::ListNode;
using milvus::cachinglayer::internal::MockListNode;
using ::testing::StrictMock;
using DLF = DListTestFriend;

class DListTest : public ::testing::Test {
 protected:
    ResourceUsage initial_limit{100, 50};
    // Set watermarks relative to the limit
    ResourceUsage low_watermark{80, 40};   // 80%
    ResourceUsage high_watermark{90, 45};  // 90%
    // Use a very long interval to disable background eviction for most tests
    EvictionConfig eviction_config_{10,    // cache_touch_window (10 ms)
                                    true,  // enable background eviction
                                    100};  // eviction_interval (100 ms)

    std::unique_ptr<DList> dlist;
    // Keep track of nodes to prevent them from being deleted prematurely
    std::vector<std::shared_ptr<MockListNode>> managed_nodes;
    // Keep track of nodes that are loading
    std::map<std::shared_ptr<MockListNode>, ResourceUsage> loading_nodes;

    DListTest() = default;

    void
    SetUp() override {
        dlist = std::make_unique<DList>(true, initial_limit, low_watermark, high_watermark, eviction_config_);
        managed_nodes.clear();
        loading_nodes.clear();
    }

    void
    TearDown() override {
        managed_nodes.clear();
        loading_nodes.clear();
        dlist.reset();
    }

    // Helper method to reserve memory with a large timeout for testing
    // This simulates the old synchronous reserveMemory behavior
    bool
    reserveLoadingMemorySync(const ResourceUsage& size,
                             std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        // Use 1 hour timeout for tests
        auto future = dlist->ReserveLoadingResourceWithTimeout(size, timeout);
        return std::move(future).get();
    }

    void
    releaseLoadingMemory(const ResourceUsage& size) {
        dlist->ReleaseLoadingResource(size);
    }

    // Helper to create a mock node, simulate loading it, and add it to the list.
    // Returns a raw pointer, but ownership is managed by the shared_ptr in managed_nodes.
    MockListNode*
    add_and_load_node(ResourceUsage size, const std::string& key = "key", cid_t cid = 0, int pin_count = 0) {
        // reserve 3x the size for loading
        ResourceUsage loading_size = 3 * size;
        // Check if adding this node would exceed capacity before creating/adding it.
        // We want to use add_and_load_node to create a DList in valid state.
        ResourceUsage current_usage = get_using_memory();
        ResourceUsage limit = DLF::get_max_memory(*dlist);
        if (!limit.CanHold(current_usage + loading_size)) {
            throw std::invalid_argument("Adding this node would exceed capacity");
        }

        auto reserve_result = reserveLoadingMemorySync(loading_size);
        if (!reserve_result) {
            throw std::invalid_argument("Failed to reserve loading memory");
        }
        auto node_ptr = std::make_shared<StrictMock<MockListNode>>(dlist.get(), key, cid);
        managed_nodes.push_back(node_ptr);
        MockListNode* node = node_ptr.get();
        releaseLoadingMemory(loading_size);

        node->test_set_state(ListNode::State::LOADED);
        node->test_set_loaded_size(size);
        node->test_set_pin_count(pin_count);

        // Manually adjust used memory and list pointers
        DLF::test_add_used_memory(dlist.get(), size);
        DLF::test_push_head(dlist.get(), node);
        node->test_set_state(ListNode::State::CACHED);

        if (pin_count == 0) {
            DLF::test_add_evictable_memory(dlist.get(), size);
        }

        return node;
    }

    [[nodiscard]] ResourceUsage
    get_using_memory() const {
        return DLF::get_using_memory(*dlist);
    }

    [[nodiscard]] ResourceUsage
    get_used_memory() const {
        return DLF::get_used_memory(*dlist);
    }

    [[nodiscard]] ResourceUsage
    get_loading_memory() const {
        return DLF::get_loading_memory(*dlist);
    }
};

TEST_F(DListTest, Initialization) {
    EXPECT_TRUE(dlist->IsEmpty());
    EXPECT_EQ(get_used_memory(), ResourceUsage{});
    EXPECT_EQ(get_loading_memory(), ResourceUsage{});
    EXPECT_EQ(DLF::get_head(*dlist), nullptr);
    EXPECT_EQ(DLF::get_tail(*dlist), nullptr);
}

TEST_F(DListTest, UpdateMaxLimitIncrease) {
    MockListNode* node1 = add_and_load_node({10, 5});
    EXPECT_EQ(get_used_memory(), node1->loaded_size());

    ResourceUsage new_limit{200, 100};
    EXPECT_TRUE(dlist->UpdateMaxLimit(new_limit));

    EXPECT_EQ(get_used_memory(), node1->loaded_size());
    DLF::verify_list(dlist.get(), {node1});
    EXPECT_EQ(get_loading_memory(), ResourceUsage{});
}

TEST_F(DListTest, UpdateMaxLimitDecreaseNoEviction) {
    MockListNode* node1 = add_and_load_node({10, 5});
    ResourceUsage current_usage = node1->loaded_size();
    ASSERT_EQ(get_used_memory(), current_usage);

    ResourceUsage new_limit{50, 25};
    dlist->UpdateLowWatermark({40, 20});
    dlist->UpdateHighWatermark({50, 25});
    EXPECT_TRUE(dlist->UpdateMaxLimit(new_limit));

    EXPECT_EQ(get_used_memory(), current_usage);
    DLF::verify_list(dlist.get(), {node1});
}

TEST_F(DListTest, UpdateMaxLimitDecreaseWithEvictionLRU) {
    MockListNode* node1 = add_and_load_node({30, 10}, "key1");
    MockListNode* node2 = add_and_load_node({20, 10}, "key2");
    ResourceUsage usage_node1 = node1->loaded_size();
    ResourceUsage usage_node2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    EXPECT_EQ(get_used_memory(), usage_node1 + usage_node2);

    // Expect node1 to be evicted
    EXPECT_CALL(*node1, unload()).Times(1);
    EXPECT_CALL(*node2, unload()).Times(0);

    ResourceUsage new_limit{20, 10};
    dlist->UpdateLowWatermark({16, 8});
    dlist->UpdateHighWatermark({20, 10});
    EXPECT_TRUE(dlist->UpdateMaxLimit(new_limit));

    EXPECT_EQ(get_used_memory(), usage_node2);
    DLF::verify_list(dlist.get(), {node2});
    EXPECT_FALSE(dlist->IsEmpty());
}

TEST_F(DListTest, UpdateMaxLimitDecreaseWithEvictionMultiple) {
    MockListNode* node1 = add_and_load_node({20, 10}, "key1");
    MockListNode* node2 = add_and_load_node({20, 10}, "key2");
    MockListNode* node3 = add_and_load_node({20, 10}, "key3");
    ResourceUsage usage_node1 = node1->loaded_size();
    ResourceUsage usage_node2 = node2->loaded_size();
    ResourceUsage usage_node3 = node3->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2, node3});
    ASSERT_EQ(get_used_memory(), usage_node1 + usage_node2 + usage_node3);

    EXPECT_CALL(*node1, unload()).Times(1);
    EXPECT_CALL(*node2, unload()).Times(1);
    EXPECT_CALL(*node3, unload()).Times(0);

    ResourceUsage new_limit{20, 10};
    dlist->UpdateLowWatermark({16, 8});
    dlist->UpdateHighWatermark({20, 10});
    EXPECT_TRUE(dlist->UpdateMaxLimit(new_limit));

    EXPECT_EQ(get_used_memory(), usage_node3);
    DLF::verify_list(dlist.get(), {node3});
}

TEST_F(DListTest, UpdateMaxLimitSkipsPinned) {
    MockListNode* node1 = add_and_load_node({33, 15}, "key1", 0, 1);
    MockListNode* node2 = add_and_load_node({20, 10}, "key2");
    ResourceUsage usage_node1 = node1->loaded_size();
    ResourceUsage usage_node2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(get_used_memory(), usage_node1 + usage_node2);

    EXPECT_CALL(*node1, unload()).Times(0);
    EXPECT_CALL(*node2, unload()).Times(1);

    ResourceUsage new_limit{33, 15};
    dlist->UpdateLowWatermark({26, 12});
    dlist->UpdateHighWatermark({33, 15});
    EXPECT_TRUE(dlist->UpdateMaxLimit(new_limit));

    EXPECT_EQ(get_used_memory(), usage_node1);
    DLF::verify_list(dlist.get(), {node1});
}

TEST_F(DListTest, UpdateMaxLimitToZero) {
    MockListNode* node1 = add_and_load_node({10, 0});
    MockListNode* node2 = add_and_load_node({0, 5});
    EXPECT_CALL(*node1, unload()).Times(1);
    EXPECT_CALL(*node2, unload()).Times(1);

    dlist->UpdateLowWatermark({0, 0});
    dlist->UpdateHighWatermark({1, 1});
    EXPECT_TRUE(dlist->UpdateMaxLimit({1, 1}));

    EXPECT_EQ(get_used_memory(), ResourceUsage{});
    EXPECT_TRUE(dlist->IsEmpty());
}

TEST_F(DListTest, UpdateMaxLimitInvalid) {
    EXPECT_THROW(dlist->UpdateMaxLimit({-10, 0}), milvus::SegcoreError);
    EXPECT_THROW(dlist->UpdateMaxLimit({0, -5}), milvus::SegcoreError);
}

TEST_F(DListTest, ReserveResourceSufficient) {
    ResourceUsage size{20, 10};
    EXPECT_TRUE(reserveLoadingMemorySync(size));
    EXPECT_EQ(get_loading_memory(), size);
}

TEST_F(DListTest, ReserveResourceRequiresEviction) {
    MockListNode* node1 = add_and_load_node({30, 15}, "key1");
    MockListNode* node2 = add_and_load_node({20, 10}, "key2");
    ResourceUsage usage_node1 = node1->loaded_size();
    ResourceUsage usage_node2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});

    ASSERT_EQ(get_used_memory(), usage_node1 + usage_node2);

    EXPECT_CALL(*node1, unload()).Times(1);
    EXPECT_CALL(*node2, unload()).Times(0);

    // Limit: 100, 50, current usage: 50, 25, reserve: 60, 30
    // Potential total: 110, 55. Need to free to low watermark 80, 40.
    // Evicting node1 ({30, 15}) is sufficient.
    // After eviction, usage: 20, 10, loading: 60, 30.
    ResourceUsage reserve_size{60, 30};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    // Reserved resources accounted in loading, used remains from survivors
    EXPECT_EQ(get_used_memory(), usage_node2);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node2});
}

TEST_F(DListTest, ReserveResourceEvictPinnedSkipped) {
    MockListNode* node_pinned = add_and_load_node({33, 15}, "key_pinned", 0, 1);
    MockListNode* node_evict = add_and_load_node({20, 10}, "key_evict");
    ResourceUsage usage_pinned = node_pinned->loaded_size();
    ResourceUsage usage_evict = node_evict->loaded_size();
    DLF::verify_list(dlist.get(), {node_pinned, node_evict});

    ASSERT_EQ(get_used_memory(), usage_pinned + usage_evict);

    EXPECT_CALL(*node_pinned, unload()).Times(0);
    EXPECT_CALL(*node_evict, unload()).Times(1);

    ResourceUsage reserve_size{60, 35};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage_pinned);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node_pinned});
}

TEST_F(DListTest, ReserveResourceEvictLockedSkipped) {
    MockListNode* node_locked = add_and_load_node({33, 15}, "key_locked");
    MockListNode* node_evict = add_and_load_node({20, 10}, "key_evict");
    ResourceUsage usage_locked = node_locked->loaded_size();
    ResourceUsage usage_evict = node_evict->loaded_size();
    DLF::verify_list(dlist.get(), {node_locked, node_evict});

    ASSERT_EQ(get_used_memory(), usage_locked + usage_evict);

    EXPECT_CALL(*node_locked, unload()).Times(0);
    EXPECT_CALL(*node_evict, unload()).Times(1);

    // Simulate locking the node during eviction attempt
    std::unique_lock locked_node_lock(node_locked->test_get_mutex());

    ResourceUsage reserve_size{60, 35};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    locked_node_lock.unlock();

    EXPECT_EQ(get_used_memory(), usage_locked);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node_locked});
}

TEST_F(DListTest, ReserveResourceInsufficientEvenWithEviction) {
    MockListNode* node1 = add_and_load_node({10, 5});
    ResourceUsage usage_node1 = node1->loaded_size();
    ASSERT_EQ(get_used_memory(), usage_node1);

    ResourceUsage reserve_size{200, 100};

    EXPECT_FALSE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage_node1);
    EXPECT_FALSE(dlist->IsEmpty());
}

TEST_F(DListTest, TouchItemMovesToHead) {
    MockListNode* node1 = add_and_load_node({10, 0}, "key1");
    MockListNode* node2 = add_and_load_node({10, 0}, "key2");
    MockListNode* node3 = add_and_load_node({10, 0}, "key3");

    DLF::verify_list(dlist.get(), {node1, node2, node3});

    {
        std::unique_lock node_lock(node1->test_get_mutex());
        dlist->touchItem(node1, true);
    }

    DLF::verify_list(dlist.get(), {node2, node3, node1});
}

TEST_F(DListTest, TouchItemRefreshWindow) {
    MockListNode* node1 = add_and_load_node({10, 0}, "key1");
    MockListNode* node2 = add_and_load_node({10, 0}, "key2");

    DLF::verify_list(dlist.get(), {node1, node2});

    {
        std::unique_lock node_lock(node1->test_get_mutex());
        dlist->touchItem(node1, true);
    }
    DLF::verify_list(dlist.get(), {node2, node1});

    {
        std::unique_lock node_lock(node1->test_get_mutex());
        dlist->touchItem(node1, true);
    }
    DLF::verify_list(dlist.get(), {node2, node1});

    // Use eviction_config from dlist
    std::this_thread::sleep_for(dlist->eviction_config().cache_touch_window + std::chrono::milliseconds(10));

    {
        std::unique_lock node_lock(node1->test_get_mutex());
        dlist->touchItem(node1, true);
    }
    DLF::verify_list(dlist.get(), {node2, node1});

    // Use eviction_config from dlist
    std::this_thread::sleep_for(dlist->eviction_config().cache_touch_window + std::chrono::milliseconds(10));

    {
        std::unique_lock node_lock(node2->test_get_mutex());
        dlist->touchItem(node2, true);
    }
    DLF::verify_list(dlist.get(), {node1, node2});
}

TEST_F(DListTest, ReleaseLoadingResource) {
    ResourceUsage initial_size{30, 15};
    DLF::test_add_used_memory(dlist.get(), initial_size);
    ASSERT_EQ(get_used_memory(), initial_size);

    ResourceUsage failed_load_size{10, 5};
    // Reserve and then release loading resource
    EXPECT_TRUE(reserveLoadingMemorySync(failed_load_size));
    EXPECT_EQ(get_loading_memory(), failed_load_size);
    dlist->ReleaseLoadingResource(failed_load_size);

    EXPECT_EQ(get_used_memory(), initial_size);
    EXPECT_EQ(get_loading_memory(), ResourceUsage{});
}

TEST_F(DListTest, ReserveResourceEvictOnlyMemoryNeeded) {
    MockListNode* node_disk_only = add_and_load_node({0, 10}, "disk_only");
    MockListNode* node_mixed = add_and_load_node({25, 5}, "mixed");
    ResourceUsage usage_disk = node_disk_only->loaded_size();
    ResourceUsage usage_mixed = node_mixed->loaded_size();
    DLF::verify_list(dlist.get(), {node_disk_only, node_mixed});
    ASSERT_EQ(get_used_memory(), usage_disk + usage_mixed);

    EXPECT_CALL(*node_disk_only, unload()).Times(0);
    EXPECT_CALL(*node_mixed, unload()).Times(1);

    // node_disk_only is at tail, but it contains no memory, and disk usage is below low watermark,
    // thus evicting it does not help. We need to evict node_mixed to free up memory.
    ResourceUsage reserve_size{80, 0};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage_disk);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node_disk_only});
}

TEST_F(DListTest, ReserveResourceEvictOnlyDiskNeeded) {
    MockListNode* node_memory_only = add_and_load_node({30, 0}, "memory_only");
    MockListNode* node_mixed = add_and_load_node({20, 15}, "mixed");
    ResourceUsage usage_memory = node_memory_only->loaded_size();
    ResourceUsage usage_mixed = node_mixed->loaded_size();
    DLF::verify_list(dlist.get(), {node_memory_only, node_mixed});
    ASSERT_EQ(get_used_memory(), usage_memory + usage_mixed);

    EXPECT_CALL(*node_memory_only, unload()).Times(0);
    EXPECT_CALL(*node_mixed, unload()).Times(1);

    // node_memory_only is at tail, but it contains no disk, and memory usage is below low watermark,
    // thus evicting it does not help. We need to evict node_mixed to free up disk.
    ResourceUsage reserve_size{0, 40};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage_memory);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node_memory_only});
}

TEST_F(DListTest, ReserveResourceEvictBothNeeded) {
    MockListNode* node1 = add_and_load_node({20, 5}, "node1");
    MockListNode* node2 = add_and_load_node({10, 10}, "node2");
    MockListNode* node3 = add_and_load_node({20, 5}, "node3");
    ResourceUsage usage1 = node1->loaded_size();
    ResourceUsage usage2 = node2->loaded_size();
    ResourceUsage usage3 = node3->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2, node3});
    ASSERT_EQ(get_used_memory(), usage1 + usage2 + usage3);

    EXPECT_CALL(*node1, unload()).Times(1);
    EXPECT_CALL(*node2, unload()).Times(1);
    EXPECT_CALL(*node3, unload()).Times(0);

    ResourceUsage reserve_size{60, 20};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage3);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node3});
}

TEST_F(DListTest, ReserveToAboveLowWatermarkNoEviction) {
    // initial 40, 20
    MockListNode* node1 = add_and_load_node({30, 5}, "node1");
    MockListNode* node2 = add_and_load_node({10, 15}, "node2");
    ResourceUsage usage1 = node1->loaded_size();
    ResourceUsage usage2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(get_used_memory(), usage1 + usage2);

    // after reserve, stay below max limit; no eviction
    ResourceUsage reserve_size{20, 10};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage1 + usage2);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node1, node2});
}

TEST_F(DListTest, ReserveToAboveHighWatermarkNoEvictionThenAutoEviction) {
    // initial 40, 20
    MockListNode* node1 = add_and_load_node({30, 15}, "node1");
    MockListNode* node2 = add_and_load_node({10, 5}, "node2");
    ResourceUsage usage1 = node1->loaded_size();
    ResourceUsage usage2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(get_used_memory(), usage1 + usage2);

    // after reserve, 55, 26, end up in 95, 46, above high watermark, no eviction
    ResourceUsage reserve_size{55, 26};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage1 + usage2);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node1, node2});

    // wait for background eviction to run, current usage 95, 46, above high watermark.
    // reserved 55, 26 is considered pinned, thus evict node 1, resulting in 65, 31, below low watermark
    EXPECT_CALL(*node1, unload()).Times(1);
    std::this_thread::sleep_for(dlist->eviction_config().eviction_interval + std::chrono::milliseconds(10));

    EXPECT_EQ(get_used_memory(), usage2);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node2});
}

TEST_F(DListTest, ReserveResourceFailsAllPinned) {
    MockListNode* node1 = add_and_load_node({30, 15}, "key1", 0, 1);
    MockListNode* node2 = add_and_load_node({20, 10}, "key2", 0, 1);
    ResourceUsage usage_node1 = node1->loaded_size();
    ResourceUsage usage_node2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(get_used_memory(), usage_node1 + usage_node2);

    EXPECT_CALL(*node1, unload()).Times(0);
    EXPECT_CALL(*node2, unload()).Times(0);

    ResourceUsage reserve_size{55, 30};
    EXPECT_FALSE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage_node1 + usage_node2);
    DLF::verify_list(dlist.get(), {node1, node2});
}

TEST_F(DListTest, ReserveResourceFailsAllLocked) {
    MockListNode* node1 = add_and_load_node({30, 15}, "key1");
    MockListNode* node2 = add_and_load_node({20, 10}, "key2");
    ResourceUsage usage_node1 = node1->loaded_size();
    ResourceUsage usage_node2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(get_used_memory(), usage_node1 + usage_node2);

    std::unique_lock lock1(node1->test_get_mutex());
    std::unique_lock lock2(node2->test_get_mutex());

    EXPECT_CALL(*node1, unload()).Times(0);
    EXPECT_CALL(*node2, unload()).Times(0);

    ResourceUsage reserve_size{55, 30};
    EXPECT_FALSE(reserveLoadingMemorySync(reserve_size));

    lock1.unlock();
    lock2.unlock();

    EXPECT_EQ(get_used_memory(), usage_node1 + usage_node2);
    DLF::verify_list(dlist.get(), {node1, node2});
}

TEST_F(DListTest, ReserveResourceFailsSpecificPinned) {
    MockListNode* node_evict = add_and_load_node({30, 15}, "evict_candidate", 0, 1);
    MockListNode* node_small = add_and_load_node({20, 10}, "small");
    ResourceUsage usage_evict = node_evict->loaded_size();
    ResourceUsage usage_small = node_small->loaded_size();
    DLF::verify_list(dlist.get(), {node_evict, node_small});
    ASSERT_EQ(get_used_memory(), usage_evict + usage_small);

    EXPECT_CALL(*node_evict, unload()).Times(0);
    EXPECT_CALL(*node_small, unload()).Times(0);

    ResourceUsage reserve_size{75, 10};
    EXPECT_FALSE(reserveLoadingMemorySync(reserve_size));

    EXPECT_EQ(get_used_memory(), usage_evict + usage_small);
    DLF::verify_list(dlist.get(), {node_evict, node_small});
}

TEST_F(DListTest, ReserveResourceFailsSpecificLocked) {
    MockListNode* node_evict = add_and_load_node({30, 15}, "evict_candidate");
    MockListNode* node_small = add_and_load_node({20, 10}, "small");
    ResourceUsage usage_evict = node_evict->loaded_size();
    ResourceUsage usage_small = node_small->loaded_size();
    DLF::verify_list(dlist.get(), {node_evict, node_small});
    ASSERT_EQ(get_used_memory(), usage_evict + usage_small);

    std::unique_lock lock_evict(node_evict->test_get_mutex());

    EXPECT_CALL(*node_evict, unload()).Times(0);
    EXPECT_CALL(*node_small, unload()).Times(0);

    ResourceUsage reserve_size{75, 10};
    EXPECT_FALSE(reserveLoadingMemorySync(reserve_size));

    lock_evict.unlock();

    EXPECT_EQ(get_used_memory(), usage_evict + usage_small);
    DLF::verify_list(dlist.get(), {node_evict, node_small});
}

TEST_F(DListTest, TouchItemHeadOutsideWindow) {
    MockListNode* node1 = add_and_load_node({10, 0}, "key1");
    MockListNode* node2 = add_and_load_node({10, 0}, "key2");
    DLF::verify_list(dlist.get(), {node1, node2});

    // Use eviction_config from dlist
    std::this_thread::sleep_for(dlist->eviction_config().cache_touch_window + std::chrono::milliseconds(10));

    {
        std::unique_lock node_lock(node2->test_get_mutex());
        dlist->touchItem(node2, true);
    }

    DLF::verify_list(dlist.get(), {node1, node2});
}

TEST_F(DListTest, RemoveItemFromList) {
    MockListNode* node1 = add_and_load_node({10, 0}, "key1");
    MockListNode* node2 = add_and_load_node({10, 0}, "key2");
    DLF::verify_list(dlist.get(), {node1, node2});

    {
        std::unique_lock node_lock(node1->test_get_mutex());
        // dlist->removeItem() only removes the node from the list, does not update used_resources_
        dlist->removeItem(node1, node1->loaded_size());
    }

    DLF::verify_list(dlist.get(), {node2});
    EXPECT_EQ(get_used_memory(), node1->loaded_size() + node2->loaded_size());
}

TEST_F(DListTest, PopItemNotPresent) {
    MockListNode* node1 = add_and_load_node({10, 0}, "key1");
    MockListNode* node2 = add_and_load_node({10, 0}, "key2");
    ResourceUsage initial_usage = get_used_memory();
    DLF::verify_list(dlist.get(), {node1, node2});

    auto orphan_node_ptr = std::make_unique<StrictMock<MockListNode>>(dlist.get(), "orphan", 0);
    MockListNode* orphan_node = orphan_node_ptr.get();

    {
        std::unique_lock node_lock(orphan_node->test_get_mutex());
        EXPECT_NO_THROW(DLF::test_pop_item(dlist.get(), orphan_node));
    }

    DLF::verify_list(dlist.get(), {node1, node2});
    EXPECT_EQ(get_used_memory(), initial_usage);
}

TEST_F(DListTest, UpdateMaxLimitIncreaseMemDecreaseDisk) {
    MockListNode* node1 = add_and_load_node({30, 15}, "node1");
    MockListNode* node2 = add_and_load_node({20, 10}, "node2");
    ResourceUsage usage1 = node1->loaded_size();
    ResourceUsage usage2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(get_used_memory(), usage1 + usage2);

    EXPECT_CALL(*node1, unload()).Times(1);
    EXPECT_CALL(*node2, unload()).Times(0);

    ResourceUsage new_limit{200, 24};
    dlist->UpdateLowWatermark({90, 20});
    dlist->UpdateHighWatermark({90, 20});
    EXPECT_TRUE(dlist->UpdateMaxLimit(new_limit));

    EXPECT_EQ(get_used_memory(), usage2);
    DLF::verify_list(dlist.get(), {node2});
    EXPECT_EQ(DLF::get_max_memory(*dlist), new_limit);
}

TEST_F(DListTest, EvictedNodeDestroyed) {
    MockListNode* node1 = add_and_load_node({30, 15}, "node1");
    MockListNode* node2 = add_and_load_node({20, 10}, "node2");
    ResourceUsage usage1 = node1->loaded_size();
    ResourceUsage usage2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(managed_nodes.size(), 2);
    ASSERT_EQ(get_used_memory(), usage1 + usage2);

    EXPECT_CALL(*node1, unload()).Times(1);
    EXPECT_CALL(*node2, unload()).Times(0);
    ResourceUsage new_limit{70, 24};
    dlist->UpdateLowWatermark({56, 20});
    dlist->UpdateHighWatermark({70, 20});
    EXPECT_TRUE(dlist->UpdateMaxLimit(new_limit));
    DLF::verify_list(dlist.get(), {node2});
    ResourceUsage memory_after_eviction = get_used_memory();
    ASSERT_EQ(memory_after_eviction, usage2);

    // destroy node1 by removing its shared_ptr
    // node1's destructor should not decrement used_resources_ again
    auto it =
        std::find_if(managed_nodes.begin(), managed_nodes.end(), [&](const auto& ptr) { return ptr.get() == node1; });
    ASSERT_NE(it, managed_nodes.end());
    managed_nodes.erase(it);

    EXPECT_EQ(get_used_memory(), memory_after_eviction);
    DLF::verify_list(dlist.get(), {node2});
}

TEST_F(DListTest, NodeInListDestroyed) {
    MockListNode* node1 = add_and_load_node({30, 15}, "node1");
    MockListNode* node2 = add_and_load_node({20, 10}, "node2");
    ResourceUsage usage1 = node1->loaded_size();
    ResourceUsage usage2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(managed_nodes.size(), 2);
    ResourceUsage memory_before_destroy = get_used_memory();
    ASSERT_EQ(memory_before_destroy, usage1 + usage2);

    // destroy node1 by removing its shared_ptr
    // node1's destructor should decrement used_resources_ by node1->size() and remove node1 from the list
    auto it =
        std::find_if(managed_nodes.begin(), managed_nodes.end(), [&](const auto& ptr) { return ptr.get() == node1; });
    ASSERT_NE(it, managed_nodes.end());
    managed_nodes.erase(it);

    EXPECT_EQ(get_used_memory(), memory_before_destroy - usage1);
    DLF::verify_list(dlist.get(), {node2});
}

// New tests for watermark updates
TEST_F(DListTest, UpdateWatermarksValid) {
    ResourceUsage new_low{70, 30};
    ResourceUsage new_high{85, 40};

    // Check initial watermarks (optional, could use friend class if needed)
    // EXPECT_EQ(DLF::get_low_watermark(*dlist), low_watermark);
    // EXPECT_EQ(DLF::get_high_watermark(*dlist), high_watermark);

    EXPECT_NO_THROW(dlist->UpdateLowWatermark(new_low));
    EXPECT_NO_THROW(dlist->UpdateHighWatermark(new_high));

    // Verify new watermarks (requires friend class accessors)
    // EXPECT_EQ(DLF::get_low_watermark(*dlist), new_low);
    // EXPECT_EQ(DLF::get_high_watermark(*dlist), new_high);

    // Verify no change in list state or usage
    EXPECT_TRUE(dlist->IsEmpty());
    EXPECT_EQ(get_used_memory(), ResourceUsage{});
}

TEST_F(DListTest, UpdateWatermarksInvalid) {
    EXPECT_THROW(dlist->UpdateLowWatermark({-10, 0}), milvus::SegcoreError);
    EXPECT_THROW(dlist->UpdateLowWatermark({0, -5}), milvus::SegcoreError);
    EXPECT_THROW(dlist->UpdateHighWatermark({-10, 0}), milvus::SegcoreError);
    EXPECT_THROW(dlist->UpdateHighWatermark({0, -5}), milvus::SegcoreError);
}

TEST_F(DListTest, ReserveResourceUsesLowWatermark) {
    // Set up: Limit 100/100, Low 80/80, High 90/90
    initial_limit = {100, 100};
    low_watermark = {80, 80};
    high_watermark = {90, 90};
    EXPECT_TRUE(dlist->UpdateMaxLimit(initial_limit));
    dlist->UpdateHighWatermark(high_watermark);
    dlist->UpdateLowWatermark(low_watermark);

    // Add nodes totaling 95/95 usage (above high watermark)
    MockListNode* node1 = add_and_load_node({30, 30}, "node1");  // Tail
    MockListNode* node2 = add_and_load_node({20, 20}, "node2");  // Head
    ResourceUsage usage1 = node1->loaded_size();
    ResourceUsage usage2 = node2->loaded_size();
    DLF::verify_list(dlist.get(), {node1, node2});
    ASSERT_EQ(get_used_memory(), usage1 + usage2);  // 95, 95

    EXPECT_CALL(*node1, unload()).Times(1);  // Evict node1 to get below low watermark
    EXPECT_CALL(*node2, unload()).Times(0);

    // Reserve 55/55. Current usage 50/50. New potential usage 105/105.
    // Max limit 100/100. Min eviction needed: 5/5.
    // Expected eviction (target low watermark): 50/50 + 55/55 - 80/80 = 25/25.
    // Evicting node1 (30/30) satisfies both min and expected.
    ResourceUsage reserve_size{55, 55};
    EXPECT_TRUE(reserveLoadingMemorySync(reserve_size));

    // Expected usage: usage2 + reserve_size = (50,50) + (10,10) = (60,60)
    EXPECT_EQ(get_used_memory(), usage2);
    EXPECT_EQ(get_loading_memory(), reserve_size);
    DLF::verify_list(dlist.get(), {node2});
}
