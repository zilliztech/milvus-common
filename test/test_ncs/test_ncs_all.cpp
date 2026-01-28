/**
 * @file test_ncs_all.cpp
 * @brief Unified parameterized tests for all NCS (Near Compute Storage) implementations.
 * 
 * All tests run on all NCS types (InMemory, Redis) using Google Test's
 * parameterized test framework. Only initialization differs per NCS type.
 */

#include <gtest/gtest.h>
#include "ncs/InMemNcsConnector.h"
#include "ncs/InMemoryNcs.h"
#ifdef USE_REDIS
#include "ncs/RedisNcs.h"
#include "ncs/RedisNcsConnector.h"
#endif
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <random>

namespace milvus {

// Anonymous namespace for internal helpers
namespace {

// ============================================================================
// NCS Type Enumeration and Configuration
// ============================================================================

enum class NcsType {
    InMemory,
#ifdef USE_REDIS
    Redis
#endif
};

std::string NcsTypeToString(NcsType type) {
    switch (type) {
        case NcsType::InMemory: return "InMemory";
#ifdef USE_REDIS
        case NcsType::Redis: return "Redis";
#endif
    }
    return "Unknown";
}

/**
 * @brief Configuration for each NCS type.
 */
struct NcsTestConfig {
    NcsType type;
    std::string kind;
    nlohmann::json config;

    static NcsTestConfig InMemory() {
        return {NcsType::InMemory, "in_memory", nlohmann::json::object()};
    }
    
#ifdef USE_REDIS
    static NcsTestConfig Redis() {
        nlohmann::json config;
        config["redis_host"] = "localhost";
        config["redis_port"] = 6379;
        return {NcsType::Redis, "redis", config};
    }
#endif
};

} // namespace (anonymous)

// ============================================================================
// Parameterized Test Fixture
// ============================================================================

class NcsTest : public ::testing::TestWithParam<NcsTestConfig> {
protected:
    static constexpr size_t kValueSize = 4096;  // 4KB per value
    
    void SetUp() override {
        config_ = GetParam();
        
        // Initialize NCS based on type
        try {
            switch (config_.type) {
                case NcsType::InMemory:
                    NcsSingleton::initNcs(InMemoryNcsFactory::KIND);
                    break;
#ifdef USE_REDIS
                case NcsType::Redis:
                    NcsSingleton::initNcs(RedisNcsFactory::KIND, config_.config);
                    break;
#endif
            }
        } catch (const std::exception& e) {
            GTEST_SKIP() << NcsTypeToString(config_.type) << " not available: " << e.what();
        }
        
        ncs_ = NcsSingleton::Instance();
        if (!ncs_) {
            GTEST_SKIP() << NcsTypeToString(config_.type) << " NCS not initialized";
        }
        
        auto createResult = ncs_->createBucket(bucketId_);
        if (createResult != NcsStatus::OK) {
            GTEST_SKIP() << "Failed to create " << NcsTypeToString(config_.type) << " bucket";
        }
        
        auto descriptor = std::make_unique<NcsDescriptor>(config_.kind, bucketId_, config_.config);
        try {
            connector_.reset(NcsConnectorFactory::Instance().createConnector(descriptor.get()));
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Failed to create " << NcsTypeToString(config_.type) << " connector: " << e.what();
        }
        
        if (!connector_) {
            GTEST_SKIP() << NcsTypeToString(config_.type) << " connector is null";
        }
    }
    
    void TearDown() override {
        connector_.reset();  // Release connector before deleting bucket
        if (ncs_) {
            ncs_->deleteBucket(bucketId_);
        }
        ncs_ = nullptr;
        NcsSingleton::reset();  // Reset singleton for next test
    }
    
    std::vector<uint8_t> generateTestData(size_t size, uint8_t pattern) {
        return std::vector<uint8_t>(size, pattern);
    }
    
    /**
     * @brief Create a connector with custom config (for high-concurrency tests).
     */
    std::unique_ptr<NcsConnector> createConnectorWithConfig(uint64_t bucketId, const nlohmann::json& customConfig) {
        ncs_->createBucket(bucketId);
        auto descriptor = std::make_unique<NcsDescriptor>(config_.kind, bucketId, customConfig);
        return std::unique_ptr<NcsConnector>(
            NcsConnectorFactory::Instance().createConnector(descriptor.get()));
    }
    
    /**
     * @brief Create a new connector to the test bucket.
     * Each connector has its own connection to the backend.
     */
    std::unique_ptr<NcsConnector> createConnector() {
        auto descriptor = std::make_unique<NcsDescriptor>(config_.kind, bucketId_, config_.config);
        return std::unique_ptr<NcsConnector>(
            NcsConnectorFactory::Instance().createConnector(descriptor.get()));
    }
    
    /**
     * @brief Run concurrent test using multiple connectors (one per thread).
     * 
     * This simulates the thread_local connector model where each thread
     * creates and uses its own connector instance.
     */
    void runConcurrentTestWithMultipleConnectors(
        size_t numThreads,
        size_t opsPerThread,
        size_t numKeys
    ) {
        std::atomic<size_t> successfulOps{0};
        std::atomic<size_t> failedOps{0};
        std::vector<std::thread> threads;
        
        // Pre-populate data using the main connector
        std::vector<uint32_t> allKeys;
        std::vector<std::vector<uint8_t>> allValues;
        std::vector<boost::span<uint8_t>> allSpans;
        
        for (size_t i = 0; i < numKeys; ++i) {
            allKeys.push_back(static_cast<uint32_t>(i));
            allValues.push_back(generateTestData(kValueSize, static_cast<uint8_t>(i % 256)));
        }
        for (auto& v : allValues) {
            allSpans.emplace_back(v.data(), v.size());
        }
        
        auto putResults = connector_->multiPut(allKeys, allSpans);
        for (const auto& r : putResults) {
            ASSERT_EQ(r, NcsStatus::OK) << "Failed to pre-populate data";
        }
        
        for (size_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([this, &successfulOps, &failedOps, numKeys, opsPerThread, t]() {
                // Each thread creates its own connector (thread_local model)
                std::unique_ptr<NcsConnector> threadConnector;
                try {
                    threadConnector = createConnector();
                } catch (const std::exception& e) {
                    failedOps.fetch_add(opsPerThread);
                    return;
                }
                
                if (!threadConnector) {
                    failedOps.fetch_add(opsPerThread);
                    return;
                }
                
                std::mt19937 rng(static_cast<unsigned>(t));
                std::uniform_int_distribution<size_t> keyDist(0, numKeys - 1);
                std::uniform_int_distribution<size_t> batchDist(1, 10);
                
                for (size_t op = 0; op < opsPerThread; ++op) {
                    size_t batchSize = batchDist(rng);
                    std::vector<uint32_t> keys;
                    std::vector<std::vector<uint8_t>> buffers;
                    std::vector<boost::span<uint8_t>> spans;
                    
                    for (size_t b = 0; b < batchSize; ++b) {
                        size_t keyIdx = keyDist(rng);
                        keys.push_back(static_cast<uint32_t>(keyIdx));
                        buffers.emplace_back(kValueSize);
                    }
                    for (auto& buf : buffers) {
                        spans.emplace_back(buf.data(), buf.size());
                    }
                    
                    auto results = threadConnector->multiGet(keys, spans);
                    
                    bool allOk = true;
                    for (size_t i = 0; i < results.size(); ++i) {
                        if (results[i] == NcsStatus::OK) {
                            uint8_t expectedPattern = static_cast<uint8_t>(keys[i] % 256);
                            if (buffers[i][0] != expectedPattern) {
                                allOk = false;
                            }
                        } else {
                            allOk = false;
                        }
                    }
                    
                    if (allOk) {
                        successfulOps.fetch_add(batchSize);
                    } else {
                        failedOps.fetch_add(batchSize);
                    }
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        EXPECT_EQ(failedOps.load(), 0) << "Some operations failed during concurrent access";
        
        connector_->multiDelete(allKeys);
    }
    
    static constexpr uint64_t bucketId_ = 1000;
    NcsTestConfig config_;
    Ncs* ncs_ = nullptr;
    std::unique_ptr<NcsConnector> connector_;
};

// ============================================================================
// Test Cases (run on all NCS types)
// ============================================================================

TEST_P(NcsTest, BasicOperations) {
    std::vector<uint32_t> keys = {1, 2, 3};
    std::vector<std::vector<uint8_t>> values = {
        generateTestData(100, 0x11),
        generateTestData(200, 0x22),
        generateTestData(300, 0x33)
    };
    std::vector<boost::span<uint8_t>> putSpans;
    for (auto& v : values) {
        putSpans.emplace_back(v.data(), v.size());
    }

    // Put
    auto putResults = connector_->multiPut(keys, putSpans);
    ASSERT_EQ(putResults.size(), keys.size());
    for (const auto& r : putResults) {
        EXPECT_EQ(r, NcsStatus::OK);
    }

    // Get
    std::vector<std::vector<uint8_t>> readBuffers;
    std::vector<boost::span<uint8_t>> readSpans;
    for (const auto& v : values) {
        readBuffers.emplace_back(v.size());
    }
    for (auto& buf : readBuffers) {
        readSpans.emplace_back(buf.data(), buf.size());
    }

    auto getResults = connector_->multiGet(keys, readSpans);
    ASSERT_EQ(getResults.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(getResults[i], NcsStatus::OK);
        EXPECT_EQ(readBuffers[i], values[i]);
    }

    // Delete
    auto deleteResults = connector_->multiDelete(keys);
    for (const auto& r : deleteResults) {
        EXPECT_EQ(r, NcsStatus::OK);
    }

    // Verify deleted
    auto verifyResults = connector_->multiGet(keys, readSpans);
    for (const auto& r : verifyResults) {
        EXPECT_EQ(r, NcsStatus::ERROR);
    }
}

TEST_P(NcsTest, BufferTooSmall) {
    std::vector<uint32_t> keys = {100};
    std::vector<uint8_t> value(300, 0xAA);
    std::vector<boost::span<uint8_t>> putSpans = {boost::span<uint8_t>(value.data(), value.size())};
    
    connector_->multiPut(keys, putSpans);
    
    // Try to read with buffer too small
    std::vector<uint8_t> smallBuffer(50);
    std::vector<boost::span<uint8_t>> smallSpans = {boost::span<uint8_t>(smallBuffer.data(), smallBuffer.size())};
    
    auto result = connector_->multiGet(keys, smallSpans);
    EXPECT_EQ(result[0], NcsStatus::ERROR);
    
    connector_->multiDelete(keys);
}

TEST_P(NcsTest, Overwrite) {
    std::vector<uint32_t> keys = {10};
    std::vector<uint8_t> value1(50, 0xAA);
    std::vector<boost::span<uint8_t>> putSpans1 = {boost::span<uint8_t>(value1.data(), value1.size())};
    
    connector_->multiPut(keys, putSpans1);
    
    // Overwrite with new value
    std::vector<uint8_t> value2(50, 0xBB);
    std::vector<boost::span<uint8_t>> putSpans2 = {boost::span<uint8_t>(value2.data(), value2.size())};
    
    connector_->multiPut(keys, putSpans2);
    
    // Verify new value
    std::vector<uint8_t> readBuffer(50);
    std::vector<boost::span<uint8_t>> readSpans = {boost::span<uint8_t>(readBuffer.data(), readBuffer.size())};
    
    auto result = connector_->multiGet(keys, readSpans);
    EXPECT_EQ(result[0], NcsStatus::OK);
    EXPECT_EQ(readBuffer, value2);
    
    connector_->multiDelete(keys);
}

TEST_P(NcsTest, LargeBucketId) {
    const uint64_t largeBucketId = 463281186922614943ULL;
    
    auto createResult = ncs_->createBucket(largeBucketId);
    EXPECT_EQ(createResult, NcsStatus::OK);
    
    bool exists = ncs_->isBucketExist(largeBucketId);
    EXPECT_TRUE(exists);
    
    auto deleteResult = ncs_->deleteBucket(largeBucketId);
    EXPECT_EQ(deleteResult, NcsStatus::OK);
}

TEST_P(NcsTest, MissingBucket) {
    // Negative test: try to create a connector for a bucket that doesn't exist
    const uint64_t missingBucket = 999999;
    
    // Ensure bucket doesn't exist
    ncs_->deleteBucket(missingBucket);
    EXPECT_FALSE(ncs_->isBucketExist(missingBucket));
    
    auto descriptor = std::make_unique<NcsDescriptor>(config_.kind, missingBucket, config_.config);
    auto connector = std::unique_ptr<NcsConnector>(
        NcsConnectorFactory::Instance().createConnector(descriptor.get()));
    
    // Connector creation should fail (return nullptr) because bucket wasn't created
    EXPECT_EQ(connector, nullptr);
}

TEST_P(NcsTest, EmptyBatch) {
    std::vector<uint32_t> emptyKeys;
    std::vector<boost::span<uint8_t>> emptySpans;
    
    auto getResults = connector_->multiGet(emptyKeys, emptySpans);
    EXPECT_TRUE(getResults.empty());
    
    auto putResults = connector_->multiPut(emptyKeys, emptySpans);
    EXPECT_TRUE(putResults.empty());
    
    auto deleteResults = connector_->multiDelete(emptyKeys);
    EXPECT_TRUE(deleteResults.empty());
}

TEST_P(NcsTest, LargeBatch) {
    const size_t batchSize = 100;
    
    std::vector<uint32_t> keys;
    std::vector<std::vector<uint8_t>> values;
    std::vector<boost::span<uint8_t>> putSpans;
    
    for (size_t i = 0; i < batchSize; ++i) {
        keys.push_back(static_cast<uint32_t>(5000 + i));
        values.push_back(generateTestData(kValueSize, static_cast<uint8_t>(i)));
    }
    for (auto& v : values) {
        putSpans.emplace_back(v.data(), v.size());
    }
    
    // Put batch
    auto putResults = connector_->multiPut(keys, putSpans);
    ASSERT_EQ(putResults.size(), batchSize);
    for (const auto& r : putResults) {
        EXPECT_EQ(r, NcsStatus::OK);
    }
    
    // Get batch
    std::vector<std::vector<uint8_t>> readBuffers(batchSize, std::vector<uint8_t>(kValueSize));
    std::vector<boost::span<uint8_t>> readSpans;
    for (auto& buf : readBuffers) {
        readSpans.emplace_back(buf.data(), buf.size());
    }
    
    auto getResults = connector_->multiGet(keys, readSpans);
    ASSERT_EQ(getResults.size(), batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
        EXPECT_EQ(getResults[i], NcsStatus::OK);
        EXPECT_EQ(readBuffers[i], values[i]);
    }
    
    // Delete batch
    auto deleteResults = connector_->multiDelete(keys);
    ASSERT_EQ(deleteResults.size(), batchSize);
    for (const auto& r : deleteResults) {
        EXPECT_EQ(r, NcsStatus::OK);
    }
}

TEST_P(NcsTest, ConcurrentAccess) {
    // Test using multiple connectors (one per thread) - simulates thread_local model
    runConcurrentTestWithMultipleConnectors(
        /*numThreads=*/16, 
        /*opsPerThread=*/100, 
        /*numKeys=*/1000);
}

TEST_P(NcsTest, HighConcurrency) {
    // High concurrency test using multiple connectors (one per thread)
    // No need for special config since each thread has its own connector
    runConcurrentTestWithMultipleConnectors(
        /*numThreads=*/80, 
        /*opsPerThread=*/50, 
        /*numKeys=*/2000);
}

TEST_P(NcsTest, MultipleBatches) {
    // Test multiple sequential batch operations
    const size_t batchSize = 100;
    const size_t numBatches = 10;
    
    std::vector<uint32_t> keys;
    std::vector<std::vector<uint8_t>> values;
    std::vector<boost::span<uint8_t>> putSpans;
    
    for (size_t i = 0; i < batchSize; ++i) {
        keys.push_back(static_cast<uint32_t>(10000 + i));
        values.push_back(generateTestData(kValueSize, static_cast<uint8_t>(i)));
    }
    for (auto& v : values) {
        putSpans.emplace_back(v.data(), v.size());
    }
    
    for (size_t batch = 0; batch < numBatches; ++batch) {
        auto putResults = connector_->multiPut(keys, putSpans);
        ASSERT_EQ(putResults.size(), batchSize);
        for (const auto& r : putResults) {
            ASSERT_EQ(r, NcsStatus::OK);
        }
        
        std::vector<std::vector<uint8_t>> readBuffers(batchSize, std::vector<uint8_t>(kValueSize));
        std::vector<boost::span<uint8_t>> readSpans;
        for (auto& buf : readBuffers) {
            readSpans.emplace_back(buf.data(), buf.size());
        }
        
        auto getResults = connector_->multiGet(keys, readSpans);
        ASSERT_EQ(getResults.size(), batchSize);
        for (size_t i = 0; i < batchSize; ++i) {
            EXPECT_EQ(getResults[i], NcsStatus::OK);
            EXPECT_EQ(readBuffers[i], values[i]);
        }
    }
    
    connector_->multiDelete(keys);
}

// ============================================================================
// Test Instantiation
// ============================================================================

namespace {
// Custom name generator for better test output
std::string NcsTestNameGenerator(const ::testing::TestParamInfo<NcsTestConfig>& info) {
    return NcsTypeToString(info.param.type);
}

// Build the list of NCS types to test
std::vector<NcsTestConfig> GetNcsTestConfigs() {
    std::vector<NcsTestConfig> configs;
    configs.push_back(NcsTestConfig::InMemory());
#ifdef USE_REDIS
    configs.push_back(NcsTestConfig::Redis());
#endif
    return configs;
}
} // namespace (anonymous)

INSTANTIATE_TEST_SUITE_P(
    AllNcsTypes,
    NcsTest,
    ::testing::ValuesIn(GetNcsTestConfigs()),
    NcsTestNameGenerator
);

} // namespace milvus
