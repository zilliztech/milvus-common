#include <gtest/gtest.h>
#include "ncs/RedisNcs.h"
#include "ncs/RedisNcsConnector.h"
#include <memory>
#include <vector>

namespace milvus {
namespace {

// Note: These tests require a Redis server running on localhost:6379
// To skip these tests if Redis is not available, they can be marked as DISABLED_

TEST(RedisNcsTest, BasicBucketOperations) {
    // Initialize RedisNcs with host and port
    json config;
    config["redis_host"] = "localhost";
    config["redis_port"] = 6379;
    
    NcsSingleton::initNcs(RedisNcsFactory::KIND, config);
    Ncs* ncs = NcsSingleton::Instance();
    
    const uint64_t bucketId = 100;
    
    // Test bucket creation
    auto createResult = ncs->createBucket(bucketId);
    EXPECT_EQ(createResult, NcsStatus::OK);
    
    // Test bucket existence check
    bool exists = ncs->isBucketExist(bucketId);
    EXPECT_TRUE(exists);
    
    // Test bucket deletion
    auto deleteResult = ncs->deleteBucket(bucketId);
    EXPECT_EQ(deleteResult, NcsStatus::OK);
    
    // Verify bucket no longer exists
    exists = ncs->isBucketExist(bucketId);
    EXPECT_FALSE(exists);
}

TEST(RedisNcsConnectorTest, MultiGetPutDelete) {
    const uint64_t bucketId = 101;
    
    // Initialize RedisNcs
    json config;
    config["redis_host"] = "localhost";
    config["redis_port"] = 6379;
    
    NcsSingleton::initNcs(RedisNcsFactory::KIND, config);
    Ncs* ncs = NcsSingleton::Instance();
    
    // Create bucket
    auto createResult = ncs->createBucket(bucketId);
    ASSERT_EQ(createResult, NcsStatus::OK);
    
    auto descriptor = std::make_unique<NcsDescriptor>("redis", bucketId, config);
    auto connector = std::unique_ptr<NcsConnector>(
        NcsConnectorFactory::Instance().createConnector(descriptor.get()));
    
    ASSERT_NE(connector, nullptr);

    // Prepare test data
    std::vector<uint32_t> keys = {1, 2, 3};
    std::vector<std::vector<uint8_t>> values = {
        std::vector<uint8_t>(100, 0x11),  // 100 bytes
        std::vector<uint8_t>(200, 0x22),  // 200 bytes
        std::vector<uint8_t>(300, 0x33)   // 300 bytes
    };
    
    // Create SpanBytes for put operation
    std::vector<SpanBytes> putBuffs;
    for (const auto& value : values) {
        putBuffs.emplace_back(const_cast<uint8_t*>(value.data()), value.size());
    }

    // Test multiPut
    auto putResults = connector->multiPut(keys, putBuffs);
    EXPECT_EQ(putResults.size(), keys.size());
    for (const auto& result : putResults) {
        EXPECT_EQ(result, NcsStatus::OK);
    }

    // Test multiGet
    std::vector<std::vector<uint8_t>> getBuffers(keys.size());
    std::vector<SpanBytes> getBuffs;
    for (size_t i = 0; i < keys.size(); ++i) {
        getBuffers[i].resize(values[i].size());
        getBuffs.emplace_back(getBuffers[i].data(), getBuffers[i].size());
    }

    auto getResults = connector->multiGet(keys, getBuffs);
    EXPECT_EQ(getResults.size(), keys.size());
    for (size_t i = 0; i < getResults.size(); ++i) {
        EXPECT_EQ(getResults[i], NcsStatus::OK);
        EXPECT_EQ(getBuffers[i], values[i]);
    }

    // Test multiDelete
    auto deleteResults = connector->multiDelete(keys);
    EXPECT_EQ(deleteResults.size(), keys.size());
    for (const auto& result : deleteResults) {
        EXPECT_EQ(result, NcsStatus::OK);
    }

    // Verify deletion - get should return empty/error
    auto verifyResults = connector->multiGet(keys, getBuffs);
    for (const auto& result : verifyResults) {
        EXPECT_NE(result, NcsStatus::OK);
    }
    
    // Cleanup
    ncs->deleteBucket(bucketId);
}

TEST(RedisNcsTest, LargeBucketId) {
    // Initialize RedisNcs with host and port
    json config;
    config["redis_host"] = "localhost";
    config["redis_port"] = 6379;
    
    NcsSingleton::initNcs(RedisNcsFactory::KIND, config);
    Ncs* ncs = NcsSingleton::Instance();
    
    const uint64_t bucketId = 463281186922614943ULL;
    
    // Test bucket creation
    auto createResult = ncs->createBucket(bucketId);
    EXPECT_EQ(createResult, NcsStatus::OK);
    
    // Test bucket existence check
    bool exists = ncs->isBucketExist(bucketId);
    EXPECT_TRUE(exists);
    
    // Test bucket deletion
    auto deleteResult = ncs->deleteBucket(bucketId);
    EXPECT_EQ(deleteResult, NcsStatus::OK);
}

} // namespace
} // namespace milvus
