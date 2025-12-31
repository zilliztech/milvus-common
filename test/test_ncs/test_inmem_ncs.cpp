#include <gtest/gtest.h>
#include "ncs/InMemNcsConnector.h"
#include "ncs/InMemoryNcs.h" // provides InMemoryNcsFactory
#include <memory>

namespace milvus {
namespace {

TEST(InMemNcsConnectorTest, BasicOperations) {
    const uint64_t bucketId = 1;
    // Register the trivial InMemoryNcs factory in the singleton and get Ncs instance
    NcsSingleton::initNcs(InMemoryNcsFactory::KIND);
    Ncs* ncs = NcsSingleton::Instance();
    auto createResult = ncs->createBucket(bucketId);
    EXPECT_EQ(createResult, NcsStatus::OK);
    
    // Create descriptor and connector
    auto descriptor = std::make_unique<NcsDescriptor>(NcsDescriptor("in_memory", bucketId, json::object()));
    auto connector = std::unique_ptr<NcsConnector>(
        NcsConnectorFactory::Instance().createConnector(descriptor.get()));
    
    ASSERT_NE(connector, nullptr);

    // Prepare test data with varying sizes
    std::vector<uint32_t> keys = {1, 2, 3};
    std::vector<std::vector<uint8_t>> values = {
        std::vector<uint8_t>(100, 0x11),  // 100 bytes
        std::vector<uint8_t>(200, 0x22),  // 200 bytes
        std::vector<uint8_t>(300, 0x33)   // 300 bytes
    };
    std::vector<SpanBytes> valueSpans;
    for (auto& value : values) {
        valueSpans.emplace_back(value.data(), value.size());
    }

    // Test multiPut
    auto putResults = connector->multiPut(keys, valueSpans);
    ASSERT_EQ(putResults.size(), keys.size());
    for (const auto& status : putResults) {
        EXPECT_EQ(status, NcsStatus::OK);
    }

    // Prepare buffers for reading
    std::vector<std::vector<uint8_t>> readBuffers;
    std::vector<SpanBytes> readSpans;
    for (const auto& value : values) {
        readBuffers.emplace_back(value.size());
        readSpans.emplace_back(readBuffers.back().data(), readBuffers.back().size());
    }

    // Test multiGet
    auto getResults = connector->multiGet(keys, readSpans);
    ASSERT_EQ(getResults.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(getResults[i], NcsStatus::OK);
        EXPECT_EQ(readBuffers[i], values[i]);
    }

    // Test with buffer too small
    std::vector<uint8_t> smallBuffer(50);
    SpanBytes smallSpan(smallBuffer.data(), smallBuffer.size());
    auto getResult = connector->multiGet({keys[2]}, {smallSpan});
    ASSERT_EQ(getResult.size(), 1);
    EXPECT_EQ(getResult[0], NcsStatus::ERROR);  // Should fail as buffer is too small

    // Test multiDelete for specific keys
    auto deleteResult = connector->multiDelete(keys);
    ASSERT_EQ(deleteResult.size(), keys.size());
    for (const auto& status : deleteResult) {
        EXPECT_EQ(status, NcsStatus::OK);
    }

    // Verify data is deleted
    auto getResultsAfterDelete = connector->multiGet(keys, readSpans);
    for (const auto& status : getResultsAfterDelete) {
        EXPECT_EQ(status, NcsStatus::ERROR);
    }

    // Test bucket deletion
    auto bucketDeleteResult = ncs->deleteBucket(bucketId);
    EXPECT_EQ(bucketDeleteResult, NcsStatus::OK);

    // Negative test: try to put into a bucket that was not created
    const uint32_t missingBucket = 999;
    auto descriptor2 = std::make_unique<NcsDescriptor>("in_memory", missingBucket, json::object());
    auto connector2 = std::unique_ptr<NcsConnector>(
        NcsConnectorFactory::Instance().createConnector(descriptor2.get()));
    // Connector creation should fail (return nullptr) because bucket wasn't created
    EXPECT_EQ(connector2, nullptr);
}

} // namespace
} // namespace milvus