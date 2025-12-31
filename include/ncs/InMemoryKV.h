#pragma once

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <unistd.h>

#include "common/SpanBytes.h"

namespace milvus {

class InMemoryKV {
public:
    static InMemoryKV* Instance();

    // Put a value into an existing bucket. Returns true on success, false if the
    // bucket does not exist.
    bool put(uint64_t bucketId, uint32_t key, const SpanBytes& value);
    bool createBucket(uint64_t bucketId);
    bool deleteBucket(uint64_t bucketId);
    bool hasBucket(uint64_t bucketId) const;
    bool get(uint64_t bucketId, uint32_t key, const SpanBytes& buff) const;
    bool deleteKey(uint64_t bucketId, uint32_t key);
private:
    InMemoryKV() = default;
    using BucketMap = std::unordered_map<uint32_t, std::vector<uint8_t>>;
    using DataMap = std::unordered_map<uint64_t, BucketMap>;

    DataMap data_;
};

} // namespace milvus
