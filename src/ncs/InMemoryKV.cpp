#include "ncs/InMemoryKV.h"
#include <cstring>

namespace milvus {

InMemoryKV* InMemoryKV::Instance() {
    static InMemoryKV instance;
    return &instance;
}

bool InMemoryKV::put(uint64_t bucketId, uint32_t key, const SpanBytes& value) {
    auto bucket_it = data_.find(bucketId);
    if (bucket_it == data_.end()) {
        // bucket not created
        return false;
    }

    auto& bucket = bucket_it->second;
    std::vector<uint8_t> value_copy(static_cast<const uint8_t*>(value.data()),
                                    static_cast<const uint8_t*>(value.data()) + value.size());
    bucket[key] = std::move(value_copy);
    return true;
}

bool InMemoryKV::createBucket(uint64_t bucketId) {
    data_.emplace(bucketId, BucketMap{});
    // If bucket already exists, emplace does nothing; treat as success.
    return true;
}

bool InMemoryKV::deleteBucket(uint64_t bucketId) {
    auto erased = data_.erase(bucketId);
    return erased > 0;
}

bool InMemoryKV::hasBucket(uint64_t bucketId) const {
    return data_.find(bucketId) != data_.end();
}

bool InMemoryKV::get(uint64_t bucketId, uint32_t key, const SpanBytes& buff) const {
    auto bucket_it = data_.find(bucketId);
    if (bucket_it == data_.end()) return false;

    auto value_it = bucket_it->second.find(key);
    if (value_it == bucket_it->second.end()) return false;

    if (value_it->second.size() > buff.size()) return false;

    std::memcpy(buff.data(), value_it->second.data(), value_it->second.size());
    return true;
}

bool InMemoryKV::deleteKey(uint64_t bucketId, uint32_t key) {
    auto bucket_it = data_.find(bucketId);
    if (bucket_it == data_.end()) return false;
    auto erased = bucket_it->second.erase(key);
    return erased > 0;
}

} // namespace milvus
