#pragma once

#include "ncs/ncs.h"
#include "ncs/InMemoryKV.h"
#include <memory>
#include <unistd.h>


namespace milvus {

class InMemoryNcs : public Ncs {
public:
    NcsStatus createBucket(uint64_t bucketId) override;
    NcsStatus deleteBucket(uint64_t bucketId) override;
    NcsBucketStatus getBucketNcsStatus(uint64_t bucketId) override;
    bool isBucketExist(uint64_t bucketId) override;
    ~InMemoryNcs() override = default;
private:
    InMemoryNcs() = default;
    friend class InMemoryNcsFactory;
};

class InMemoryNcsFactory : public NcsFactory {
public:
    static const std::string KIND;
    std::unique_ptr<Ncs> createNcs(const nlohmann::json& params = nlohmann::json{}) override;
    const std::string& getKind() const override;
};

} // namespace milvus
