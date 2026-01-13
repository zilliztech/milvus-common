#pragma once

#ifdef USE_REDIS

#include "ncs/ncs.h"
#include "ncs/RedisTypes.h"
#include "log/Log.h"
#include <memory>
#include <string>
#include <mutex>
#include <hiredis/hiredis.h>

namespace milvus {

class RedisNcs : public Ncs {
public:
    NcsStatus createBucket(uint64_t bucketId) override;
    NcsStatus deleteBucket(uint64_t bucketId) override;
    NcsBucketStatus getBucketNcsStatus(uint64_t bucketId) override;
    bool isBucketExist(uint64_t bucketId) override;
    ~RedisNcs() override;
    
private:
    explicit RedisNcs(const std::string& host, int port);
    ncs::RedisContextPtr context_;
    std::string host_;
    int port_ = 0;
    std::mutex mutex_;
    
    friend class RedisNcsFactory;
};

class RedisNcsFactory : public NcsFactory {
public:
    static const std::string KIND;
    std::unique_ptr<Ncs> createNcs(const nlohmann::json& params = nlohmann::json{}) override;
    const std::string& getKind() const override;
};

} // namespace milvus

#endif // USE_REDIS
