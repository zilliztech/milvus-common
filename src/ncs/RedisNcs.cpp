#ifdef USE_REDIS

#include "ncs/RedisNcs.h"
#include "log/Log.h"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace milvus {

// RedisNcs implementation
RedisNcs::RedisNcs(const std::string& host, int port) 
    : context_(nullptr, redisFree), host_(host), port_(port) {
    context_.reset(redisConnect(host.c_str(), port));
    if (context_ == nullptr || context_->err) {
        std::string error_msg = context_ ? context_->errstr : "can't allocate redis context";
        LOG_ERROR("[RedisNcs] Redis connection error: {}", error_msg);
        context_.reset();
        throw std::runtime_error("Failed to connect to Redis at " + host + ":" + std::to_string(port));
    }
    LOG_INFO("[RedisNcs] Connected to Redis at {}:{}", host, port);
}

RedisNcs::~RedisNcs() {
    // unique_ptr automatically calls redisFree
}

NcsStatus RedisNcs::createBucket(uint64_t bucketId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) {
        LOG_ERROR("[RedisNcs] Redis context is null");
        return NcsStatus::ERROR;
    }
    
    std::string key = "bucket_" + std::to_string(bucketId) + "_valid";
    ncs::RedisReplyPtr reply(
        (redisReply*)redisCommand(context_.get(), "SET %s true", key.c_str()),
        freeReplyObject
    );
    
    if (reply == nullptr) {
        LOG_ERROR("[RedisNcs] Failed to create bucket {}: {}", bucketId, context_->errstr);
        return NcsStatus::ERROR;
    }
    
    if (reply->type == REDIS_REPLY_STATUS && std::string(reply->str) == "OK") {
        LOG_INFO("[RedisNcs] Created bucket {}", bucketId);
        return NcsStatus::OK;
    }
    
    LOG_ERROR("[RedisNcs] Failed to create bucket {}. Reply type: {}, str: {}", 
             bucketId, reply->type, (reply->str ? reply->str : "null"));
    return NcsStatus::ERROR;
}

NcsStatus RedisNcs::deleteBucket(uint64_t bucketId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) {
        LOG_ERROR("[RedisNcs] Redis context is null");
        return NcsStatus::ERROR;
    }
    
    std::string pattern = "bucket_" + std::to_string(bucketId) + "_*";
    std::vector<std::string> keysToDelete;
    
    // Use SCAN to find all keys matching the pattern
    int cursor = 0;
    do {
        ncs::RedisReplyPtr reply(
            (redisReply*)redisCommand(context_.get(), "SCAN %d MATCH %s COUNT 100", cursor, pattern.c_str()),
            freeReplyObject
        );
        
        if (reply == nullptr || reply->type != REDIS_REPLY_ARRAY) {
            LOG_ERROR("[RedisNcs] Failed to scan keys for bucket {}", bucketId);
            return NcsStatus::ERROR;
        }
        
        // Parse cursor
        cursor = std::atoi(reply->element[0]->str);
        
        // Parse keys
        redisReply* keysArray = reply->element[1];
        for (size_t i = 0; i < keysArray->elements; ++i) {
            keysToDelete.push_back(keysArray->element[i]->str);
        }
    } while (cursor != 0);
    
    // Delete all found keys using UNLINK (async delete)
    if (!keysToDelete.empty()) {
        // Build UNLINK command
        std::string cmd = "UNLINK";
        for (const auto& key : keysToDelete) {
            cmd += " " + key;
        }
        
        ncs::RedisReplyPtr reply(
            (redisReply*)redisCommand(context_.get(), cmd.c_str()),
            freeReplyObject
        );
        if (reply == nullptr) {
            LOG_ERROR("[RedisNcs] Failed to delete keys for bucket {}: {}", 
                     bucketId, context_->errstr);
            return NcsStatus::ERROR;
        }
        
        LOG_INFO("[RedisNcs] Deleted {} keys for bucket {}", reply->integer, bucketId);
    }
    
    return NcsStatus::OK;
}

NcsBucketStatus RedisNcs::getBucketNcsStatus(uint64_t /*bucketId*/) {
    return NcsBucketStatus();
}

bool RedisNcs::isBucketExist(uint64_t bucketId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!context_) {
        LOG_ERROR("[RedisNcs] Redis context is null");
        return false;
    }
    
    std::string key = "bucket_" + std::to_string(bucketId) + "_valid";
    ncs::RedisReplyPtr reply(
        (redisReply*)redisCommand(context_.get(), "EXISTS %s", key.c_str()),
        freeReplyObject
    );
    
    if (reply == nullptr) {
        LOG_ERROR("[RedisNcs] Failed to check bucket existence: {}", context_->errstr);
        return false;
    }
    
    return (reply->type == REDIS_REPLY_INTEGER && reply->integer == 1);
}

// RedisNcsFactory implementation
const std::string RedisNcsFactory::KIND = "redis";

std::unique_ptr<Ncs> RedisNcsFactory::createNcs(const nlohmann::json& params) {
    if (!params.contains("redis_host")) {
        throw std::runtime_error("RedisNcsFactory: 'redis_host' is required in params");
    }
    if (!params.contains("redis_port")) {
        throw std::runtime_error("RedisNcsFactory: 'redis_port' is required in params");
    }
    
    std::string host = params["redis_host"].get<std::string>();
    int port = params["redis_port"].get<int>();
    
    return std::unique_ptr<Ncs>(new RedisNcs(host, port));
}

const std::string& RedisNcsFactory::getKind() const {
    return KIND;
}

// Register the factory on startup
namespace {
    struct RegisterRedisNcsFactory {
        RegisterRedisNcsFactory() {
            NcsFactoryRegistry::Instance().registerFactory(
                std::make_unique<RedisNcsFactory>());
        }
    } registerRedisNcsFactory;
}

} // namespace milvus

#endif // USE_REDIS
