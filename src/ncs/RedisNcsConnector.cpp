#include "ncs/RedisNcsConnector.h"
#include "log/Log.h"
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

namespace milvus {

// RedisNcsConnector implementation
RedisNcsConnector::RedisNcsConnector(uint64_t bucketId, const std::string& host, int port)
    : NcsConnector(bucketId), context_(nullptr), host_(host), port_(port) {
    context_ = redisConnect(host.c_str(), port);
    if (context_ == nullptr || context_->err) {
        if (context_) {
            LOG_ERROR("[RedisNcsConnector] Redis connection error: {}", context_->errstr);
            redisFree(context_);
            context_ = nullptr;
        } else {
            LOG_ERROR("[RedisNcsConnector] Redis connection error: can't allocate redis context");
        }
        throw std::runtime_error("Failed to connect to Redis at " + host + ":" + std::to_string(port));
    }
    LOG_DEBUG("[RedisNcsConnector] Connected to Redis at {}:{} for bucket {}", host, port, bucketId);
}

RedisNcsConnector::~RedisNcsConnector() {
    if (context_) {
        redisFree(context_);
        context_ = nullptr;
    }
}

std::vector<NcsStatus> RedisNcsConnector::multiGet(
    const std::vector<uint32_t>& keys, 
    const std::vector<SpanBytes>& buffs) {
    
    std::vector<NcsStatus> results(keys.size(), NcsStatus::ERROR);
    
    if (!context_) {
        LOG_ERROR("[RedisNcsConnector] Redis context is null");
        return results;
    }
    
    if (keys.size() != buffs.size()) {
        LOG_ERROR("[RedisNcsConnector] Keys and buffers size mismatch");
        return results;
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
        
        redisReply* reply = (redisReply*)redisCommand(context_, "GET %s", redisKey.c_str());
        
        if (reply == nullptr) {
            LOG_ERROR("[RedisNcsConnector] Failed to GET key {}: {}", redisKey, context_->errstr);
            continue;
        }
        
        if (reply->type == REDIS_REPLY_STRING) {
            size_t dataSize = reply->len;
            if (dataSize <= buffs[i].size()) {
                std::memcpy(buffs[i].data(), reply->str, dataSize);
                results[i] = NcsStatus::OK;
            } else {
                LOG_ERROR("[RedisNcsConnector] Buffer too small for key {}: need {}, have {}", 
                         redisKey, dataSize, buffs[i].size());
            }
        } else if (reply->type == REDIS_REPLY_NIL) {
            LOG_WARN("[RedisNcsConnector] Key {} does not exist", redisKey);
        } else {
            LOG_ERROR("[RedisNcsConnector] Unexpected reply type for key {}", redisKey);
        }
        
        freeReplyObject(reply);
    }
    
    return results;
}

std::vector<NcsStatus> RedisNcsConnector::multiPut(
    const std::vector<uint32_t>& keys, 
    const std::vector<SpanBytes>& buffs) {
    
    std::vector<NcsStatus> results(keys.size(), NcsStatus::ERROR);
    
    if (!context_) {
        LOG_ERROR("[RedisNcsConnector] Redis context is null");
        return results;
    }
    
    if (keys.size() != buffs.size()) {
        LOG_ERROR("[RedisNcsConnector] Keys and buffers size mismatch");
        return results;
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
        
        redisReply* reply = (redisReply*)redisCommand(context_, "SET %s %b", 
            redisKey.c_str(), buffs[i].data(), buffs[i].size());
        
        if (reply == nullptr) {
            LOG_ERROR("[RedisNcsConnector] Failed to SET key {}: {}", redisKey, context_->errstr);
            continue;
        }
        
        if (reply->type == REDIS_REPLY_STATUS && std::string(reply->str) == "OK") {
            results[i] = NcsStatus::OK;
        } else {
            LOG_ERROR("[RedisNcsConnector] Failed to SET key {}", redisKey);
        }
        
        freeReplyObject(reply);
    }
    
    return results;
}

std::vector<NcsStatus> RedisNcsConnector::multiDelete(const std::vector<uint32_t>& keys) {
    std::vector<NcsStatus> results(keys.size(), NcsStatus::ERROR);
    
    if (!context_) {
        LOG_ERROR("[RedisNcsConnector] Redis context is null");
        return results;
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
        
        redisReply* reply = (redisReply*)redisCommand(context_, "DEL %s", redisKey.c_str());
        
        if (reply == nullptr) {
            LOG_ERROR("[RedisNcsConnector] Failed to DEL key {}: {}", redisKey, context_->errstr);
            continue;
        }
        
        if (reply->type == REDIS_REPLY_INTEGER) {
            results[i] = NcsStatus::OK;
        } else {
            LOG_ERROR("[RedisNcsConnector] Failed to DEL key {}", redisKey);
        }
        
        freeReplyObject(reply);
    }
    
    return results;
}

// RedisNcsConnectorCreator implementation
const std::string RedisNcsConnectorCreator::KIND = "redis";

NcsConnector* RedisNcsConnectorCreator::factoryMethod(const NcsDescriptor* descriptor) {
    const json& extras = descriptor->getExtras();
    
    if (!extras.contains("redis_host")) {
        throw std::runtime_error("RedisNcsConnectorCreator: 'redis_host' is required in descriptor extras");
    }
    if (!extras.contains("redis_port")) {
        throw std::runtime_error("RedisNcsConnectorCreator: 'redis_port' is required in descriptor extras");
    }
    
    std::string host = extras["redis_host"].get<std::string>();
    int port = extras["redis_port"].get<int>();
    
    return new RedisNcsConnector(descriptor->getbucketId(), host, port);
}

const std::string& RedisNcsConnectorCreator::getKind() const {
    return KIND;
}

// Register the connector creator on startup
namespace {
    struct RegisterRedisNcsConnectorCreator {
        RegisterRedisNcsConnectorCreator() {
            NcsConnectorFactory::Instance().registerCreator(
                std::make_unique<RedisNcsConnectorCreator>());
        }
    } registerRedisNcsConnectorCreator;
}

} // namespace milvus
