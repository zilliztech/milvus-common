#include "ncs/RedisNcsConnector.h"
#include "log/Log.h"
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

namespace milvus {

// ============================================================================
// RedisNcsConnector implementation
// ============================================================================

RedisNcsConnector::RedisNcsConnector(uint64_t bucketId, const std::string& host, int port)
    : NcsConnector(bucketId), ctx_(nullptr), host_(host), port_(port) {
    
    ctx_ = redisConnect(host.c_str(), port);
    if (ctx_ == nullptr || ctx_->err) {
        if (ctx_) {
            LOG_ERROR("[RedisNcsConnector] Connection error: {}", ctx_->errstr);
            redisFree(ctx_);
            ctx_ = nullptr;
        } else {
            LOG_ERROR("[RedisNcsConnector] Cannot allocate redis context");
        }
        throw std::runtime_error("Failed to connect to Redis at " + host + ":" + std::to_string(port));
    }
    
    LOG_DEBUG("[RedisNcsConnector] Created connector for bucket {} to {}:{}", 
              bucketId, host, port);
}

RedisNcsConnector::~RedisNcsConnector() {
    if (ctx_) {
        redisFree(ctx_);
        ctx_ = nullptr;
    }
    LOG_DEBUG("[RedisNcsConnector] Destroyed connector for bucket {}", bucketId_);
}

bool RedisNcsConnector::ensureConnected() {
    if (ctx_ != nullptr && !ctx_->err) {
        return true;
    }
    
    if (ctx_) {
        redisFree(ctx_);
    }
    
    ctx_ = redisConnect(host_.c_str(), port_);
    if (ctx_ == nullptr || ctx_->err) {
        if (ctx_) {
            LOG_ERROR("[RedisNcsConnector] Reconnection error: {}", ctx_->errstr);
            redisFree(ctx_);
            ctx_ = nullptr;
        }
        return false;
    }
    
    LOG_DEBUG("[RedisNcsConnector] Reconnected to {}:{}", host_, port_);
    return true;
}

std::vector<NcsStatus> RedisNcsConnector::multiGet(
    const std::vector<uint32_t>& keys, 
    const std::vector<SpanBytes>& buffs) {
    
    std::vector<NcsStatus> results(keys.size(), NcsStatus::ERROR);
    
    if (keys.empty()) {
        return results;
    }
    
    if (keys.size() != buffs.size()) {
        LOG_ERROR("[RedisNcsConnector] Keys and buffers size mismatch");
        return results;
    }
    
    if (!ensureConnected()) {
        LOG_ERROR("[RedisNcsConnector] Not connected to Redis");
        return results;
    }
    
    // Pipeline all GET commands
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
        
        if (redisAppendCommand(ctx_, "GET %s", redisKey.c_str()) != REDIS_OK) {
            LOG_ERROR("[RedisNcsConnector] Failed to append GET command for key {}", redisKey);
            for (size_t j = 0; j < i; ++j) {
                redisReply* reply = nullptr;
                redisGetReply(ctx_, (void**)&reply);
                if (reply) freeReplyObject(reply);
            }
            return results;
        }
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        redisReply* reply = nullptr;
        
        if (redisGetReply(ctx_, (void**)&reply) != REDIS_OK) {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to get reply for key {}: {}", 
                     redisKey, ctx_->errstr ? ctx_->errstr : "unknown error");
            continue;
        }
        
        if (reply == nullptr) {
            continue;
        }
        
        if (reply->type == REDIS_REPLY_STRING) {
            size_t dataSize = reply->len;
            if (dataSize <= buffs[i].size()) {
                std::memcpy(buffs[i].data(), reply->str, dataSize);
                results[i] = NcsStatus::OK;
            } else {
                std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
                LOG_ERROR("[RedisNcsConnector] Buffer too small for key {}: need {}, have {}", 
                         redisKey, dataSize, buffs[i].size());
            }
        } else if (reply->type == REDIS_REPLY_NIL) {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_WARN("[RedisNcsConnector] Key {} does not exist", redisKey);
        } else {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Unexpected reply type {} for key {}", reply->type, redisKey);
        }
        
        freeReplyObject(reply);
    }
    
    return results;
}

std::vector<NcsStatus> RedisNcsConnector::multiPut(
    const std::vector<uint32_t>& keys, 
    const std::vector<SpanBytes>& buffs) {
    
    std::vector<NcsStatus> results(keys.size(), NcsStatus::ERROR);
    
    if (keys.empty()) {
        return results;
    }
    
    if (keys.size() != buffs.size()) {
        LOG_ERROR("[RedisNcsConnector] Keys and buffers size mismatch");
        return results;
    }
    
    if (!ensureConnected()) {
        LOG_ERROR("[RedisNcsConnector] Not connected to Redis");
        return results;
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
        
        if (redisAppendCommand(ctx_, "SET %s %b", 
                              redisKey.c_str(), buffs[i].data(), buffs[i].size()) != REDIS_OK) {
            LOG_ERROR("[RedisNcsConnector] Failed to append SET command for key {}", redisKey);
            for (size_t j = 0; j < i; ++j) {
                redisReply* reply = nullptr;
                redisGetReply(ctx_, (void**)&reply);
                if (reply) freeReplyObject(reply);
            }
            return results;
        }
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        redisReply* reply = nullptr;
        
        if (redisGetReply(ctx_, (void**)&reply) != REDIS_OK) {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to get reply for key {}: {}", 
                     redisKey, ctx_->errstr ? ctx_->errstr : "unknown error");
            continue;
        }
        
        if (reply == nullptr) {
            continue;
        }
        
        if (reply->type == REDIS_REPLY_STATUS && 
            reply->str && std::string(reply->str) == "OK") {
            results[i] = NcsStatus::OK;
        } else {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to SET key {}", redisKey);
        }
        
        freeReplyObject(reply);
    }
    
    return results;
}

std::vector<NcsStatus> RedisNcsConnector::multiDelete(const std::vector<uint32_t>& keys) {
    std::vector<NcsStatus> results(keys.size(), NcsStatus::ERROR);
    
    if (keys.empty()) {
        return results;
    }
    
    if (!ensureConnected()) {
        LOG_ERROR("[RedisNcsConnector] Not connected to Redis");
        return results;
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
        
        if (redisAppendCommand(ctx_, "DEL %s", redisKey.c_str()) != REDIS_OK) {
            LOG_ERROR("[RedisNcsConnector] Failed to append DEL command for key {}", redisKey);
            for (size_t j = 0; j < i; ++j) {
                redisReply* reply = nullptr;
                redisGetReply(ctx_, (void**)&reply);
                if (reply) freeReplyObject(reply);
            }
            return results;
        }
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        redisReply* reply = nullptr;
        
        if (redisGetReply(ctx_, (void**)&reply) != REDIS_OK) {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to get reply for key {}: {}", 
                     redisKey, ctx_->errstr ? ctx_->errstr : "unknown error");
            continue;
        }
        
        if (reply == nullptr) {
            continue;
        }
        
        if (reply->type == REDIS_REPLY_INTEGER) {
            results[i] = NcsStatus::OK;
        } else {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
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
    
    // Check if bucket exists by querying Redis directly
    std::string bucketKey = "bucket_" + std::to_string(descriptor->getbucketId()) + "_valid";
    redisContext* ctx = redisConnect(host.c_str(), port);
    if (ctx == nullptr || ctx->err) {
        if (ctx) {
            LOG_ERROR("[RedisNcsConnectorCreator] Failed to connect to Redis: {}", ctx->errstr);
            redisFree(ctx);
        }
        return nullptr;
    }
    
    redisReply* reply = (redisReply*)redisCommand(ctx, "EXISTS %s", bucketKey.c_str());
    bool bucketExists = (reply != nullptr && reply->type == REDIS_REPLY_INTEGER && reply->integer == 1);
    if (reply) freeReplyObject(reply);
    redisFree(ctx);
    
    if (!bucketExists) {
        LOG_ERROR("[RedisNcsConnectorCreator] Bucket {} does not exist", descriptor->getbucketId());
        return nullptr;
    }
    
    LOG_INFO("[RedisNcsConnectorCreator] Creating connector for bucket {} with host={}:{}",
             descriptor->getbucketId(), host, port);
    
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
