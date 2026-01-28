#ifdef USE_REDIS

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
    : NcsConnector(bucketId), ctx_(nullptr, redisFree), host_(host), port_(port) {
    
    ctx_.reset(redisConnect(host.c_str(), port));
    if (ctx_ == nullptr || ctx_->err) {
        std::string error_msg = ctx_ ? ctx_->errstr : "Cannot allocate redis context";
        LOG_ERROR("[RedisNcsConnector] Connection error: {}", error_msg);
        ctx_.reset();
        throw std::runtime_error("Failed to connect to Redis at " + host + ":" + std::to_string(port));
    }
    
    LOG_DEBUG("[RedisNcsConnector] Created connector for bucket {} to {}:{}", 
              bucketId, host, port);
}

RedisNcsConnector::~RedisNcsConnector() {
    LOG_DEBUG("[RedisNcsConnector] Destroyed connector for bucket {}", bucketId_);
}

bool RedisNcsConnector::ensureConnected() {
    if (ctx_ != nullptr && !ctx_->err) {
        return true;
    }
    
    ctx_.reset(redisConnect(host_.c_str(), port_));
    if (ctx_ == nullptr || ctx_->err) {
        std::string error_msg = ctx_ ? ctx_->errstr : "Cannot allocate redis context";
        LOG_ERROR("[RedisNcsConnector] Reconnection error: {}", error_msg);
        ctx_.reset();
        return false;
    }
    
    LOG_DEBUG("[RedisNcsConnector] Reconnected to {}:{}", host_, port_);
    return true;
}

ncs::RedisReplyPtr RedisNcsConnector::getSafeReply() {
    redisReply* raw_reply = nullptr;
    if (redisGetReply(ctx_.get(), (void**)&raw_reply) != REDIS_OK) {
        return ncs::RedisReplyPtr(nullptr, freeReplyObject);
    }
    return ncs::RedisReplyPtr(raw_reply, freeReplyObject);
}

std::vector<NcsStatus> RedisNcsConnector::multiGet(
    const std::vector<uint32_t>& keys, 
    const std::vector<boost::span<uint8_t>>& buffs) {
    
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
        
        if (redisAppendCommand(ctx_.get(), "GET %s", redisKey.c_str()) != REDIS_OK) {
            LOG_ERROR("[RedisNcsConnector] Failed to append GET command for key {}", redisKey);
            // Connection is in bad state, force reconnect on next call
            ctx_.reset();
            return results;
        }
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        auto reply = getSafeReply();
        
        if (!reply) {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to get reply for key {}: {}", 
                     redisKey, ctx_->errstr ? ctx_->errstr : "unknown error");
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
    }
    
    return results;
}

std::vector<NcsStatus> RedisNcsConnector::multiPut(
    const std::vector<uint32_t>& keys, 
    const std::vector<boost::span<uint8_t>>& buffs) {
    
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
        
        if (redisAppendCommand(ctx_.get(), "SET %s %b", 
                              redisKey.c_str(), buffs[i].data(), buffs[i].size()) != REDIS_OK) {
            LOG_ERROR("[RedisNcsConnector] Failed to append SET command for key {}", redisKey);
            // Connection is in bad state, force reconnect on next call
            ctx_.reset();
            return results;
        }
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        auto reply = getSafeReply();
        
        if (!reply) {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to get reply for key {}: {}", 
                     redisKey, ctx_->errstr ? ctx_->errstr : "unknown error");
            continue;
        }
        
        if (reply->type == REDIS_REPLY_STATUS && 
            reply->str && std::string(reply->str) == "OK") {
            results[i] = NcsStatus::OK;
        } else {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to SET key {}", redisKey);
        }
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
        
        if (redisAppendCommand(ctx_.get(), "DEL %s", redisKey.c_str()) != REDIS_OK) {
            LOG_ERROR("[RedisNcsConnector] Failed to append DEL command for key {}", redisKey);
            // Connection is in bad state, force reconnect on next call
            ctx_.reset();
            return results;
        }
    }
    
    for (size_t i = 0; i < keys.size(); ++i) {
        auto reply = getSafeReply();
        
        if (!reply) {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to get reply for key {}: {}", 
                     redisKey, ctx_->errstr ? ctx_->errstr : "unknown error");
            continue;
        }
        
        if (reply->type == REDIS_REPLY_INTEGER) {
            results[i] = NcsStatus::OK;
        } else {
            std::string redisKey = "bucket_" + std::to_string(bucketId_) + "_" + std::to_string(keys[i]);
            LOG_ERROR("[RedisNcsConnector] Failed to DEL key {}", redisKey);
        }
    }
    
    return results;
}

// RedisNcsConnectorCreator implementation
const std::string RedisNcsConnectorCreator::KIND = "redis";

NcsConnector* RedisNcsConnectorCreator::factoryMethod(const NcsDescriptor* descriptor) {
    const nlohmann::json& extras = descriptor->getExtras();
    
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
    ncs::RedisContextPtr ctx(
        redisConnect(host.c_str(), port),
        redisFree
    );
    if (ctx == nullptr || ctx->err) {
        if (ctx) {
            LOG_ERROR("[RedisNcsConnectorCreator] Failed to connect to Redis: {}", ctx->errstr);
        }
        return nullptr;
    }
    
    ncs::RedisReplyPtr reply(
        (redisReply*)redisCommand(ctx.get(), "EXISTS %s", bucketKey.c_str()),
        freeReplyObject
    );
    bool bucketExists = (reply != nullptr && reply->type == REDIS_REPLY_INTEGER && reply->integer == 1);
    
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

#endif // USE_REDIS
