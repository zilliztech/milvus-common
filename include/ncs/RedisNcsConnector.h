#pragma once

#include "ncs/ncs.h"
#include "log/Log.h"
#include <memory>
#include <string>
#include <vector>
#include <hiredis/hiredis.h>

namespace milvus {

/**
 * @brief Redis-based NCS connector with single connection.
 * 
 * This connector uses a single Redis connection and is NOT thread-safe.
 * Each thread should have its own connector instance.
 * 
 * Thread-safety is achieved at a higher level (e.g., NCSReader) by using
 * thread_local connectors, one per thread.
 * 
 * Uses Redis pipelining for efficient batch operations.
 */
class RedisNcsConnector : public NcsConnector {
public:
    ~RedisNcsConnector() override;
    std::vector<NcsStatus> multiGet(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) override;
    std::vector<NcsStatus> multiPut(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) override;
    std::vector<NcsStatus> multiDelete(const std::vector<uint32_t>& keys) override;

private:
    explicit RedisNcsConnector(uint64_t bucketId, const std::string& host, int port);
    
    /**
     * @brief Ensure connection is valid, reconnect if needed.
     * @return true if connection is valid, false otherwise.
     */
    bool ensureConnected();
    
    redisContext* ctx_ = nullptr;
    std::string host_;
    int port_;
    
    friend class RedisNcsConnectorCreator;
};

class RedisNcsConnectorCreator : public NcsConnectorCreator {
public:
    static const std::string KIND;
    NcsConnector* factoryMethod(const NcsDescriptor* descriptor) override;
    const std::string& getKind() const override;
};

} // namespace milvus
