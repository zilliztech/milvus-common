#pragma once

#include "ncs/ncs.h"
#include "log/Log.h"
#include <memory>
#include <string>
#include <hiredis/hiredis.h>

namespace milvus {

class RedisNcsConnector : public NcsConnector {
public:
    ~RedisNcsConnector() override;
    std::vector<NcsStatus> multiGet(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) override;
    std::vector<NcsStatus> multiPut(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) override;
    std::vector<NcsStatus> multiDelete(const std::vector<uint32_t>& keys) override;

private:
    explicit RedisNcsConnector(uint64_t bucketId, const std::string& host, int port);
    redisContext* context_;
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
