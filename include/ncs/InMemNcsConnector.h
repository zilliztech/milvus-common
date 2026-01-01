#pragma once

#include "ncs/ncs.h"
#include "ncs/InMemoryKV.h"
#include <memory>

namespace milvus {

using std::make_unique;

class InMemNcsConnector : public NcsConnector {
public:
    friend class InMemoryNcsConnectorCreator;
    
    // Interface implementations
    std::vector<NcsStatus> multiGet(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) override;
    std::vector<NcsStatus> multiPut(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) override;
    std::vector<NcsStatus> multiDelete(const std::vector<uint32_t>& keys) override;

private:
    explicit InMemNcsConnector(uint64_t bucketId);  // Private constructor
};

class InMemoryNcsConnectorCreator : public NcsConnectorCreator {
public:
    NcsConnector* factoryMethod(const NcsDescriptor* descriptor) override;
    const std::string& getKind() const override;

private:
    static const std::string KIND;
};

} // namespace milvus