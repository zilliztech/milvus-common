#include "ncs/InMemNcsConnector.h"
#include "log/Log.h"
#include <assert.h>
#include <memory>

namespace milvus {

using std::make_unique;


InMemNcsConnector::InMemNcsConnector(uint64_t bucketId)
: NcsConnector(bucketId) {
    // Constructor performs no blocking checks; factory will validate bucket.
}


std::vector<NcsStatus> InMemNcsConnector::multiGet(const std::vector<uint32_t>& keys, 
                                              const std::vector<SpanBytes>& buffs) {
    std::vector<NcsStatus> results;
    results.reserve(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        bool success = InMemoryKV::Instance()->get(bucketId_, keys[i], buffs[i]);
        results.push_back(success ? NcsStatus::OK : NcsStatus::ERROR);
    }

    return results;
}

std::vector<NcsStatus> InMemNcsConnector::multiPut(const std::vector<uint32_t>& keys, 
                                              const std::vector<SpanBytes>& buffs) {
    std::vector<NcsStatus> results;
    results.reserve(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        bool ok = InMemoryKV::Instance()->put(bucketId_, keys[i], buffs[i]);
        results.push_back(ok ? NcsStatus::OK : NcsStatus::ERROR);
    }

    return results;
}

std::vector<NcsStatus> InMemNcsConnector::multiDelete(const std::vector<uint32_t>& keys) {
    std::vector<NcsStatus> results;
    results.reserve(keys.size());

    for (const auto& key : keys) {
        bool success = InMemoryKV::Instance()->deleteKey(bucketId_, key);
        results.push_back(success ? NcsStatus::OK : NcsStatus::ERROR);
    }
    
    return results;
}

const std::string InMemoryNcsConnectorCreator::KIND = "in_memory";

NcsConnector* InMemoryNcsConnectorCreator::factoryMethod(const NcsDescriptor* descriptor) {
    if (descriptor->getKind() != KIND) {
        LOG_ERROR("[NCS] InMemoryNcsConnectorCreator received incompatible descriptor kind: {}", descriptor->getKind());    
        return nullptr;
    }
    // If bucket doesn't exist in backing KV, fail early and return nullptr.
    if (!InMemoryKV::Instance()->hasBucket(descriptor->getbucketId())) {
        LOG_ERROR("[NCS] InMemNcsConnector creation failed: bucket {} does not exist.", descriptor->getbucketId());
        return nullptr;
    }
    return new InMemNcsConnector(descriptor->getbucketId());
}

const std::string& InMemoryNcsConnectorCreator::getKind() const {
    return KIND;
}

// Register the connector type on startup
namespace {
    struct RegisterInMemConnector {
        RegisterInMemConnector() {
            NcsConnectorFactory::Instance().registerCreator(
                std::make_unique<InMemoryNcsConnectorCreator>());
        }
    } registerInMemConnector;
}

} // namespace milvus