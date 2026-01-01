#include "ncs/ncs.h"
#include <cstdint>
#include <numeric>
#include "log/Log.h"

namespace milvus {

// NcsFactoryRegistry implementation
NcsFactoryRegistry& NcsFactoryRegistry::Instance() {
    static NcsFactoryRegistry instance;
    return instance;
}

void NcsFactoryRegistry::registerFactory(std::unique_ptr<NcsFactory> factory) {
    auto kind = factory->getKind();
    registry_[kind] = std::move(factory);
}

std::unique_ptr<Ncs> NcsFactoryRegistry::createNcs(const std::string& kind, const json& params) {
    LOG_DEBUG("[NCS] Creating NCS of kind: {} with params: {}", kind, params.dump());
    auto it = registry_.find(kind);
    if (it != registry_.end()) {
        return it->second->createNcs(params);
    }
    
    std::string registered_kinds_str = 
        std::accumulate(registry_.begin(), registry_.end(),
            std::string(),
            [](const std::string& a, const auto& b) {
                return a.empty() ? b.first : a + ", " + b.first;
            });
    LOG_WARN("[NCS] No registered factory for NCS kind: {}. Registered kinds: {}", kind, registered_kinds_str);
    return nullptr;
}

bool NcsFactoryRegistry::hasKind(const std::string& kind) const {
    return registry_.find(kind) != registry_.end();
}

// NcsSingleton implementation
void NcsSingleton::initNcs(const std::string& kind, const json& extras) {
    if (!NcsFactoryRegistry::Instance().hasKind(kind)) {
        throw std::runtime_error("NCS Factory kind '" + kind + "' is not registered.");
    }
    kind_ = kind;
    extras_ = extras;
}

Ncs* NcsSingleton::Instance() {
    if(kind_.empty()){
        throw std::runtime_error("NCS Factory have not been set yet. Use initNcs() first.");
    }
    if(!instance_){
        instance_ = NcsFactoryRegistry::Instance().createNcs(kind_, extras_);
        if (!instance_) {
            throw std::runtime_error("NCS Factory kind '" + kind_ + "' is not registered.");
        }
    }
    return instance_.get();
}

// NcsConnectorFactory implementation
NcsConnector* NcsConnectorFactory::createConnector(const NcsDescriptor* descriptor) {
    LOG_DEBUG("[NCS] Creating NcsConnector of kind: {} with extra params: {}", descriptor->getKind(), descriptor->getExtras().dump());
    auto it = registry_.find(descriptor->getKind());
    if (it != registry_.end()) {
        return it->second->factoryMethod(descriptor);
    }

    std::string registered_kinds_str = 
        std::accumulate(registry_.begin(), registry_.end(),
            std::string(),
            [](const std::string& a, const auto& b) {
                return a.empty() ? b.first : a + ", " + b.first;
            });
    LOG_WARN("[NCS] No registered creator for NCS kind: {}. registered kinds: {}", descriptor->getKind(), registered_kinds_str);
    return nullptr;
}

void NcsConnectorFactory::registerCreator(std::unique_ptr<NcsConnectorCreator> creator) {
    auto kind = creator->getKind();
    registry_[kind] = std::move(creator);
}

// NcsConnectorFactory singleton implementation
NcsConnectorFactory& NcsConnectorFactory::Instance() {
    static NcsConnectorFactory instance;
    return instance;
}

// Already defaulted in header

// NcsDescriptor implementation
NcsDescriptor::NcsDescriptor(const std::string& ncsKind, uint64_t bucketId, const json& extras)
    : ncsKind_(ncsKind), bucketId_(bucketId), extras_(extras) {
}

const std::string& NcsDescriptor::getKind() const {
    return ncsKind_;
}

uint64_t NcsDescriptor::getbucketId() const {
    return bucketId_;
}

// NcsConnector constructor is already defined in header

// Already defined above

} // namespace milvus