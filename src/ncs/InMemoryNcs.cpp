#include "ncs/InMemoryNcs.h"
#include <memory>

namespace milvus {

// InMemoryNcs implementation
NcsStatus InMemoryNcs::createBucket(uint64_t bucketId) {
    bool ok = InMemoryKV::Instance()->createBucket(bucketId);
    return ok ? NcsStatus::OK : NcsStatus::ERROR;
}

NcsStatus InMemoryNcs::deleteBucket(uint64_t bucketId) {
    bool ok = InMemoryKV::Instance()->deleteBucket(bucketId);
    return ok ? NcsStatus::OK : NcsStatus::ERROR;
}

NcsBucketStatus InMemoryNcs::getBucketNcsStatus(uint64_t /*bucketId*/) {
    return NcsBucketStatus();
}

bool InMemoryNcs::isBucketExist(uint64_t bucketId) {
    return InMemoryKV::Instance()->hasBucket(bucketId);
}

// InMemoryNcsFactory implementation
const std::string InMemoryNcsFactory::KIND = "in_memory";

std::unique_ptr<Ncs> InMemoryNcsFactory::createNcs(const json& params) {
    return std::unique_ptr<Ncs>(new InMemoryNcs());
}

const std::string& InMemoryNcsFactory::getKind() const {
    return KIND;
}

// Register the factory on startup
namespace {
    struct RegisterInMemoryNcsFactory {
        RegisterInMemoryNcsFactory() {
            NcsFactoryRegistry::Instance().registerFactory(
                std::make_unique<InMemoryNcsFactory>());
        }
    } registerInMemoryNcsFactory;
}

} // namespace milvus
