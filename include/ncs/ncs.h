#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include "nlohmann/json.hpp"

#include "common/SpanBytes.h"


namespace milvus {

using std::string;
using std::vector;
using std::unique_ptr;
using std::unordered_map;
using json = nlohmann::json;

enum class NcsStatus {
    OK,
    ERROR
};

class NcsDescriptor {
public: 
    NcsDescriptor(const std::string& ncsKind, uint64_t bucketId, const json& extras);
    virtual ~NcsDescriptor() = default;
    const std::string& getKind() const;
    uint64_t getbucketId() const;
    const json& getExtras() const { return extras_; }
    
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(NcsDescriptor, ncsKind_, bucketId_, extras_)
    NcsDescriptor() = default;
private:
    std::string ncsKind_;
    uint64_t bucketId_;
    json extras_;
};

class NcsBucketStatus{
    // TBD (capacity, occupancy, etc)
};


class Ncs{
public:
    virtual NcsStatus createBucket(uint64_t bucketId) = 0;
    virtual NcsStatus deleteBucket(uint64_t bucketId) = 0;
    virtual NcsBucketStatus getBucketNcsStatus(uint64_t bucketId) = 0;
    virtual bool isBucketExist(uint64_t bucketId) = 0;
    virtual ~Ncs() = default;
protected:
    Ncs() = default;
};

class NcsFactory {
public:
    virtual std::unique_ptr<Ncs> createNcs(const json& params = json{}) = 0;
    virtual const std::string& getKind() const = 0;
    virtual ~NcsFactory() = default;
};

class NcsFactoryRegistry {
public:
    static NcsFactoryRegistry& Instance();
    void registerFactory(std::unique_ptr<NcsFactory> factory);
    std::unique_ptr<Ncs> createNcs(const std::string& kind, const json& params = json{});
    bool hasKind(const std::string& kind) const;
    ~NcsFactoryRegistry() = default;

private:
    NcsFactoryRegistry() = default;
    std::unordered_map<std::string, std::unique_ptr<NcsFactory>> registry_;
};

class NcsSingleton final{
public:
    static void initNcs(const std::string& kind, const json& extras = json{});
    static Ncs* Instance();
    
private:
    inline static std::string kind_;
    inline static json extras_;
    inline static std::unique_ptr<Ncs> instance_ = nullptr;
};


class NcsConnector {
public:
    virtual ~NcsConnector() = default;
    virtual std::vector<NcsStatus> multiGet(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) = 0;
    virtual std::vector<NcsStatus> multiPut(const std::vector<uint32_t>& keys, const std::vector<SpanBytes>& buffs) = 0;
    virtual std::vector<NcsStatus> multiDelete(const std::vector<uint32_t>& keys) = 0;

protected:
    explicit NcsConnector(uint64_t bucketId) : bucketId_(bucketId) {}
    const uint64_t bucketId_;
};

class NcsConnectorCreator {
public:
    virtual ~NcsConnectorCreator() = default;
    virtual NcsConnector* factoryMethod(const NcsDescriptor* descriptor) = 0;
    virtual const std::string& getKind() const = 0;
};



class NcsConnectorFactory {
public:
    static NcsConnectorFactory& Instance();
    NcsConnector* createConnector(const NcsDescriptor* descriptor);
    void registerCreator(std::unique_ptr<NcsConnectorCreator> creator);
    ~NcsConnectorFactory() = default;  

private:
    NcsConnectorFactory() = default;  
    std::unordered_map<std::string, std::unique_ptr<NcsConnectorCreator>> registry_;
};
 
} // namespace milvus