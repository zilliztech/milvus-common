#pragma once

/**
 * @file ncs.h
 * @brief Near Compute Storage (NCS) abstraction layer.
 * 
 * This header defines the interfaces for NCS implementations (InMemory, Redis).
 * All implementations must adhere to the contracts specified in this file.
 * 
 * ## Implementation Requirements
 * 
 * ### Thread Safety - Single-Threaded Connector Model
 * 
 * **NcsConnector instances are NOT required to be thread-safe.**
 * 
 * Concurrency is handled at a higher level by the consumer (e.g., NCSReader in DiskANN).
 * Each thread should create and use its own NcsConnector instance via thread_local storage.
 * 
 * #### Why this design?
 * - Simplifies NcsConnector implementations (no need for connection pools or mutexes)
 * - Eliminates lock contention in high-throughput scenarios
 * - Natural isolation - each thread has its own connection to the backend
 * - Backend connections (for example: Redis) are typically not thread-safe anyway
 * 
 * #### Concurrency Model (handled by NCSReader):
 * ```
 * Thread 1 ──► NcsConnector instance 1 ──►┐
 * Thread 2 ──► NcsConnector instance 2 ──►├──► NCS Backend (for example: Redis)
 * Thread 3 ──► NcsConnector instance 3 ──►┘
 * ```
 * 
 * #### Consumer Responsibilities:
 * - NCSReader uses `thread_local` storage to create one connector per thread
 * - Each thread's connector is created lazily on first access
 * - Connectors are destroyed when the NCSReader is destroyed (for current thread)
 *   or when threads exit
 * 
 * #### Implementation Note for Backends:
 * - **Redis**: Each connector holds a single `redisContext*` (not thread-safe)
 * - **InMemory**: Uses shared unordered_map; put/delete operations are NOT thread-safe
 *   (acceptable if data is populated before concurrent reads begin)
 * 
 * ### Bucket Management
 * - Buckets must be created via `Ncs::createBucket()` before any operations.
 * - `NcsConnectorCreator::factoryMethod()` MUST return `nullptr` if the bucket does not exist.
 *   Implementations should query the backend directly to verify bucket existence.
 * 
 * ### Error Handling
 * - `multiGet` returns `NcsStatus::ERROR` for keys that don't exist or if buffer is too small.
 * - `multiPut` returns `NcsStatus::ERROR` if the write operation fails.
 * - `multiDelete` returns `NcsStatus::OK` even if the key doesn't exist (idempotent delete).
 * 
 * ### Performance Guidelines
 * - Implementations should use batching/pipelining for `multi*` operations when possible.
 * - Redis: Use `redisAppendCommand()` + `redisGetReply()` for pipelining.
 * - Avoid sequential operations in loops; prefer bulk commands supported by the backend.
 */

#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include "nlohmann/json.hpp"

#include <boost/core/span.hpp>


namespace milvus {

enum class NcsStatus {
    OK,
    ERROR
};

class NcsDescriptor {
public: 
    NcsDescriptor(const std::string& ncsKind, uint64_t bucketId, const nlohmann::json& extras);
    virtual ~NcsDescriptor() = default;
    const std::string& getKind() const;
    uint64_t getbucketId() const;
    const nlohmann::json& getExtras() const { return extras_; }
    
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(NcsDescriptor, ncsKind_, bucketId_, extras_)
    NcsDescriptor() = default;
private:
    std::string ncsKind_;
    uint64_t bucketId_ = 0;
    nlohmann::json extras_;
};

class NcsBucketStatus{
    // TBD (capacity, occupancy, etc)
};


/**
 * @brief Abstract interface for NCS bucket management.
 * 
 * Provides bucket lifecycle operations. Each NCS implementation (InMemory, Redis)
 * must implement this interface.
 * 
 * Thread Safety: Implementations must ensure thread-safe bucket operations.
 */
class Ncs{
public:
    /**
     * @brief Create a new bucket.
     * @param bucketId Unique identifier for the bucket.
     * @return NcsStatus::OK on success, NcsStatus::ERROR on failure.
     */
    virtual NcsStatus createBucket(uint64_t bucketId) = 0;
    
    /**
     * @brief Delete a bucket and all its contents.
     * @param bucketId The bucket to delete.
     * @return NcsStatus::OK on success (or if bucket doesn't exist), NcsStatus::ERROR on failure.
     */
    virtual NcsStatus deleteBucket(uint64_t bucketId) = 0;
    
    /**
     * @brief Get status information about a bucket.
     * @param bucketId The bucket to query.
     * @return Bucket status information.
     */
    virtual NcsBucketStatus getBucketNcsStatus(uint64_t bucketId) = 0;
    
    /**
     * @brief Check if a bucket exists.
     * @param bucketId The bucket to check.
     * @return true if bucket exists, false otherwise.
     */
    virtual bool isBucketExist(uint64_t bucketId) = 0;
    
    virtual ~Ncs() = default;
protected:
    Ncs() = default;
};

class NcsFactory {
public:
    virtual std::unique_ptr<Ncs> createNcs(const nlohmann::json& params = nlohmann::json{}) = 0;
    virtual const std::string& getKind() const = 0;
    virtual ~NcsFactory() = default;
};

class NcsFactoryRegistry {
public:
    static NcsFactoryRegistry& Instance();
    void registerFactory(std::unique_ptr<NcsFactory> factory);
    std::unique_ptr<Ncs> createNcs(const std::string& kind, const nlohmann::json& params = nlohmann::json{});
    bool hasKind(const std::string& kind) const;
    ~NcsFactoryRegistry() = default;

private:
    NcsFactoryRegistry() = default;
    std::unordered_map<std::string, std::unique_ptr<NcsFactory>> registry_;
};

class NcsSingleton final{
public:
    static void initNcs(const std::string& kind, const nlohmann::json& extras = nlohmann::json{});
    static Ncs* Instance();
    
    /**
     * @brief Reset the singleton instance. For testing purposes only.
     * 
     * This destroys the current NCS instance and clears the configuration,
     * allowing a new NCS type to be initialized via initNcs().
     * 
     * WARNING: Only use in test code. Do not use in production.
     */
    static void reset() {
        instance_.reset();
        kind_.clear();
        extras_.clear();
    }
    
private:
    inline static std::string kind_;
    inline static nlohmann::json extras_;
    inline static std::unique_ptr<Ncs> instance_ = nullptr;
};


/**
 * @brief Abstract interface for NCS data operations within a bucket.
 * 
 * Provides key-value operations (get, put, delete) for a specific bucket.
 * 
 * ## Thread Safety
 * NcsConnector instances are NOT required to be thread-safe.
 * 
 * ## Performance Requirements
 * Implementations SHOULD use batching/pipelining for multi* operations:
 * - Redis: Use pipelining instead of sequential GET/SET commands
 */
class NcsConnector {
public:
    virtual ~NcsConnector() = default;
    
    // Delete copy and move operations - connectors are not copyable/movable
    NcsConnector(const NcsConnector&) = delete;
    NcsConnector& operator=(const NcsConnector&) = delete;
    NcsConnector(NcsConnector&&) = delete;
    NcsConnector& operator=(NcsConnector&&) = delete;
    
    /**
     * @brief Read multiple key-value pairs.
     * @param keys Vector of keys to read.
     * @param buffs Vector of buffers to receive the values. Must be same size as keys.
     *              Each buffer must be large enough to hold the corresponding value.
     * @return Vector of status codes, one per key:
     *         - NcsStatus::OK if read succeeded
     *         - NcsStatus::ERROR if key doesn't exist or buffer too small
     */
    virtual std::vector<NcsStatus> multiGet(const std::vector<uint32_t>& keys, const std::vector<boost::span<uint8_t>>& buffs) = 0;
    
    /**
     * @brief Write multiple key-value pairs.
     * @param keys Vector of keys to write.
     * @param buffs Vector of buffers containing the values. Must be same size as keys.
     * @return Vector of status codes, one per key:
     *         - NcsStatus::OK if write succeeded
     *         - NcsStatus::ERROR if write failed
     * 
     * Note: If a key already exists, its value is overwritten.
     */
    virtual std::vector<NcsStatus> multiPut(const std::vector<uint32_t>& keys, const std::vector<boost::span<uint8_t>>& buffs) = 0;
    
    /**
     * @brief Delete multiple keys.
     * @param keys Vector of keys to delete.
     * @return Vector of status codes, one per key:
     *         - NcsStatus::OK if delete succeeded (including if key didn't exist)
     *         - NcsStatus::ERROR if delete operation failed
     * 
     * Note: Deleting a non-existent key is not an error (idempotent).
     */
    virtual std::vector<NcsStatus> multiDelete(const std::vector<uint32_t>& keys) = 0;

protected:
    explicit NcsConnector(uint64_t bucketId) : bucketId_(bucketId) {}
    const uint64_t bucketId_;
};

/**
 * @brief Factory interface for creating NcsConnector instances.
 * 
 * Each NCS implementation must provide a concrete NcsConnectorCreator.
 * 
 * ## Implementation Requirements
 * - `factoryMethod()` MUST check bucket existence before creating a connector.
 * - If the bucket does not exist, `factoryMethod()` MUST return `nullptr`.
 * - Bucket existence check MUST query the backend directly (e.g., Redis EXISTS)
 *   rather than using NcsSingleton, to avoid coupling.
 */
class NcsConnectorCreator {
public:
    virtual ~NcsConnectorCreator() = default;
    
    /**
     * @brief Create an NcsConnector for the given descriptor.
     * @param descriptor Contains bucket ID, NCS kind, and configuration.
     * @return Pointer to new NcsConnector, or nullptr if:
     *         - The bucket does not exist
     *         - Required configuration is missing
     *         - Connection to backend fails
     * 
     * Note: Caller takes ownership of the returned pointer.
     */
    virtual NcsConnector* factoryMethod(const NcsDescriptor* descriptor) = 0;
    
    /**
     * @brief Get the NCS kind this creator handles (e.g., "redis", "in_memory").
     */
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