// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <folly/Expected.h>
#include <folly/Unit.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <variant>

namespace milvus::cachinglayer {

struct ResourceUsage;

enum class LRUCacheError { Pinned, NotFound, InsufficientResource, InvalidArgument, Conflict, Closed };

namespace internal {
class DList;

// The index, nodes, synchronization and accounting are defined in LRUCache.cpp.
class LRUCacheCore {
    struct CacheState;
    struct Entry;

 public:
    using Result = folly::Expected<folly::Unit, LRUCacheError>;
    using KeyView = std::variant<uint64_t, std::string_view>;
    using ValuePtr = std::unique_ptr<void, void (*)(void*)>;

    class PinState {
     public:
        PinState();
        PinState(const PinState&) = delete;
        PinState&
        operator=(const PinState&) = delete;
        PinState(PinState&& other) noexcept;
        PinState&
        operator=(PinState&& other) noexcept;
        ~PinState();

        explicit
        operator bool() const noexcept;
        const void*
        get() const;

     private:
        friend class LRUCacheCore;
        PinState(std::shared_ptr<CacheState> state, Entry* entry);
        std::shared_ptr<CacheState> state_;
        Entry* entry_{nullptr};
    };

    explicit LRUCacheCore(std::shared_ptr<DList> dlist);
    LRUCacheCore(const LRUCacheCore&) = delete;
    LRUCacheCore&
    operator=(const LRUCacheCore&) = delete;
    ~LRUCacheCore();

    [[nodiscard]] PinState
    Get(const KeyView& key) const;
    [[nodiscard]] Result
    Put(const KeyView& key, ValuePtr& value, const ResourceUsage& bytes);
    [[nodiscard]] Result
    Erase(const KeyView& key);
    [[nodiscard]] size_t
    Size() const;

 private:
    const std::shared_ptr<CacheState> state_;
};
}  // namespace internal

}  // namespace milvus::cachinglayer
