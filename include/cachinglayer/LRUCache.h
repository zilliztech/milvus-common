// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <array>
#include <climits>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/LRUCacheCore.h"

namespace milvus::cachinglayer {

template <typename Key, typename Value>
class LRUCache;

// A move-only pin on a resident Value, independent of the cache's Key type.
// An empty Pin represents a cache miss. Keep the Pin alive while using the Value.
template <typename Value>
class Pin {
    using Core = internal::LRUCacheCore;

 public:
    Pin() = default;
    Pin(const Pin&) = delete;
    Pin&
    operator=(const Pin&) = delete;
    Pin(Pin&&) noexcept = default;
    Pin&
    operator=(Pin&&) noexcept = default;

    explicit
    operator bool() const noexcept {
        return static_cast<bool>(handle_);
    }
    const Value&
    operator*() const {
        return *static_cast<const Value*>(handle_.get());
    }
    const Value*
    operator->() const {
        return static_cast<const Value*>(handle_.get());
    }

 private:
    template <typename Key, typename T>
    friend class LRUCache;
    explicit Pin(Core::PinState handle) : handle_(std::move(handle)) {
    }
    Core::PinState handle_;
};

// Values and keys must not reenter a cache from their destructors. Get never
// loads; recency follows the shared domain's unpin/touch-window policy.
template <typename Key, typename Value>
class LRUCache {
    static_assert(std::is_integral_v<Key> || std::is_same_v<Key, std::string>,
                  "LRUCache currently supports integral and std::string keys");
    using Core = internal::LRUCacheCore;

 public:
    using Result = Core::Result;

    explicit LRUCache(std::shared_ptr<internal::DList> dlist) : core_(std::move(dlist)) {
    }
    LRUCache(const LRUCache&) = delete;
    LRUCache&
    operator=(const LRUCache&) = delete;

    [[nodiscard]] Pin<Value>
    Get(const Key& key) const {
        return WithKey(key, [this](const Core::KeyView& view) { return Pin<Value>(core_.Get(view)); });
    }

    // Transfer ownership only on success, including when the backend throws.
    // bytes covers the Value payload; the core adds key/index/node overhead.
    [[nodiscard]] Result
    Put(const Key& key, std::unique_ptr<Value>& value, const ResourceUsage& bytes) {
        struct Input {
            std::unique_ptr<Value>& source;
            Core::ValuePtr erased;
            ~Input() {
                source.reset(static_cast<Value*>(erased.release()));
            }
        } input{value, Core::ValuePtr(const_cast<std::remove_cv_t<Value>*>(value.release()), [](void* pointer) {
                    // The core only stores the opaque pointer. Restore the exact
                    // Value type and its default_delete access rights on destruction.
                    std::default_delete<Value>{}(static_cast<Value*>(pointer));
                })};
        return WithKey(key, [&](const Core::KeyView& view) { return core_.Put(view, input.erased, bytes); });
    }

    [[nodiscard]] Result
    Erase(const Key& key) {
        return WithKey(key, [this](const Core::KeyView& view) { return core_.Erase(view); });
    }

    [[nodiscard]] size_t
    Size() const {
        return core_.Size();
    }

 private:
    template <typename Fn>
    static auto
    WithKey(const Key& key, Fn&& fn) {
        if constexpr (std::is_same_v<Key, std::string>) {
            return fn(Core::KeyView{std::string_view(key)});
        } else if constexpr (sizeof(Key) <= sizeof(uint64_t)) {
            return fn(Core::KeyView{static_cast<uint64_t>(key)});
        } else {
            // Preserve extended integral keys without narrowing or depending on
            // padding bytes. This view is consumed synchronously by the core.
            std::array<char, sizeof(Key)> encoded{};
            auto value = static_cast<std::make_unsigned_t<Key>>(key);
            for (auto& byte : encoded) {
                byte = static_cast<char>(value);
                value >>= CHAR_BIT;
            }
            return fn(Core::KeyView{std::string_view(encoded.data(), encoded.size())});
        }
    }

    Core core_;
};

}  // namespace milvus::cachinglayer
