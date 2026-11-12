/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (C) 2026 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <folly/hash/Hash.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace milvus::cachinglayer::internal {

// Hash table with fixed power-of-two buckets, intrusive chains and striped mutexes.
// Lookups return shared ownership so nodes remain alive after the index lock is
// released. Index methods never acquire node or LRU locks.
class ChainedHashTable {
 public:
    struct Config {
        uint32_t buckets_power{10};  // 1024 buckets; no automatic rehash.
        uint32_t locks_power{6};     // 64 mutexes shared by the buckets.
    };

    template <typename T>
    struct Hook {
        std::shared_ptr<T> next;
        const size_t hash;  // Raw key hash; bucket selection applies the mixing.
    };

    template <typename Key, typename T, Hook<T> T::*HookPtr>
    class Container {
     public:
        explicit Container(const Config config) {
            if (config.buckets_power >= std::numeric_limits<size_t>::digits ||
                config.locks_power > config.buckets_power) {
                throw std::invalid_argument("Invalid ChainedHashTable bucket/lock powers");
            }
            buckets_.resize(size_t{1} << config.buckets_power);
            const auto num_locks = size_t{1} << config.locks_power;
            locks_ = std::make_unique<std::mutex[]>(num_locks);
            locks_mask_ = num_locks - 1;
        }

        ~Container() {
            // Break chains iteratively, including if a caller retains a node.
            // Destruction, as with standard containers, requires exclusive access.
            for (auto& head : buckets_) {
                while (head) {
                    auto node = std::move(head);
                    head = std::move((node.get()->*HookPtr).next);
                }
            }
        }
        Container(const Container&) = delete;
        Container&
        operator=(const Container&) = delete;

        std::shared_ptr<T>
        find(const Key& key) const {
            const auto hash = std::hash<Key>{}(key);
            const auto bucket = getBucket(hash);
            std::lock_guard lock(locks_[bucket & locks_mask_]);
            const auto* link = &buckets_[bucket];
            while (*link) {
                const auto& hook = (link->get()->*HookPtr);
                if (hook.hash == hash && (*link)->getKey() == key) {
                    // Only retain the wrapper here. Pinning/value access happens
                    // after releasing this lock, under the node's own lock.
                    return *link;
                }
                link = &hook.next;
            }
            return {};
        }

        // The caller must establish key absence and serialize insertions until
        // this operation completes. The node must not already be linked.
        void
        prepend(const std::shared_ptr<T>& node) {
            auto& hook = (node.get()->*HookPtr);
            const auto bucket = getBucket(hook.hash);
            std::lock_guard lock(locks_[bucket & locks_mask_]);
            hook.next = std::move(buckets_[bucket]);
            buckets_[bucket] = node;
            num_keys_.fetch_add(1, std::memory_order_relaxed);
        }

        std::shared_ptr<T>
        remove(T& node) {
            auto& hook = node.*HookPtr;
            const auto bucket = getBucket(hook.hash);
            std::lock_guard lock(locks_[bucket & locks_mask_]);
            auto* link = &buckets_[bucket];
            // Cleanup uses cached hash and pointer identity, so user Hash/Equal
            // cannot throw during eviction or remove a replacement for this key.
            while (*link && link->get() != &node) {
                link = &((link->get()->*HookPtr).next);
            }
            if (!*link) {
                return {};
            }
            auto result = std::move(*link);
            *link = std::move(hook.next);
            num_keys_.fetch_sub(1, std::memory_order_relaxed);
            return result;
        }

        // Used by cache shutdown: retain the head before waiting on its node lock.
        std::shared_ptr<T>
        getBucketHead(const size_t bucket) const {
            std::lock_guard lock(locks_[bucket & locks_mask_]);
            return buckets_[bucket];
        }

        size_t
        size() const {
            return num_keys_.load(std::memory_order_relaxed);
        }

        size_t
        getNumBuckets() const {
            return buckets_.size();
        }

     private:
        size_t
        getBucket(const size_t hash) const noexcept {
            // Mix high bits into low bits before masking: integer hashers may
            // return aligned offsets unchanged. All index operations use this path.
            return folly::hash::twang_mix64(hash) & (buckets_.size() - 1);
        }

        std::vector<std::shared_ptr<T>> buckets_;
        std::unique_ptr<std::mutex[]> locks_;
        size_t locks_mask_;
        std::atomic<size_t> num_keys_{0};
    };
};

}  // namespace milvus::cachinglayer::internal
