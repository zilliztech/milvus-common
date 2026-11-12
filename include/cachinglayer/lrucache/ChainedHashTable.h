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
#include <cassert>
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

struct ChainedHashTableConfig {
    uint32_t buckets_power{10};  // 1024 buckets; no automatic rehash.
    uint32_t locks_power{6};     // 64 mutexes shared by the buckets.
};

template <typename T>
struct ChainedHashTableHook {
    std::shared_ptr<T> next;
    const size_t hash;  // Raw key hash; bucket selection applies the mixing.
};

// Hash table with fixed power-of-two buckets, intrusive chains and striped mutexes.
// Lookups return shared ownership so nodes remain alive after the index lock is
// released. Index methods never acquire node or LRU locks.
template <typename Key, typename T, ChainedHashTableHook<T> T::*HookPtr>
class ChainedHashTable {
 public:
    class Bucket {
     public:
        const size_t stripe;

        std::mutex&
        mutex() const noexcept {
            return table_.locks_[stripe];
        }

        // Caller holds mutex() and looks up a key with this bucket view's raw hash.
        std::shared_ptr<T>
        findLocked(const Key& key) const {
            const auto* link = &table_.buckets_[index_];
            while (*link) {
                const auto& hook = (link->get()->*HookPtr);
                if (hook.hash == hash_ && (*link)->getKey() == key) {
                    return *link;
                }
                link = &hook.next;
            }
            return {};
        }

     private:
        friend class ChainedHashTable;
        Bucket(const ChainedHashTable& table, size_t index, size_t hash) noexcept
            : stripe(index & table.locks_mask_), table_(table), index_(index), hash_(hash) {
        }
        const ChainedHashTable& table_;
        const size_t index_;
        const size_t hash_;
    };

    explicit ChainedHashTable(const ChainedHashTableConfig config) {
        if (config.buckets_power >= std::numeric_limits<size_t>::digits || config.locks_power > config.buckets_power) {
            throw std::invalid_argument("Invalid ChainedHashTable bucket/lock powers");
        }
        buckets_.resize(size_t{1} << config.buckets_power);
        const auto num_locks = size_t{1} << config.locks_power;
        locks_ = std::make_unique<std::mutex[]>(num_locks);
        locks_mask_ = num_locks - 1;
    }

    ~ChainedHashTable() {
        // Break chains iteratively, including if a caller retains a node.
        // Destruction, as with standard containers, requires exclusive access.
        for (auto& head : buckets_) {
            while (head) {
                auto node = std::move(head);
                head = std::move((node.get()->*HookPtr).next);
            }
        }
    }
    ChainedHashTable(const ChainedHashTable&) = delete;
    ChainedHashTable&
    operator=(const ChainedHashTable&) = delete;

    [[nodiscard]] Bucket
    getBucket(const size_t hash) const noexcept {
        // Mix high bits into low bits before masking, including for aligned offsets.
        return Bucket(*this, folly::hash::twang_mix64(hash) & (buckets_.size() - 1), hash);
    }

    std::shared_ptr<T>
    find(const Key& key) const {
        const auto bucket = getBucket(std::hash<Key>{}(key));
        std::lock_guard lock(bucket.mutex());
        return bucket.findLocked(key);
    }

    // Caller established absence under bucket.mutex() and still holds it.
    void
    prependLocked(const Bucket& bucket, const std::shared_ptr<T>& node) noexcept {
        auto& hook = node.get()->*HookPtr;
        assert(&bucket.table_ == this && bucket.hash_ == hook.hash);
        auto& head = buckets_[bucket.index_];
        hook.next = std::move(head);
        head = node;
        num_keys_.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] std::shared_ptr<T>
    insertIfAbsent(const std::shared_ptr<T>& node) {
        const auto bucket = getBucket((node.get()->*HookPtr).hash);
        std::lock_guard lock(bucket.mutex());
        if (auto existing = bucket.findLocked(node->getKey())) {
            return existing;
        }
        prependLocked(bucket, node);
        return node;
    }

    std::shared_ptr<T>
    remove(T& node) {
        auto& hook = node.*HookPtr;
        const auto bucket = getBucket(hook.hash);
        std::lock_guard lock(bucket.mutex());
        auto* link = &buckets_[bucket.index_];
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

    size_t
    getNumStripes() const noexcept {
        return locks_mask_ + 1;
    }

 private:
    std::vector<std::shared_ptr<T>> buckets_;
    std::unique_ptr<std::mutex[]> locks_;
    size_t locks_mask_;
    std::atomic<size_t> num_keys_{0};
};

}  // namespace milvus::cachinglayer::internal
