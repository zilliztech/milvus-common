// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <thread>
#include <type_traits>
#include <vector>

#include "common/OpContext.h"

using milvus::OpContext;

static_assert(!std::is_copy_constructible_v<OpContext>);
static_assert(!std::is_move_constructible_v<OpContext>);
static_assert(std::is_const_v<decltype(OpContext::sequence_id)>);
static_assert(std::is_trivially_copyable_v<OpContext::InheritSequence>);

TEST(OpContext, RootsAllocateIncreasingSequences) {
    const OpContext first;
    folly::CancellationSource cancel;
    const OpContext second(cancel.getToken());
    const OpContext third;
    EXPECT_GT(first.sequence_id, 0);
    EXPECT_GT(second.sequence_id, first.sequence_id);
    EXPECT_GT(third.sequence_id, second.sequence_id);
    cancel.requestCancellation();
    EXPECT_TRUE(second.cancellation_token.isCancellationRequested());
}

TEST(OpContext, ConcurrentRootsHaveDistinctSequences) {
    constexpr size_t threads = 8;
    constexpr size_t per_thread = 1000;
    std::array<uint64_t, threads * per_thread> ids{};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (size_t i = 0; i < per_thread; ++i) {
                ids[t * per_thread + i] = OpContext{}.sequence_id;
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (size_t t = 0; t < threads; ++t) {
        EXPECT_TRUE(std::is_sorted(ids.begin() + t * per_thread, ids.begin() + (t + 1) * per_thread));
    }
    std::sort(ids.begin(), ids.end());
    EXPECT_GT(ids.front(), 0);
    EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
}

TEST(OpContext, ChildrenInheritOnlySequence) {
    folly::CancellationSource parent_cancel;
    OpContext parent(parent_cancel.getToken());
    parent.build_id = 42;
    parent.runtime_load_priority = 1;
    parent.coload_fields = {7};
    parent.storage_usage.scanned_total_bytes = 100;
    parent_cancel.requestCancellation();

    const OpContext child(OpContext::InheritSequence{parent});
    const OpContext sibling(OpContext::InheritSequence{parent});
    const OpContext grandchild(OpContext::InheritSequence{child});
    EXPECT_EQ(child.sequence_id, parent.sequence_id);
    EXPECT_EQ(sibling.sequence_id, parent.sequence_id);
    EXPECT_EQ(grandchild.sequence_id, parent.sequence_id);
    EXPECT_FALSE(child.cancellation_token.canBeCancelled());
    EXPECT_EQ(child.build_id, 0);
    EXPECT_FALSE(child.runtime_load_priority.has_value());
    EXPECT_TRUE(child.coload_fields.empty());
    EXPECT_EQ(child.storage_usage.scanned_total_bytes.load(), 0);
}

TEST(OpContext, InheritanceSurvivesParentDestruction) {
    uint64_t parent_id = 0;
    const auto inherited = [&] {
        const OpContext parent;
        parent_id = parent.sequence_id;
        return OpContext::InheritSequence{parent};
    }();
    const OpContext unrelated;
    const OpContext child(inherited);
    EXPECT_EQ(child.sequence_id, parent_id);
    EXPECT_LT(child.sequence_id, unrelated.sequence_id);
}

TEST(OpContext, ChildUsesExplicitCancellationToken) {
    const OpContext parent;
    folly::CancellationSource cancel;
    const OpContext child(OpContext::InheritSequence{parent}, cancel.getToken());
    cancel.requestCancellation();
    EXPECT_EQ(child.sequence_id, parent.sequence_id);
    EXPECT_TRUE(child.cancellation_token.isCancellationRequested());
    EXPECT_FALSE(parent.cancellation_token.isCancellationRequested());
}
