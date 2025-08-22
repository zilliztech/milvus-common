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

#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cachinglayer/Manager.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/lrucache/DList.h"

namespace milvus {

using namespace cachinglayer;

namespace cachinglayer::internal {
class DListTestFriend {
 public:
    static ResourceUsage
    get_using_memory(const DList& dlist) {
        return dlist.total_loaded_size_.load() + dlist.total_loading_size_.load();
    }
    static ResourceUsage
    get_used_memory(const DList& dlist) {
        return dlist.total_loaded_size_.load();
    }
    static ResourceUsage
    get_loading_memory(const DList& dlist) {
        return dlist.total_loading_size_.load();
    }
    static ResourceUsage
    get_max_memory(const DList& dlist) {
        std::lock_guard lock(dlist.list_mtx_);
        return dlist.max_resource_limit_;
    }
    static ListNode*
    get_head(const DList& dlist) {
        std::lock_guard lock(dlist.list_mtx_);
        return dlist.head_;
    }
    static ListNode*
    get_tail(const DList& dlist) {
        std::lock_guard lock(dlist.list_mtx_);
        return dlist.tail_;
    }
    static void
    test_push_head(DList* dlist, ListNode* node) {
        std::lock_guard lock(dlist->list_mtx_);
        dlist->pushHead(node);
    }
    static void
    test_pop_item(DList* dlist, ListNode* node) {
        std::lock_guard lock(dlist->list_mtx_);
        dlist->popItem(node);
    }
    static void
    test_add_used_memory(DList* dlist, const ResourceUsage& size) {
        std::lock_guard lock(dlist->list_mtx_);
        dlist->total_loaded_size_ += size;
    }
    static void
    test_add_loading_memory(DList* dlist, const ResourceUsage& size) {
        std::lock_guard lock(dlist->list_mtx_);
        dlist->total_loading_size_ += size;
    }
    static void
    test_sub_loading_memory(DList* dlist, const ResourceUsage& size) {
        std::lock_guard lock(dlist->list_mtx_);
        dlist->total_loading_size_ -= size;
    }
    static void
    test_add_evictable_memory(DList* dlist, const ResourceUsage& size) {
        dlist->evictable_size_ += size;
        // If there are waiters, try to satisfy them
        if (!dlist->waiting_queue_empty_) {
            std::unique_lock<std::mutex> lock(dlist->list_mtx_);
            dlist->notifyWaitingRequests();
        }
    }

    // nodes are from tail to head
    static void
    verify_list(DList* dlist, std::vector<ListNode*> nodes) {
        std::lock_guard lock(dlist->list_mtx_);
        EXPECT_EQ(nodes.front(), dlist->tail_);
        EXPECT_EQ(nodes.back(), dlist->head_);
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto current = nodes[i];
            auto expected_prev = i == 0 ? nullptr : nodes[i - 1];
            auto expected_next = i == nodes.size() - 1 ? nullptr : nodes[i + 1];
            EXPECT_EQ(current->prev_, expected_prev);
            EXPECT_EQ(current->next_, expected_next);
        }
    }

    static void
    verify_integrity(DList* dlist) {
        std::lock_guard lock(dlist->list_mtx_);

        ResourceUsage total_size;
        EXPECT_EQ(dlist->tail_->prev_, nullptr);
        ListNode* current = dlist->tail_;
        ListNode* prev = nullptr;

        while (current != nullptr) {
            EXPECT_EQ(current->prev_, prev);
            total_size += current->loaded_size();
            prev = current;
            current = current->next_;
        }

        EXPECT_EQ(prev, dlist->head_);
        EXPECT_EQ(dlist->head_->next_, nullptr);

        EXPECT_EQ(total_size, dlist->total_loaded_size_.load());
    }
};
}  // namespace cachinglayer::internal

}  // namespace milvus
