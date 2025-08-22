// Copyright (C) 2019-2025 Zilliz. All rights reserved.
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

#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include <atomic>
#include <chrono>
#include <memory>

#include "cachinglayer/Utils.h"
#include "common/EasyAssert.h"

namespace milvus::cachinglayer::internal {

class DList;

// ListNode is not movable/copyable.
class ListNode {
 public:
    // RAII class to unpin the node.
    class NodePin {
     public:
        // NodePin is movable but not copyable.
        NodePin(NodePin&&) noexcept;
        NodePin&
        operator=(NodePin&&) noexcept;
        NodePin(const NodePin&) = delete;
        NodePin&
        operator=(const NodePin&) = delete;
        ~NodePin();

     private:
        explicit NodePin(ListNode* node);
        friend class ListNode;
        ListNode* node_;
    };
    ListNode() = default;
    ListNode(DList* dlist, bool evictable);
    virtual ~ListNode();

    // ListNode is not movable/copyable because it contains a shared_mutex.
    // ListNode also should not be movable/copyable because that would make
    // all NodePin::node_ dangling pointers.
    ListNode(const ListNode&) = delete;
    ListNode&
    operator=(const ListNode&) = delete;
    ListNode(ListNode&&) = delete;
    ListNode&
    operator=(ListNode&&) = delete;

    // bool in return value: whether the caller needs to load this cell.
    // - If the cell is already loaded, return false and an immediately ready future with a NodePin, the node is pinned
    //   upon return.
    // - If the cell is in error state, return false and an immediately ready future with an exception.
    // - If the cell is already being loaded by another thread, return false and a future that will be ready when the
    //   cell is loaded. The node will not be pinned until the future is ready.
    // - Otherwise, the cell is not loaded and not being loaded, return true and a future that will be ready when the
    //   cell is loaded. The caller needs to load this cell and call mark_loaded() to set the cell as loaded.
    //   The node will not be pinned until the future is ready.
    std::pair<bool, folly::SemiFuture<NodePin>>
    pin();

    const ResourceUsage&
    loaded_size() const;

    // Manually evicts the cell if it is not pinned.
    // Returns true if the cell ends up in a state other than LOADED.
    bool
    manual_evict();

    // State transition diagram:
    // +------------+           +---------+
    // | NOT_LOADED | <-------> | LOADING |
    // +------------+           +---------+
    //      ^   ^                   |
    //      |   |                   v
    //      |   |               +---------+
    //      |   +-------------- | LOADED  |
    //      |                   +---------+
    //      |                        |
    //      |                        v
    //      |                 +---------------------------+
    //      +---------------> | CACHED && pin_count == 0  |
    //                        +---------------------------+
    //                               ^        |
    //                               |        v
    //                        +---------------------------+
    //                        | CACHED && pin_count != 0  |
    //                        +---------------------------+
    // NOT_LOADED: The cell is not loaded.
    // LOADING: The cell is loading.
    // LOADED: The cell is loaded but not cached in the LRU list, typically used when the cell is non-evictable.
    // CACHED: The cell is loaded and cached in the LRU list. `pin_count == 0` means the cell can be evicted.
    enum class State { NOT_LOADED, LOADING, LOADED, CACHED };

 protected:
    // will be called during eviction, implementation should release all resources.
    virtual void
    clear_data();

    virtual std::string
    key() const = 0;

    template <typename Fn>
    void
    mark_loaded(Fn&& cb, bool requesting_thread) {
        std::unique_ptr<folly::SharedPromise<folly::Unit>> promise = nullptr;
        {
            std::unique_lock<std::shared_mutex> lock(mtx_);
            if (requesting_thread) {
                switch (state_) {
                    case State::LOADING: {
                        // no need to touch() here: node is pinned thus not eligible for eviction.
                        // we can delay touch() to when unpin() is called.
                        cb();
                        state_ = State::LOADED;
                        promise = std::move(load_promise_);
                        break;
                    }
                    case State::LOADED:
                    case State::CACHED: {
                        // This state can only happen when this node is loaded/cached as a bonus.
                        // touch() has been called by the bonus loading thread.
                        promise = std::move(load_promise_);
                        break;
                    }
                    default:
                        ThrowInfo(ErrorCode::UnexpectedError,
                                  "Programming error: "
                                  "mark_loaded(requesting_thread=true) "
                                  "called on a {} cell",
                                  state_to_string(state_));
                }
            } else {
                switch (state_) {
                    case State::NOT_LOADED: {
                        // Even though this thread did not request loading this cell, translator still
                        // decided to download it because the adjacent cells are requested.
                        cb();
                        state_ = State::LOADED;
                        // memory of this cell is not reserved, touch() to track it.
                        if (evictable_) {
                            touch_to_dlist(true);
                        }
                        break;
                    }
                    case State::LOADING: {
                        // another thread has explicitly requested loading this cell, we did it first
                        // thus we set up the state first.
                        cb();
                        state_ = State::LOADED;
                        // the node that marked LOADING has already reserved memory, do not double count.
                        if (evictable_) {
                            touch_to_dlist(false);
                        }
                    }
                    default:;  // LOADED/CACHED: cell has been loaded by another thread, do nothing.
                }
            }
        }
        if (promise) {
            promise->setValue(folly::Unit());
        }
    }

    void
    set_error(folly::exception_wrapper error);

    State state_{State::NOT_LOADED};

    static std::string
    state_to_string(State state);

    // loaded_size_ must be set to the real size of the cell when the state is transferred to LOADED/CACHED.
    ResourceUsage loaded_size_{};

 private:
    friend class DList;
    friend class NodePin;

    friend class MockListNode;
    friend class DListTest;
    friend class DListTestFriend;
    friend class ListNodeTestFriend;
    friend class ListNodeTest;

    // called by DList during eviction. must be called under the lock of mtx_.
    // Made virtual for mock testing.
    virtual void
    unload();

    void
    unpin();

    // must be called under the lock of mtx_.
    void
    touch_to_dlist(bool update_evictable_size);

    mutable std::shared_mutex mtx_;
    // if a ListNode is in a DList, last_touch_ is the time when the node was lastly pushed
    // to the head of the DList. Thus all ListNodes in a DList are sorted by last_touch_.
    // last_touch_ should only be updated by DList::touchItem, except for the initialization case.
    std::chrono::steady_clock::time_point last_touch_;
    // a nullptr dlist_ means this node is not in any DList, and is not prone to cache management.
    DList* dlist_;
    ListNode* prev_ = nullptr;
    ListNode* next_ = nullptr;
    std::atomic<int> pin_count_{0};

    std::unique_ptr<folly::SharedPromise<folly::Unit>> load_promise_{nullptr};
    bool evictable_{false};
};

}  // namespace milvus::cachinglayer::internal
