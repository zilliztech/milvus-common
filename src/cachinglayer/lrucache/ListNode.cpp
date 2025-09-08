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
#include "cachinglayer/lrucache/ListNode.h"

#include <fmt/core.h>
#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"
#include "common/EasyAssert.h"
#include "log/Log.h"

namespace milvus::cachinglayer::internal {

ListNode::NodePin::NodePin(ListNode* node) : node_(node) {
    // The pin_count_ is incremented in ListNode::pin() before this constructor is called.
}

ListNode::NodePin::~NodePin() {
    if (node_) {
        node_->unpin();
    }
}

ListNode::NodePin::NodePin(NodePin&& other) noexcept : NodePin(nullptr) {
    std::swap(node_, other.node_);
}

ListNode::NodePin&
ListNode::NodePin::operator=(NodePin&& other) noexcept {
    std::swap(node_, other.node_);
    return *this;
}

ListNode::ListNode(DList* dlist, bool evictable)
    : last_touch_(dlist ? (std::chrono::steady_clock::now() - 2 * dlist->eviction_config().cache_touch_window)
                        : std::chrono::steady_clock::now()),
      dlist_(dlist),
      evictable_(evictable) {
}

ListNode::~ListNode() {
    std::unique_lock<std::shared_mutex> lock(mtx_);
    switch (state_) {
        case State::CACHED: {
            dlist_->removeItem(this, loaded_size_);
            // fall through
        }
        case State::LOADED: {
            // data cleanup should be handled in child class, e.g. CacheCell::~CacheCell()
            state_ = State::NOT_LOADED;
            break;
        }
        case State::LOADING: {
            // NOTE:
            // The LOADING state occurs during pin() and set_cell(), while LoadingResource +/- is handled in RunLoad().
            // We believe RunLoad() will handle all situations, including exceptions, so there's no need to handle
            // LoadingResource here.
            // If that edge case actually occurs, it shouldn't be the fault of ~ListNode() - the system must have bugs.
            LOG_ERROR("[MCL] ListNode destroyed while loading");
            break;
        }
        default:;  // do nothing
    }
}

bool
ListNode::manual_evict() {
    std::unique_lock<std::shared_mutex> lock(mtx_);
    switch (state_) {
        case State::CACHED: {
            // even if pin_count_ > 0, removeItem is still ok since it can be readded to dlist_ when pin_count_ == 0.
            dlist_->removeItem(this, loaded_size_);
            // fall through
        }
        case State::LOADED: {
            if (pin_count_.load() > 0) {
                LOG_ERROR(
                    "[MCL] manual_evict() called on a LOADED and pinned cell "
                    "{}, "
                    "aborting eviction.",
                    key());
                return false;
            }
            auto saved_loaded_size = loaded_size_;
            unload();
            dlist_->RefundLoadedResource(saved_loaded_size);
            return true;
        }
        case State::LOADING: {
            LOG_ERROR("[MCL] manual_evict() called on a {} cell {}", state_to_string(state_), key());
            return false;
        }
        default: {
            return false;
        }
    }
}

const ResourceUsage&
ListNode::loaded_size() const {
    return loaded_size_;
}

std::pair<bool, folly::SemiFuture<ListNode::NodePin>>
ListNode::pin() {
    // must be called with lock acquired, and state must not be NOT_LOADED.
    auto read_op = [this]() -> std::pair<bool, folly::SemiFuture<NodePin>> {
        // pin the cell now so that we can avoid taking the lock again in deferValue.
        auto old_pin_count = pin_count_.fetch_add(1);
        switch (state_) {
            case State::CACHED: {
                if (old_pin_count == 0) {
                    // node became inevictable, freeze it if it is in dlist
                    dlist_->freezeItem(this, loaded_size_);
                }
                // fall through
            }
            case State::LOADED: {
                auto p = NodePin(this);
                return std::make_pair(false, std::move(p));
            }
            case State::LOADING: {
                auto p = NodePin(this);
                return std::make_pair(false, load_promise_->getSemiFuture().deferValue(
                                                 [this, p = std::move(p)](auto&&) mutable { return std::move(p); }));
            }
            default:
                ThrowInfo(ErrorCode::UnexpectedError, "Programming error: read_op called on a {} cell",
                          state_to_string(state_));
        }
    };

    {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        if (state_ != State::NOT_LOADED) {
            return read_op();
        }
    }

    std::unique_lock<std::shared_mutex> lock(mtx_);
    if (state_ != State::NOT_LOADED) {
        return read_op();
    }

    // need to load. state_ == State::NOT_LOADED
    load_promise_ = std::make_unique<folly::SharedPromise<folly::Unit>>();
    state_ = State::LOADING;

    // pin the cell now so that we can avoid taking the lock again in deferValue.
    pin_count_.fetch_add(1);
    auto p = NodePin(this);
    return std::make_pair(true, load_promise_->getSemiFuture().deferValue(
                                    [this, p = std::move(p)](auto&&) mutable { return std::move(p); }));
}

void
ListNode::set_error(folly::exception_wrapper error) {
    std::unique_ptr<folly::SharedPromise<folly::Unit>> promise = nullptr;
    {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        switch (state_) {
            case State::LOADING: {
                state_ = State::NOT_LOADED;
                if (load_promise_) {
                    promise = std::move(load_promise_);
                }
                break;
            }
            case State::CACHED:
            case State::LOADED: {
                // may be successfully loaded/cached by another thread as a bonus, they will update used memory.
                return;
            }
            default:
                ThrowInfo(ErrorCode::UnexpectedError, "Programming error: set_error() called on a {} cell",
                          state_to_string(state_));
        }
    }
    // Notify waiting threads about the error
    // setException may call continuation of bound futures inline, and those continuation may also need to acquire the
    // lock, which may cause deadlock. So we release the lock before calling setException.
    if (promise) {
        promise->setException(std::move(error));
    }
}

std::string
ListNode::state_to_string(State state) {
    switch (state) {
        case State::NOT_LOADED:
            return "NOT_LOADED";
        case State::LOADING:
            return "LOADING";
        case State::LOADED:
            return "LOADED";
        case State::CACHED:
            return "CACHED";
    }
    throw std::invalid_argument("Invalid state");
}

void
ListNode::unpin() {
    std::unique_lock<std::shared_mutex> lock(mtx_);
    if (pin_count_.fetch_sub(1) == 1) {
        if (evictable_) {
            touch_to_dlist(state_ == State::LOADED || state_ == State::CACHED);
        }
    }
}

// ListNode::touch_to_dlist() should only be called when evictable_ is true
void
ListNode::touch_to_dlist(bool update_evictable_memory) {
    std::optional<ResourceUsage> size = std::nullopt;
    if (update_evictable_memory) {
        size = loaded_size_;
    }
    dlist_->touchItem(this, false, size);
    state_ = State::CACHED;
}

void
ListNode::unload() {
    state_ = State::NOT_LOADED;  // reset state_ to NOT_LOADED to avoid double refund from dlist_
}

}  // namespace milvus::cachinglayer::internal
