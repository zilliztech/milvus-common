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

#include <folly/CancellationToken.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/futures/SharedPromise.h>

#include <atomic>
#include <functional>
#include <memory>
#include <unordered_set>

#include "cachinglayer/Utils.h"
#include "common/OpContext.h"

namespace milvus::cachinglayer::internal {

// Loads use a separate executor: a synchronous requester must never occupy the
// worker needed to complete its download. Coroutine loads release workers on suspension.
folly::Executor&
SharedLoadExecutor();
// Dedicated, lazily grown pool; the caller can configure its concurrency through Manager.
folly::CPUThreadPoolExecutor&
SharedLoadThreadPool();

struct SharedLoad {
    explicit SharedLoad(const OpContext* caller);
    folly::CancellationSource cancellation;
    OpContext context;
    std::unordered_set<cid_t> cids;
    folly::SharedPromise<folly::Unit> drained;
    // Protected by the owning CacheSlot's loads_mutex_.
    size_t waiters = 0;
    bool closing = false;
    bool finished = false;
};

// A request owns its result, never the shared producer's promise. Keep the
// producer and its pins alive until completion even when this waiter exits early.
// on_exit is called exactly once, before making the request result visible.
template <typename T>
folly::SemiFuture<T>
WaitForSharedLoad(folly::SemiFuture<T> input, folly::CancellationToken token, std::shared_ptr<void> owner,
                  std::function<void(bool)> on_exit = {}) {
    struct Waiter {
        folly::Promise<T> promise;
        std::atomic<bool> finished{false};
        std::function<void(bool)> on_exit;
        std::unique_ptr<folly::CancellationCallback> callback;
        void
        Complete(folly::Try<T>&& value, bool cancelled) {
            if (finished.exchange(true))
                return;
            auto exit = std::move(on_exit);
            if (exit)
                exit(cancelled);
            promise.setTry(std::move(value));
        }
    };
    auto waiter = std::make_shared<Waiter>();
    waiter->on_exit = std::move(on_exit);
    auto result = waiter->promise.getSemiFuture();
    // Install completion before cancellation so an already-ready load wins.
    // Callers check pre-cancelled tokens before pinning.
    std::move(input)
        .via(&folly::InlineExecutor::instance())
        .thenTry(
            [waiter, owner = std::move(owner)](folly::Try<T>&& value) { waiter->Complete(std::move(value), false); });
    std::weak_ptr<Waiter> weak = waiter;
    waiter->callback = std::make_unique<folly::CancellationCallback>(token, [weak] {
        if (auto current = weak.lock()) {
            current->Complete(folly::Try<T>(folly::make_exception_wrapper<SegcoreError>(ErrorCode::FollyCancel,
                                                                                        "Cache wait cancelled")),
                              true);
        }
    });
    return result;
}

}  // namespace milvus::cachinglayer::internal
