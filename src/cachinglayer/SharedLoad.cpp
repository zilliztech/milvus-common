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

#include "cachinglayer/SharedLoad.h"

#include <folly/executors/thread_factory/InitThreadFactory.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>

#include <algorithm>
#include <thread>

namespace milvus::cachinglayer::internal {
namespace {
thread_local bool load_worker = false;
}

folly::CPUThreadPoolExecutor&
SharedLoadThreadPool() {
    // Match the default CPU-scaled load concurrency without eagerly creating all
    // workers. Applications with a CPU quota or a different load budget should
    // pass their effective concurrency to Manager::SetLoadPoolThreads().
    static folly::CPUThreadPoolExecutor executor(
        std::make_pair(size_t{5} * std::max(1u, std::thread::hardware_concurrency()), size_t{0}),
        std::make_shared<folly::InitThreadFactory>(std::make_shared<folly::NamedThreadFactory>("mcl_load"),
                                                   [] { load_worker = true; }));
    return executor;
}

folly::Executor&
SharedLoadExecutor() {
    // A synchronous translator may load another slot. Do not queue that nested
    // producer behind the workers blocked waiting for it. It still owns its token
    // and context; the outer producer retains its resources until the nested call drains.
    if (load_worker)
        return folly::InlineExecutor::instance();
    return SharedLoadThreadPool();
}

SharedLoad::SharedLoad(const OpContext* caller) : context(cancellation.getToken()) {
    (void)caller;
}
}  // namespace milvus::cachinglayer::internal
