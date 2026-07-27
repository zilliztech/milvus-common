// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common/EasyAssert.h"

#include <atomic>
#include <boost/stacktrace.hpp>
#include <iostream>
#include <sstream>

#include "fmt/format.h"

namespace milvus {

namespace {
std::atomic<UntypedCgoExceptionObserver> untyped_cgo_exception_observer{nullptr};
}  // namespace

void
RegisterUntypedCgoExceptionObserver(UntypedCgoExceptionObserver observer) {
    untyped_cgo_exception_observer.store(observer, std::memory_order_release);
}

namespace impl {
void
NotifyUntypedCgoException(const char* what) noexcept {
    if (auto observer = untyped_cgo_exception_observer.load(std::memory_order_acquire)) {
        try {
            observer(what);
        } catch (...) {
            // The observer is metrics/logging only. It runs inside
            // FailureCStatus's exception-to-CStatus conversion, so a throwing
            // observer must never replace the original failure or let an
            // exception escape the cgo boundary (which would terminate the
            // process). Swallow and carry on with the conversion.
        }
    }
}
}  // namespace impl

}  // namespace milvus

namespace milvus::impl {

std::string
EasyStackTrace() {
    std::string output;
#ifdef BOOST_STACKTRACE_USE_BACKTRACE
    auto stack_info = boost::stacktrace::stacktrace();
    std::ostringstream ss;
    ss << stack_info;
    output = std::string(ss.str());
#endif
    return output;
}

void
EasyAssertInfo(bool value, std::string_view expr_str, std::string_view filename, int lineno,
               std::string_view extra_info, ErrorCode error_code) {
    // enable error code
    if (!value) {
        std::string info;
        if (!expr_str.empty()) {
            info += fmt::format("Assert \"{}\" ", expr_str);
        }
        if (!extra_info.empty()) {
            info += " => " + std::string(extra_info);
        }
        info += fmt::format(" at {}:{}\n", std::string(filename), std::to_string(lineno));
        std::cout << info << std::endl;

        throw SegcoreError(error_code, std::string(info));
    }
}

}  // namespace milvus::impl
