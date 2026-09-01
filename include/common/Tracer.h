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

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "common/TracerBase.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_id.h"
#include "opentelemetry/trace/trace_id.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace {
class Span;
class Tracer;
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE

#define TRACE_SERVICE_SEGCORE "segcore"

namespace milvus::tracer {

struct TraceConfig {
    std::string exporter;
    float sampleFraction;
    std::string jaegerURL;
    std::string otlpEndpoint;
    std::string otlpMethod;
    std::string otlpHeaders;
    bool oltpSecure;

    int nodeID;
};

struct TraceContext {
    const uint8_t* traceID = nullptr;
    const uint8_t* spanID = nullptr;
    uint8_t traceFlags = 0;
};

struct OwnedTraceContext {
    std::array<uint8_t, opentelemetry::trace::TraceId::kSize> trace_id{};
    std::array<uint8_t, opentelemetry::trace::SpanId::kSize> span_id{};
    uint8_t trace_flags = 0;
    bool has_value = false;

    OwnedTraceContext() = default;

    explicit OwnedTraceContext(const TraceContext& ctx) {
        if (ctx.traceID == nullptr || ctx.spanID == nullptr) {
            return;
        }
        const auto source_trace_id = opentelemetry::trace::TraceId({ctx.traceID, opentelemetry::trace::TraceId::kSize});
        const auto source_span_id = opentelemetry::trace::SpanId({ctx.spanID, opentelemetry::trace::SpanId::kSize});
        if (!source_trace_id.IsValid() || !source_span_id.IsValid()) {
            return;
        }
        std::copy_n(ctx.traceID, trace_id.size(), trace_id.begin());
        std::copy_n(ctx.spanID, span_id.size(), span_id.begin());
        trace_flags = ctx.traceFlags;
        has_value = true;
    }

    [[nodiscard]] bool
    HasValue() const {
        return has_value;
    }

    [[nodiscard]] TraceContext
    AsTraceContext() const& {
        if (!has_value) {
            return {};
        }
        return TraceContext{trace_id.data(), span_id.data(), trace_flags};
    }

    [[nodiscard]] TraceContext
    AsTraceContext() const&& = delete;

    void
    Clear() {
        trace_id.fill(0);
        span_id.fill(0);
        trace_flags = 0;
        has_value = false;
    }
};
namespace trace = opentelemetry::trace;

void
initTelemetry(const TraceConfig& cfg);

bool
IsTraceEnabled();

std::shared_ptr<trace::Tracer>
GetTracer();

std::shared_ptr<trace::Span>
StartSpan(const std::string& name, TraceContext* ctx = nullptr);

std::shared_ptr<trace::Span>
StartSpan(const char* name, TraceContext* ctx = nullptr);

std::shared_ptr<trace::Span>
StartSpan(const std::string& name, const std::shared_ptr<trace::Span>& span);

std::shared_ptr<trace::Span>
StartSpan(const char* name, const std::shared_ptr<trace::Span>& span);

void
SetRootSpan(std::shared_ptr<trace::Span> span);

std::shared_ptr<trace::Span>
GetRootSpan();

void
CloseRootSpan();

void
AddEvent(const std::string& event_label);

bool
EmptyTraceID(const TraceContext* ctx);

bool
EmptySpanID(const TraceContext* ctx);

std::string
BytesToHexStr(const uint8_t* data, const size_t len);

std::string
GetIDFromHexStr(const std::string& hexStr);

std::string
GetTraceIDAsHexStr(const TraceContext* ctx);

std::string
GetSpanIDAsHexStr(const TraceContext* ctx);

std::map<std::string, std::string>
parseHeaders(const std::string& headers);

using SpanPtr = std::shared_ptr<trace::Span>;

// Starts a child span from a parent span and ends that child span when the guard leaves scope.
// If the parent is empty or tracing is disabled, the guard stays empty and has no effect.
class ScopedSpan {
 public:
    ScopedSpan() = default;

    ScopedSpan(const std::string& name, const SpanPtr& parent) {
        if (parent != nullptr && IsTraceEnabled()) {
            span_ = StartSpan(name, parent);
        }
    }

    ScopedSpan(const char* name, const SpanPtr& parent) {
        if (parent != nullptr && IsTraceEnabled()) {
            span_ = StartSpan(name, parent);
        }
    }

    ScopedSpan(const ScopedSpan&) = delete;
    ScopedSpan&
    operator=(const ScopedSpan&) = delete;

    ScopedSpan(ScopedSpan&& other) noexcept : span_(std::move(other.span_)) {
    }

    ScopedSpan&
    operator=(ScopedSpan&& other) noexcept {
        if (this != &other) {
            End();
            span_ = std::move(other.span_);
        }
        return *this;
    }

    ~ScopedSpan() {
        End();
    }

    const SpanPtr&
    Get() const {
        return span_;
    }

    ScopedSpan
    StartChild(const std::string& name) const {
        return ScopedSpan(name, span_);
    }

    ScopedSpan
    StartChild(const char* name) const {
        return ScopedSpan(name, span_);
    }

    void
    End() {
        if (span_ != nullptr) {
            span_->End();
            span_.reset();
        }
    }

 private:
    SpanPtr span_;
};

// Temporarily replaces a span slot with a nested/current span, then restores the previous span on destruction.
// This guard does not start or end spans; use ScopedSpan for span lifetime ownership.
class NestedSpanGuard {
 public:
    NestedSpanGuard() = default;

    NestedSpanGuard(SpanPtr* span_slot, const SpanPtr& current)
        : span_slot_(span_slot), previous_(span_slot != nullptr ? *span_slot : SpanPtr{}) {
        if (span_slot_ != nullptr) {
            *span_slot_ = current;
        }
    }

    NestedSpanGuard(SpanPtr& span_slot, const SpanPtr& current) : NestedSpanGuard(&span_slot, current) {
    }

    NestedSpanGuard(NestedSpanGuard&&) = delete;
    NestedSpanGuard&
    operator=(NestedSpanGuard&&) = delete;

    NestedSpanGuard(const NestedSpanGuard&) = delete;
    NestedSpanGuard&
    operator=(const NestedSpanGuard&) = delete;

    ~NestedSpanGuard() {
        Restore();
    }

 private:
    void
    Restore() {
        if (span_slot_ != nullptr) {
            *span_slot_ = previous_;
            span_slot_ = nullptr;
        }
    }

    SpanPtr* span_slot_ = nullptr;
    SpanPtr previous_;
};

struct AutoSpan {
    explicit AutoSpan(const std::string& name, TraceContext* ctx = nullptr, bool is_root_span = false);

    // Creates a span with a parent span. If set_as_temp_root is true, this span will temporarily
    // replace the current thread-local root span. The original root span will be saved and restored
    // when this AutoSpan is destroyed.
    explicit AutoSpan(const std::string& name, const std::shared_ptr<trace::Span>& span, bool set_as_temp_root = false);

    // const char* overloads. Call sites pass string literals, and most of the span names used on
    // the segcore expression path exceed libstdc++'s 15-char SSO buffer (e.g.
    // "PhyConjunctFilterExpr::Eval" is 27 chars), so the std::string temporary is heap-allocated
    // by the caller before the constructor can early-return on IsTraceEnabled(). These overloads
    // forward to the const char* StartSpan() variants added in #103 and keep the disabled path
    // allocation-free.
    explicit AutoSpan(const char* name, TraceContext* ctx = nullptr, bool is_root_span = false);
    explicit AutoSpan(const char* name, const std::shared_ptr<trace::Span>& span, bool set_as_temp_root = false);

    // Sets an attribute on the underlying span.
    // When tracing is disabled, span_ is always nullptr (the constructor returns early),
    // so this short-circuits to a strict no-op: a single branch, fully inlined, with no
    // shared_ptr refcount traffic and no virtual call.
    //
    // key is nostd::string_view rather than const std::string&: call sites pass string literals,
    // and the argument is materialised before the span_ check, so a std::string parameter charges
    // a heap allocation to the disabled path for any key past the 15-char SSO buffer -
    // "json_filter_expr_type" (21 chars) is already one of them. nostd::string_view is what
    // Span::SetAttribute() takes anyway, and is constructible from both const char* and
    // std::string, so no call site has to change.
    template <typename T>
    void
    SetAttribute(opentelemetry::nostd::string_view key, const T& value) {
        if (span_ != nullptr) {
            span_->SetAttribute(key, value);
        }
    }

    // Returns by const reference. When tracing is disabled span_ is null and this returns the
    // process-global noop_span; returning it by value made every call copy that one shared_ptr,
    // i.e. an atomic increment and decrement on a single control block shared by every thread.
    // Callers use the result as `span.GetSpan()->...` within the full-expression, so a reference
    // is sufficient: noop_span is a file-static and span_ outlives the call as a member.
    const std::shared_ptr<trace::Span>&
    GetSpan();

    ~AutoSpan();

 private:
    std::shared_ptr<trace::Span> span_;
    bool is_root_span_;
    bool set_as_temp_root_ = false;
    std::shared_ptr<trace::Span> previous_root_;
};

}  // namespace milvus::tracer
