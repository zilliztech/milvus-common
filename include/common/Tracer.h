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

#include "common/TracerBase.h"
#include "opentelemetry/trace/span_id.h"
#include "opentelemetry/trace/trace_id.h"

namespace opentelemetry::trace {
class Span;
class Tracer;
}  // namespace opentelemetry::trace

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
StartSpan(const std::string& name, const std::shared_ptr<trace::Span>& span);

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

struct AutoSpan {
    explicit AutoSpan(const std::string& name, TraceContext* ctx = nullptr, bool is_root_span = false);

    // Creates a span with a parent span. If set_as_temp_root is true, this span will temporarily
    // replace the current thread-local root span. The original root span will be saved and restored
    // when this AutoSpan is destroyed.
    explicit AutoSpan(const std::string& name, const std::shared_ptr<trace::Span>& span, bool set_as_temp_root = false);

    std::shared_ptr<trace::Span>
    GetSpan();

    ~AutoSpan();

 private:
    std::shared_ptr<trace::Span> span_;
    bool is_root_span_;
    bool set_as_temp_root_ = false;
    std::shared_ptr<trace::Span> previous_root_;
};

}  // namespace milvus::tracer
