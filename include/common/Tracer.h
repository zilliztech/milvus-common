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

#include <map>
#include <memory>
#include <string>

#include "common/TracerBase.h"
#include "opentelemetry/trace/provider.h"

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
namespace trace = opentelemetry::trace;

void
initTelemetry(const TraceConfig& cfg);

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
