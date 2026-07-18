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
#include <folly/CancellationToken.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "common/Tracer.h"

namespace milvus {

/**
 * @brief Operation context for tracking operation-specific metadata and resources.
 *
 * OpContext provides a unified container for operation-related information including
 * resource usage tracking, performance metrics, and extensible metadata. It is designed
 * to be passed through operation call chains to maintain context and enable observability.
 */
struct OpContext {
    // Storage Usage tracking
    struct {
        std::atomic<int64_t> scanned_cold_bytes{0};
        std::atomic<int64_t> scanned_total_bytes{0};
    } storage_usage;

    folly::CancellationToken cancellation_token;

    // Runtime load priority that overrides the translator's cached priority.
    // Maps to proto::common::LoadPriority (milvus-proto): HIGH = 0, LOW = 1.
    std::optional<int32_t> runtime_load_priority;

    // Coalesced-read hint: the field ids (FieldId::get() values) this operation
    // will read. Set once by the caller at operation entry. A per-field cache
    // reader may use it to co-load the sibling hinted fields of the same
    // column group in one IO instead of a separate read per field. Empty = no
    // hint (each field loaded independently). Raw int64 keeps common/
    // independent of segcore's FieldId type.
    std::vector<int64_t> coload_fields;

    std::optional<tracer::OwnedTraceContext> trace_context;

    void
    SetTraceContext(const tracer::TraceContext& ctx) {
        tracer::OwnedTraceContext snapshot(ctx);
        if (!snapshot.HasValue()) {
            trace_context.reset();
            return;
        }
        trace_context = snapshot;
    }

    void
    ClearTraceContext() {
        trace_context.reset();
    }

    [[nodiscard]] std::optional<tracer::TraceContext>
    MakeTraceContextView() const {
        if (!trace_context.has_value() || !trace_context->HasValue()) {
            return std::nullopt;
        }
        return trace_context->AsTraceContext();
    }

    // Trace parent propagated across Milvus/Knowhere/Cardinal boundaries.
    // This slot is a single-threaded parent handoff point, not a concurrent current-span stack.
    // Code that enters parallel work must copy the parent into per-worker state before nesting spans.
    std::shared_ptr<tracer::trace::Span> trace_span = nullptr;

    [[nodiscard]] std::shared_ptr<tracer::trace::Span>
    GetTraceSpan() const {
        return trace_span;
    }

    [[nodiscard]] static std::shared_ptr<tracer::trace::Span>
    GetTraceSpan(const OpContext* op_context) {
        return op_context != nullptr ? op_context->GetTraceSpan() : nullptr;
    }

    OpContext() = default;
    OpContext(const folly::CancellationToken& cancellation_token) : cancellation_token(cancellation_token) {
    }
    ~OpContext() = default;
    OpContext(const OpContext&) = delete;
    OpContext&
    operator=(const OpContext&) = delete;
    OpContext(OpContext&&) = delete;
    OpContext&
    operator=(OpContext&&) = delete;
};

}  // namespace milvus
