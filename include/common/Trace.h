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

#include <memory>
#include <string>
#include <utility>

#include "common/OpContext.h"
#include "common/Tracer.h"

namespace milvus::tracer {

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

    NestedSpanGuard(OpContext* op_context, const SpanPtr& current)
        : NestedSpanGuard(op_context != nullptr ? &op_context->trace_span : nullptr, current) {
    }

    NestedSpanGuard(NestedSpanGuard&& other) noexcept
        : span_slot_(other.span_slot_), previous_(std::move(other.previous_)) {
        other.span_slot_ = nullptr;
    }

    NestedSpanGuard&
    operator=(NestedSpanGuard&& other) noexcept {
        if (this != &other) {
            Restore();
            span_slot_ = other.span_slot_;
            previous_ = std::move(other.previous_);
            other.span_slot_ = nullptr;
        }
        return *this;
    }

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

}  // namespace milvus::tracer
