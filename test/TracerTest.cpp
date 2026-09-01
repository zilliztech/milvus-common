#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#include "common/EasyAssert.h"
#include "common/Tracer.h"

using namespace milvus;
using namespace milvus::tracer;
using namespace opentelemetry::trace;

TEST(Tracer, Init) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);
    auto span = StartSpan("test");
    ASSERT_TRUE(span->IsRecording());

    config = std::make_shared<TraceConfig>();
    config->exporter = "jaeger";
    config->jaegerURL = "http://localhost:14268/api/traces";
    config->nodeID = 1;
    initTelemetry(*config);
    span = StartSpan("test");
    ASSERT_TRUE(span->IsRecording());
}

TEST(Tracer, Span) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);

    auto ctx = std::make_shared<TraceContext>();
    ctx->traceID =
        new uint8_t[16]{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10};
    ctx->spanID = new uint8_t[8]{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
    ctx->traceFlags = 1;
    auto span = StartSpan("test", ctx.get());

    ASSERT_TRUE(span->GetContext().trace_id() == trace::TraceId({ctx->traceID, 16}));

    delete[] ctx->traceID;
    delete[] ctx->spanID;
}

TEST(Tracer, GetTraceID) {
    auto trace_id = GetTraceID();
    ASSERT_TRUE(trace_id.empty());

    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);

    auto span = StartSpan("test");
    SetRootSpan(span);
    trace_id = GetTraceID();
    ASSERT_TRUE(trace_id.size() == 32);

    CloseRootSpan();
    trace_id = GetTraceID();
    ASSERT_TRUE(trace_id.empty());
}

TEST(Tracer, ParseHeaders) {
    // Test empty headers
    auto headers_map = parseHeaders("");
    ASSERT_TRUE(headers_map.empty());

    // Test simple JSON headers
    std::string json_headers = R"({"Authorization": "Bearer token123", "Content-Type": "application/json"})";
    headers_map = parseHeaders(json_headers);
    ASSERT_EQ(headers_map.size(), 2);
    ASSERT_EQ(headers_map["Authorization"], "Bearer token123");
    ASSERT_EQ(headers_map["Content-Type"], "application/json");

    // Test JSON with whitespace
    std::string json_headers_with_spaces = R"({ "key1" : "value1" , "key2" : "value2" })";
    headers_map = parseHeaders(json_headers_with_spaces);
    ASSERT_EQ(headers_map.size(), 2);
    ASSERT_EQ(headers_map["key1"], "value1");
    ASSERT_EQ(headers_map["key2"], "value2");

    // Test invalid JSON
    std::string invalid_json = "invalid json string";
    headers_map = parseHeaders(invalid_json);
    ASSERT_TRUE(headers_map.empty());

    // Test empty JSON object
    std::string empty_json = "{}";
    headers_map = parseHeaders(empty_json);
    ASSERT_TRUE(headers_map.empty());
}

TEST(Tracer, OTLPHttpExporter) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "otlp";
    config->otlpMethod = "http";
    config->otlpEndpoint = "http://localhost:4318/v1/traces";
    config->otlpHeaders = R"({"Authorization": "Bearer test-token", "Content-Type": "application/json"})";
    config->nodeID = 1;

    initTelemetry(*config);
    auto span = StartSpan("test_otlp_http");
    ASSERT_TRUE(span->IsRecording());

    // Test with empty headers
    config->otlpHeaders = "";
    initTelemetry(*config);
    span = StartSpan("test_otlp_http_empty_headers");
    ASSERT_TRUE(span->IsRecording());

    // Test with invalid JSON headers
    config->otlpHeaders = "invalid json";
    initTelemetry(*config);
    span = StartSpan("test_otlp_http_invalid_headers");
    ASSERT_TRUE(span->IsRecording());
}

TEST(Tracer, OTLPGrpcExporter) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "otlp";
    config->otlpMethod = "grpc";
    config->otlpEndpoint = "localhost:4317";
    config->otlpHeaders = R"({"Authorization": "Bearer grpc-token"})";
    config->oltpSecure = false;
    config->nodeID = 1;

    initTelemetry(*config);
    auto span = StartSpan("test_otlp_grpc");
    ASSERT_TRUE(span->IsRecording());

    // Test with secure connection
    config->oltpSecure = true;
    initTelemetry(*config);
    span = StartSpan("test_otlp_grpc_secure");
    ASSERT_TRUE(span->IsRecording());

    // Test with empty headers
    config->otlpHeaders = "";
    config->oltpSecure = false;
    initTelemetry(*config);
    span = StartSpan("test_otlp_grpc_empty_headers");
    ASSERT_TRUE(span->IsRecording());
}

TEST(Tracer, OTLPLegacyConfiguration) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "otlp";
    config->otlpMethod = "";  // legacy configuration
    config->otlpEndpoint = "localhost:4317";
    config->otlpHeaders = R"({"legacy": "header"})";
    config->oltpSecure = false;
    config->nodeID = 1;

    initTelemetry(*config);
    auto span = StartSpan("test_otlp_legacy");
    ASSERT_TRUE(span->IsRecording());
}

TEST(Tracer, OTLPInvalidMethod) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "otlp";
    config->otlpMethod = "invalid_method";
    config->otlpEndpoint = "localhost:4317";
    config->nodeID = 1;

    initTelemetry(*config);
    auto span = StartSpan("test_otlp_invalid");
    // Should fall back to noop provider when export creation fails
    // Span should never be nullptr, but should be non-recording
    ASSERT_NE(span, nullptr);
    ASSERT_FALSE(span->IsRecording());
}

TEST(Tracer, OTLPComplexHeaders) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "otlp";
    config->otlpMethod = "http";
    config->otlpEndpoint = "http://localhost:4318/v1/traces";
    config->otlpHeaders = R"({
        "Authorization": "Bearer complex-token-123",
        "X-Custom-Header": "custom-value",
        "User-Agent": "Milvus-Tracer/1.0",
        "Accept": "application/json"
    })";
    config->nodeID = 1;

    initTelemetry(*config);
    auto span = StartSpan("test_otlp_complex_headers");
    ASSERT_TRUE(span->IsRecording());
}

TEST(Tracer, OTLPEmptyExporter) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "";  // empty exporter
    config->nodeID = 1;

    initTelemetry(*config);
    auto span = StartSpan("test_empty_exporter");
    // Should fall back to noop provider
    // Span should never be nullptr, but should be non-recording
    ASSERT_NE(span, nullptr);
    ASSERT_FALSE(span->IsRecording());
}

TEST(Tracer, OTLPInvalidExporter) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "invalid_exporter";
    config->nodeID = 1;

    initTelemetry(*config);
    auto span = StartSpan("test_invalid_exporter");
    // Should fall back to noop provider
    // Span should never be nullptr, but should be non-recording
    ASSERT_NE(span, nullptr);
    ASSERT_FALSE(span->IsRecording());
}

TEST(Tracer, OTLPHeadersParsingEdgeCases) {
    // Test with whitespace in JSON
    std::string json_with_spaces = R"({ "key1" : "value1" , "key2" : "value2" })";
    auto headers_map = parseHeaders(json_with_spaces);
    ASSERT_EQ(headers_map.size(), 2);
    ASSERT_EQ(headers_map["key1"], "value1");
    ASSERT_EQ(headers_map["key2"], "value2");

    // Test with nested JSON (should fail gracefully)
    std::string nested_json = R"({"key": {"nested": "value"}})";
    headers_map = parseHeaders(nested_json);
    ASSERT_TRUE(headers_map.empty());

    // Test with array JSON (should fail gracefully)
    std::string array_json = R"(["header1", "header2"])";
    headers_map = parseHeaders(array_json);
    ASSERT_TRUE(headers_map.empty());

    // Test with null JSON
    std::string null_json = "null";
    headers_map = parseHeaders(null_json);
    ASSERT_TRUE(headers_map.empty());
}

TEST(Tracer, IsTraceEnabled) {
    // Enable tracing
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_TRUE(IsTraceEnabled());

    // Disable tracing with noop exporter
    config = std::make_shared<TraceConfig>();
    config->exporter = "noop";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_FALSE(IsTraceEnabled());

    // Disable tracing with empty exporter
    config = std::make_shared<TraceConfig>();
    config->exporter = "";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_FALSE(IsTraceEnabled());
}

TEST(Tracer, DisabledTracingReturnsNoopSpan) {
    // Disable tracing
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "noop";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_FALSE(IsTraceEnabled());

    // StartSpan should return noop span, not nullptr
    auto span = StartSpan("test_disabled");
    ASSERT_NE(span, nullptr);
    ASSERT_FALSE(span->IsRecording());

    // Should be safe to call methods on the noop span
    span->SetAttribute("key", "value");
    span->AddEvent("event");
    span->End();

    // StartSpan with TraceContext should also return noop span
    auto ctx = std::make_shared<TraceContext>();
    ctx->traceID = new uint8_t[16]{0};
    ctx->spanID = new uint8_t[8]{0};
    ctx->traceFlags = 0;
    span = StartSpan("test_disabled_with_ctx", ctx.get());
    ASSERT_NE(span, nullptr);
    ASSERT_FALSE(span->IsRecording());
    delete[] ctx->traceID;
    delete[] ctx->spanID;

    // StartSpan with parent span should also return noop span
    auto parent = StartSpan("parent");
    auto child = StartSpan("child", parent);
    ASSERT_NE(child, nullptr);
    ASSERT_FALSE(child->IsRecording());
}

TEST(Tracer, AutoSpanGetSpanReturnsNoopWhenDisabled) {
    // Disable tracing
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "noop";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_FALSE(IsTraceEnabled());

    // AutoSpan::GetSpan should return noop span, not nullptr
    {
        AutoSpan span("test_auto_span", nullptr, false);
        auto inner_span = span.GetSpan();
        ASSERT_NE(inner_span, nullptr);
        ASSERT_FALSE(inner_span->IsRecording());

        // Should be safe to call methods - this is the critical test
        // that prevents NPE in code like: span.GetSpan()->SetAttribute(...)
        inner_span->SetAttribute("data_type", 1);
        inner_span->SetAttribute("key", "value");
        inner_span->AddEvent("event");
    }

    // Test with parent span parameter
    {
        auto parent = StartSpan("parent");
        AutoSpan span("test_auto_span_with_parent", parent, false);
        auto inner_span = span.GetSpan();
        ASSERT_NE(inner_span, nullptr);
        ASSERT_FALSE(inner_span->IsRecording());
        inner_span->SetAttribute("test", 123);
    }

    // Test with root span flag
    {
        AutoSpan span("test_root_span", nullptr, true);
        auto inner_span = span.GetSpan();
        ASSERT_NE(inner_span, nullptr);
        inner_span->SetAttribute("is_root", true);
    }
}

TEST(Tracer, AutoSpanGetSpanReturnsRealSpanWhenEnabled) {
    // Enable tracing
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_TRUE(IsTraceEnabled());

    // AutoSpan::GetSpan should return real recording span
    {
        AutoSpan span("test_enabled_auto_span", nullptr, false);
        auto inner_span = span.GetSpan();
        ASSERT_NE(inner_span, nullptr);
        ASSERT_TRUE(inner_span->IsRecording());
        inner_span->SetAttribute("data_type", 1);
    }
}

TEST(Tracer, AutoSpanSetAttributeNoopWhenDisabled) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "noop";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_FALSE(IsTraceEnabled());

    // When tracing is disabled, SetAttribute short-circuits on the nullptr span_
    // and must be a safe no-op for any value type.
    AutoSpan span("test_auto_span", nullptr, false);
    span.SetAttribute("data_type", 1);
    span.SetAttribute("key", std::string("value"));
    span.SetAttribute("ratio", 0.5);
    span.SetAttribute("is_root", true);
}

TEST(Tracer, AutoSpanSetAttributeWhenEnabled) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_TRUE(IsTraceEnabled());

    AutoSpan span("test_enabled_auto_span", nullptr, false);
    span.SetAttribute("data_type", 1);
    span.SetAttribute("key", std::string("value"));
    span.SetAttribute("ratio", 0.5);
    span.SetAttribute("is_root", true);
}

TEST(Tracer, NoopSpanMethodsAreSafe) {
    // Disable tracing
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "";
    config->nodeID = 1;
    initTelemetry(*config);

    auto span = StartSpan("noop_test");
    ASSERT_NE(span, nullptr);

    // All these operations should be safe (no crash, no side effects)
    span->SetAttribute("string_attr", "value");
    span->SetAttribute("int_attr", 42);
    span->SetAttribute("double_attr", 3.14);
    span->SetAttribute("bool_attr", true);
    span->AddEvent("test_event");
    span->SetStatus(opentelemetry::trace::StatusCode::kOk, "ok");
    span->UpdateName("new_name");

    auto ctx = span->GetContext();
    ASSERT_FALSE(ctx.IsValid());

    span->End();
}

TEST(Tracer, AutoSpanCStringOverloadDisabled) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "noop";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_FALSE(IsTraceEnabled());

    // The const char* overloads must behave exactly like the std::string ones: no span is
    // started while tracing is off, and GetSpan() still hands back the noop span so that
    // `span.GetSpan()->...` stays safe.
    AutoSpan ctx_span("PhyConjunctFilterExpr::Eval", nullptr, false);
    ASSERT_NE(ctx_span.GetSpan(), nullptr);

    AutoSpan child_span("PhyBinaryArithOpEvalRangeExpr::Eval", GetRootSpan(), true);
    ASSERT_NE(child_span.GetSpan(), nullptr);
}

TEST(Tracer, AutoSpanCStringOverloadEnabled) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_TRUE(IsTraceEnabled());

    AutoSpan ctx_span("PhyConjunctFilterExpr::Eval", nullptr, false);
    ASSERT_NE(ctx_span.GetSpan(), nullptr);
    ASSERT_TRUE(ctx_span.GetSpan()->IsRecording());

    AutoSpan child_span("PhyBinaryArithOpEvalRangeExpr::Eval", ctx_span.GetSpan(), true);
    ASSERT_NE(child_span.GetSpan(), nullptr);
    ASSERT_TRUE(child_span.GetSpan()->IsRecording());
}

namespace {

std::atomic<std::uintptr_t> bench_sink{0};

// Returns the mean per-op latency in nanoseconds observed by one thread.
double
RunDisabledPathBench(int num_threads, int iters_per_thread, bool force_copy) {
    std::atomic<bool> go{false};
    std::atomic<int> ready{0};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&] {
            std::uintptr_t sink = 0;
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
            }
            for (int i = 0; i < iters_per_thread; ++i) {
                AutoSpan span("PhyConjunctFilterExpr::Eval", GetRootSpan(), true);
                if (force_copy) {
                    auto copied = span.GetSpan();
                    sink += reinterpret_cast<std::uintptr_t>(copied.get());
                } else {
                    sink += reinterpret_cast<std::uintptr_t>(span.GetSpan().get());
                }
            }
            bench_sink.fetch_add(sink, std::memory_order_relaxed);
        });
    }

    while (ready.load(std::memory_order_acquire) < num_threads) {
    }
    auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    return static_cast<double>(ns) / iters_per_thread;
}

}  // namespace

// Measures the cost of reaching an AutoSpan's span while tracing is disabled — the production
// default — as the thread count grows. This is a measurement, not an assertion, so it is
// DISABLED by default. Run it explicitly with:
//
//   ./<test-binary> --gtest_also_run_disabled_tests \
//                   --gtest_filter='Tracer.DISABLED_AutoSpanDisabledPathContention'
//
// Two variants are timed per thread count:
//   copy   - `auto s = span.GetSpan();` forces a shared_ptr copy, i.e. an atomic increment and
//            decrement on the single process-global noop_span control block. This is what every
//            `span.GetSpan()->SetAttribute(...)` call site did before this change.
//   borrow - `span.GetSpan()->...` with GetSpan() returning by const reference, which touches no
//            refcount at all.
//
// The gap between the two columns is the contention removed here, and it should widen
// superlinearly with the thread count. On a build without this change both columns copy, so they
// converge - which is itself a useful control.
TEST(Tracer, DISABLED_AutoSpanDisabledPathContention) {
    auto config = std::make_shared<TraceConfig>();
    config->exporter = "noop";
    config->nodeID = 1;
    initTelemetry(*config);
    ASSERT_FALSE(IsTraceEnabled());

    constexpr int kIters = 200000;
    std::printf("%8s %14s %14s %10s\n", "threads", "copy ns/op", "borrow ns/op", "ratio");
    for (int threads : {1, 2, 4, 8, 16, 24, 48}) {
        auto copy_ns = RunDisabledPathBench(threads, kIters, /*force_copy=*/true);
        auto borrow_ns = RunDisabledPathBench(threads, kIters, /*force_copy=*/false);
        std::printf("%8d %14.1f %14.1f %10.1fx\n", threads, copy_ns, borrow_ns,
                    borrow_ns > 0 ? copy_ns / borrow_ns : 0.0);
    }
    SUCCEED();
}
