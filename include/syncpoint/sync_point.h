// Adapted from RocksDB's SyncPoint implementation
#pragma once

#include <assert.h>

#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef ENABLE_SYNCPOINT
#define TEST_SYNC_POINT(x)
#define TEST_IDX_SYNC_POINT(x, index)
#define TEST_SYNC_POINT_CALLBACK(x, y)
#define INIT_SYNC_POINT_SINGLETONS()
#else

namespace milvus {

class Slice {
 public:
    Slice() : data_(""), size_(0) {
    }
    Slice(const char* d, size_t n) : data_(d), size_(n) {
    }
    Slice(const std::string& s) : data_(s.data()), size_(s.size()) {
    }
    Slice(const char* s) : data_(s), size_(strlen(s)) {
    }

    const char*
    data() const {
        return data_;
    }
    size_t
    size() const {
        return size_;
    }
    std::string
    ToString() const {
        return std::string(data_, size_);
    }

 private:
    const char* data_;
    size_t size_;
};

// This class provides facility to reproduce race conditions deterministically
// in unit tests.
// Developer could specify sync points in the codebase via TEST_SYNC_POINT.
// Each sync point represents a position in the execution stream of a thread.
// In the unit test, 'Happens After' relationship among sync points could be
// setup via SyncPoint::LoadDependency, to reproduce a desired interleave of
// threads execution.

class SyncPoint {
 public:
    static SyncPoint*
    GetInstance();

    SyncPoint(const SyncPoint&) = delete;
    SyncPoint&
    operator=(const SyncPoint&) = delete;
    ~SyncPoint();

    struct SyncPointPair {
        std::string predecessor;
        std::string successor;
    };

    // call once at the beginning of a test to setup the dependency between
    // sync points. Specifically, execution will not be allowed to proceed past
    // each successor until execution has reached the corresponding predecessor,
    // in any thread.
    void
    LoadDependency(const std::vector<SyncPointPair>& dependencies);

    // call once at the beginning of a test to setup the dependency between
    // sync points and setup markers indicating the successor is only enabled
    // when it is processed on the same thread as the predecessor.
    // When adding a marker, it implicitly adds a dependency for the marker pair.
    void
    LoadDependencyAndMarkers(const std::vector<SyncPointPair>& dependencies, const std::vector<SyncPointPair>& markers);

    // The argument to the callback is passed through from
    // TEST_SYNC_POINT_CALLBACK(); nullptr if TEST_SYNC_POINT or
    // TEST_IDX_SYNC_POINT was used.
    void
    SetCallBack(const std::string& point, const std::function<void(void*)>& callback);

    // Clear callback function by point
    void
    ClearCallBack(const std::string& point);

    // Clear all call back functions.
    void
    ClearAllCallBacks();

    // Block at a specific sync point until it's unblocked
    void
    BlockAtPoint(const std::string& point);

    // Unblock a specific sync point
    void
    UnblockPoint(const std::string& point);

    // Clear all blocked points and wake up all waiting threads
    void
    ClearAllBlockedPoints();

    // enable sync point processing (disabled on startup)
    void
    EnableProcessing();

    // disable sync point processing
    void
    DisableProcessing();

    // remove the execution trace of all sync points
    void
    ClearTrace();

    // Reset all state (dependencies, callbacks, blocked points, traces)
    void
    Reset();

    // triggered by TEST_SYNC_POINT, blocking execution until all predecessors
    // are executed.
    // And/or call registered callback function, with argument `cb_arg`
    void
    Process(const Slice& point, void* cb_arg = nullptr);

    // template gets length of const string at compile time,
    //  avoiding strlen() at runtime
    template <size_t kLen>
    void
    Process(const char (&point)[kLen], void* cb_arg = nullptr) {
        static_assert(kLen > 0, "Must not be empty");
        assert(point[kLen - 1] == '\0');
        Process(Slice(point, kLen - 1), cb_arg);
    }

    // TODO: it might be useful to provide a function that blocks until all
    // sync points are cleared.

    // We want this to be public so we can
    // subclass the implementation
    struct Data;

 private:
    // Singleton
    SyncPoint();
    Data* impl_;
};

}  // namespace milvus

// Use TEST_SYNC_POINT to specify sync points inside code base.
// Sync points can have happens-after dependency on other sync points,
// configured at runtime via SyncPoint::LoadDependency. This could be
// utilized to re-produce race conditions between threads.
// TEST_SYNC_POINT is no op in release build.
#define TEST_SYNC_POINT(x) milvus::SyncPoint::GetInstance()->Process(x)
#define TEST_IDX_SYNC_POINT(x, index) milvus::SyncPoint::GetInstance()->Process(x + std::to_string(index))
#define TEST_SYNC_POINT_CALLBACK(x, y) milvus::SyncPoint::GetInstance()->Process(x, y)
#define INIT_SYNC_POINT_SINGLETONS() (void)milvus::SyncPoint::GetInstance();
#endif  // ENABLE_SYNCPOINT
