// Adapted from RocksDB's SyncPoint implementation

#include <assert.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "syncpoint/sync_point.h"

#pragma once

#ifdef ENABLE_SYNCPOINT
namespace milvus {

struct SyncPoint::Data {
    Data() : enabled_(false) {
    }
    // Enable proper deletion by subclasses
    virtual ~Data() {
    }
    // successor/predecessor map loaded from LoadDependency
    std::unordered_map<std::string, std::vector<std::string>> successors_;
    std::unordered_map<std::string, std::vector<std::string>> predecessors_;
    std::unordered_map<std::string, std::function<void(void*)>> callbacks_;
    std::unordered_map<std::string, std::vector<std::string>> markers_;
    std::unordered_map<std::string, std::thread::id> marked_thread_id_;

    std::mutex mutex_;
    std::condition_variable cv_;
    // sync points that have been passed through
    std::unordered_set<std::string> cleared_points_;
    // sync points that should block execution
    std::unordered_set<std::string> blocked_points_;
    std::atomic<bool> enabled_;
    int num_callbacks_running_ = 0;

    void
    LoadDependency(const std::vector<SyncPointPair>& dependencies);
    void
    LoadDependencyAndMarkers(const std::vector<SyncPointPair>& dependencies, const std::vector<SyncPointPair>& markers);
    bool
    PredecessorsAllCleared(const std::string& point);
    void
    SetCallBack(const std::string& point, const std::function<void(void*)>& callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_[point] = callback;
    }

    void
    ClearCallBack(const std::string& point);
    void
    ClearAllCallBacks();
    void
    EnableProcessing() {
        enabled_ = true;
    }
    void
    DisableProcessing() {
        enabled_ = false;
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();  // Wake up all blocked threads
    }
    void
    ClearTrace() {
        std::lock_guard<std::mutex> lock(mutex_);
        cleared_points_.clear();
    }

    void
    BlockAtPoint(const std::string& point) {
        std::lock_guard<std::mutex> lock(mutex_);
        blocked_points_.insert(point);
    }

    void
    UnblockPoint(const std::string& point) {
        std::lock_guard<std::mutex> lock(mutex_);
        blocked_points_.erase(point);
        cv_.notify_all();
    }

    void
    ClearAllBlockedPoints() {
        std::lock_guard<std::mutex> lock(mutex_);
        blocked_points_.clear();
        cv_.notify_all();
    }

    void
    Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        successors_.clear();
        predecessors_.clear();
        callbacks_.clear();
        markers_.clear();
        marked_thread_id_.clear();
        cleared_points_.clear();
        blocked_points_.clear();
        cv_.notify_all();
    }
    bool
    DisabledByMarker(const std::string& point, std::thread::id thread_id) {
        auto marked_point_iter = marked_thread_id_.find(point);
        return marked_point_iter != marked_thread_id_.end() && thread_id != marked_point_iter->second;
    }
    void
    Process(const Slice& point, void* cb_arg);
};
}  // namespace milvus
#endif  // ENABLE_SYNCPOINT
