// Adapted from RocksDB's SyncPoint implementation

#include "syncpoint/sync_point_impl.h"

#ifdef ENABLE_SYNCPOINT
namespace milvus {

void
SyncPoint::Data::LoadDependency(const std::vector<SyncPointPair>& dependencies) {
    std::lock_guard<std::mutex> lock(mutex_);
    successors_.clear();
    predecessors_.clear();
    cleared_points_.clear();
    for (const auto& dependency : dependencies) {
        successors_[dependency.predecessor].push_back(dependency.successor);
        predecessors_[dependency.successor].push_back(dependency.predecessor);
    }
    cv_.notify_all();
}

void
SyncPoint::Data::LoadDependencyAndMarkers(const std::vector<SyncPointPair>& dependencies,
                                          const std::vector<SyncPointPair>& markers) {
    std::lock_guard<std::mutex> lock(mutex_);
    successors_.clear();
    predecessors_.clear();
    cleared_points_.clear();
    markers_.clear();
    marked_thread_id_.clear();
    for (const auto& dependency : dependencies) {
        successors_[dependency.predecessor].push_back(dependency.successor);
        predecessors_[dependency.successor].push_back(dependency.predecessor);
    }
    for (const auto& marker : markers) {
        successors_[marker.predecessor].push_back(marker.successor);
        predecessors_[marker.successor].push_back(marker.predecessor);
        markers_[marker.predecessor].push_back(marker.successor);
    }
    cv_.notify_all();
}

bool
SyncPoint::Data::PredecessorsAllCleared(const std::string& point) {
    for (const auto& pred : predecessors_[point]) {
        if (cleared_points_.count(pred) == 0) {
            return false;
        }
    }
    return true;
}

void
SyncPoint::Data::ClearCallBack(const std::string& point) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (num_callbacks_running_ > 0) {
        cv_.wait(lock);
    }
    callbacks_.erase(point);
}

void
SyncPoint::Data::ClearAllCallBacks() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (num_callbacks_running_ > 0) {
        cv_.wait(lock);
    }
    callbacks_.clear();
}

void
SyncPoint::Data::Process(const Slice& point, void* cb_arg) {
    if (!enabled_) {
        return;
    }

    // Must convert to std::string for remaining work.  Take
    //  heap hit.
    std::string point_string(point.ToString());
    std::unique_lock<std::mutex> lock(mutex_);
    auto thread_id = std::this_thread::get_id();

    auto marker_iter = markers_.find(point_string);
    if (marker_iter != markers_.end()) {
        for (auto& marked_point : marker_iter->second) {
            marked_thread_id_.emplace(marked_point, thread_id);
        }
    }

    if (DisabledByMarker(point_string, thread_id)) {
        return;
    }

    // Check if this point should block
    while (blocked_points_.count(point_string) > 0) {
        cv_.wait(lock);
        if (!enabled_) {
            return;
        }
    }

    while (!PredecessorsAllCleared(point_string)) {
        cv_.wait(lock);
        if (DisabledByMarker(point_string, thread_id)) {
            return;
        }
        // Also check if blocking was cleared
        if (!enabled_) {
            return;
        }
    }

    auto callback_pair = callbacks_.find(point_string);
    if (callback_pair != callbacks_.end()) {
        num_callbacks_running_++;
        mutex_.unlock();
        callback_pair->second(cb_arg);
        mutex_.lock();
        num_callbacks_running_--;
    }
    cleared_points_.insert(point_string);
    cv_.notify_all();
}
}  // namespace milvus
#endif  // ENABLE_SYNCPOINT
