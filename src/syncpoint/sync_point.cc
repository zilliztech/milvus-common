// Adapted from RocksDB's SyncPoint implementation

#include "syncpoint/sync_point.h"

#include <fcntl.h>

#include "syncpoint/sync_point_impl.h"

#ifdef ENABLE_SYNCPOINT
namespace milvus {

SyncPoint*
SyncPoint::GetInstance() {
    static SyncPoint sync_point;
    return &sync_point;
}

SyncPoint::SyncPoint() : impl_(new Data) {
}

SyncPoint::~SyncPoint() {
    delete impl_;
}

void
SyncPoint::LoadDependency(const std::vector<SyncPointPair>& dependencies) {
    impl_->LoadDependency(dependencies);
}

void
SyncPoint::LoadDependencyAndMarkers(const std::vector<SyncPointPair>& dependencies,
                                    const std::vector<SyncPointPair>& markers) {
    impl_->LoadDependencyAndMarkers(dependencies, markers);
}

void
SyncPoint::SetCallBack(const std::string& point, const std::function<void(void*)>& callback) {
    impl_->SetCallBack(point, callback);
}

void
SyncPoint::ClearCallBack(const std::string& point) {
    impl_->ClearCallBack(point);
}

void
SyncPoint::ClearAllCallBacks() {
    impl_->ClearAllCallBacks();
}

void
SyncPoint::EnableProcessing() {
    impl_->EnableProcessing();
}

void
SyncPoint::DisableProcessing() {
    impl_->DisableProcessing();
}

void
SyncPoint::ClearTrace() {
    impl_->ClearTrace();
}

void
SyncPoint::BlockAtPoint(const std::string& point) {
    impl_->BlockAtPoint(point);
}

void
SyncPoint::UnblockPoint(const std::string& point) {
    impl_->UnblockPoint(point);
}

void
SyncPoint::ClearAllBlockedPoints() {
    impl_->ClearAllBlockedPoints();
}

void
SyncPoint::Reset() {
    impl_->Reset();
}

void
SyncPoint::Process(const Slice& point, void* cb_arg) {
    impl_->Process(point, cb_arg);
}
}  // namespace milvus
#endif  // ENABLE_SYNCPOINT
