#include "filemanager/impl/LocalFileManager.h"

#include "filemanager/impl/LocalInputStream.h"
#include "filemanager/impl/LocalOutputStream.h"

namespace milvus {

bool
LocalFileManager::LoadFile(const std::string& filename) {
    return true;
}

bool
LocalFileManager::AddFile(const std::string& filename) {
    files.insert(filename);
    return true;
}

std::optional<bool>
LocalFileManager::IsExisted(const std::string& filename) {
    return std::make_optional<bool>(files.find(filename) != files.end());
}

bool
LocalFileManager::RemoveFile(const std::string& filename) {
    files.erase(filename);
    return true;
}

std::shared_ptr<InputStream>
LocalFileManager::OpenInputStream(const std::string& filename) {
    return std::make_shared<LocalInputStream>(filename);
}

std::shared_ptr<OutputStream>
LocalFileManager::OpenOutputStream(const std::string& filename) {
    return std::make_shared<LocalOutputStream>(filename);
}

}  // namespace milvus
