#pragma once

#ifdef USE_REDIS

#include <memory>
#include <hiredis/hiredis.h>

namespace milvus {
namespace ncs {

using RedisReplyPtr = std::unique_ptr<redisReply, decltype(&freeReplyObject)>;
using RedisContextPtr = std::unique_ptr<redisContext, decltype(&redisFree)>;

} // namespace ncs
} // namespace milvus

#endif // USE_REDIS
