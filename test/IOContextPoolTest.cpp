#include <fcntl.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "knowhere/io_context_pool.h"
#include "knowhere/io_reader.h"
#ifdef MILVUS_COMMON_WITH_LIBAIO
#include "knowhere/aio_context_pool.h"
#endif
#ifdef WITH_IO_URING
#include "knowhere/uring_context_pool.h"
#endif

class IOContextPoolTestFixture : public ::testing::Test {
 protected:
    void
    SetUp() override {
        IOContextPool::ResetGlobalForTest();
    }

    void
    TearDown() override {
        IOContextPool::ResetGlobalForTest();
    }
};

#ifdef WITH_IO_URING
#ifdef MILVUS_COMMON_WITH_LIBAIO
TEST_F(IOContextPoolTestFixture, InitShouldFallbackToAioWhenUringUnavailable) {
    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        struct rlimit lim;
        lim.rlim_cur = 3;
        lim.rlim_max = 3;
        if (setrlimit(RLIMIT_NOFILE, &lim) != 0) {
            _exit(20);
        }

        IOContextPoolConfig cfg;
        cfg.num_ctx = 1;
        cfg.max_events = 128;
        const bool ok = IOContextPool::InitGlobal(cfg);
        if (!ok) {
            _exit(21);
        }

        auto pool = IOContextPool::GetGlobal();
        if (pool == nullptr || pool->Backend() != IOBackend::AIO) {
            _exit(22);
        }
        _exit(0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
}
#endif
#endif

TEST_F(IOContextPoolTestFixture, BackendIsSelectedAtInit) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 2;
    cfg.max_events = 128;

    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);

    auto backend = pool->Backend();
#ifdef WITH_IO_URING
    ASSERT_EQ(backend, IOBackend::IO_URING);
#else
    ASSERT_EQ(backend, IOBackend::AIO);
#endif
}

TEST_F(IOContextPoolTestFixture, InvalidConfigRejected) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 0;
    cfg.max_events = 128;

    ASSERT_FALSE(IOContextPool::InitGlobal(cfg));
}

TEST_F(IOContextPoolTestFixture, ReinitWithDifferentConfigShouldFail) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 2;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    IOContextPoolConfig mismatch = cfg;
    mismatch.num_ctx = 4;

    ASSERT_FALSE(IOContextPool::InitGlobal(mismatch));
}

TEST_F(IOContextPoolTestFixture, ResetGlobalForTestShouldClearSingletonState) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 2;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    IOContextPoolConfig mismatch = cfg;
    mismatch.num_ctx = 4;

    ASSERT_FALSE(IOContextPool::InitGlobal(mismatch));
    IOContextPool::ResetGlobalForTest();
    ASSERT_TRUE(IOContextPool::InitGlobal(mismatch));
}

#ifdef MILVUS_COMMON_WITH_LIBAIO
TEST_F(IOContextPoolTestFixture, DefaultConfigShouldMatchLegacyAioPoolSize) {
    IOContextPoolConfig cfg;
    ASSERT_EQ(cfg.num_ctx, default_pool_size);
    ASSERT_EQ(cfg.max_events, default_max_events);
}
#else
TEST_F(IOContextPoolTestFixture, DefaultConfigShouldNotUseSingleContext) {
    IOContextPoolConfig cfg;
    ASSERT_GT(cfg.num_ctx, 1u);
    ASSERT_EQ(cfg.max_events, 128u);
}
#endif

TEST_F(IOContextPoolTestFixture, ReaderCanBeConstructed) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 2;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    IOReader reader;
#ifdef WITH_IO_URING
    ASSERT_EQ(reader.Backend(), IOBackend::IO_URING);
#else
    ASSERT_EQ(reader.Backend(), IOBackend::AIO);
#endif
}

TEST_F(IOContextPoolTestFixture, UnifiedPopPushShouldUseSelectedBackend) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);

    auto handle = pool->Pop();
    ASSERT_EQ(handle.backend, pool->Backend());
#ifdef WITH_IO_URING
    ASSERT_EQ(handle.backend, IOBackend::IO_URING);
    ASSERT_NE(handle.uring, nullptr);
#endif
#if !defined(WITH_IO_URING) && defined(MILVUS_COMMON_WITH_LIBAIO)
    ASSERT_EQ(handle.backend, IOBackend::AIO);
    ASSERT_NE(handle.aio, nullptr);
#endif
    pool->Push(handle);

    auto second = pool->Pop();
    ASSERT_EQ(second.backend, pool->Backend());
    pool->Push(second);
}

TEST_F(IOContextPoolTestFixture, PushShouldRejectHandleFromDifferentBackend) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);

    auto handle = pool->Pop();
    ASSERT_EQ(handle.backend, pool->Backend());

    IOContextHandle mismatched = handle;
    mismatched.backend = handle.backend == IOBackend::AIO ? IOBackend::IO_URING : IOBackend::AIO;
    pool->Push(mismatched);
    pool->Push(handle);

    auto second = pool->Pop();
    ASSERT_EQ(second.backend, pool->Backend());
    pool->Push(second);
}

TEST_F(IOContextPoolTestFixture, IoReaderSpanShouldUseCompatSpanType) {
    EXPECT_TRUE((std::is_same_v<IOReaderSpan<int>, knowhere_compat::span<int>>));
#if defined(__cpp_lib_span)
    EXPECT_FALSE((std::is_same_v<IOReaderSpan<int>, std::span<int>>));
#endif
}

#ifdef MILVUS_COMMON_WITH_LIBAIO
TEST_F(IOContextPoolTestFixture, LegacyAioInitStillWorksViaUnifiedPath) {
#ifdef WITH_IO_URING
    ASSERT_FALSE(AioContextPool::InitGlobalAioPool(2, 128));
    ASSERT_EQ(AioContextPool::GetGlobalAioPool(), nullptr);
#else
    ASSERT_TRUE(AioContextPool::InitGlobalAioPool(2, 64));
    auto p = AioContextPool::GetGlobalAioPool();
    ASSERT_NE(p, nullptr);
    auto io_pool = IOContextPool::GetGlobal();
    ASSERT_NE(io_pool, nullptr);
    ASSERT_EQ(io_pool->Backend(), IOBackend::AIO);
    ASSERT_EQ(io_pool->MaxEventsPerCtx(), 64u);
#endif
}

TEST_F(IOContextPoolTestFixture, LegacyAioValidationReinitMismatchShouldFail) {
    ASSERT_TRUE(AioContextPool::InitGlobalAioPoolWithValidation(2, 128));
    ASSERT_FALSE(AioContextPool::InitGlobalAioPoolWithValidation(4, 128));
}

TEST_F(IOContextPoolTestFixture, LegacyAioValidationShouldRejectZeroMaxEvents) {
    ASSERT_FALSE(AioContextPool::InitGlobalAioPoolWithValidation(1, 0));
    ASSERT_TRUE(AioContextPool::InitGlobalAioPoolWithValidation(1, default_max_events));

    auto pool = AioContextPool::GetGlobalAioPoolDirect();
    ASSERT_NE(pool, nullptr);
    ASSERT_EQ(pool->max_events_per_ctx(), default_max_events);
}

TEST_F(IOContextPoolTestFixture, LegacyAioPopShouldReturnNullAfterShutdown) {
    ASSERT_TRUE(AioContextPool::InitGlobalAioPoolWithValidation(1, 128));
    auto pool = AioContextPool::GetGlobalAioPoolDirect();
    ASSERT_NE(pool, nullptr);

    auto first = pool->pop();
    ASSERT_NE(first, nullptr);

    auto blocked = std::async(std::launch::async, [&]() { return pool->pop(); });
    ASSERT_EQ(blocked.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    pool->Shutdown();

    ASSERT_EQ(blocked.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_EQ(blocked.get(), nullptr);
}
#endif

#ifdef WITH_IO_URING
TEST_F(IOContextPoolTestFixture, LegacyUringInitStillWorksViaUnifiedPath) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto io_pool = IOContextPool::GetGlobal();
    ASSERT_NE(io_pool, nullptr);
    if (io_pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    ASSERT_TRUE(UringContextPool::InitGlobalUringPool(1, 128));
    auto p = UringContextPool::GetGlobalUringPool();
    ASSERT_NE(p, nullptr);
}

TEST_F(IOContextPoolTestFixture, LegacyUringValidationReinitMismatchShouldFail) {
    ASSERT_TRUE(UringContextPool::InitGlobalUringPoolWithValidation(1, 64));
    ASSERT_FALSE(UringContextPool::InitGlobalUringPoolWithValidation(2, 64));
}

TEST_F(IOContextPoolTestFixture, LegacyUringInitShouldHonorRequestedConfig) {
    ASSERT_TRUE(UringContextPool::InitGlobalUringPool(1, 64));

    auto io_pool = IOContextPool::GetGlobal();
    ASSERT_NE(io_pool, nullptr);
    ASSERT_EQ(io_pool->Backend(), IOBackend::IO_URING);
    ASSERT_EQ(io_pool->MaxEventsPerCtx(), 64u);
}
#endif

namespace {
constexpr size_t kIOReaderTestBlockSize = 4096;

struct AlignedBufferDeleter {
    void
    operator()(std::byte* ptr) const {
        std::free(ptr);
    }
};

using AlignedBuffer = std::unique_ptr<std::byte, AlignedBufferDeleter>;

AlignedBuffer
AllocateAlignedBuffer(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, kIOReaderTestBlockSize, size) != 0) {
        return nullptr;
    }
    std::memset(ptr, 0, size);
    return AlignedBuffer(static_cast<std::byte*>(ptr));
}

int
OpenIOReaderTestFile(char* path, bool needs_direct_io) {
    int tmp_fd = ::mkstemp(path);
    if (tmp_fd < 0) {
        return -1;
    }
    ::close(tmp_fd);

    int flags = O_CREAT | O_TRUNC | O_RDWR;
#ifdef O_DIRECT
    if (needs_direct_io) {
        flags |= O_DIRECT;
    }
#else
    if (needs_direct_io) {
        return -1;
    }
#endif
    return ::open(path, flags, 0644);
}
}  // namespace

TEST_F(IOContextPoolTestFixture, ReadAsyncShouldSubmitFirstBatchBeforeFutureWait) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 1;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    const bool needs_direct_io = pool->Backend() == IOBackend::AIO;

    char path[] = "/tmp/io_reader_eager_submit_XXXXXX";
    int fd = OpenIOReaderTestFile(path, needs_direct_io);
    if (fd < 0 && needs_direct_io) {
        ::unlink(path);
        GTEST_SKIP() << "direct I/O is not available for AIO ReadAsync test";
    }
    ASSERT_GE(fd, 0);

    auto content = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto buffer = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(buffer, nullptr);
    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x3});

    const auto written = ::pwrite(fd, content.get(), kIOReaderTestBlockSize, 0);
    if (written < 0 && needs_direct_io && errno == EINVAL) {
        ::close(fd);
        ::unlink(path);
        GTEST_SKIP() << "filesystem does not support direct I/O";
    }
    ASSERT_EQ(written, static_cast<ssize_t>(kIOReaderTestBlockSize));
    ASSERT_EQ(::fsync(fd), 0);

    auto reader = IOReader(fd);
    std::vector<std::byte*> buffers{buffer.get()};
    std::vector<size_t> offsets{0};

    auto fut = reader.ReadAsync(std::move(buffers), kIOReaderTestBlockSize, std::move(offsets));
    ASSERT_EQ(::close(fd), 0);
    fd = -1;

    ASSERT_TRUE(fut.get());
    ASSERT_EQ(std::memcmp(buffer.get(), content.get(), kIOReaderTestBlockSize), 0);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, ReadAsyncShouldReadMultipleBatches) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 1;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    const bool needs_direct_io = pool->Backend() == IOBackend::AIO;

    char path[] = "/tmp/io_reader_multi_batch_XXXXXX";
    int fd = OpenIOReaderTestFile(path, needs_direct_io);
    if (fd < 0 && needs_direct_io) {
        ::unlink(path);
        GTEST_SKIP() << "direct I/O is not available for AIO ReadAsync test";
    }
    ASSERT_GE(fd, 0);

    constexpr size_t kTotalSize = kIOReaderTestBlockSize * 2;
    auto content = AllocateAlignedBuffer(kTotalSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x4});
    std::fill(content.get() + kIOReaderTestBlockSize, content.get() + kTotalSize, std::byte{0x5});

    const auto written = ::pwrite(fd, content.get(), kTotalSize, 0);
    if (written < 0 && needs_direct_io && errno == EINVAL) {
        ::close(fd);
        ::unlink(path);
        GTEST_SKIP() << "filesystem does not support direct I/O";
    }
    ASSERT_EQ(written, static_cast<ssize_t>(kTotalSize));
    ASSERT_EQ(::fsync(fd), 0);

    auto reader = IOReader(fd);
    std::vector<std::byte*> buffers{first.get(), second.get()};
    std::vector<size_t> offsets{0, kIOReaderTestBlockSize};

    auto fut = reader.ReadAsync(std::move(buffers), kIOReaderTestBlockSize, std::move(offsets));
    ASSERT_TRUE(fut.get());
    ASSERT_EQ(std::memcmp(first.get(), content.get(), kIOReaderTestBlockSize), 0);
    ASSERT_EQ(std::memcmp(second.get(), content.get() + kIOReaderTestBlockSize, kIOReaderTestBlockSize), 0);

    ::close(fd);
    ::unlink(path);
}
