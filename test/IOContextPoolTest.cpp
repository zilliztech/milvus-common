#include <fcntl.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

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

TEST_F(IOContextPoolTestFixture, ReadAsyncShouldBeDeferredFuture) {
    const char path[] = "/tmp/io_reader_async_mode_test.bin";
    int fd = ::open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
    ASSERT_GE(fd, 0);

    constexpr size_t kSize = 4096;
    std::vector<std::byte> content(kSize, std::byte{0x3});
    ASSERT_EQ(::write(fd, content.data(), static_cast<size_t>(kSize)), static_cast<ssize_t>(kSize));

    auto reader = IOReader(fd);
    std::vector<std::byte> buffer(kSize);
    std::vector<std::byte*> buffers{buffer.data()};
    std::vector<size_t> offsets{0};

    auto fut = reader.ReadAsync(std::move(buffers), kSize, std::move(offsets));
    ASSERT_EQ(fut.wait_for(std::chrono::milliseconds(0)), std::future_status::deferred);

    ::close(fd);
    ::unlink(path);
}
#endif
