#include <fcntl.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "knowhere/io_completion_reader.h"
#include "knowhere/io_context_pool.h"
#include "knowhere/io_reader.h"
#include "syncpoint/sync_point.h"
#include "../src/knowhere/io_reader_internal.h"
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

#if defined(MILVUS_COMMON_WITH_LIBAIO) && defined(ENABLE_SYNCPOINT)
TEST_F(IOContextPoolTestFixture, PartialUringInitShouldFallbackToAio) {
    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    int init_calls = 0;
    sync_point->SetCallBack("UringContextPool::Ctor:BeforeInit", [&](void* arg) {
        if (init_calls++ == 1) {
            *static_cast<int*>(arg) = -EMFILE;
        }
    });
    sync_point->EnableProcessing();

    IOContextPoolConfig cfg;
    cfg.num_ctx = 2;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    ASSERT_EQ(pool->Backend(), IOBackend::AIO);
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
    ASSERT_TRUE(backend == IOBackend::IO_URING || backend == IOBackend::AIO);
#else
    ASSERT_EQ(backend, IOBackend::AIO);
#endif
}

#ifdef WITH_IO_URING
TEST_F(IOContextPoolTestFixture, RequiredIoUringBackendShouldBeSelected) {
    const char* require_uring = std::getenv("KNOWHERE_REQUIRE_IO_URING");
    if (require_uring == nullptr || std::strcmp(require_uring, "1") != 0) {
        GTEST_SKIP() << "io_uring backend is not required by this environment";
    }

    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    ASSERT_EQ(pool->Backend(), IOBackend::IO_URING);
}
#endif

TEST_F(IOContextPoolTestFixture, InvalidConfigRejected) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 0;
    cfg.max_events = 128;

    ASSERT_FALSE(IOContextPool::InitGlobal(cfg));
}

TEST_F(IOContextPoolTestFixture, GetGlobalShouldNotSelectBackendBeforeExplicitInit) {
    ASSERT_EQ(IOContextPool::GetGlobal(), nullptr);
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

TEST_F(IOContextPoolTestFixture, GetGlobalOrInitShouldRejectDifferentConfig) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    IOContextPoolConfig mismatch = cfg;
    mismatch.num_ctx = 2;

    ASSERT_EQ(IOContextPool::GetGlobalOrInit(mismatch), nullptr);
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
    ASSERT_EQ(reader.Backend(), IOContextPool::GetGlobal()->Backend());
}

TEST_F(IOContextPoolTestFixture, ReaderShouldNotImplicitlyInitializeGlobalPool) {
    ASSERT_EQ(IOContextPool::GetGlobal(), nullptr);

    IOReader reader;
    ASSERT_EQ(IOContextPool::GetGlobal(), nullptr);
    ASSERT_FALSE(reader.IsReady());
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
    if (pool->Backend() == IOBackend::IO_URING) {
        ASSERT_EQ(handle.backend, IOBackend::IO_URING);
        ASSERT_NE(handle.uring, nullptr);
    }
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

TEST_F(IOContextPoolTestFixture, HandleDestructorShouldReturnCheckedOutContext) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);

    {
        auto handle = pool->Pop();
        ASSERT_TRUE(handle.HasContext());
    }

    auto second = std::async(std::launch::async, [&] { return pool->Pop(); });
    ASSERT_EQ(second.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto handle = second.get();
    ASSERT_TRUE(handle.HasContext());
    pool->Push(handle);
}

TEST_F(IOContextPoolTestFixture, MoveAssignHandleShouldReturnPreviousContext) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 2;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);

    auto first = pool->Pop();
    auto second = pool->Pop();
    ASSERT_TRUE(first.HasContext());
    ASSERT_TRUE(second.HasContext());

    second = std::move(first);
    ASSERT_TRUE(second.HasContext());

    auto returned = std::async(std::launch::async, [&] { return pool->Pop(); });
    ASSERT_EQ(returned.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto returned_handle = returned.get();
    ASSERT_TRUE(returned_handle.HasContext());

    pool->Push(returned_handle);
    pool->Push(second);
}

TEST_F(IOContextPoolTestFixture, RetireHandleShouldMakePoolFailFast) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);

    auto handle = pool->Pop();
    ASSERT_TRUE(handle.HasContext());
    ASSERT_TRUE(pool->Release(handle, IOContextReleaseDisposition::Retire));

    auto failed = std::async(std::launch::async, [&] { return pool->Pop(); });
    ASSERT_EQ(failed.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_FALSE(failed.get().HasContext());
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

    IOContextHandle mismatched;
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

TEST_F(IOContextPoolTestFixture, InterruptedSyscallRetryShouldHaveBound) {
    size_t retry = 0;

    EXPECT_TRUE(knowhere_internal::ShouldRetryInterruptedSyscall(retry, 2));
    EXPECT_EQ(retry, 1u);
    EXPECT_TRUE(knowhere_internal::ShouldRetryInterruptedSyscall(retry, 2));
    EXPECT_EQ(retry, 2u);
    EXPECT_FALSE(knowhere_internal::ShouldRetryInterruptedSyscall(retry, 2));
    EXPECT_EQ(retry, 3u);
}

#ifdef ENABLE_SYNCPOINT
TEST_F(IOContextPoolTestFixture, ResetFailureShouldMakePoolFailFast) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
#ifdef WITH_IO_URING
    sync_point->SetCallBack("UringContextPool::ResetCheckedOut:BeforeInit", [](void* arg) {
        *static_cast<int*>(arg) = -EMFILE;
    });
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
    sync_point->SetCallBack("AioContextPool::ResetCheckedOut:BeforeSetup", [](void* arg) {
        *static_cast<int*>(arg) = -EAGAIN;
    });
#endif
    sync_point->EnableProcessing();

    auto handle = pool->Pop();
    ASSERT_TRUE(handle.HasContext());
    ASSERT_FALSE(pool->Reset(handle));

    auto blocked = std::async(std::launch::async, [&] { return pool->Pop(); });
    ASSERT_EQ(blocked.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto failed = blocked.get();
    ASSERT_FALSE(failed.HasContext());

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
}
#endif

#if defined(MILVUS_COMMON_WITH_LIBAIO) && defined(ENABLE_SYNCPOINT)
TEST_F(IOContextPoolTestFixture, AioDestroyFailureShouldMakePoolFailFast) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::AIO) {
        GTEST_SKIP() << "AIO backend unavailable";
    }

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    int destroy_calls = 0;
    sync_point->SetCallBack("AioContextPool::DestroyContext:AfterDestroy", [&](void* arg) {
        if (destroy_calls++ == 0) {
            *static_cast<int*>(arg) = -EINVAL;
        }
    });
    sync_point->EnableProcessing();

    auto handle = pool->Pop();
    ASSERT_TRUE(handle.HasContext());
    ASSERT_FALSE(pool->Reset(handle));

    auto failed = std::async(std::launch::async, [&] { return pool->Pop(); });
    ASSERT_EQ(failed.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_FALSE(failed.get().HasContext());

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
}
#endif

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

TEST_F(IOContextPoolTestFixture, LegacyAioPoolShouldRejectDoublePush) {
    ASSERT_TRUE(AioContextPool::InitGlobalAioPoolWithValidation(1, 128));
    auto pool = AioContextPool::GetGlobalAioPoolDirect();
    ASSERT_NE(pool, nullptr);

    auto first = pool->pop();
    ASSERT_NE(first, nullptr);
    pool->push(first);
    pool->push(first);

    auto checked_out = pool->pop();
    ASSERT_EQ(checked_out, first);

    auto blocked = std::async(std::launch::async, [&]() { return pool->pop(); });
    ASSERT_EQ(blocked.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    pool->push(checked_out);
    ASSERT_EQ(blocked.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto second = blocked.get();
    ASSERT_EQ(second, first);
    pool->push(second);
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
    if (!UringContextPool::InitGlobalUringPool(1, 64)) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    auto io_pool = IOContextPool::GetGlobal();
    ASSERT_NE(io_pool, nullptr);
    ASSERT_EQ(io_pool->Backend(), IOBackend::IO_URING);
    ASSERT_EQ(io_pool->MaxEventsPerCtx(), 64u);
}

TEST_F(IOContextPoolTestFixture, LegacyUringDirectDefaultShouldProvideMultipleContexts) {
    auto pool = UringContextPool::GetGlobalUringPoolDirect();
    ASSERT_NE(pool, nullptr);
    if (!pool->IsUsable()) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    auto* first = pool->pop();
    ASSERT_NE(first, nullptr);

    auto second_future = std::async(std::launch::async, [&] { return pool->pop(); });
    if (second_future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
        pool->push(first);
        first = nullptr;
        auto* second = second_future.get();
        if (second != nullptr) {
            pool->push(second);
        }
        FAIL() << "direct uring default should expose more than one context";
    }

    auto* second = second_future.get();
    ASSERT_NE(second, nullptr);
    pool->push(second);
    pool->push(first);
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

TEST_F(IOContextPoolTestFixture, ReadAsyncShouldReturnFalseOnShortRead) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    const bool needs_direct_io = pool->Backend() == IOBackend::AIO;

    char path[] = "/tmp/io_reader_short_read_XXXXXX";
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
    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0xa});

    const auto written = ::pwrite(fd, content.get(), kIOReaderTestBlockSize, 0);
    if (written < 0 && needs_direct_io && errno == EINVAL) {
        ::close(fd);
        ::unlink(path);
        GTEST_SKIP() << "filesystem does not support direct I/O";
    }
    ASSERT_EQ(written, static_cast<ssize_t>(kIOReaderTestBlockSize));
    ASSERT_EQ(::fsync(fd), 0);

    IOReader reader(fd, pool);
    std::vector<std::byte*> buffers{buffer.get()};
    std::vector<size_t> offsets{kIOReaderTestBlockSize};

    auto fut = reader.ReadAsync(std::move(buffers), kIOReaderTestBlockSize, std::move(offsets));
    ASSERT_FALSE(fut.get());

    ::close(fd);
    ::unlink(path);
}

#if defined(WITH_IO_URING) && defined(ENABLE_SYNCPOINT)
TEST_F(IOContextPoolTestFixture, ReadAsyncSubmitFailureShouldResetUringHandle) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 1;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_reader_submit_failure_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    constexpr size_t kTotalSize = kIOReaderTestBlockSize * 2;
    auto content = AllocateAlignedBuffer(kTotalSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto retry = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(retry, nullptr);

    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x8});
    std::fill(content.get() + kIOReaderTestBlockSize, content.get() + kTotalSize, std::byte{0x9});
    ASSERT_EQ(::pwrite(fd, content.get(), kTotalSize, 0), static_cast<ssize_t>(kTotalSize));
    ASSERT_EQ(::fsync(fd), 0);

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    int submit_calls = 0;
    sync_point->SetCallBack("IOReader::SubmitUring:Before", [&](void* arg) {
        if (submit_calls++ == 1) {
            *static_cast<int*>(arg) = -EIO;
        }
    });
    sync_point->EnableProcessing();

    auto reader = IOReader(fd, pool);
    std::vector<std::byte*> buffers{first.get(), second.get()};
    std::vector<size_t> offsets{0, kIOReaderTestBlockSize};
    auto failed = reader.ReadAsync(std::move(buffers), kIOReaderTestBlockSize, std::move(offsets));
    ASSERT_FALSE(failed.get());

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    std::vector<std::byte*> retry_buffers{retry.get()};
    std::vector<size_t> retry_offsets{0};
    auto retried = reader.ReadAsync(std::move(retry_buffers), kIOReaderTestBlockSize, std::move(retry_offsets));
    ASSERT_TRUE(retried.get());
    ASSERT_EQ(std::memcmp(retry.get(), content.get(), kIOReaderTestBlockSize), 0);

    ::close(fd);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, DroppedReadAsyncFutureShouldResetPartialUringSubmit) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 128;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_reader_dropped_future_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    constexpr size_t kTotalSize = kIOReaderTestBlockSize * 2;
    auto content = AllocateAlignedBuffer(kTotalSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto retry = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(retry, nullptr);

    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0xb});
    std::fill(content.get() + kIOReaderTestBlockSize, content.get() + kTotalSize, std::byte{0xc});
    ASSERT_EQ(::pwrite(fd, content.get(), kTotalSize, 0), static_cast<ssize_t>(kTotalSize));
    ASSERT_EQ(::fsync(fd), 0);

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    int submit_calls = 0;
    sync_point->SetCallBack("IOReader::SubmitUring:Before", [&](void* arg) {
        if (submit_calls++ == 0) {
            *static_cast<int*>(arg) = 1;
        }
    });
    sync_point->EnableProcessing();

    auto reader = IOReader(fd, pool);
    {
        std::vector<std::byte*> buffers{first.get(), second.get()};
        std::vector<size_t> offsets{0, kIOReaderTestBlockSize};
        auto dropped = reader.ReadAsync(std::move(buffers), kIOReaderTestBlockSize, std::move(offsets));
    }

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    std::vector<std::byte*> retry_buffers{retry.get()};
    std::vector<size_t> retry_offsets{0};
    auto retried = reader.ReadAsync(std::move(retry_buffers), kIOReaderTestBlockSize, std::move(retry_offsets));
    ASSERT_TRUE(retried.get());
    ASSERT_EQ(std::memcmp(retry.get(), content.get(), kIOReaderTestBlockSize), 0);

    ::close(fd);
    ::unlink(path);
}
#endif

#if defined(MILVUS_COMMON_WITH_LIBAIO) && defined(ENABLE_SYNCPOINT)
TEST_F(IOContextPoolTestFixture, DroppedAioReadAsyncFutureShouldResetContext) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 2;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::AIO) {
        GTEST_SKIP() << "AIO backend unavailable";
    }

    char path[] = "/tmp/io_reader_aio_dropped_future_XXXXXX";
    int fd = OpenIOReaderTestFile(path, true);
    if (fd < 0 && errno == EINVAL) {
        GTEST_SKIP() << "direct I/O is not available for AIO ReadAsync test";
    }
    ASSERT_GE(fd, 0);

    auto content = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto retry = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(retry, nullptr);
    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x11});
    ASSERT_EQ(::pwrite(fd, content.get(), kIOReaderTestBlockSize, 0), static_cast<ssize_t>(kIOReaderTestBlockSize));
    ASSERT_EQ(::fsync(fd), 0);

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    int reset_calls = 0;
    sync_point->SetCallBack("AioContextPool::ResetCheckedOut:BeforeSetup", [&](void* arg) {
        ++reset_calls;
        *static_cast<int*>(arg) = 0;
    });
    sync_point->EnableProcessing();

    IOReader reader(fd, pool);
    {
        std::vector<std::byte*> buffers{first.get()};
        std::vector<size_t> offsets{0};
        auto dropped = reader.ReadAsync(std::move(buffers), kIOReaderTestBlockSize, std::move(offsets));
    }
    ASSERT_GT(reset_calls, 0);

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    std::vector<std::byte*> retry_buffers{retry.get()};
    std::vector<size_t> retry_offsets{0};
    auto retried = reader.ReadAsync(std::move(retry_buffers), kIOReaderTestBlockSize, std::move(retry_offsets));
    ASSERT_TRUE(retried.get());
    ASSERT_EQ(std::memcmp(retry.get(), content.get(), kIOReaderTestBlockSize), 0);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, AioLaterBatchCleanupFailureShouldResetContext) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 2;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::AIO) {
        GTEST_SKIP() << "AIO backend unavailable";
    }

    char path[] = "/tmp/io_reader_aio_later_cleanup_XXXXXX";
    int fd = OpenIOReaderTestFile(path, true);
    if (fd < 0 && errno == EINVAL) {
        GTEST_SKIP() << "direct I/O is not available for AIO ReadAsync test";
    }
    ASSERT_GE(fd, 0);

    constexpr size_t kBlocks = 4;
    constexpr size_t kTotalSize = kIOReaderTestBlockSize * kBlocks;
    auto content = AllocateAlignedBuffer(kTotalSize);
    ASSERT_NE(content, nullptr);
    std::fill(content.get(), content.get() + kTotalSize, std::byte{0x4});
    ASSERT_EQ(::pwrite(fd, content.get(), kTotalSize, 0), static_cast<ssize_t>(kTotalSize));
    ASSERT_EQ(::fsync(fd), 0);

    std::vector<AlignedBuffer> owned_buffers;
    std::vector<std::byte*> buffers;
    std::vector<size_t> offsets;
    for (size_t i = 0; i < kBlocks; ++i) {
        auto buffer = AllocateAlignedBuffer(kIOReaderTestBlockSize);
        ASSERT_NE(buffer, nullptr);
        buffers.push_back(buffer.get());
        offsets.push_back(i * kIOReaderTestBlockSize);
        owned_buffers.push_back(std::move(buffer));
    }

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    int submit_calls = 0;
    int wait_calls = 0;
    int reset_calls = 0;
    sync_point->SetCallBack("IOReader::SubmitAioBatch:BeforeSubmit", [&](void* arg) {
        if (submit_calls++ == 1) {
            *static_cast<size_t*>(arg) = 1;
        }
    });
    sync_point->SetCallBack("IOReader::WaitAioBatch:BeforeReturn", [&](void* arg) {
        if (wait_calls++ == 1) {
            *static_cast<bool*>(arg) = true;
        }
    });
    sync_point->SetCallBack("AioContextPool::ResetCheckedOut:BeforeSetup", [&](void* arg) {
        ++reset_calls;
        *static_cast<int*>(arg) = 0;
    });
    sync_point->EnableProcessing();

    IOReader reader(fd, pool);
    auto failed = reader.ReadAsync(std::move(buffers), kIOReaderTestBlockSize, std::move(offsets));
    ASSERT_FALSE(failed.get());
    ASSERT_GT(reset_calls, 0);

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    auto retry = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(retry, nullptr);
    std::vector<std::byte*> retry_buffers{retry.get()};
    std::vector<size_t> retry_offsets{0};
    auto retried = reader.ReadAsync(std::move(retry_buffers), kIOReaderTestBlockSize, std::move(retry_offsets));
    ASSERT_TRUE(retried.get());
    ASSERT_EQ(std::memcmp(retry.get(), content.get(), kIOReaderTestBlockSize), 0);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}
#endif

#ifdef WITH_IO_URING
TEST_F(IOContextPoolTestFixture, CompletionReaderReturnsSubmittedRequestIds) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 8;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    constexpr size_t kTotalSize = kIOReaderTestBlockSize * 2;
    auto content = AllocateAlignedBuffer(kTotalSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x1});
    std::fill(content.get() + kIOReaderTestBlockSize, content.get() + kTotalSize, std::byte{0x2});

    ASSERT_EQ(::pwrite(fd, content.get(), kTotalSize, 0), static_cast<ssize_t>(kTotalSize));
    ASSERT_EQ(::fsync(fd), 0);

    IOCompletionReader reader(fd, pool);
    std::array<std::byte*, 1> first_buffers{first.get()};
    std::array<size_t, 1> first_offsets{0};
    std::array<std::byte*, 1> second_buffers{second.get()};
    std::array<size_t, 1> second_offsets{kIOReaderTestBlockSize};

    const auto request_1 = reader.Submit(std::span<std::byte* const>(first_buffers.data(), first_buffers.size()),
                                         kIOReaderTestBlockSize,
                                         std::span<const size_t>(first_offsets.data(), first_offsets.size()));
    const auto request_2 = reader.Submit(std::span<std::byte* const>(second_buffers.data(), second_buffers.size()),
                                         kIOReaderTestBlockSize,
                                         std::span<const size_t>(second_offsets.data(), second_offsets.size()));
    ASSERT_NE(request_1, request_2);

    std::vector<IOCompletionReader::Completion> completions;
    completions.push_back(reader.WaitCompleted());

    auto polled = reader.PollCompleted();
    completions.insert(completions.end(), polled.begin(), polled.end());
    while (completions.size() < 2) {
        completions.push_back(reader.WaitCompleted());
    }

    ASSERT_EQ(completions.size(), 2u);
    ASSERT_TRUE(std::all_of(completions.begin(), completions.end(), [](const auto& c) { return c.ok; }));

    std::vector<IOCompletionReader::RequestId> ids{completions[0].request_id, completions[1].request_id};
    std::sort(ids.begin(), ids.end());
    ASSERT_EQ(ids[0], std::min(request_1, request_2));
    ASSERT_EQ(ids[1], std::max(request_1, request_2));

    ASSERT_EQ(std::memcmp(first.get(), content.get(), kIOReaderTestBlockSize), 0);
    ASSERT_EQ(std::memcmp(second.get(), content.get() + kIOReaderTestBlockSize, kIOReaderTestBlockSize), 0);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, CompletionReaderWaitsForAllBuffersInRequest) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 8;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_multi_buffer_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    constexpr size_t kTotalSize = kIOReaderTestBlockSize * 2;
    auto content = AllocateAlignedBuffer(kTotalSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0xd});
    std::fill(content.get() + kIOReaderTestBlockSize, content.get() + kTotalSize, std::byte{0xe});
    ASSERT_EQ(::pwrite(fd, content.get(), kTotalSize, 0), static_cast<ssize_t>(kTotalSize));
    ASSERT_EQ(::fsync(fd), 0);

    IOCompletionReader reader(fd, pool);
    std::array<std::byte*, 2> buffers{first.get(), second.get()};
    std::array<size_t, 2> offsets{0, kIOReaderTestBlockSize};

    const auto request = reader.Submit(std::span<std::byte* const>(buffers.data(), buffers.size()),
                                       kIOReaderTestBlockSize,
                                       std::span<const size_t>(offsets.data(), offsets.size()));
    auto completion = reader.WaitCompleted();
    ASSERT_EQ(completion.request_id, request);
    ASSERT_TRUE(completion.ok);
    ASSERT_TRUE(reader.PollCompleted().empty());

    ASSERT_EQ(std::memcmp(first.get(), content.get(), kIOReaderTestBlockSize), 0);
    ASSERT_EQ(std::memcmp(second.get(), content.get() + kIOReaderTestBlockSize, kIOReaderTestBlockSize), 0);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, CompletionReaderReportsShortReadFailure) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 8;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_short_read_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    auto content = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto buffer = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(buffer, nullptr);
    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x12});
    ASSERT_EQ(::pwrite(fd, content.get(), kIOReaderTestBlockSize, 0), static_cast<ssize_t>(kIOReaderTestBlockSize));
    ASSERT_EQ(::fsync(fd), 0);

    IOCompletionReader reader(fd, pool);
    std::array<std::byte*, 1> buffers{buffer.get()};
    std::array<size_t, 1> offsets{kIOReaderTestBlockSize};
    const auto request = reader.Submit(std::span<std::byte* const>(buffers.data(), buffers.size()),
                                       kIOReaderTestBlockSize,
                                       std::span<const size_t>(offsets.data(), offsets.size()));
    auto completion = reader.WaitCompleted();
    ASSERT_EQ(completion.request_id, request);
    ASSERT_FALSE(completion.ok);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, CompletionReaderReportsNegativeCqeFailure) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 8;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_negative_cqe_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    auto buffer = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(buffer, nullptr);

    IOCompletionReader reader(fd, pool);
    ASSERT_EQ(::close(fd), 0);
    fd = -1;

    std::array<std::byte*, 1> buffers{buffer.get()};
    std::array<size_t, 1> offsets{0};
    const auto request = reader.Submit(std::span<std::byte* const>(buffers.data(), buffers.size()),
                                       kIOReaderTestBlockSize,
                                       std::span<const size_t>(offsets.data(), offsets.size()));
    auto completion = reader.WaitCompleted();
    ASSERT_EQ(completion.request_id, request);
    ASSERT_FALSE(completion.ok);

    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, CompletionReaderRejectsBatchLargerThanCapacity) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 1;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_capacity_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    IOCompletionReader reader(fd, pool);
    std::array<std::byte*, 2> buffers{first.get(), second.get()};
    std::array<size_t, 2> offsets{0, kIOReaderTestBlockSize};

    EXPECT_THROW(reader.Submit(std::span<std::byte* const>(buffers.data(), buffers.size()), kIOReaderTestBlockSize,
                               std::span<const size_t>(offsets.data(), offsets.size())),
                 std::invalid_argument);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}

#ifdef ENABLE_SYNCPOINT
TEST_F(IOContextPoolTestFixture, CompletionReaderRejectsOutstandingRequestsBeyondCapacity) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 2;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_outstanding_capacity_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    auto content = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto third = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(third, nullptr);
    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0xf});
    ASSERT_EQ(::pwrite(fd, content.get(), kIOReaderTestBlockSize, 0), static_cast<ssize_t>(kIOReaderTestBlockSize));
    ASSERT_EQ(::fsync(fd), 0);

    IOCompletionReader reader(fd, pool);

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    sync_point->SetCallBack("IOCompletionReader::ProcessAvailableCompletions:Skip", [](void* arg) {
        *static_cast<bool*>(arg) = true;
    });
    sync_point->EnableProcessing();

    std::array<size_t, 1> offsets{0};
    std::array<std::byte*, 1> first_buffers{first.get()};
    std::array<std::byte*, 1> second_buffers{second.get()};
    std::array<std::byte*, 1> third_buffers{third.get()};
    ASSERT_NE(reader.Submit(std::span<std::byte* const>(first_buffers.data(), first_buffers.size()),
                            kIOReaderTestBlockSize, std::span<const size_t>(offsets.data(), offsets.size())),
              0u);
    ASSERT_NE(reader.Submit(std::span<std::byte* const>(second_buffers.data(), second_buffers.size()),
                            kIOReaderTestBlockSize, std::span<const size_t>(offsets.data(), offsets.size())),
              0u);
    EXPECT_THROW(reader.Submit(std::span<std::byte* const>(third_buffers.data(), third_buffers.size()),
                               kIOReaderTestBlockSize, std::span<const size_t>(offsets.data(), offsets.size())),
                 std::runtime_error);

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, CompletionReaderSubmitFailureKeepsExistingRequestObservable) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 8;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_submit_failure_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    constexpr size_t kTotalSize = kIOReaderTestBlockSize * 2;
    auto content = AllocateAlignedBuffer(kTotalSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto second = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x6});
    std::fill(content.get() + kIOReaderTestBlockSize, content.get() + kTotalSize, std::byte{0x7});

    ASSERT_EQ(::pwrite(fd, content.get(), kTotalSize, 0), static_cast<ssize_t>(kTotalSize));
    ASSERT_EQ(::fsync(fd), 0);

    IOCompletionReader reader(fd, pool);
    std::array<std::byte*, 1> first_buffers{first.get()};
    std::array<size_t, 1> first_offsets{0};
    const auto request_1 = reader.Submit(std::span<std::byte* const>(first_buffers.data(), first_buffers.size()),
                                         kIOReaderTestBlockSize,
                                         std::span<const size_t>(first_offsets.data(), first_offsets.size()));

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    int forced_submits = 0;
    int skipped_polls = 0;
    sync_point->SetCallBack("IOCompletionReader::ProcessAvailableCompletions:Skip", [&](void* arg) {
        if (skipped_polls++ == 0) {
            *static_cast<bool*>(arg) = true;
        }
    });
    sync_point->SetCallBack("IOCompletionReader::SubmitRing:Before", [&](void* arg) {
        if (forced_submits++ == 0) {
            *static_cast<int*>(arg) = -EIO;
        }
    });
    sync_point->EnableProcessing();

    std::array<std::byte*, 1> second_buffers{second.get()};
    std::array<size_t, 1> second_offsets{kIOReaderTestBlockSize};
    EXPECT_THROW(reader.Submit(std::span<std::byte* const>(second_buffers.data(), second_buffers.size()),
                               kIOReaderTestBlockSize,
                               std::span<const size_t>(second_offsets.data(), second_offsets.size())),
                 std::runtime_error);

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    auto completion = reader.WaitCompleted();
    EXPECT_EQ(completion.request_id, request_1);
    EXPECT_FALSE(completion.ok);
    EXPECT_THROW(reader.WaitCompleted(), std::runtime_error);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}

TEST_F(IOContextPoolTestFixture, CompletionReaderDestructorResetKeepsPoolReusable) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = 1;
    cfg.max_events = 8;
    ASSERT_TRUE(IOContextPool::InitGlobal(cfg));

    auto pool = IOContextPool::GetGlobal();
    ASSERT_NE(pool, nullptr);
    if (pool->Backend() != IOBackend::IO_URING) {
        GTEST_SKIP() << "io_uring backend unavailable";
    }

    char path[] = "/tmp/io_completion_reader_destructor_reset_XXXXXX";
    int fd = OpenIOReaderTestFile(path, false);
    ASSERT_GE(fd, 0);

    auto content = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto first = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    auto retry = AllocateAlignedBuffer(kIOReaderTestBlockSize);
    ASSERT_NE(content, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(retry, nullptr);
    std::fill(content.get(), content.get() + kIOReaderTestBlockSize, std::byte{0x13});
    ASSERT_EQ(::pwrite(fd, content.get(), kIOReaderTestBlockSize, 0), static_cast<ssize_t>(kIOReaderTestBlockSize));
    ASSERT_EQ(::fsync(fd), 0);

    auto* sync_point = milvus::SyncPoint::GetInstance();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();
    sync_point->SetCallBack("IOCompletionReader::DrainOutstandingNoThrow:Skip", [](void* arg) {
        *static_cast<bool*>(arg) = true;
    });
    sync_point->EnableProcessing();

    {
        IOCompletionReader reader(fd, pool);
        std::array<std::byte*, 1> buffers{first.get()};
        std::array<size_t, 1> offsets{0};
        ASSERT_NE(reader.Submit(std::span<std::byte* const>(buffers.data(), buffers.size()),
                                kIOReaderTestBlockSize, std::span<const size_t>(offsets.data(), offsets.size())),
                  0u);
    }

    sync_point->DisableProcessing();
    sync_point->ClearAllCallBacks();
    sync_point->ClearTrace();

    IOCompletionReader retry_reader(fd, pool);
    std::array<std::byte*, 1> retry_buffers{retry.get()};
    std::array<size_t, 1> retry_offsets{0};
    const auto request = retry_reader.Submit(std::span<std::byte* const>(retry_buffers.data(), retry_buffers.size()),
                                             kIOReaderTestBlockSize,
                                             std::span<const size_t>(retry_offsets.data(), retry_offsets.size()));
    auto completion = retry_reader.WaitCompleted();
    ASSERT_EQ(completion.request_id, request);
    ASSERT_TRUE(completion.ok);
    ASSERT_EQ(std::memcmp(retry.get(), content.get(), kIOReaderTestBlockSize), 0);

    ASSERT_EQ(::close(fd), 0);
    ::unlink(path);
}
#endif
#endif
