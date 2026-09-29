#include <dlfcn.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <libaio.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <system_error>

#include "knowhere/aio_context_pool.h"

namespace {
constexpr size_t kMaxEvents = 8;
std::atomic<int> setup_error{0};
std::atomic<int> destroy_error{0};
std::atomic<size_t> destroy_calls{0};
std::atomic<size_t> setup_calls{0};
std::function<void()> before_destroy;
std::mutex live_mutex;
std::set<io_context_t> live_contexts;
bool invalid_destroy = false;

void
CheckAllContextsDestroyed() {
    std::lock_guard lock(live_mutex);
    if (invalid_destroy || !live_contexts.empty()) {
        std::fputs("AIO pool leaked or destroyed a context twice at shutdown\n", stderr);
        std::_Exit(1);
    }
}

struct ReadInput {
    static constexpr size_t kPageSize = 4096;
    int fd = -1;
    std::unique_ptr<void, decltype(&std::free)> buffer{nullptr, &std::free};
    iocb cb{};

    ReadInput() {
        char name[] = "/tmp/aio_pool_XXXXXX";
        fd = mkstemp(name);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category());
        }
        unlink(name);
        if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_DIRECT) != 0) {
            close(fd);
            throw std::system_error(errno, std::generic_category());
        }
        if (ftruncate(fd, kPageSize) != 0) {
            close(fd);
            throw std::system_error(errno, std::generic_category());
        }
        buffer.reset(std::aligned_alloc(kPageSize, kPageSize));
        if (!buffer) {
            close(fd);
            throw std::bad_alloc();
        }
        io_prep_pread(&cb, fd, buffer.get(), kPageSize, 0);
    }
    ~ReadInput() {
        close(fd);
    }
};

void
ReadPage(io_context_t ctx) {
    ReadInput input;
    iocb* cb = &input.cb;
    ASSERT_EQ(io_submit(ctx, 1, &cb), 1);
    io_event event{};
    ASSERT_EQ(io_getevents(ctx, 1, 1, &event, nullptr), 1);
    EXPECT_EQ(event.res, ReadInput::kPageSize);
}

class AioContextPoolTest : public ::testing::Test {
 protected:
    std::shared_ptr<AioContextPool> pool;
    io_context_t held = nullptr;
    io_context_t ctx = nullptr;

    void
    SetUp() override {
        pool = AioContextPool::GetGlobalAioPool();
        held = pool->pop();
        ctx = pool->pop();
    }
    void
    TearDown() override {
        setup_error = 0;
        destroy_error = 0;
        before_destroy = {};
        if (ctx)
            pool->push(ctx);
        if (held)
            pool->push(held);
    }
};
}  // namespace

extern "C" int
io_setup(int count, io_context_t* ctx) {
    ++setup_calls;
    if (int error = setup_error.exchange(0); error != 0)
        return error;
    static auto real_setup = reinterpret_cast<int (*)(int, io_context_t*)>(dlsym(RTLD_NEXT, "io_setup"));
    if (!real_setup)
        std::abort();
    const int ret = real_setup(count, ctx);
    if (ret == 0) {
        std::lock_guard lock(live_mutex);
        live_contexts.insert(*ctx);
    }
    return ret;
}

extern "C" int
io_destroy(io_context_t ctx) {
    ++destroy_calls;
    if (before_destroy)
        before_destroy();
    if (int error = destroy_error.exchange(0); error != 0)
        return error;
    // Remove before destruction: the kernel may immediately reuse the handle
    // address for a context created concurrently by another thread.
    {
        std::lock_guard lock(live_mutex);
        if (live_contexts.erase(ctx) != 1)
            invalid_destroy = true;
    }
    static auto real_destroy = reinterpret_cast<int (*)(io_context_t)>(dlsym(RTLD_NEXT, "io_destroy"));
    if (!real_destroy)
        std::abort();
    return real_destroy(ctx);
}

TEST_F(AioContextPoolTest, ReplacesContextWithUnreapedReads) {
    ReadInput input;
    iocb* cb = &input.cb;
    ASSERT_EQ(io_submit(ctx, 1, &cb), 1);
    const auto calls = destroy_calls.load();
    pool->DestroyAndRecreate(ctx);
    if (ctx != nullptr) {
        io_event event{};
        ASSERT_EQ(io_getevents(ctx, 1, 1, &event, nullptr), 1);
    }
    ASSERT_EQ(ctx, nullptr);
    EXPECT_EQ(destroy_calls, calls + 1);
    ctx = pool->pop();
    ReadPage(ctx);
}

TEST_F(AioContextPoolTest, FailedSetupReducesCapacityWithoutRetry) {
    // Run permanent capacity loss in a fresh process so other tests retain
    // their two-context singleton pool, including with shuffled test order.
    EXPECT_EXIT(([&] {
                    const auto verify = [&] {
                        const auto calls = setup_calls.load();
                        setup_error = -ENOMEM;
                        pool->DestroyAndRecreate(ctx);
                        ASSERT_EQ(ctx, nullptr);
                        EXPECT_EQ(setup_calls, calls + 1);
                        std::promise<void> entered;
                        auto waiter = std::async(std::launch::async, [&] {
                            entered.set_value();
                            return pool->pop();
                        });
                        entered.get_future().wait();
                        EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
                        EXPECT_EQ(setup_calls, calls + 1);
                        const auto available = held;
                        pool->push(held);
                        held = nullptr;
                        ctx = waiter.get();
                        EXPECT_EQ(ctx, available);
                        EXPECT_EQ(setup_calls, calls + 1);
                        ReadPage(ctx);
                    };
                    verify();
                    pool.reset();
                    std::exit(::testing::Test::HasFailure() ? 1 : 0);
                }()),
                ::testing::ExitedWithCode(0), "");
}

TEST_F(AioContextPoolTest, FailedDestroyAbandonsOldContextAndTriesReplacement) {
    const auto abandoned = ctx;
    const auto calls = setup_calls.load();
    destroy_error = -EINVAL;
    pool->DestroyAndRecreate(ctx);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_EQ(setup_calls, calls + 1);
    {
        std::lock_guard lock(live_mutex);
        EXPECT_EQ(live_contexts.count(abandoned), 1);
    }
    ctx = pool->pop();
    EXPECT_NE(ctx, abandoned);
    ReadPage(ctx);
    // The production pool intentionally forgets this handle. The test owns
    // cleanup so its real kernel resource does not survive the test case.
    EXPECT_EQ(io_destroy(abandoned), 0);
}

TEST_F(AioContextPoolTest, RepeatedReplacementPreservesCapacity) {
    for (int i = 0; i < 16; ++i) {
        pool->DestroyAndRecreate(ctx);
        ASSERT_EQ(ctx, nullptr);
        ctx = pool->pop();
        ReadPage(ctx);
    }
}

TEST_F(AioContextPoolTest, WakesWaiterAfterReplacement) {
    std::promise<void> entered;
    auto waiter = std::async(std::launch::async, [&] {
        entered.set_value();
        return pool->pop();
    });
    entered.get_future().wait();
    ASSERT_EQ(waiter.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    pool->DestroyAndRecreate(ctx);
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    ctx = waiter.get();
    ReadPage(ctx);
}

TEST_F(AioContextPoolTest, ExhaustedPoolWakesAllWaitersWithoutRetry) {
    EXPECT_EXIT(([&] {
                    const auto verify = [&] {
                        // Keep one context borrowed while the other slot is lost.
                        setup_error = -ENOMEM;
                        pool->DestroyAndRecreate(ctx);
                        std::promise<void> first_entered;
                        std::promise<void> second_entered;
                        const auto wait = [&](std::promise<void>& entered) {
                            entered.set_value();
                            try {
                                pool->pop();
                                return false;
                            } catch (const std::runtime_error&) {
                                return true;
                            }
                        };
                        auto first = std::async(std::launch::async, [&] { return wait(first_entered); });
                        auto second = std::async(std::launch::async, [&] { return wait(second_entered); });
                        first_entered.get_future().wait();
                        second_entered.get_future().wait();
                        EXPECT_EQ(first.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
                        EXPECT_EQ(second.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
                        const auto calls = setup_calls.load();
                        setup_error = -ENOMEM;
                        pool->DestroyAndRecreate(held);
                        EXPECT_EQ(ctx, nullptr);
                        EXPECT_EQ(held, nullptr);
                        if (first.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
                            second.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
                            std::_Exit(2);
                        }
                        EXPECT_TRUE(first.get());
                        EXPECT_TRUE(second.get());
                        EXPECT_THROW(pool->pop(), std::runtime_error);
                        EXPECT_EQ(setup_calls, calls + 1);
                    };
                    verify();
                    pool.reset();
                    std::exit(::testing::Test::HasFailure() ? 1 : 0);
                }()),
                ::testing::ExitedWithCode(0), "");
}

TEST_F(AioContextPoolTest, DoesNotHoldPoolLockDuringDestroy) {
    std::promise<void> destroying;
    std::promise<void> resume;
    auto resume_signal = resume.get_future();
    before_destroy = [&] {
        destroying.set_value();
        resume_signal.wait();
    };
    auto recovery = std::async(std::launch::async, [&] { return pool->DestroyAndRecreate(ctx); });
    destroying.get_future().wait();
    auto other = std::async(std::launch::async, [&] {
        pool->push(held);
        held = pool->pop();
    });
    const auto status = other.wait_for(std::chrono::seconds(2));
    resume.set_value();
    EXPECT_EQ(status, std::future_status::ready);
    recovery.get();
    other.get();
    before_destroy = {};
    ctx = pool->pop();
    ReadPage(ctx);
}

TEST_F(AioContextPoolTest, ConcurrentReplacementKeepsDistinctBorrowedContexts) {
    const auto replace = [&](io_context_t& borrowed) {
        for (int i = 0; i < 8; ++i) {
            pool->DestroyAndRecreate(borrowed);
            if (borrowed != nullptr) {
                throw std::runtime_error("Context replacement failed");
            }
            borrowed = pool->pop();
            ReadPage(borrowed);
        }
    };
    auto first = std::async(std::launch::async, [&] { replace(ctx); });
    auto second = std::async(std::launch::async, [&] { replace(held); });
    first.get();
    second.get();
    EXPECT_NE(ctx, held);
}

int
main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    std::atexit(CheckAllContextsDestroyed);
    AioContextPool::InitGlobalAioPool(2, kMaxEvents);
    return RUN_ALL_TESTS();
}
