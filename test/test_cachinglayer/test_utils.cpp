// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"
#include "cachinglayer_test_utils.h"
#include "syncpoint/sync_point.h"

namespace fs = std::filesystem;
using milvus::cachinglayer::internal::getContainerMemLimit;
using milvus::cachinglayer::internal::getCurrentProcessMemoryUsage;
using milvus::cachinglayer::internal::getSystemDiskInfo;
using milvus::cachinglayer::internal::getSystemMemoryInfo;
using milvus::cachinglayer::internal::SystemResourceInfo;

namespace {

// Helper RAII type to restore env vars after test
class EnvGuard {
 public:
    EnvGuard(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name_);
        if (old) {
            old_value_ = old;
        }
        if (value) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
    }

    ~EnvGuard() {
        if (old_value_.has_value()) {
            setenv(name_, old_value_->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

 private:
    const char* name_;
    std::optional<std::string> old_value_;
};

// Create a unique temporary directory for tests.
fs::path
CreateTempDir(const std::string& prefix) {
    fs::path base = fs::temp_directory_path();
    for (int i = 0; i < 100; ++i) {
        auto candidate = base / (prefix + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                                 "_" + std::to_string(i));
        std::error_code ec;
        if (fs::create_directory(candidate, ec) && !ec) {
            return candidate;
        }
    }
    throw std::runtime_error("Failed to create temporary directory for tests");
}

}  // namespace

TEST(Utils, SemiInlineGetPreservesValuesAndErrors) {
    using milvus::cachinglayer::SemiInlineGet;
    auto value = std::make_unique<int>(42);
    const auto* address = value.get();
    EXPECT_EQ(SemiInlineGet(folly::makeSemiFuture(std::move(value))).get(), address);
    EXPECT_EQ(*SemiInlineGet(folly::makeSemiFuture().deferValue([](auto) { return std::make_unique<int>(7); })), 7);
    EXPECT_EQ(SemiInlineGet(folly::makeSemiFuture()), folly::unit);
    EXPECT_EQ(SemiInlineGet(folly::makeSemiFuture().deferValue([](auto) {})), folly::unit);
    EXPECT_THROW(SemiInlineGet(folly::makeSemiFuture<int>(std::runtime_error("ready failure"))), std::runtime_error);
    EXPECT_THROW(
        SemiInlineGet(folly::makeSemiFuture().deferValue([](auto) { throw std::runtime_error("deferred failure"); })),
        std::runtime_error);
}

// getContainerMemLimit() basic env variable precedence: MEM_LIMIT only
TEST(Utils, GetContainerMemLimitEnvOnly) {
#ifdef __linux__
    EnvGuard guard_root("MCL_CGROUP_ROOT", nullptr);
    EnvGuard guard_proc("MCL_PROC_CGROUP", nullptr);

    const char* limit_str = "123456789";
    EnvGuard guard_mem("MEM_LIMIT", limit_str);

    int64_t limit = getContainerMemLimit();
    EXPECT_EQ(limit, std::stoll(limit_str));
#else
    GTEST_SKIP() << "getContainerMemLimit is only implemented on Linux";
#endif
}

// MEM_LIMIT should override any cgroup limits when present.
TEST(Utils, GetContainerMemLimitMemLimitOverridesCgroup) {
#ifdef __linux__
    fs::path root = CreateTempDir("mcl_mem_override_");

    // Direct v1 limit (large)
    fs::create_directories(root / "memory");
    {
        std::ofstream ofs(root / "memory" / "memory.limit_in_bytes");
        ofs << "1073741824";  // 1 GiB
    }

    // Direct v2 limit (even larger)
    {
        std::ofstream ofs(root / "memory.max");
        ofs << "2147483648";  // 2 GiB
    }

    // Simple cgroup file (content doesn't really matter here)
    fs::path fake_proc = root / "proc_self_cgroup";
    {
        std::ofstream ofs(fake_proc);
        ofs << "2:memory:/my.slice" << std::endl;
    }

    // MEM_LIMIT smaller than any cgroup value should win
    EnvGuard guard_root("MCL_CGROUP_ROOT", root.c_str());
    EnvGuard guard_proc("MCL_PROC_CGROUP", fake_proc.c_str());
    EnvGuard guard_mem("MEM_LIMIT", "268435456");  // 256 MiB

    int64_t limit = getContainerMemLimit();
    EXPECT_EQ(limit, 268435456);

    fs::remove_all(root);
#else
    GTEST_SKIP() << "getContainerMemLimit is only implemented on Linux";
#endif
}

// Only direct cgroup v1 root path is present.
TEST(Utils, GetContainerMemLimitCgroupV1RootOnly) {
#ifdef __linux__
    fs::path root = CreateTempDir("mcl_cgroup_v1_root_");
    // Direct v1 limit
    fs::create_directories(root / "memory");
    {
        std::ofstream ofs(root / "memory" / "memory.limit_in_bytes");
        ofs << "1073741824";  // 1 GiB
    }

    // No process-specific v1 path for this test

    // Fake /proc/self/cgroup without memory controller to ensure only direct v1 path is used
    fs::path fake_proc = root / "proc_self_cgroup";
    {
        std::ofstream ofs(fake_proc);
        ofs << "1:name=systemd:/" << std::endl;
    }

    EnvGuard guard_root("MCL_CGROUP_ROOT", root.c_str());
    EnvGuard guard_proc("MCL_PROC_CGROUP", fake_proc.c_str());
    EnvGuard guard_mem("MEM_LIMIT", nullptr);

    int64_t limit = getContainerMemLimit();
    EXPECT_EQ(limit, 1073741824);

    fs::remove_all(root);
#else
    GTEST_SKIP() << "getContainerMemLimit is only implemented on Linux";
#endif
}

// Process-specific cgroup v1 path should override the direct v1 root limit (we use the minimum).
TEST(Utils, GetContainerMemLimitCgroupV1ProcSpecific) {
#ifdef __linux__
    fs::path root = CreateTempDir("mcl_cgroup_v1_proc_");
    // Direct v1 limit
    fs::create_directories(root / "memory");
    {
        std::ofstream ofs(root / "memory" / "memory.limit_in_bytes");
        ofs << "1073741824";  // 1 GiB
    }

    // Process-specific v1 path with smaller limit
    fs::create_directories(root / "memory" / "my.slice");
    {
        std::ofstream ofs(root / "memory" / "my.slice" / "memory.limit_in_bytes");
        ofs << "536870912";  // 512 MiB, smaller than direct limit
    }

    // Fake /proc/self/cgroup line for memory controller (v1 style)
    fs::path fake_proc = root / "proc_self_cgroup";
    {
        std::ofstream ofs(fake_proc);
        ofs << "2:memory:/my.slice" << std::endl;
    }

    EnvGuard guard_root("MCL_CGROUP_ROOT", root.c_str());
    EnvGuard guard_proc("MCL_PROC_CGROUP", fake_proc.c_str());
    EnvGuard guard_mem("MEM_LIMIT", nullptr);

    int64_t limit = getContainerMemLimit();
    // We expect the minimum from the available limits (512 MiB here)
    EXPECT_EQ(limit, 536870912);

    fs::remove_all(root);
#else
    GTEST_SKIP() << "getContainerMemLimit is only implemented on Linux";
#endif
}

// Only direct cgroup v2 root path is present.
TEST(Utils, GetContainerMemLimitCgroupV2RootOnly) {
#ifdef __linux__
    fs::path root = CreateTempDir("mcl_cgroup_v2_root_");

    // Root v2 memory.max with concrete limit
    {
        std::ofstream ofs(root / "memory.max");
        ofs << "268435456";  // 256 MiB
    }

    // Fake /proc/self/cgroup line for unified cgroup v2 hierarchy,
    // but no process-specific memory.max file for this test.
    fs::path fake_proc = root / "proc_self_cgroup";
    {
        std::ofstream ofs(fake_proc);
        ofs << "0::/user.slice" << std::endl;
    }

    EnvGuard guard_root("MCL_CGROUP_ROOT", root.c_str());
    EnvGuard guard_proc("MCL_PROC_CGROUP", fake_proc.c_str());
    EnvGuard guard_mem("MEM_LIMIT", nullptr);

    int64_t limit = getContainerMemLimit();
    EXPECT_EQ(limit, 268435456);

    fs::remove_all(root);
#else
    GTEST_SKIP() << "getContainerMemLimit is only implemented on Linux";
#endif
}

// Process-specific cgroup v2 path should be honored when present.
TEST(Utils, GetContainerMemLimitCgroupV2ProcSpecific) {
#ifdef __linux__
    fs::path root = CreateTempDir("mcl_cgroup_v2_proc_");

    // Root v2 memory.max is often "max" (unlimited); we simulate that.
    {
        std::ofstream ofs(root / "memory.max");
        ofs << "max";
    }

    // Process-specific v2 path
    fs::create_directories(root / "user.slice");
    {
        std::ofstream ofs(root / "user.slice" / "memory.max");
        ofs << "268435456";  // 256 MiB
    }

    // Fake /proc/self/cgroup line for unified cgroup v2 hierarchy
    fs::path fake_proc = root / "proc_self_cgroup";
    {
        std::ofstream ofs(fake_proc);
        ofs << "0::/user.slice" << std::endl;
    }

    EnvGuard guard_root("MCL_CGROUP_ROOT", root.c_str());
    EnvGuard guard_proc("MCL_PROC_CGROUP", fake_proc.c_str());
    EnvGuard guard_mem("MEM_LIMIT", nullptr);

    int64_t limit = getContainerMemLimit();
    EXPECT_EQ(limit, 268435456);

    fs::remove_all(root);
#else
    GTEST_SKIP() << "getContainerMemLimit is only implemented on Linux";
#endif
}

// Verify that getSystemMemoryInfo prefers container limit when smaller than host memory.
TEST(Utils, GetSystemMemoryInfoUsesContainerLimitWhenSmaller) {
#ifdef __linux__
    fs::path root = CreateTempDir("mcl_sysmem_");

    // Minimal cgroup v2-like setup providing a small limit
    {
        std::ofstream ofs(root / "memory.max");
        ofs << "max";
    }
    fs::create_directories(root / "test.slice");
    {
        std::ofstream ofs(root / "test.slice" / "memory.max");
        ofs << "134217728";  // 128 MiB
    }
    fs::path fake_proc = root / "proc_self_cgroup";
    {
        std::ofstream ofs(fake_proc);
        ofs << "0::/test.slice" << std::endl;
    }

    EnvGuard guard_root("MCL_CGROUP_ROOT", root.c_str());
    EnvGuard guard_proc("MCL_PROC_CGROUP", fake_proc.c_str());
    EnvGuard guard_mem("MEM_LIMIT", nullptr);

    SystemResourceInfo info = getSystemMemoryInfo();
    // total_bytes should be the container limit (128 MiB)
    EXPECT_EQ(info.total_bytes, 134217728);
    EXPECT_GE(info.used_bytes, 0);

    fs::remove_all(root);
#else
    GTEST_SKIP() << "getSystemMemoryInfo is only fully implemented on Linux";
#endif
}

TEST(Utils, GetSystemDiskInfoBasic) {
    // Empty path -> unlimited
    SystemResourceInfo info_empty = getSystemDiskInfo("");
    EXPECT_GT(info_empty.total_bytes, 0);
    EXPECT_EQ(info_empty.used_bytes, 0);

    // Existing path (likely root filesystem)
    SystemResourceInfo info_root = getSystemDiskInfo("/");
    EXPECT_GT(info_root.total_bytes, 0);
    EXPECT_GE(info_root.used_bytes, 0);
    EXPECT_LE(info_root.used_bytes, info_root.total_bytes);
}

TEST(Utils, GetCurrentProcessMemoryUsageNonNegative) {
    int64_t usage = getCurrentProcessMemoryUsage();
    EXPECT_GE(usage, 0);
}

TEST(Utils, ProcessMemoryUsageHandlesVariableFieldOffsets) {
#ifdef __linux__
    const auto root = CreateTempDir("mcl_status_");
    const auto path = root / "status";
    EnvGuard guard("MCL_PROC_STATUS", path.c_str());
    for (const auto& [content, expected] : std::vector<std::pair<std::string, int64_t>>{
             {"Name:\tqn\nVmRSS:\t100 kB\nRssAnon:\t50 kB\nRssFile:\t30 kB\nRssShmem:\t20 kB\n", 70 * 1024},
             {"Name:\tlonger-process-name\nVmSize:\t123456789 kB\nRssFile: 300 kB\nVmRSS: 1000 kB\n", 700 * 1024}}) {
        std::ofstream(path) << content;
        EXPECT_EQ(getCurrentProcessMemoryUsage(), expected);
    }
    for (const auto& content :
         {"VmRSS: invalid kB\nRssFile: 1 kB\n", "VmRSS: 10 kB\n", "VmRSS: 9223372036854775807 kB\nRssFile: 1 kB\n"}) {
        std::ofstream(path) << content;
        EXPECT_EQ(getCurrentProcessMemoryUsage(), 0);
    }
    fs::remove_all(root);
#else
    GTEST_SKIP() << "proc status is Linux-specific";
#endif
}

#if defined(__linux__) && defined(ENABLE_SYNCPOINT)
TEST(Utils, DListCachesMemoryLimitForFiveSeconds) {
    const auto root = CreateTempDir("mcl_limit_cache_");
    const auto proc = root / "cgroup";
    std::ofstream(proc) << "0::/test.slice\n";
    const auto limit_path = root / "memory.max";
    std::ofstream(limit_path) << "67108864\n";
    EnvGuard root_guard("MCL_CGROUP_ROOT", root.c_str());
    EnvGuard proc_guard("MCL_PROC_CGROUP", proc.c_str());
    EnvGuard mem_guard("MEM_LIMIT", nullptr);
    auto cleanup = folly::makeGuard([&] { fs::remove_all(root); });
    auto dlist = std::make_shared<milvus::cachinglayer::internal::DList>(
        true, milvus::cachinglayer::ResourceUsage{100, 0}, milvus::cachinglayer::ResourceUsage{80, 0},
        milvus::cachinglayer::ResourceUsage{90, 0}, milvus::cachinglayer::EvictionConfig{});
    const auto worker = milvus::cachinglayer::internal::DListTestFriend::background_thread_id(*dlist);
    std::promise<int64_t> first_sample;
    auto first = first_sample.get_future();
    std::promise<void> refreshed;
    auto refresh = refreshed.get_future();
    bool saw_first = false;
    bool saw_refresh = false;
    auto* sync = milvus::SyncPoint::GetInstance();
    sync->Reset();
    auto reset = folly::makeGuard([&] { sync->Reset(); });
    sync->SetCallBack("DList::MemorySnapshot", [&](void* value) {
        if (std::this_thread::get_id() != worker) {
            return;
        }
        const auto total = static_cast<SystemResourceInfo*>(value)->total_bytes;
        if (!saw_first) {
            saw_first = true;
            first_sample.set_value(total);
        }
        if (total == 134217728 && !saw_refresh) {
            saw_refresh = true;
            refreshed.set_value();
        }
    });
    sync->EnableProcessing();
    const auto start = std::chrono::steady_clock::now();
    std::ofstream(limit_path) << "134217728\n";
    ASSERT_EQ(first.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(first.get(), 67108864);
    EXPECT_EQ(refresh.wait_until(start + std::chrono::seconds(4)), std::future_status::timeout);
    EXPECT_EQ(refresh.wait_until(start + std::chrono::seconds(7)), std::future_status::ready);
}
#endif
