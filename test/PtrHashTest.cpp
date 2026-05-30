#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ptrhash/ptrhash.hpp"

namespace {

using ptrhash::BucketFunction;
using ptrhash::MappedPtrHash;
using ptrhash::PtrHash;
using ptrhash::PtrHashParams;
using ptrhash::PtrHashView;
using ptrhash::PtrHashWithHasher;

std::vector<uint64_t>
MakeIntegerKeys(size_t n) {
    std::vector<uint64_t> keys;
    keys.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        keys.push_back(i * 11400714819323198485ull + 0x9e3779b97f4a7c15ull);
    }
    return keys;
}

std::vector<uint64_t>
MakeMultipleKeys(size_t n, uint64_t multiplier) {
    std::vector<uint64_t> keys;
    keys.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        keys.push_back(i * multiplier);
    }
    return keys;
}

std::string
MakeTempPath(const std::string& label) {
    static std::atomic<uint64_t> counter{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "/tmp/milvus_common_ptrhash_" + label + "_" + std::to_string(now) + "_" +
           std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

class TempFile {
 public:
    explicit TempFile(std::string label) : path_(MakeTempPath(label)) {
    }

    ~TempFile() {
        std::remove(path_.c_str());
    }

    const std::string&
    path() const {
        return path_;
    }

 private:
    std::string path_;
};

template <typename Hash>
size_t
IndexKey(const Hash& hash, uint64_t key) {
    return hash.index(key);
}

template <typename Hash>
size_t
IndexNoRemapKey(const Hash& hash, uint64_t key) {
    return hash.index_no_remap(key);
}

template <typename Hash>
size_t
IndexKey(const Hash& hash, const std::string& key) {
    return hash.index(std::string_view(key.data(), key.size()));
}

template <typename Hash>
size_t
IndexNoRemapKey(const Hash& hash, const std::string& key) {
    return hash.index_no_remap(std::string_view(key.data(), key.size()));
}

template <typename Hash>
size_t
IndexKey(const Hash& hash, std::string_view key) {
    return hash.index(key);
}

template <typename Hash>
size_t
IndexNoRemapKey(const Hash& hash, std::string_view key) {
    return hash.index_no_remap(key);
}

template <typename Hash, typename Key>
auto
IndexKey(const Hash& hash, const Key& key) -> decltype(hash.index(key)) {
    return hash.index(key);
}

template <typename Hash, typename Key>
void
ExpectMinimalPerfect(const Hash& hash, const std::vector<Key>& keys) {
    ASSERT_EQ(hash.n(), keys.size());
    std::vector<uint8_t> seen(keys.size(), 0);
    for (const auto& key : keys) {
        const size_t index = IndexKey(hash, key);
        ASSERT_LT(index, keys.size());
        EXPECT_FALSE(seen[index]) << "duplicate minimal index " << index;
        seen[index] = 1;
        EXPECT_LT(IndexNoRemapKey(hash, key), hash.max_index());
    }
    for (size_t i = 0; i < seen.size(); ++i) {
        EXPECT_TRUE(seen[i]) << "missing minimal index " << i;
    }
}

template <typename Hash>
void
ExpectMinimalPerfectHashes(const Hash& hash, const std::vector<uint64_t>& hashes) {
    ASSERT_EQ(hash.n(), hashes.size());
    std::vector<uint8_t> seen(hashes.size(), 0);
    for (uint64_t key_hash : hashes) {
        const size_t index = hash.index_hash(key_hash);
        ASSERT_LT(index, hashes.size());
        EXPECT_FALSE(seen[index]) << "duplicate minimal index " << index;
        seen[index] = 1;
        EXPECT_LT(hash.index_no_remap_hash(key_hash), hash.max_index());
    }
}

template <typename LhsHash, typename RhsHash, typename Key>
void
ExpectSameQueryResults(const LhsHash& lhs, const RhsHash& rhs, const std::vector<Key>& keys) {
    for (const auto& key : keys) {
        EXPECT_EQ(IndexKey(lhs, key), IndexKey(rhs, key));
        EXPECT_EQ(IndexNoRemapKey(lhs, key), IndexNoRemapKey(rhs, key));
    }
}

template <typename Key>
void
ExpectSameQueries(const PtrHash& lhs, const PtrHash& rhs, const std::vector<Key>& keys) {
    ASSERT_EQ(lhs.serialize(), rhs.serialize());
    ExpectSameQueryResults(lhs, rhs, keys);
}

template <typename LhsHash, typename RhsHash>
void
ExpectSameHashQueries(const LhsHash& lhs, const RhsHash& rhs, const std::vector<uint64_t>& hashes) {
    for (uint64_t key_hash : hashes) {
        EXPECT_EQ(lhs.index_hash(key_hash), rhs.index_hash(key_hash));
        EXPECT_EQ(lhs.index_no_remap_hash(key_hash), rhs.index_no_remap_hash(key_hash));
    }
}

template <typename Hash, typename Key>
std::vector<size_t>
CollectIndexes(const Hash& hash, const std::vector<Key>& keys) {
    std::vector<size_t> indexes;
    indexes.reserve(keys.size());
    for (const auto& key : keys) {
        indexes.push_back(IndexKey(hash, key));
    }
    return indexes;
}

template <typename Hash>
std::vector<size_t>
CollectHashIndexes(const Hash& hash, const std::vector<uint64_t>& hashes) {
    std::vector<size_t> indexes;
    indexes.reserve(hashes.size());
    for (uint64_t key_hash : hashes) {
        indexes.push_back(hash.index_hash(key_hash));
    }
    return indexes;
}

uint64_t
StableFingerprint(const std::vector<uint8_t>& bytes) {
    uint64_t hash = 1469598103934665603ull;
    for (uint8_t byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

template <typename Key, typename Value>
struct StaticMapTables {
    std::vector<Key> keys_by_index;
    std::vector<Value> values_by_index;
};

template <typename Hash, typename Key, typename Value>
StaticMapTables<Key, Value>
MakeStaticMapTables(const Hash& hash, const std::vector<Key>& keys, const std::vector<Value>& values) {
    StaticMapTables<Key, Value> tables;
    tables.keys_by_index.resize(hash.n());
    tables.values_by_index.resize(hash.n());
    for (size_t i = 0; i < keys.size(); ++i) {
        const size_t index = IndexKey(hash, keys[i]);
        tables.keys_by_index[index] = keys[i];
        tables.values_by_index[index] = values[i];
    }
    return tables;
}

template <typename Hash, typename Key, typename Value>
std::optional<Value>
StaticMapFind(const Hash& hash, const StaticMapTables<Key, Value>& tables, const Key& query) {
    if (hash.n() == 0) {
        return std::nullopt;
    }
    const size_t index = IndexKey(hash, query);
    if (!(tables.keys_by_index[index] == query)) {
        return std::nullopt;
    }
    return tables.values_by_index[index];
}

template <typename Key, typename Value, typename PtrHashType, typename UnorderedHasher = std::hash<Key>>
void
ExpectStaticMapMatchesUnorderedMap(const PtrHashType& hash, const std::vector<Key>& keys,
                                   const std::vector<Value>& values, const std::vector<Key>& queries,
                                   UnorderedHasher unordered_hasher = UnorderedHasher()) {
    ASSERT_EQ(keys.size(), values.size());

    std::unordered_map<Key, Value, UnorderedHasher> expected(0, unordered_hasher);
    for (size_t i = 0; i < keys.size(); ++i) {
        expected.emplace(keys[i], values[i]);
    }
    const auto tables = MakeStaticMapTables(hash, keys, values);

    for (const auto& query : queries) {
        const auto actual = StaticMapFind(hash, tables, query);
        const auto expected_it = expected.find(query);
        if (expected_it == expected.end()) {
            EXPECT_FALSE(actual.has_value());
        } else {
            ASSERT_TRUE(actual.has_value());
            EXPECT_EQ(*actual, expected_it->second);
        }
    }
}

uint32_t
ReadU32(const std::vector<uint8_t>& bytes, size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

uint64_t
ReadU64(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
    }
    return value;
}

void
WriteU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void
WriteU64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

PtrHash
Deserialize(const std::vector<uint8_t>& bytes) {
    return PtrHash::deserialize(bytes.data(), bytes.size());
}

size_t
CountMappingsForPath(const std::string& path) {
#if defined(__linux__)
    std::ifstream maps("/proc/self/maps");
    size_t count = 0;
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find(path) != std::string::npos) {
            ++count;
        }
    }
    return count;
#else
    (void)path;
    return 0;
#endif
}

struct IdentityHasher {
    uint64_t
    operator()(uint64_t key) const {
        return key;
    }
};

struct CustomKey {
    uint64_t tenant = 0;
    std::string name;

    bool
    operator==(const CustomKey& other) const {
        return tenant == other.tenant && name == other.name;
    }
};

struct CustomKeyHasher {
    uint64_t
    operator()(const CustomKey& key) const {
        uint64_t hash = key.tenant * 0x9e3779b97f4a7c15ull;
        for (unsigned char byte : key.name) {
            hash ^= static_cast<uint64_t>(byte);
            hash *= 1099511628211ull;
        }
        return hash;
    }
};

struct ConstantCustomKeyHasher {
    uint64_t
    operator()(const CustomKey&) const {
        return 7;
    }
};

std::vector<uint64_t>
MakeRandomIntegerKeys(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::unordered_set<uint64_t> seen;
    std::vector<uint64_t> keys;
    keys.reserve(n);
    while (keys.size() < n) {
        const uint64_t key = rng();
        if (seen.insert(key).second) {
            keys.push_back(key);
        }
    }
    return keys;
}

std::string
MakeRandomString(std::mt19937_64& rng) {
    const size_t length = static_cast<size_t>(rng() % 32);
    std::string value;
    value.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        const char c = static_cast<char>('a' + (rng() % 26));
        value.push_back(c);
    }
    return value;
}

std::vector<std::string>
MakeRandomStringKeys(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::unordered_set<std::string> seen;
    std::vector<std::string> keys;
    keys.reserve(n);
    while (keys.size() < n) {
        std::string key = MakeRandomString(rng);
        key += "#" + std::to_string(rng());
        if (seen.insert(key).second) {
            keys.push_back(std::move(key));
        }
    }
    return keys;
}

constexpr size_t kSerializedHeaderSize = 88;
constexpr size_t kSerializedNOffset = 16;
constexpr size_t kSerializedSlotsTotalOffset = 24;
constexpr size_t kSerializedBucketsTotalOffset = 32;
constexpr size_t kSerializedPilotCountOffset = 48;
constexpr size_t kSerializedRemapCountOffset = 56;
constexpr size_t kSerializedPartsOffset = 64;
constexpr size_t kSerializedSlotsPerPartOffset = 72;
constexpr size_t kSerializedBucketsPerPartOffset = 80;

}  // namespace

TEST(PtrHashTest, BuildIntegerKeysReturnsMinimalPermutation) {
    std::vector<uint64_t> keys = {0, 1, 42, 999, std::numeric_limits<uint64_t>::max(), 1ull << 63, (1ull << 63) + 17};
    auto hash = PtrHash::build(keys);
    ExpectMinimalPerfect(hash, keys);
}

TEST(PtrHashTest, BuildRandomSizedIntegerSetsReturnsMinimalPermutation) {
    for (size_t n :
         {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5}, size_t{6}, size_t{7}, size_t{8}, size_t{9},
          size_t{10}, size_t{30}, size_t{100}, size_t{300}, size_t{1000}, size_t{3000}, size_t{10000}, size_t{30000}}) {
        auto keys = MakeIntegerKeys(n);
        auto hash = PtrHash::build(keys);
        ExpectMinimalPerfect(hash, keys);
    }
}

TEST(PtrHashTest, BuildMultipleIntegerSetsReturnsMinimalPermutation) {
    for (uint64_t multiplier : {uint64_t{1}, uint64_t{1} << 40, uint64_t{1000000000000}, uint64_t{94143178827}}) {
        for (size_t n :
             {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5}, size_t{6}, size_t{7}, size_t{8},
              size_t{9}, size_t{10}, size_t{30}, size_t{100}, size_t{300}, size_t{1000}, size_t{3000}, size_t{10000}}) {
            auto keys = MakeMultipleKeys(n, multiplier);
            auto hash = PtrHash::build(keys);
            ExpectMinimalPerfect(hash, keys);
        }
    }
}

TEST(PtrHashTest, IntegerKeyTypesCompileAndQuery) {
    auto check = [](const auto& keys) {
        auto hash = PtrHash::build(keys);
        ExpectMinimalPerfect(hash, keys);
    };

    check(std::vector<uint8_t>{0, 1, std::numeric_limits<uint8_t>::max()});
    check(std::vector<uint16_t>{0, 1, 257, std::numeric_limits<uint16_t>::max()});
    check(std::vector<uint32_t>{0, 1, 65537, std::numeric_limits<uint32_t>::max()});
    check(std::vector<uint64_t>{0, 1, 1ull << 40, std::numeric_limits<uint64_t>::max()});
    check(std::vector<size_t>{0, 1, 4097, static_cast<size_t>(std::numeric_limits<uint32_t>::max())});
    check(std::vector<int8_t>{std::numeric_limits<int8_t>::min(), -1, 0, 1, std::numeric_limits<int8_t>::max()});
    check(std::vector<int16_t>{std::numeric_limits<int16_t>::min(), -1, 0, 1, std::numeric_limits<int16_t>::max()});
    check(std::vector<int32_t>{std::numeric_limits<int32_t>::min(), -1, 0, 1, std::numeric_limits<int32_t>::max()});
    check(std::vector<int64_t>{std::numeric_limits<int64_t>::min(), -1, 0, 1, std::numeric_limits<int64_t>::max()});
}

TEST(PtrHashTest, IndexSumMatchesMinimalPermutation) {
    for (size_t n : {size_t{2}, size_t{10}, size_t{100}, size_t{1000}, size_t{10000}}) {
        auto keys = MakeIntegerKeys(n);
        auto hash = PtrHash::build(keys);
        size_t sum = 0;
        for (uint64_t key : keys) {
            sum += hash.index(key);
        }
        EXPECT_EQ(sum, n * (n - 1) / 2);
    }
}

TEST(PtrHashTest, BuildUint32KeysReturnsMinimalPermutation) {
    std::vector<uint32_t> keys = {0, 1, 42, 65535, 65536, 1234567890u, std::numeric_limits<uint32_t>::max()};
    auto hash = PtrHash::build(keys);
    ExpectMinimalPerfect(hash, keys);
}

TEST(PtrHashTest, BuildInt32KeysReturnsMinimalPermutation) {
    std::vector<int32_t> keys = {
        0, 1, -1, 42, -42, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    auto hash = PtrHash::build(keys);
    ExpectMinimalPerfect(hash, keys);
}

TEST(PtrHashTest, BuildStringKeysReturnsMinimalPermutation) {
    std::vector<std::string> keys = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"};
    auto hash = PtrHash::build(keys);
    ExpectMinimalPerfect(hash, keys);
}

TEST(PtrHashTest, StringKeyCornerCases) {
    std::vector<std::string> keys;
    keys.emplace_back("");
    keys.emplace_back("a");
    keys.emplace_back("abc");
    keys.emplace_back("prefix");
    keys.emplace_back("prefix_suffix");
    keys.emplace_back("a\0", 2);
    keys.emplace_back("a\0b", 3);
    keys.emplace_back("\0leading", 8);
    keys.emplace_back(4096, 'x');
    keys.back()[2048] = '\0';

    auto hash = PtrHash::build(keys);
    ExpectMinimalPerfect(hash, keys);
}

TEST(PtrHashTest, StringViewKeyCornerCases) {
    std::vector<std::string> backing;
    backing.emplace_back("");
    backing.emplace_back("short");
    backing.emplace_back("a\0", 2);
    backing.emplace_back("a\0b", 3);
    backing.emplace_back(1024, 'q');
    backing.back()[511] = '\0';

    std::vector<std::string_view> keys;
    keys.reserve(backing.size());
    for (const auto& key : backing) {
        keys.emplace_back(key.data(), key.size());
    }

    auto hash = PtrHash::build(keys);
    ExpectMinimalPerfect(hash, keys);
}

TEST(PtrHashTest, BuildPrehashedKeysReturnsMinimalPermutation) {
    auto hashes = MakeIntegerKeys(32);
    auto hash = PtrHash::build_hashes(hashes);
    ExpectMinimalPerfectHashes(hash, hashes);
}

TEST(PtrHashTest, PtrHashWithHasherUsesExternalHasher) {
    auto keys = MakeIntegerKeys(64);
    auto hash = PtrHashWithHasher<uint64_t, IdentityHasher>::build(keys, IdentityHasher{});
    ExpectMinimalPerfect(hash, keys);
}

TEST(PtrHashTest, PtrHashWithHasherSupportsCustomKeyStaticMapPattern) {
    std::vector<CustomKey> keys = {{1, "alpha"}, {1, "beta"}, {2, "alpha"}, {3, "tenant-three"}, {99, "omega"}};
    std::vector<std::string> values = {"a1", "b1", "a2", "t3", "o99"};
    std::vector<CustomKey> queries = keys;
    queries.push_back({1, "missing"});
    queries.push_back({4, "alpha"});
    queries.push_back({99, "omega-x"});

    auto hash = PtrHashWithHasher<CustomKey, CustomKeyHasher>::build(keys, CustomKeyHasher{});
    ExpectStaticMapMatchesUnorderedMap(hash, keys, values, queries, CustomKeyHasher{});
}

TEST(PtrHashTest, PtrHashWithHasherRejectsExternalHashCollisions) {
    std::vector<CustomKey> keys = {{1, "alpha"}, {2, "beta"}};

    std::unordered_map<CustomKey, int, ConstantCustomKeyHasher> unordered;
    unordered.emplace(keys[0], 10);
    unordered.emplace(keys[1], 20);
    ASSERT_EQ(unordered.size(), 2);

    EXPECT_THROW((PtrHashWithHasher<CustomKey, ConstantCustomKeyHasher>::build(keys, ConstantCustomKeyHasher{})),
                 std::invalid_argument);
}

TEST(PtrHashTest, EmptyHashHasZeroSizeAndQueryThrows) {
    auto hash = PtrHash::build(std::vector<uint64_t>{});
    EXPECT_EQ(hash.n(), 0);
    EXPECT_EQ(hash.max_index(), 0);
    EXPECT_THROW(hash.index(1), std::out_of_range);
    EXPECT_THROW(hash.index_no_remap(1), std::out_of_range);
}

TEST(PtrHashTest, SingleKeyMapsToZero) {
    auto hash = PtrHash::build(std::vector<uint64_t>{42});
    EXPECT_EQ(hash.n(), 1);
    EXPECT_EQ(hash.index(42), 0);
    EXPECT_LT(hash.index_no_remap(42), hash.max_index());
    EXPECT_GT(hash.max_index(), hash.n());
}

TEST(PtrHashTest, SupportsBucketFunctions) {
    auto keys = MakeIntegerKeys(100);
    for (BucketFunction bucket_function :
         {BucketFunction::Linear, BucketFunction::SquareEps, BucketFunction::CubicEps}) {
        PtrHashParams params;
        params.bucket_function = bucket_function;
        auto hash = PtrHash::build(keys, params);
        ExpectMinimalPerfect(hash, keys);
    }
}

TEST(PtrHashTest, SupportsBuildThreadSettings) {
    auto keys = MakeIntegerKeys(120000);
    for (size_t build_threads : {size_t{1}, size_t{2}}) {
        PtrHashParams params;
        params.build_threads = build_threads;
        auto hash = PtrHash::build(keys, params);
        ExpectMinimalPerfect(hash, keys);
    }
}

TEST(PtrHashTest, RejectsInvalidParams) {
    const std::vector<uint64_t> keys = {1, 2, 3};

    PtrHashParams params;
    params.alpha = 0.0;
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);

    params = PtrHashParams();
    params.alpha = 1.01;
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);

    params = PtrHashParams();
    params.alpha = 1e-300;
    EXPECT_THROW(PtrHash::build(keys, params), std::overflow_error);

    params = PtrHashParams();
    params.lambda = 0.0;
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);

    params = PtrHashParams();
    params.lambda = std::numeric_limits<double>::infinity();
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);

    params = PtrHashParams();
    params.max_pilot = 256;
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);
}

TEST(PtrHashTest, RejectsInvalidParamsForEmptyInput) {
    const std::vector<uint64_t> keys;

    PtrHashParams params;
    params.alpha = 0.0;
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);

    params = PtrHashParams();
    params.lambda = 0.0;
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);

    params = PtrHashParams();
    params.max_pilot = 256;
    EXPECT_THROW(PtrHash::build(keys, params), std::invalid_argument);
}

TEST(PtrHashTest, RejectsDuplicateKeys) {
    EXPECT_THROW(PtrHash::build(std::vector<uint64_t>{1, 2, 1}), std::invalid_argument);
    EXPECT_THROW(PtrHash::build(std::vector<std::string>{"", ""}), std::invalid_argument);

    std::vector<std::string> backing;
    backing.emplace_back("a\0b", 3);
    backing.emplace_back("a\0b", 3);
    std::vector<std::string_view> views;
    for (const auto& key : backing) {
        views.emplace_back(key.data(), key.size());
    }
    EXPECT_THROW(PtrHash::build(views), std::invalid_argument);
}

TEST(PtrHashTest, RejectsDuplicatePrehashedValues) {
    EXPECT_THROW(PtrHash::build_hashes(std::vector<uint64_t>{11, 17, 11}), std::invalid_argument);
}

TEST(PtrHashTest, SerializedHeaderUsesPtrHashMagicAndVersionOne) {
    auto hash = PtrHash::build(std::vector<uint64_t>{1, 2, 3});
    const auto& bytes = hash.serialize();
    ASSERT_GE(bytes.size(), 12);

    const std::array<uint8_t, 8> expected_magic = {'P', 'T', 'R', 'H', 'A', 'S', 'H', '\0'};
    EXPECT_TRUE(std::equal(expected_magic.begin(), expected_magic.end(), bytes.begin()));
    EXPECT_EQ(ReadU32(bytes, 8), 1u);
}

TEST(PtrHashTest, GoldenMappingsStayStableForRepresentativeKeyKinds) {
    PtrHashParams params;
    params.seed = 0x123456789abcdef0ull;
    params.build_threads = 1;

    params.bucket_function = BucketFunction::SquareEps;
    std::vector<uint64_t> integer_keys = {
        0, 1, 42, 999, std::numeric_limits<uint64_t>::max(), 1ull << 63, (1ull << 63) + 17};
    auto integer_hash = PtrHash::build(integer_keys, params);
    ExpectMinimalPerfect(integer_hash, integer_keys);

    params.bucket_function = BucketFunction::CubicEps;
    std::vector<std::string> string_keys = {
        "", "alpha", "beta", "gamma", "delta", std::string("a\0b", 3), "longer-string-value"};
    auto string_hash = PtrHash::build(string_keys, params);
    ExpectMinimalPerfect(string_hash, string_keys);

    params.bucket_function = BucketFunction::Linear;
    std::vector<uint64_t> prehashed_keys = {11, 0x100000001ull, 0xabcdef1234567890ull,
                                            99, 123456789,      0xfedcba9876543210ull};
    auto prehashed_hash = PtrHash::build_hashes(prehashed_keys, params);
    ExpectMinimalPerfectHashes(prehashed_hash, prehashed_keys);

#if defined(__SIZEOF_INT128__)
    EXPECT_EQ(CollectIndexes(integer_hash, integer_keys), (std::vector<size_t>{3, 5, 2, 0, 6, 1, 4}));
    EXPECT_EQ(StableFingerprint(integer_hash.serialize()), 0xbff1bd6e24232c11ull);

    EXPECT_EQ(CollectIndexes(string_hash, string_keys), (std::vector<size_t>{5, 2, 3, 4, 6, 1, 0}));
    EXPECT_EQ(StableFingerprint(string_hash.serialize()), 0x3cbc60b3f568fd3bull);

    EXPECT_EQ(CollectHashIndexes(prehashed_hash, prehashed_keys), (std::vector<size_t>{1, 3, 2, 4, 5, 0}));
    EXPECT_EQ(StableFingerprint(prehashed_hash.serialize()), 0xd47690f139b34e71ull);
#endif
}

TEST(PtrHashTest, SerializedBufferUsesExactCapacity) {
    auto hash = PtrHash::build(MakeIntegerKeys(1000));
    const auto& bytes = hash.serialize();
    EXPECT_EQ(bytes.size(), bytes.capacity());
}

TEST(PtrHashTest, DeserializeRoundTripPreservesQueries) {
    auto keys = MakeIntegerKeys(128);
    auto hash = PtrHash::build(keys);
    auto roundtrip = Deserialize(hash.serialize());
    ExpectSameQueries(hash, roundtrip, keys);
}

TEST(PtrHashTest, SaveLoadRoundTripPreservesQueries) {
    auto keys = MakeIntegerKeys(128);
    auto hash = PtrHash::build(keys);
    TempFile file("save_load");
    hash.save(file.path());

    auto loaded = PtrHash::load(file.path());
    ExpectSameQueries(hash, loaded, keys);
}

TEST(PtrHashTest, SaveLoadRoundTripPreservesStringAndPrehashedQueries) {
    std::vector<std::string> string_keys = {"", "alpha", "beta", std::string("a\0b", 3), "gamma"};
    auto string_hash = PtrHash::build(string_keys);
    TempFile string_file("save_load_string");
    string_hash.save(string_file.path());

    auto loaded_string = PtrHash::load(string_file.path());
    ExpectSameQueries(string_hash, loaded_string, string_keys);
    EXPECT_THROW(loaded_string.index(uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(loaded_string.index_hash(1), std::invalid_argument);

    std::vector<std::string> backing = {"view-alpha", "view-beta", std::string("view\0gamma", 10)};
    std::vector<std::string_view> string_views;
    string_views.reserve(backing.size());
    for (const auto& key : backing) {
        string_views.emplace_back(key.data(), key.size());
    }
    auto string_view_hash = PtrHash::build(string_views);
    TempFile string_view_file("save_load_string_view");
    string_view_hash.save(string_view_file.path());

    auto loaded_string_view = PtrHash::load(string_view_file.path());
    ExpectSameQueries(string_view_hash, loaded_string_view, string_views);
    EXPECT_THROW(loaded_string_view.index(uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(loaded_string_view.index_hash(1), std::invalid_argument);

    auto prehashed_keys = MakeIntegerKeys(32);
    auto prehashed_hash = PtrHash::build_hashes(prehashed_keys);
    TempFile prehashed_file("save_load_prehashed");
    prehashed_hash.save(prehashed_file.path());

    auto loaded_prehashed = PtrHash::load(prehashed_file.path());
    ASSERT_EQ(prehashed_hash.serialize(), loaded_prehashed.serialize());
    ExpectSameHashQueries(prehashed_hash, loaded_prehashed, prehashed_keys);
    EXPECT_THROW(loaded_prehashed.index(uint64_t{prehashed_keys.front()}), std::invalid_argument);
    EXPECT_THROW(loaded_prehashed.index(std::string_view("not-a-prehash")), std::invalid_argument);
}

TEST(PtrHashTest, DeserializeRejectsCorruptInput) {
    auto hash = PtrHash::build(std::vector<uint64_t>{1, 2, 3, 4});
    auto bytes = hash.serialize();
    ASSERT_GE(bytes.size(), 88);

    auto bad = bytes;
    bad[0] = 'X';
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);

    bad = bytes;
    WriteU32(bad, 8, 2);
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);

    bad = bytes;
    WriteU32(bad, 12, 0);
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);

    bad = bytes;
    WriteU32(bad, 12, 4 | (99u << 8));
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);

    bad = bytes;
    WriteU32(bad, 12, 4 | (99u << 16));
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);

    bad = bytes;
    WriteU64(bad, 48, 0);
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);

    bad.assign(bytes.begin(), bytes.begin() + 10);
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);

    bad = bytes;
    bad.pop_back();
    EXPECT_THROW(Deserialize(bad), std::invalid_argument);
}

TEST(PtrHashTest, DeserializeRejectsNullNonEmptyInput) {
    EXPECT_THROW(PtrHash::deserialize(nullptr, 1), std::invalid_argument);
    EXPECT_THROW(PtrHashView::from_bytes(nullptr, 1), std::invalid_argument);
}

TEST(PtrHashTest, DeserializeRejectsImpossibleEmptyLayout) {
    auto hash = PtrHash::build(std::vector<uint64_t>{});
    auto bytes = hash.serialize();
    ASSERT_EQ(bytes.size(), kSerializedHeaderSize);

    WriteU64(bytes, kSerializedPartsOffset, 1);
    EXPECT_THROW(Deserialize(bytes), std::invalid_argument);
}

TEST(PtrHashTest, DeserializeRejectsNonEmptyLayoutWithZeroBuckets) {
    auto hash = PtrHash::build(std::vector<uint64_t>{});
    auto bytes = hash.serialize();
    ASSERT_EQ(bytes.size(), kSerializedHeaderSize);

    WriteU64(bytes, kSerializedNOffset, 1);
    WriteU64(bytes, kSerializedSlotsTotalOffset, 1);
    WriteU64(bytes, kSerializedBucketsTotalOffset, 0);
    WriteU64(bytes, kSerializedPilotCountOffset, 0);
    WriteU64(bytes, kSerializedRemapCountOffset, 0);
    WriteU64(bytes, kSerializedPartsOffset, 1);
    WriteU64(bytes, kSerializedSlotsPerPartOffset, 1);
    WriteU64(bytes, kSerializedBucketsPerPartOffset, 0);
    EXPECT_THROW(Deserialize(bytes), std::invalid_argument);
}

TEST(PtrHashTest, DeserializeRejectsSerializedSizeOverflow) {
    auto hash = PtrHash::build(std::vector<uint64_t>{});
    auto bytes = hash.serialize();
    ASSERT_EQ(bytes.size(), kSerializedHeaderSize);

    const uint64_t huge = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    WriteU64(bytes, kSerializedNOffset, 1);
    WriteU64(bytes, kSerializedSlotsTotalOffset, 1);
    WriteU64(bytes, kSerializedBucketsTotalOffset, huge);
    WriteU64(bytes, kSerializedPilotCountOffset, huge);
    WriteU64(bytes, kSerializedRemapCountOffset, 0);
    WriteU64(bytes, kSerializedPartsOffset, 1);
    WriteU64(bytes, kSerializedSlotsPerPartOffset, 1);
    WriteU64(bytes, kSerializedBucketsPerPartOffset, huge);
    EXPECT_THROW(Deserialize(bytes), std::overflow_error);
}

TEST(PtrHashTest, MutatedSerializedBytesDoNotCrashOrReturnOutOfRange) {
    auto hash = PtrHash::build(MakeIntegerKeys(128));
    const auto& bytes = hash.serialize();

    for (size_t offset = 0; offset < bytes.size(); ++offset) {
        auto mutated = bytes;
        mutated[offset] ^= 0x5a;
        try {
            auto loaded = Deserialize(mutated);
            ASSERT_GT(loaded.n(), 0);
            EXPECT_LT(loaded.index(uint64_t{42}), loaded.n());
            EXPECT_LT(loaded.index_no_remap(uint64_t{42}), loaded.max_index());
        } catch (const std::invalid_argument&) {
        } catch (const std::overflow_error&) {
        }
    }
}

TEST(PtrHashTest, DeserializeRejectsOutOfRangeRemapEntries) {
    auto hash = PtrHash::build(MakeIntegerKeys(128));
    auto bytes = hash.serialize();
    const uint64_t n = ReadU64(bytes, kSerializedNOffset);
    const uint64_t pilot_count = ReadU64(bytes, kSerializedPilotCountOffset);
    const uint64_t remap_count = ReadU64(bytes, kSerializedRemapCountOffset);
    ASSERT_GT(remap_count, 0);
    ASSERT_LE(n, std::numeric_limits<uint32_t>::max());
    ASSERT_GE(bytes.size(), kSerializedHeaderSize + pilot_count + sizeof(uint32_t));

    WriteU32(bytes, kSerializedHeaderSize + static_cast<size_t>(pilot_count), static_cast<uint32_t>(n));
    EXPECT_THROW(Deserialize(bytes), std::invalid_argument);
}

TEST(PtrHashTest, RepeatedBuildsStayMinimalForSkewedDistributions) {
    std::vector<std::vector<uint64_t>> datasets;
    datasets.emplace_back(MakeIntegerKeys(4096));

    std::vector<uint64_t> high_bits;
    high_bits.reserve(4096);
    for (uint64_t i = 0; i < 4096; ++i) {
        high_bits.push_back((i << 32) | ((i * 17) & 0xffffu));
    }
    datasets.emplace_back(std::move(high_bits));

    std::vector<uint64_t> descending;
    descending.reserve(4096);
    for (uint64_t i = 0; i < 4096; ++i) {
        descending.push_back(std::numeric_limits<uint64_t>::max() - i * 8191);
    }
    datasets.emplace_back(std::move(descending));

    for (size_t dataset = 0; dataset < datasets.size(); ++dataset) {
        for (uint64_t seed : {uint64_t{0}, uint64_t{1}, uint64_t{0x9e3779b97f4a7c15ull}}) {
            for (BucketFunction bucket_function :
                 {BucketFunction::Linear, BucketFunction::SquareEps, BucketFunction::CubicEps}) {
                SCOPED_TRACE("dataset=" + std::to_string(dataset) + " seed=" + std::to_string(seed));
                PtrHashParams params;
                params.seed = seed;
                params.bucket_function = bucket_function;
                params.build_threads = 1;
                auto hash = PtrHash::build(datasets[dataset], params);
                ExpectMinimalPerfect(hash, datasets[dataset]);
            }
        }
    }
}

TEST(PtrHashTest, RandomizedStaticMapDifferentialMatchesUnorderedMap) {
    const std::vector<size_t> sizes = {0, 1, 7, 64, 257};
    const std::vector<BucketFunction> bucket_functions = {BucketFunction::Linear, BucketFunction::SquareEps,
                                                          BucketFunction::CubicEps};
    const std::vector<size_t> thread_counts = {1, 2, 4};

    for (size_t n : sizes) {
        for (BucketFunction bucket_function : bucket_functions) {
            for (size_t threads : thread_counts) {
                SCOPED_TRACE("integer n=" + std::to_string(n) + " bucket=" +
                             std::to_string(static_cast<int>(bucket_function)) + " threads=" + std::to_string(threads));
                PtrHashParams params;
                params.seed = 0x72616e646f6d0000ull + n + threads;
                params.bucket_function = bucket_function;
                params.build_threads = threads;

                auto keys = MakeRandomIntegerKeys(n, params.seed);
                std::vector<uint64_t> values;
                values.reserve(keys.size());
                for (size_t i = 0; i < keys.size(); ++i) {
                    values.push_back(keys[i] ^ (0x9e3779b97f4a7c15ull + i));
                }

                std::vector<uint64_t> queries = keys;
                std::unordered_set<uint64_t> key_set(keys.begin(), keys.end());
                for (uint64_t candidate : MakeRandomIntegerKeys(128, params.seed ^ 0x5eed5eedull)) {
                    if (key_set.find(candidate) == key_set.end()) {
                        queries.push_back(candidate);
                    }
                }

                auto hash = PtrHash::build(keys, params);
                ExpectStaticMapMatchesUnorderedMap(hash, keys, values, queries);
            }
        }
    }

    const std::vector<size_t> string_sizes = {0, 1, 9, 96};
    for (size_t n : string_sizes) {
        for (BucketFunction bucket_function : bucket_functions) {
            for (size_t threads : {size_t{1}, size_t{2}}) {
                SCOPED_TRACE("string n=" + std::to_string(n) + " bucket=" +
                             std::to_string(static_cast<int>(bucket_function)) + " threads=" + std::to_string(threads));
                PtrHashParams params;
                params.seed = 0x737472696e670000ull + n + threads;
                params.bucket_function = bucket_function;
                params.build_threads = threads;

                auto keys = MakeRandomStringKeys(n, params.seed);
                std::vector<uint64_t> values;
                values.reserve(keys.size());
                for (size_t i = 0; i < keys.size(); ++i) {
                    values.push_back((i + 1) * 17);
                }

                std::vector<std::string> queries = keys;
                std::unordered_set<std::string> key_set(keys.begin(), keys.end());
                for (const auto& candidate : MakeRandomStringKeys(128, params.seed ^ 0xabcddcbaull)) {
                    if (key_set.find(candidate) == key_set.end()) {
                        queries.push_back(candidate);
                    }
                }

                auto hash = PtrHash::build(keys, params);
                ExpectStaticMapMatchesUnorderedMap(hash, keys, values, queries);
            }
        }
    }
}

TEST(PtrHashTest, ConcurrentQueriesPreserveResults) {
    auto keys = MakeIntegerKeys(10000);
    auto hash = PtrHash::build(keys);

    std::vector<size_t> expected;
    expected.reserve(keys.size());
    for (uint64_t key : keys) {
        expected.push_back(hash.index(key));
    }

    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;
    for (size_t thread = 0; thread < 4; ++thread) {
        threads.emplace_back([&, thread] {
            for (size_t round = 0; round < 20; ++round) {
                for (size_t i = thread; i < keys.size(); i += 4) {
                    if (hash.index(keys[i]) != expected[i]) {
                        ok.store(false, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(ok.load(std::memory_order_relaxed));
}

TEST(PtrHashTest, NonMemberQueriesStayInRange) {
    auto integer_hash = PtrHash::build(std::vector<uint64_t>{10, 20, 30, 40});
    for (uint64_t key : {uint64_t{0}, uint64_t{1}, uint64_t{999999}, std::numeric_limits<uint64_t>::max()}) {
        EXPECT_LT(integer_hash.index(key), integer_hash.n());
        EXPECT_LT(integer_hash.index_no_remap(key), integer_hash.max_index());
    }

    auto string_hash = PtrHash::build(std::vector<std::string>{"alpha", "beta", "gamma"});
    for (std::string_view key : {std::string_view(""), std::string_view("delta"), std::string_view("alpha\0x", 7)}) {
        EXPECT_LT(string_hash.index(key), string_hash.n());
        EXPECT_LT(string_hash.index_no_remap(key), string_hash.max_index());
    }

    auto prehashed = PtrHash::build_hashes(std::vector<uint64_t>{100, 200, 300});
    for (uint64_t key_hash : {uint64_t{0}, uint64_t{100}, uint64_t{999}}) {
        EXPECT_LT(prehashed.index_hash(key_hash), prehashed.n());
        EXPECT_LT(prehashed.index_no_remap_hash(key_hash), prehashed.max_index());
    }
}

TEST(PtrHashTest, StaticMapPatternChecksKeysBeforeReturningValues) {
    std::vector<uint64_t> integer_keys = {10, 20, 30, 40, 50};
    std::vector<std::string> integer_values = {"ten", "twenty", "thirty", "forty", "fifty"};
    std::vector<uint64_t> integer_queries = integer_keys;
    integer_queries.insert(integer_queries.end(), {0, 1, 21, 9999, std::numeric_limits<uint64_t>::max()});

    auto integer_hash = PtrHash::build(integer_keys);
    ExpectStaticMapMatchesUnorderedMap(integer_hash, integer_keys, integer_values, integer_queries);

    std::vector<std::string> string_keys = {"", "alpha", "beta", "gamma", std::string("a\0b", 3), "prefix"};
    std::vector<uint64_t> string_values = {0, 1, 2, 3, 4, 5};
    std::vector<std::string> string_queries = string_keys;
    string_queries.push_back("delta");
    string_queries.push_back(std::string("a\0c", 3));
    string_queries.push_back("prefix_suffix");

    auto string_hash = PtrHash::build(string_keys);
    ExpectStaticMapMatchesUnorderedMap(string_hash, string_keys, string_values, string_queries);
}

TEST(PtrHashTest, MappedPtrHashOpenPreservesQueries) {
    auto keys = MakeIntegerKeys(128);
    auto hash = PtrHash::build(keys);
    TempFile file("mapped");
    hash.save(file.path());

#if defined(__unix__) || defined(__APPLE__)
    auto mapped = MappedPtrHash::open(file.path());
    ExpectMinimalPerfect(mapped, keys);
#else
    EXPECT_THROW(MappedPtrHash::open(file.path()), std::runtime_error);
#endif
}

TEST(PtrHashTest, MappedPtrHashOpenPreservesStringAndPrehashedQueries) {
#if defined(__unix__) || defined(__APPLE__)
    std::vector<std::string> string_keys = {"mapped-alpha", "mapped-beta", std::string("mapped\0gamma", 12), ""};
    auto string_hash = PtrHash::build(string_keys);
    TempFile string_file("mapped_string");
    string_hash.save(string_file.path());

    auto mapped_string = MappedPtrHash::open(string_file.path());
    ExpectSameQueryResults(string_hash, mapped_string, string_keys);
    EXPECT_THROW(mapped_string.index(uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(mapped_string.index_hash(1), std::invalid_argument);

    std::vector<std::string> backing = {"mapped-view-alpha", "mapped-view-beta", std::string("mapped-view\0x", 13)};
    std::vector<std::string_view> string_views;
    string_views.reserve(backing.size());
    for (const auto& key : backing) {
        string_views.emplace_back(key.data(), key.size());
    }
    auto string_view_hash = PtrHash::build(string_views);
    TempFile string_view_file("mapped_string_view");
    string_view_hash.save(string_view_file.path());

    auto mapped_string_view = MappedPtrHash::open(string_view_file.path());
    ExpectSameQueryResults(string_view_hash, mapped_string_view, string_views);
    EXPECT_THROW(mapped_string_view.index(uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(mapped_string_view.index_hash(1), std::invalid_argument);

    auto prehashed_keys = MakeIntegerKeys(32);
    auto prehashed_hash = PtrHash::build_hashes(prehashed_keys);
    TempFile prehashed_file("mapped_prehashed");
    prehashed_hash.save(prehashed_file.path());

    auto mapped_prehashed = MappedPtrHash::open(prehashed_file.path());
    ExpectSameHashQueries(prehashed_hash, mapped_prehashed, prehashed_keys);
    EXPECT_THROW(mapped_prehashed.index(uint64_t{prehashed_keys.front()}), std::invalid_argument);
    EXPECT_THROW(mapped_prehashed.index(std::string_view("not-a-prehash")), std::invalid_argument);
#else
    GTEST_SKIP() << "mmap loading is only available on POSIX platforms";
#endif
}

TEST(PtrHashTest, MappedPtrHashOpenWithOffsetAndRejectsBadOffset) {
    auto keys = MakeIntegerKeys(128);
    auto hash = PtrHash::build(keys);
    const auto& bytes = hash.serialize();
    const std::array<char, 7> prefix = {'p', 'r', 'e', 'f', 'i', 'x', ':'};
    TempFile file("mapped_offset");

    {
        std::ofstream out(file.path(), std::ios::binary);
        ASSERT_TRUE(out);
        out.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(out.good());
    }

#if defined(__unix__) || defined(__APPLE__)
    auto mapped = MappedPtrHash::open(file.path(), prefix.size());
    ExpectMinimalPerfect(mapped, keys);
    EXPECT_THROW(MappedPtrHash::open(file.path(), prefix.size() + bytes.size() + 1), std::invalid_argument);
#else
    EXPECT_THROW(MappedPtrHash::open(file.path(), prefix.size()), std::runtime_error);
#endif
}

TEST(PtrHashTest, MappedPtrHashRejectsEmptyFileBeforeMapping) {
    TempFile file("mapped_empty");
    {
        std::ofstream out(file.path(), std::ios::binary);
        ASSERT_TRUE(out);
    }

#if defined(__unix__) || defined(__APPLE__)
    try {
        (void)MappedPtrHash::open(file.path());
        FAIL() << "expected empty mmap file to be rejected";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ("PtrHash mmap file is empty", e.what());
    }
#else
    EXPECT_THROW(MappedPtrHash::open(file.path()), std::runtime_error);
#endif
}

TEST(PtrHashTest, MappedPtrHashRejectsCorruptFileWithoutLeakingMapping) {
    TempFile file("mapped_corrupt");
    {
        std::ofstream out(file.path(), std::ios::binary);
        ASSERT_TRUE(out);
        std::string bytes(4096, 'x');
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(out.good());
    }

#if defined(__unix__) || defined(__APPLE__)
#if defined(__linux__)
    ASSERT_EQ(CountMappingsForPath(file.path()), 0);
#endif
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_THROW(MappedPtrHash::open(file.path()), std::invalid_argument);
    }
#if defined(__linux__)
    EXPECT_EQ(CountMappingsForPath(file.path()), 0);
#endif
#else
    EXPECT_THROW(MappedPtrHash::open(file.path()), std::runtime_error);
#endif
}

TEST(PtrHashTest, MappedPtrHashMoveKeepsValidView) {
    auto keys = MakeIntegerKeys(128);
    auto hash = PtrHash::build(keys);
    TempFile file("mapped_move");
    hash.save(file.path());

#if defined(__unix__) || defined(__APPLE__)
    auto mapped = MappedPtrHash::open(file.path());
    auto moved = std::move(mapped);
    ExpectMinimalPerfect(moved, keys);

    MappedPtrHash assigned;
    assigned = std::move(moved);
    ExpectMinimalPerfect(assigned, keys);
#else
    EXPECT_THROW(MappedPtrHash::open(file.path()), std::runtime_error);
#endif
}

TEST(PtrHashTest, QueryTypeMismatchThrows) {
    auto integer_hash = PtrHash::build(std::vector<uint64_t>{1, 2, 3});
    EXPECT_THROW(integer_hash.index(std::string_view("1")), std::invalid_argument);

    auto string_hash = PtrHash::build(std::vector<std::string>{"1", "2", "3"});
    EXPECT_THROW(string_hash.index(uint64_t{1}), std::invalid_argument);

    auto prehashed = PtrHash::build_hashes(std::vector<uint64_t>{1, 2, 3});
    EXPECT_THROW(prehashed.index(uint64_t{1}), std::invalid_argument);
    EXPECT_THROW(prehashed.index(std::string_view("1")), std::invalid_argument);
    EXPECT_NO_THROW((void)prehashed.index_hash(1));
}

TEST(PtrHashTest, CopyAndMoveKeepValidView) {
    auto keys = MakeIntegerKeys(128);
    PtrHash original = PtrHash::build(keys);

    PtrHash copied(original);
    ExpectSameQueries(original, copied, keys);

    PtrHash copy_assigned;
    copy_assigned = original;
    ExpectSameQueries(original, copy_assigned, keys);

    PtrHash moved(std::move(copied));
    ExpectMinimalPerfect(moved, keys);

    PtrHash move_assigned;
    move_assigned = std::move(copy_assigned);
    ExpectMinimalPerfect(move_assigned, keys);
}
