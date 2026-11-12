# Milvus Common

## Overview

Milvus Common is a core component library that serves as a bridge between Milvus and Knowhere, designed to decouple these two major components and provide common functionality. This library contains essential utilities and interfaces that are shared between Milvus and Knowhere, making the codebase more modular and maintainable.

## Components

### Caching Layer

### Common Utilities
- `Monitor`: Metrics collection and monitoring

### Log Module


## Build from source

### Prepare

```
pip install conan==2.25.1
conan profile detect
conan remote add default-conan-local2 https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2
```

### Build
```
make
```

Useful Makefile variables:

- `BUILD_TYPE`: CMake/Conan build type, default `Release`.
- `CONAN_CPPSTD`: C++ standard passed to Conan, default `20`.
- `CONAN_EXTRA`: extra arguments passed to `conan install`.
- `CMAKE_EXTRA`: extra arguments passed to `cmake -S . -B build`, for example `-DCMAKE_VERBOSE_MAKEFILE=ON`.
- `NPROC`: parallel job count for Conan dependency builds and CMake builds.

Example:

```
make BUILD_TYPE=Release CONAN_CPPSTD=gnu20 NPROC=4
```

### Run test

Use `make test` to enable the Conan unit-test option that installs GTest.

```
make test
```

## CacheMap

After `Manager::ConfigureTieredStorage()` initializes an eviction-enabled domain,
standalone keyed caches can share its memory/disk budget and LRU with CacheSlot cells:

```cpp
auto cache = milvus::cachinglayer::Manager::GetInstance()
                 .CreateCacheMap<std::string, int>();
auto value = std::make_unique<int>(42);
auto result = cache->Put("answer", value, {sizeof(int), 0});
if (!result) {
    // value remains owned by the caller; inspect result.error().
    return;
}

auto handle = cache->Get("answer");
if (handle) {
    assert(*handle == 42);
    // The value cannot be evicted while this handle exists.

    auto replacement = std::make_unique<int>(43);
    auto replacement_result = cache->Put("answer", replacement, {sizeof(int), 0});
    assert(!replacement_result && replacement_result.error() == milvus::cachinglayer::CacheMapError::Pinned);
}
handle = {};  // Unpin before replacing or erasing the key.
```

- Keys must be copy-constructible, equality-comparable and support `std::hash<Key>`.
  Custom keys can define `operator==` and specialize `std::hash`; each cache has its own key namespace.
- `Get` returns a move-only, read-only `CacheMap<Key, Value>::Pin` with typed ownership.
  Callers can use `auto pin = cache->Get(key)`. Pins can move between caches with the
  same Key/Value types. A miss returns an empty Pin (`!pin`); it neither inserts an entry nor loads data.
- `Put` consumes the input `unique_ptr` only on success. `Erase` also rejects pinned entries.
  A successful `Put` publishes the Value without pinning it. Concurrent eviction can
  remove it before `Put` returns, so callers must handle a subsequent `Get` miss.
- Resource usage describes only the Value payload. Keys, map buckets, index nodes and
  Entry objects are managed by the cache without being charged to the shared capacity.
  A replacement retains the stored Key and reserves the new Value payload while the
  old payload remains charged, so it can fail for capacity.
  Zero usage (`{0, 0}`) is accepted and means no payload charge in this domain; the
  byte budget does not limit entry count or total process memory. Unpinned zero-usage
  entries are removed by `Erase`, cache destruction or configured time-based background
  eviction; size-based reservation eviction does not select them.
- The index is a trimmed CacheLib `ChainedHashTable` with fixed buckets and striped mutexes.
  `CreateCacheMap<Key, Value>({.buckets_power = 10, .locks_power = 6})` selects
  1024 buckets and 64 mutexes (the defaults). It does not rehash; size the bucket
  count for the expected resident key count to avoid long chains.
- `Get` retains an Entry under its index mutex, releases that mutex, then pins under
  the node lock. If eviction won the race, it returns a miss. It does not acquire
  the global LRU lock. Unpin reuses the CacheSlot touch-window path; moving an entry
  in the LRU or notifying pending reservations can still acquire the global lock.
- Capacity admission does not wait, but mutex acquisition and eviction can block.
  Key operations must not reenter the cache; Key and Value destructors must not throw
  or reenter a cache.
- Recency retains the shared domain's unpin/touch-window policy. CacheSlot cells keep
  their existing array lookup and do not acquire a key or register in a KV index.
- The new ListNode eviction-lifetime hook requires downstream binaries using that interface to be rebuilt.
