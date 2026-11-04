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

## Keyed LRU cache

After `Manager::ConfigureTieredStorage()` initializes an eviction-enabled domain,
standalone keyed caches can share its memory/disk budget and LRU with CacheSlot cells:

```cpp
auto cache = milvus::cachinglayer::Manager::GetInstance()
                 .CreateLRUCache<std::string, int>();
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
}

auto replacement = std::make_unique<int>(43);
assert(cache->Put("answer", replacement, {sizeof(int), 0}).error() ==
       milvus::cachinglayer::LRUCacheError::Pinned);
handle = {};  // Unpin before replacing or erasing the key.
```

- Keys currently support integral types and `std::string`. Each cache has its own key namespace.
- `Get` returns a move-only, read-only `Pin<Value>` independent of the Key type.
  A miss returns an empty Pin (`!pin`); it neither inserts an entry nor loads data.
- `Put` consumes the input `unique_ptr` only on success. `Erase` also rejects pinned entries.
- Resource usage describes the complete Value payload. Entry, index allocations and a
  conservative string-key capacity charge are added by the cache. A replacement reserves
  space for the new entry while the old one remains charged, so it can fail for capacity.
- Capacity admission does not wait, but mutex acquisition and eviction can block.
  Values must not throw or reenter a cache from their destructors.
- Recency retains the shared domain's unpin/touch-window policy. CacheSlot cells keep
  their existing array lookup and do not acquire a key or register in a KV index.
- New ListNode virtual hooks require downstream binaries using that interface to be rebuilt.
