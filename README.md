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
using Cache = milvus::cachinglayer::CacheMap<std::string, int>;

Cache::PinResult GetAnswer(Cache& cache) {
    auto found = cache.Get("answer");
    if (!found || found.value()) {
        return found;
    }
    const milvus::cachinglayer::ResourceUsage estimate{sizeof(int), 0};
    auto reserved = cache.Reserve(estimate);
    if (!reserved) {
        return folly::makeUnexpected(reserved.error());
    }
    SCOPE_EXIT { cache.UnReserve(estimate); };
    auto value = std::make_unique<int>(42);
    return cache.Put("answer", value, {sizeof(*value), 0});
}

auto cache = milvus::cachinglayer::Manager::GetInstance()
                 .CreateCacheMap<std::string, int>();
auto result = GetAnswer(*cache);
```

- Keys must be copy-constructible, equality-comparable and support `std::hash<Key>`.
  Custom keys can define `operator==` and specialize `std::hash`; each cache has its own key namespace.
- `Get` and `Put` return `CacheMap<Key, Value>::PinResult` (`folly::Expected<Pin, milvus::SegcoreError>`).
  `Get` returns a successful result with an empty `Pin` on a miss. Check `result` for
  an operation error, then `result.value()` to see whether the key was found. A hit
  yields a move-only, read-only Pin. `Put` consumes the input `unique_ptr` on success
  and returns a valid Pin acquired during publication.
- `GetOrCreate(key, create)` returns the resident Pin on a hit. On a miss,
  one caller runs `create()` outside cache locks. Concurrent callers for that
  key receive their own Pin to the same value or the same creation error.
  If the cache closes first, those callers receive `FollyCancel` even if the factory later
  returns an error or throws. If creation completes first, callers receive its result even if
  the cache closes before they wake.
  Different keys can load concurrently, including keys with colliding hashes.
  The factory returns `Cache::CreateResult` containing `std::pair{std::move(value), actual_bytes}`
  or, for example, `folly::makeUnexpected(milvus::SegcoreError(milvus::ErrorCode::FileReadFailed, "metadata read failed"))`.
  If the factory returns a valid value and resource usage, `GetOrCreate` rechecks the key
  and pins the value from a concurrent `Put` if it is still resident. If the factory returns
  an error while the cache is open, `GetOrCreate` returns that error even if a concurrent `Put` succeeded.
  A later call can retry a failed factory. Keep factory dependencies acyclic.
- On a miss, call `Reserve(estimate, timeout)` before loading. It returns
  `Result` (`folly::Expected<folly::Unit, milvus::SegcoreError>`). Keep the cache
  alive and call `UnReserve(estimate)` exactly once after every successful
  reservation, including when loading or `Put` fails. A scope guard can handle
  early returns. Returning the estimate lets DList retry queued requests from
  any cache in the same Manager.
- Asynchronous loaders can `co_await cache.ReserveAsync(estimate, timeout)`.
  It returns `folly::SemiFuture<Result>`. The initial admission check and any
  immediate eviction run on the calling thread. Keep the cache alive and
  consume the result even if the load is canceled. When using an executor,
  keep it accepting work until the result is delivered. A dropped future can
  still leave a granted reservation charged. Call `UnReserve(estimate)` once
  for every successful result, including when the caller skips the load.
- A `GetOrCreate` factory can call `Reserve` before loading. Hold any successful
  reservation until `GetOrCreate` returns, then call `UnReserve` with its original estimate.
- `Put` only inserts absent keys. A duplicate returns `InvalidParameter` whether
  or not the current value is pinned. The caller keeps the input `unique_ptr`.
  After erasure or eviction the same key can be inserted again.
  `Erase` returns `UnexpectedError` for a pinned key and `OutOfRange` for an absent key.
  Closing the cache cancels in-flight operations with `FollyCancel`. Invalid input
  returns `InvalidParameter`. Insufficient capacity returns `InsufficientResource`.
- On failure, `result.error().get_error_code()` gives the code and
  `result.error().what()` gives the diagnostic message. Factory errors and caught
  `SegcoreError` exceptions keep their original code and message. Allocation failures map
  to `MemAllocateFailed`, invalid arguments to `InvalidParameter`, and range or length
  errors to `OutOfRange`. Folly cancellations map to `FollyCancel` and other future
  errors to `FollyOtherException`. Unclassified exceptions return `UnexpectedError`.
  Exception diagnostics are logged. Once `Put` has published the Value, it returns
  its Pin even if notifying waiters fails.
- Resource usage describes only the Value payload. Keys, map buckets, index nodes and
  Entry objects are managed by the cache without being charged to the shared capacity.
  Admission uses the estimate. `Put` charges the payload size passed by the
  caller, even if it exceeds the estimate. When the actual size is unavailable,
  pass the resident estimate to `Put` to use the same fallback as CacheSlot.
  Passing `{0, 0}` publishes a zero-charge entry. Unpinned zero-usage
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
- `Reserve` defaults to a zero timeout and tries admission once. A positive
  timeout waits up to that duration, and a negative timeout waits until
  capacity is available or the cache closes. Reservation can evict unpinned
  entries, including the key the caller later passes to `Put`.
  Key operations must not reenter the cache; Key and Value destructors must not throw
  or reenter a cache.
- Recency retains the shared domain's unpin/touch-window policy. CacheSlot cells keep
  their existing array lookup and do not acquire a key or register in a KV index.
- The new ListNode eviction-lifetime hook requires downstream binaries using that interface to be rebuilt.
