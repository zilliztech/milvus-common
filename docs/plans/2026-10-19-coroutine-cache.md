# Coroutine cache loading implementation plan

**Goal:** Make cache pinning, loading-resource waits and warmup awaitable without blocking the executor.

**Architecture:** `PinCells`, `PinAllCells`, `Warmup`, `Manager::CreateCacheSlot` and `Translator::get_cells` return Folly coroutine tasks. Keep the existing DList reservations and ListNode shared promises, awaiting their results inside the coroutine. Explicit synchronous `*Direct` methods remain blocking adapters. Milvus translators and their callers migrate separately before updating this dependency.

**Stack:** C++20, Folly Task, GTest, Conan 2, CMake.

## 1. Baseline and regression

- Build and run the existing tests with `make test NPROC=8` and capped child builders.
- Add a one-worker regression to `test/test_cachinglayer/test_cache_slot.cpp`: a pin waiting for DList capacity must let another task on the same worker run. Use a bounded external cleanup path so the old implementation fails rather than hanging.
- Run that test against the original implementation and record its failure.

## 2. Coroutine contracts and implementation

- Change `include/cachinglayer/Translator.h` to return `folly::coro::Task` from `get_cells` and document borrowed context/ID lifetimes and draining on failure.
- Change `include/cachinglayer/CacheSlot.h`: own pin arguments and slot lifetime, await DList reservation, translator loading and shared pin futures, preserve reservation release and error publication, and schedule background warmup as a coroutine.
- Change `include/cachinglayer/Manager.h` so slot creation awaits required warmup. Keep unrelated synchronous reservation APIs unchanged.
- Adapt existing translators and async call sites in `test/test_cachinglayer/test_cache_slot.cpp`.

## 3. Behavioral verification

- Add a gated asynchronous translator to verify single-worker progress, concurrent-load deduplication, load failure and retry, context cancellation, slot lifetime, eviction while pinned and reload after eviction.
- Exercise foreground/background warmup and creation through Manager.
- Run the full caching-layer and common test suites. Run formatting and `git diff --check`.
- Document ownership, cancellation, synchronous adapters and downstream migration in a design document; keep loading-overhead reservation invariants unchanged.

## 4. Commit and independent review

- Create one signed-off commit for the coroutine cache conversion.
- Request an independent static production-code review of that commit.
- Address findings by amending the commit, rerun relevant checks and request review again until no actionable findings remain.
