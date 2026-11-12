# Caching Layer Design and Usage

- Last updated: 2026-11-12
- Implementation baseline: `a88c21f` on `master`
- Scope: current architecture, usage contracts, and considerations for extending the caching layer

Update this document when the corresponding implementation or contract changes. Source links point to repository files rather than fixed line numbers. The current implementation uses CacheSlot; the acquisition-interface discussion below describes extension requirements, not an additional existing API.

## 1. Components and Ownership

| Component | Responsibility |
| --- | --- |
| [Manager](../../include/cachinglayer/Manager.h) | Initializes the shared DList, creates CacheSlots, and exposes resource/configuration management. |
| [Translator](../../include/cachinglayer/Translator.h) | Maps caller IDs to cells, estimates batch loading resources, and implements synchronous/asynchronous loading. |
| [CacheSlot / CellAccessor](../../include/cachinglayer/CacheSlot.h) | Owns a fixed set of cells, coordinates loading and pinning, and provides protected access. |
| [ListNode](../../include/cachinglayer/lrucache/ListNode.h) | Stores loading state and pin count, and coordinates waiters for the same node. |
| [DList](../../include/cachinglayer/lrucache/DList.h) | Manages the shared eviction list, memory/disk budgets, loading reservations, and capacity waiters. |
| [LoadingOverhead](../../include/cachinglayer/LoadingOverhead.h) | Defines loading overhead and shared-overhead Group accounting policies. |
| [TieredStorageConfig](../../include/cachinglayer/TieredStorageConfig.h) | Provides configuration snapshots and runtime-configurable settings. |

Call `Manager::ConfigureTieredStorage()` once before creating CacheSlots. Policy-based eviction requires both the Translator's `support_eviction` and the Manager's eviction switch. Whether a Slot reserves its own loading resources depends on the eviction switch; these are separate decisions.

Each Slot owns its Translator and cell array. Cell IDs index that array, which does not resize after construction. Multiple Slots share DList, but DList does not own their cells. Cells are not registered in a global hash index.

## 2. Access and Loading

Callers obtain a `CellAccessor` through `PinCells()`, `PinAllCells()`, or synchronous `Pin*Direct()` methods. The ordinary path for evictable cells is:

1. Resolve caller IDs to cell IDs and pin the required nodes.
2. For a resident node, increment its pin count and return a NodePin. Resources transitioning from evictable to pinned are removed from the evictable total.
3. For an unloaded node, one request transitions it to `LOADING` and takes responsibility for loading. Other ordinary requests finding `LOADING` wait on its shared promise instead of starting another load of that node.
4. `RunLoad()` estimates the batch. When self-reservation is enabled and the estimate is nonzero, it reserves resources before invoking the Translator's loader.
5. Install successfully loaded cells, account for their actual resident usage through `CellByteSize()`, and complete the promises. Scope cleanup releases loading reservations; failure paths must also clean up reservations and notify waiters.
6. The caller keeps its accessor throughout data use. Once the final pin is released, the cell can become an eviction candidate again.

Both `LOADED` and `CACHED` indicate loaded data. `CACHED` relates to DList management, not the absence of readers. Evictability also depends on the pin count; the state name alone does not establish access protection.

A Translator can request additional cells through `bonus_cells_to_be_loaded()`. Reservation with bonus cells first uses a zero timeout; if admission fails, the request falls back to essential cells and re-estimates the batch. Bonus sets from different loads can overlap. Ordinary request coalescing does not imply that all sources of loading are always deduplicated.

## 3. Access Lifetime

CellAccessor holds both a `shared_ptr` to the Slot and NodePins for its cells. Pins are destroyed before the Slot reference, keeping nodes alive during unpin. Returned cell pointers do not carry independent ownership: keep the accessor alive throughout their use. Accessing a cell outside the accessor's protected set is undefined behavior.

Slots and ListNodes hold raw DList pointers. Keeping a Slot alive does not keep DList alive; its owner must outlive the relevant Slots, accesses, and asynchronous operations. Changes to shutdown ordering or handles that outlive the outer cache must account for unlinking, resource refunds, and final destruction, not just Value survival.

`TryBorrowCell()` is a special fast path. It returns a non-null pointer only after the Slot enters its permanently resident, no-pin mode; otherwise callers must use a Pin method. Borrowed pointers require the Slot to remain alive. This is not a general unprotected read path for evictable objects.

Replacing strong ownership with `weak_ptr` changes cleanup responsibilities. If a failed `lock()` causes final release to skip cleanup, shutdown must already have completed unlinking and accounting handoff. Otherwise the list can retain dangling pointers or resource charges can leak. A mutex's `try_lock()` does not establish object lifetime.

## 4. Estimates, Reservations, and Actual Usage

`ResourceUsage` tracks memory and file resources separately. `Translator::estimated_loading_usage()` estimates a complete batch using:

- `resident`: final resources retained by the cells.
- `private_overhead`: temporary peak resources reserved independently for each request.
- `shared_overhead`: temporary resources whose entire lifetime is covered by an external shared runtime limiter.

Estimates must conservatively cover the interval from successful reservation until loading completes or fails, including actual batch concurrency. Final object size alone is not a loading-peak estimate. Temporary batch peaks need not be additive, so a reduced request must be re-estimated. Accounting actual resident usage with `CellByteSize()` does not repair a memory peak that already exceeded the estimate.

Only shared overhead is eligible for Group aggregation. A Group reflects an external Budget or executor limit; it does not enforce that limit itself. Buffers retained while queued or after worker completion are not necessarily bounded by worker count. See [Loading Overhead Management and Accounting](20260723-loading-overhead-management-and-accounting.md) for policies, binding, rollback, and runtime updates.

`RunLoad()` currently loads directly when `self_reserve_` is false or all three estimates are zero. These paths do not prove that loading consumes no resources. The caller and Translator must justify the absence of a reservation rather than use a zero estimate to hide real overhead.

Important boundaries:

- Reserving after constructing a Value controls cache admission, not the allocation that already happened. Loading-peak control requires reservation before the controlled allocations.
- Same-key load coalescing avoids duplicate I/O; reservations bound resource use across requests. Neither replaces the other.
- Replacement can keep old and new values alive simultaneously. Failure must preserve the ownership contract and release reservations accurately.
- Refunding resident resources while a Pin still retains the data separates actual usage from the budget. Destroying the outer cache does not necessarily free the resources.
- Dynamic caches must state whether keys, Entries, index nodes, and bucket arrays are charged. A zero-byte payload is not a zero-memory entry.

## 5. Concurrency, Scheduling, and Eviction

Node locks protect node operations and loading state. DList's lock protects its shared list and associated reservation operations. Ordinary node paths can acquire the node lock before accessing DList. Eviction holds the DList lock and tries node locks without blocking, skipping unavailable nodes to avoid the opposite blocking lock order. Review both paths when changing either ordering.

DList selects candidates in shared-list order, but recency follows an unpin/touch-window policy rather than moving nodes on every read. Do not infer strict per-access LRU ordering. Pinned nodes cannot be evicted. Batch eviction affects other Slots sharing the domain, so time under the shared lock matters as well as throughput.

Completing a Folly promise/future can run continuations inline. Notifications and promise completion must not retain locks that those continuations can reacquire. Returning from the loader does not by itself establish that all cache locks have been released.

Asynchronous API contracts also matter:

- `PinCells()` retains the Slot and resolved IDs in its future. Loading starts when the future is awaited or driven. A borrowed `OpContext` must outlive completion.
- The default `get_cells_async()` calls synchronous `get_cells()`; it does not automatically offload blocking I/O. Overrides must await/offload appropriately and drain issued work before returning or throwing.
- `Pin*Direct()` and synchronous creation block. Do not occupy an executor needed by the load itself, which can exhaust the workers required for progress.
- Call Warmup once before ordinary pinning. Background warmup uses its own cancellation context; without a prefetch pool, it falls back to foreground execution.

## 6. Prefer CacheSlot When Integrating with LRU

Prefer `Translator + CacheSlot` for new cached data types over directly manipulating ListNode, resource counters, and the eviction list. CacheSlot already coordinates ordinary load coalescing, reservation before loading, publication, accessor lifetime protection, and failure cleanup. Reuse this path when the data maps naturally to a stable set of cells.

Dynamic keys or an entry set that cannot be determined in advance may justify bypassing CacheSlot. Explain that need explicitly. Joining DList does not automatically provide a complete cache contract: DList supplies accounting and eviction mechanisms, not caller-specific loading coordination or ownership.

### Responsibilities When Bypassing CacheSlot

| Responsibility | Required behavior | Insufficient substitute |
| --- | --- | --- |
| Same-key loading | Share one in-flight load and its result; define failure, cancellation, and retry behavior. | Loading independently and serializing Put afterward: duplicate I/O and allocations have already occurred. |
| Admission before allocation | Reserve conservative resident and temporary-peak estimates before the controlled allocations. | Constructing Value before Reserve: this controls cache admission, not loading peaks. |
| Publication and access protection | Return a handle/accessor that retains a pin on successful acquisition, with no eviction gap between publication and delivery; retain protection throughout use. | Put followed by Get: the value can be evicted between them or before Put returns. |
| Lifetime and shutdown | Keep values, nodes, and final-release dependencies alive; define unlinking, accounting handoff, and destruction order. | Keeping only Value/Entry alive, or skipping cleanup when a weak reference expires. |
| Accounting and failure cleanup | Charge actual residency, release reservations on success/failure/cancellation, and assign accounting responsibility for surviving data. | Refunding only on success, or refunding at cache destruction while Pins retain the data. |
| Index and eviction consistency | Coordinate index ownership, DList membership, node locks, and pin counts; keep nodes alive until locks are released and prevent detached entries from rejoining accidentally. | Dropping the last index reference while still using the node/lock, or updating only one structure. |
| Shared-domain concurrency | Establish lock order, allow safe notification/callback reentry, avoid loading I/O under global locks, and assess batch eviction impact on other Slots. | Holding a whole-map write lock across the loader, or treating atomic counters as an atomic multi-state update. |

These responsibilities do not require copying CacheSlot's internal structure, but they require explicit implementation and verification. An API providing only best-effort prewarming or resident-cache admission must state that narrower contract; it is not a complete on-demand acquisition interface.

### Acquisition and Access Handles

Fixed cell arrays and dynamic key caches need different indexing, but both must support obtaining and safely using data. For on-demand loading, the following conceptual flow illustrates how acquisition and lifetime protection fit together:

```text
Pin(identity) -> Handle / Accessor protecting the use lifetime
  hit: pin the resident data and return its access handle
  miss: coalesce requests -> estimate and reserve -> load -> publish and return a pinned handle
  failure: clean up intermediate state, notify waiters, and define subsequent retries
```

The flow above is a conceptual contract, not an additional existing API. Pinning may include loading nonresident data; it need not mean only incrementing a counter. Loading behavior can be supplied by a Translator or a registered object rather than passed as a callback on every acquisition. The existing `PinCells()` / `CellAccessor` interface follows this separation.

An identity or metadata handle identifies reusable data but does not necessarily keep it resident. The returned access handle retains the pin throughout use and releases it when the handle is destroyed or explicitly released, according to the API contract. Pinning prevents eviction; it does not by itself provide read/write exclusion for mutable data. Define that synchronization separately.

Define resource estimation, ownership transfer, cancellation, failure delivery, and retry behavior as part of the interface contract. Cache keys should identify the reusable data and include version information when needed. Independent loads should be able to make progress concurrently within resource limits.

Distinguish acquisition for immediate use from admission for future reuse. Acquisition must protect the returned data throughout its use; admission alone need not guarantee that data remains resident. Expose insertion, replacement, invalidation, or prewarming only when required by consumers, and document their effects on active readers and resource accounting.

Choose index structures based on expected cardinality and access patterns. Evaluate lookup, insertion, removal, and bulk cleanup costs, including metadata memory and time spent holding shared locks. Index membership and eviction order serve different purposes; changes must preserve their consistency without assuming that their traversal orders match. Use workload estimates or measurements to justify capacity settings and performance trade-offs.

## 7. Changes and Validation

When changing a public interface, update this document to explain caller usage, ownership, reservation/refund timing, and state remaining after failure, cancellation, or shutdown. Separate implemented behavior from proposals. Public virtual-function or layout changes must identify whether downstream binaries need rebuilding.

Implementation validation should cover interleavings that can break these contracts: concurrent misses for one cell/key, failure followed by retry, waiter cancellation, lookup racing eviction, handles outliving outer owners, callback reentry, and bonus loading under insufficient capacity. Validate batch peaks and exception cleanup, not just final counters returning to zero.

[SyncPoint](../../include/syncpoint/sync_point.h) supports deterministic thread ordering. `ENABLE_SYNCPOINT` is off by default, making its macros empty; enabled builds still require runtime processing to be enabled. Callbacks release only SyncPoint's internal lock, not caller-held business locks. Clean up synchronization state after tests and avoid sleep-based guesses about ordering.

Review-only work remains static: do not launch builds or tests to validate a hypothesis. Separate code facts, conditional cost estimates, and unverified consumer assumptions in the report.
