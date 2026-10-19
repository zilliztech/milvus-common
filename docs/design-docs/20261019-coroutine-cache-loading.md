# Coroutine cache loading

## Contract

`CacheSlot::PinCells`, `PinAllCells`, `Warmup`, `Manager::CreateCacheSlot` and
`Translator::get_cells` return `folly::coro::Task`. Calling these functions does
not start I/O; callers await or schedule the returned task. A cache hit needs no
executor hop. The caller chooses the executor; the cache does not create a new
load pool.

Pin tasks own their requested IDs and retain the slot until completion. The
returned `CellAccessor` retains the slot and its pins. `OpContext` remains
borrowed and must outlive the operation. The DList accounting domain must outlive
all its slots, as before.

`Translator::get_cells` borrows the context and requested IDs until its task
completes. Implementations must finish or drain issued work before returning or
throwing. Synchronous decoding can run directly inside the coroutine; blocking
I/O requires the storage implementation's asynchronous API or an explicit
blocking executor chosen by the caller.

## Pin and load sequence

1. Map and deduplicate IDs, then pin resident cells and claim unloaded cells.
2. Reuse the existing ListNode shared promises when another caller owns a load.
3. Await DList admission for the claimed cells, including the existing
   best-effort bonus-cell reservation and essential-only retry.
4. Await the translator, publish cells and notify shared waiters.
5. Release the loading reservation exactly once, including on failure.
6. Await outstanding pins and return the accessor, or propagate the load error.

No ordinary mutex is retained across a suspension. DList reservation futures
and ListNode notification futures remain implementation primitives; the load
orchestration itself uses coroutines. Failed loads reset their claimed cells and
notify all waiters so a later pin can retry. Pins prevent eviction while an
operation is suspended or an accessor remains live.

The caller that claims cells publishes load errors, including failures to
allocate the loading coroutine or its synchronous adapter before execution.

Resource waiting retains the existing `OpContext` cancellation and timeout
contract. A caller waiting on another caller's load does not take ownership of
that load's cancellation. Already-issued work must finish before its resources
and borrowed state are released.

## Warmup

Synchronous warmup means slot creation awaits completion, without blocking the
executor. Background warmup schedules a coroutine on the configured prefetch
executor. It retains the previous weak ownership before starting and strong
ownership during loading, and uses its own context with the slot's warmup
cancellation token. Missing prefetch executors still fall back to awaited
foreground warmup. Disabled warmup performs no load.

The `skip_pin` optimization is published only after successful warmup of a
non-evictable slot.

Prefetch worker count bounds running CPU work, not suspended loads. Downstream
storage admission must bound concurrent I/O and its temporary resources. Any
loading-overhead Group bound derived from the old worker count must be updated
to reflect that admission limit before migrating the caller.

## Synchronous boundaries and migration

The explicitly synchronous `PinCellsDirect` and `PinOneCellDirect` adapters
remain available. They may block and must not run on an executor needed by the
load they wait for. Cache orchestration and warmup do not use these adapters.

This changes the C++ source interface. Downstream translators must change their
override to return a task and propagate asynchronous storage operations with
`co_await`. Callers must await pinning, slot creation and warmup, or use a
`blockingWait` at an outer synchronous boundary. Do not update the Milvus
dependency until its overrides and callers have been migrated. No storage
format or protobuf changes are involved.

## Verification

Exercise resource waits and delayed translator completion on a single-worker
executor, concurrent pins of the same cell, failure and retry, cancellation,
slot lifetime, eviction, and both warmup modes. Existing loading-overhead,
pinning and shard-accounting regressions must continue to pass.

This change makes no measured claim about coroutine frame sizes, allocations,
throughput or latency.
