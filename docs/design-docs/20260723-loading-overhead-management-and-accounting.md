# Design Document: Loading Overhead Management and Accounting

- **Date**: 2026-07-23
- **Status**: Under Review
- **Component**: Milvus Common Caching Layer
- **Scope**: `LoadingOverhead`, `Translator`, `CacheSlot`, and `DList`
- **Related PRs**: [zilliztech/milvus-common#110](https://github.com/zilliztech/milvus-common/pull/110)

---

## 1. Overview

### 1.1 Background

Loading a cache cell consumes two different kinds of resources:

- **loaded resource** is the final memory or file usage retained after the cell becomes resident;
- **loading overhead** is temporary memory or file usage that exists only while the load is running, such as read buffers, decode buffers, conversion buffers, or index construction scratch space.

Loaded resource must always be reserved per request because it remains resident independently of the mechanism that performs the load. Loading overhead is different. Concurrent requests are often constrained by a shared runtime resource such as a transient-memory budget or a thread pool. In that case, summing every request's conservative peak can be much larger than the maximum temporary resource that can be active at once.

For example, ten queued loads may each estimate 256 MiB of transient memory, while a four-worker executor can run only four of them concurrently. Reserving 2.5 GiB of overhead rejects useful work even though the runtime executor bounds the active peak near 1 GiB. Conversely, using a stale or undersized shared bound can admit work without reserving enough capacity in the caching layer.

This design introduces explicit loading-overhead Groups. A Group aggregates active loading demand for one resource dimension and translates that demand into the amount booked by `DList`. The runtime Budget or executor remains external to milvus-common; the Group mirrors its configured concurrency model for admission accounting.

### 1.2 Goals

- Keep final loaded resource and temporary loading overhead as separate estimates.
- Aggregate loading overhead across CacheSlots governed by the same runtime limiter.
- Support multiple independent Groups in one process.
- Keep memory and file accounting independent.
- Support fixed, passthrough, Budget-backed, and executor-backed policies.
- Allow an authoritative owner to replace a Group policy at runtime.
- Serialize Group binding, admission, rollback, release, policy replacement, and waiter retry with `DList` accounting.
- Preserve request-local passthrough behavior for dimensions that do not join a Group.
- Keep fractional `loading_resource_factor` accounting additive across different reserve and release partitions.
- Make failure and zero-byte success unambiguous.

### 1.3 Non-Goals

- Owning or enforcing the actual transient-memory Budget.
- Owning, resizing, or scheduling the actual executor/thread pool.
- Providing a transactional policy-update protocol with pending policy generations.
- Guaranteeing strict DList hard-limit consistency during a runtime policy transition.
- Sharing one Group across multiple `DList` accounting domains.
- Coordinating loading overhead across processes or QueryNodes.
- Allowing one resource dimension of one request to be split across multiple Groups.
- Replacing the existing DList eviction or waiting-queue policy.

### 1.4 Terminology

| Term | Meaning |
|---|---|
| Loaded resource | Final resource retained after loading completes. |
| Loading overhead | Temporary resource used only during the load. |
| Dimension | Memory or file/disk. The two dimensions are managed independently. |
| Group | Shared accounting state for one dimension and one runtime concurrency domain. |
| Binding | A CacheSlot's association with a Group, plus an optional per-runtime-unit bound. |
| Runtime unit | One unit of runtime concurrency, such as one executor task or one oversized Budget acquisition. |
| Policy owner | The Milvus component that owns the corresponding Budget/executor/configuration and serializes updates. |
| Active demand | Sum of conservative loading-overhead estimates for admitted requests in a Group. |
| Group target | Amount the current policy wants DList to book for the Group. |
| Booked overhead | Amount currently represented in DList for the Group. It may temporarily differ from the latest target after a policy update. |
| Request-local passthrough | Overhead for an unbound dimension; it is charged directly for every request. |

---

## 2. Motivation and Prior Model

The previous `LoadingOverheadTracker` model used string Group names, per-CacheSlot upper-bound contributions, numeric registration handles, and grow-only maximum bound selection. It solved static shared capping, but it made dynamic ownership ambiguous:

1. Group identity and lifetime were implicit in registration side effects.
2. A Group policy could not be replaced authoritatively; repeated registration only enlarged the bound.
3. An idle CacheSlot could retain a stale contribution.
4. Lazy refresh from request paths could publish stale configuration out of order.
5. String lookup and a separate Tracker lock complicated the DList admission transaction.
6. A static bound did not explain whether it represented a Budget, an executor, or intentional passthrough.

The new model separates identity, membership, policy, and request accounting:

- the policy owner asks DList to create a Group, retains the DList-issued opaque Handle, and distributes that same Handle downstream;
- Translator carries the Handle in Meta but does not register membership or define/mutate Group policy;
- `DList` serializes all Group mutations under the same lock as admission accounting;
- the Group persists independently of whether any CacheSlot is currently bound;
- runtime changes replace one immutable policy value on the existing Group.

The design intentionally chooses a smaller eventual-reconciliation protocol instead of the transactional/pending-policy design explored in PR #109. The trade-off is described explicitly in [Section 8](#8-runtime-policy-updates).

---

## 3. Architecture and Ownership

```text
Milvus policy owner
  | owns Budget / executor / dynamic configuration
  | asks DList to create a Group and receives its Handle
  | retains the Handle for policy updates
  | distributes the same Handle to Translator construction
  v
Manager
  | public creation and update entry points
  v
DList ------------------------------------------------------+
  | logical owner of DList-scoped Groups and Handles        |
  | owns admission, eviction, waiters, total_loading_size_  |
  | serializes every Group mutation with list_mtx_          |
  v                                                        |
LoadingOverheadGroup                                       |
  | owns policy, bindings, active demand, booked overhead   |
  +---------------------------------------------------------+

Translator
  | reports {loaded resource, loading overhead}
  | carries the DList-issued Group Handle in Meta
  | does not register membership itself
  v
CacheSlot
  | snapshots the Handle and binding metadata
  | Bind(handle) registers membership with DList
  | Reserve / Release always use that same Handle
  | Unbind(handle) removes membership at destruction
  +-------------------------------------------------------> DList / Group
```

### 3.1 Responsibility Split

| Component | Owns | Does not own |
|---|---|---|
| Policy owner in Milvus | Runtime Budget/executor, serialized configuration updates, and the control-plane copy of the Group Handle | Per-request DList bookkeeping or membership state |
| `Manager` | Public entry points that ask the process-wide DList to create or update a Group | Translator/CacheSlot membership lifetime |
| `Translator` | Conservative per-cell estimates and transport of the DList-issued Handle in Meta | Membership registration or runtime policy publication |
| `CacheSlot` | Stable Handle/config snapshot, DList membership lifetime, and the request Reserve/Release pair | Shared target calculation or Group creation |
| `LoadingOverheadGroup` | DList-scoped policy, binding metadata, active demand, and booked amount | Eviction, waiting, runtime acquisition, or external identity resolution |
| `DList` | Logical ownership of Groups/Handles, membership registration, admission, eviction, waiters, scaling, and `total_loading_size_` | Budget/executor lifecycle |

### 3.2 One Group, One Accounting Domain

`LoadingOverheadGroup` has no independent mutex and does not store its creating `DList`. Its state is safe because every supported mutation is executed while holding the creating DList's `list_mtx_`.

The `shared_ptr` Handle determines object lifetime, but it does not transfer logical ownership away from DList. Conceptually, the Handle is a capability issued by one DList for joining and operating on one DList-scoped Group.

Therefore, a Group handle has the following precondition:

> A Group must be created, bound, updated, reserved, and released through the same DList accounting domain.

Production currently has one `Manager`-owned DList, so this is a natural process-level contract. Reusing a handle across directly constructed DLists is unsupported and can corrupt both synchronization and bookkeeping.

### 3.3 Handle Distribution and Membership Registration

Group creation and CacheSlot membership are two different operations. The complete control and data path is:

```text
1. Policy owner
   -> Manager::CreateLoadingOverheadGroup(...)
   -> DList creates a DList-scoped Group
   <- LoadingOverheadGroupHandle

2. Policy owner
   -> retains the Handle for UpdateLoadingOverheadGroup(...)
   -> injects the same Handle into Translator::Meta::loading_overhead_config

3. Translator
   -> carries the Handle and per-runtime-unit metadata
   -> does not mutate Group membership

4. CacheSlot construction
   -> snapshots the config
   -> DList::BindLoadingOverheadGroups(config)
   -> registers this CacheSlot as a Group membership

5. CacheSlot request path
   -> DList Reserve / Release with the same snapshotted Handle

6. CacheSlot destruction
   -> DList::UnbindLoadingOverheadGroups(original_config)
   -> removes the membership
```

Holding a Handle is therefore not equivalent to being registered. The Handle identifies and authorizes access to a DList-scoped Group; `BindLoadingOverheadGroups()` creates membership, and `UnbindLoadingOverheadGroups()` ends it. In the current implementation, Translator is the Handle carrier and CacheSlot is the actual membership registrant.

---

## 4. Public Model

### 4.1 Translator Estimates

`Translator::estimated_byte_size_of_cell()` continues to return two values:

```cpp
std::pair<ResourceUsage, ResourceUsage>
estimated_byte_size_of_cell(cid_t cid) const;
```

The first value is final loaded resource. The second value is temporary loading overhead and excludes the final loaded resource.

Both estimates are admission-safety inputs and must be conservative. For a Group-managed dimension, the loading-overhead estimate must cover the request's actual transient usage from successful DList admission until the paired DList release. Underestimation can make the Group target smaller than the runtime peak and break admission safety.

### 4.2 Group Creation

A Group is explicitly created by the Policy owner through Manager/DList before any CacheSlot registers membership:

```cpp
auto group = Manager::CreateLoadingOverheadGroup(
    LoadingOverheadDimension::kMemory,
    LoadingOverheadPolicy::Budget(capacity_bytes));
```

Creation returns an opaque `std::shared_ptr<LoadingOverheadGroup>`. This DList-issued Handle is the Group identity and capability; there is no global string registry or numeric registration handle.

The Policy owner keeps one copy for later policy replacement and distributes the same Handle to the relevant Translator or CacheSlot factory. The current implementation transports it through `Translator::Meta::loading_overhead_config` and lets CacheSlot perform the actual DList membership registration.

`Manager::CreateLoadingOverheadGroup()` returns `nullptr` when the Manager's DList has not been configured. An invalid dimension also fails creation at the DList boundary.

### 4.3 Policies

The policy is immutable as a value and replaceable at the Group level.

| Policy | Bound | Intended use |
|---|---:|---|
| `Fixed(upper_bound)` | `upper_bound` | Static compatibility or explicitly configured cap. |
| `Passthrough()` | `INT64_MAX` | Named Group with no cap; active overhead is still aggregated in Group state. |
| `Budget(capacity)` | `capacity == 0 ? unlimited : max(capacity, max_runtime_unit)` | Runtime transient-memory Budget. A single oversized unit must still make progress safely. |
| `Executor(workers)` | `saturating_multiply(workers, max_runtime_unit)` | Executor whose configured concurrency bounds simultaneous transient work. |

All constructor inputs must be non-negative. `Budget(0)` means runtime throttling is disabled and therefore does not cap Group demand. `Executor(0)` produces a zero Group target; a zero-byte DList reservation can still be a successful admission result.

### 4.4 Bindings

Each resource dimension binds independently:

```cpp
struct LoadingOverheadGroupBinding {
    LoadingOverheadGroupHandle group;
    std::optional<int64_t> max_runtime_unit;
};

struct LoadingOverheadConfig {
    std::optional<LoadingOverheadGroupBinding> memory;
    std::optional<LoadingOverheadGroupBinding> file;
};
```

The configuration stored in Translator Meta is a delivery mechanism, not the registration itself. CacheSlot copies the configuration during construction and calls `DList::BindLoadingOverheadGroups()` with the Handle. That Bind call is the point at which the CacheSlot becomes a Group member and contributes its `max_runtime_unit` metadata.

`max_runtime_unit` is a conservative bound for one runtime unit from that binding. It is not the sum of an arbitrary multi-cell load request. Budget and Executor policies require every binding to provide it. Fixed and Passthrough policies do not.

The Group caches the maximum bound across all currently attached bindings, including idle CacheSlots. Duplicate values are retained in a multiset so unbinding one CacheSlot does not accidentally remove another CacheSlot's contribution.

An absent dimension does not participate in any Group and remains request-local passthrough. This differs from an explicit `Passthrough()` Group:

- absent dimension: no shared Group state; every request charges its complete overhead directly;
- explicit Passthrough Group: shared state exists and may later receive another compatible policy.

### 4.5 Policy Replacement

```cpp
auto result = Manager::UpdateLoadingOverheadGroup(
    group,
    LoadingOverheadPolicy::Executor(configured_workers));
```

The result is one of:

- `kApplied`: the Group now uses the replacement policy;
- `kIncompatiblePolicy`: a Budget/Executor policy was requested while one or more bindings lack `max_runtime_unit`;
- `kInvalidArgument`: the handle is invalid or the Manager DList is unavailable.

`kApplied` describes logical policy publication. It does not mean DList has immediately reconciled the Group's already-booked amount to the new target.

---

## 5. Accounting Model

The following model is applied independently to memory and file Groups.

### 5.1 Group State

For one Group, define:

| Symbol | Implementation field | Meaning |
|---|---|---|
| `S` | `sum_of_overhead_` | Sum of conservative overhead from currently admitted requests. |
| `R` | `max_runtime_unit_` | Maximum runtime-unit bound across attached bindings. |
| `Q` | `overhead_reserved_` | Unscaled Group overhead currently booked in DList. |
| `B(P, R)` | derived from `policy_` | Bound produced by policy `P`. |
| `T(S, P, R)` | `computeTarget()` | Desired Group target under the current policy. |

The target is:

```text
T(S, P, R) = min(max(S, 0), B(P, R))
```

The built-in policy bounds are:

```text
Fixed(U):       B = U
Passthrough:    B = INT64_MAX
Budget(C):      B = (C == 0) ? INT64_MAX : max(C, R)
Executor(W):    B = saturating_multiply(W, R)
```

`S` answers "how much conservative active demand exists?" while `Q` answers "how much of that Group is currently represented in DList?" Keeping both values is required for dynamic policy replacement and exact rollback.

### 5.2 Reserve Transition

For an admitted request with Group-managed overhead `o`:

```text
S1 = S0 + o
delta_reserve = max(T(S1, P, R) - Q0, 0)
Q1 = Q0 + delta_reserve
```

The Group returns `delta_reserve` to DList. If admission fails, rollback restores both state variables:

```text
S <- S - o
Q <- Q - delta_reserve
```

This means a request may add positive active demand while requiring zero additional Group reservation because another request has already brought the Group to its policy bound.

### 5.3 Release Transition

For the paired release:

```text
S1 = max(S0 - o, 0)
delta_release = max(Q0 - T(S1, P, R), 0)
Q1 = Q0 - delta_release
```

The release delta is determined by the Group's aggregate state, not by the delta originally charged to this particular request. Individual requests therefore do not own slices of the shared Group reservation. Correctness is aggregate:

> With paired Reserve/Release operations, total Group bytes booked and released converge to zero regardless of request completion order.

Negative state is treated as an invariant violation, logged, and clamped to zero to keep later accounting usable.

### 5.4 Example: Fixed or Budget Bound

Assume a 200-byte Group bound and three requests with overhead `150`, `100`, and `100`:

| Operation | Active sum `S` | Target `T` | Booked `Q` | DList Group delta |
|---|---:|---:|---:|---:|
| Reserve 150 | 150 | 150 | 150 | +150 |
| Reserve 100 | 250 | 200 | 200 | +50 |
| Reserve 100 | 350 | 200 | 200 | +0 |
| Release 100 | 250 | 200 | 200 | -0 |
| Release 100 | 150 | 150 | 150 | -50 |
| Release 150 | 0 | 0 | 0 | -150 |

The three request estimates total 350 bytes, but DList books at most 200 bytes for the shared overhead.

### 5.5 Combining Loaded, Passthrough, and Grouped Resource

For each request, DList separates two components:

```text
request_local = loaded_resource
              + overhead from every unbound dimension

group_change  = change in Q for every bound dimension
```

The unscaled value returned to `CacheSlot` is:

```text
reserved = loaded_resource + passthrough_overhead + group_delta
```

The result also contains an explicit `success` flag. `reserved == 0` is not a failure sentinel because a valid policy may produce a zero target.

### 5.6 Fractional Loading Factor

DList applies `loading_resource_factor` to admission bookkeeping. `ResourceUsage` scaling rounds each dimension, so scaling is not distributive:

```text
Scale(a) + Scale(b) may not equal Scale(a + b)
```

Request-local resource and Group resource are therefore scaled separately. For Group state, DList uses absolute booked endpoints:

```text
scaled_group_delta = Scale(Q_after) - Scale(Q_before)
```

It must not use:

```text
Scale(Q_after - Q_before)
```

For a factor of `1.5` and logical targets `0 -> 1 -> 2 -> 0`:

```text
scaled targets: 0 -> 2 -> 3 -> 0
correct deltas:   +2   +1   -3   (sum = 0)
```

Scaling each logical delta separately would produce `+2 +2 -3`, leaking one phantom byte. Endpoint scaling keeps the Group component of `total_loading_size_` equal to `Scale(Q)` after every transaction.

### 5.7 Metrics Are Not Admission Accounting

Two values intentionally describe different things:

- `DList::total_loading_size_` is the factor-adjusted amount used for admission and includes Group aggregation.
- `cache_loading_bytes` is the stable per-request estimate `loaded_resource + loading_overhead` for active CacheSlot loads.

The metric increments and decrements the same request-local estimate even if a Group policy changes mid-flight. It therefore measures active requested work, not the current shared Group booking, and cannot drift because policy replacement changed the DList delta.

---

## 6. Request Lifecycle

### 6.1 Normal Flow

```text
Translator estimates cells
        |
        v
CacheSlot combines essential and bonus cells
        |
        v
DList locks list_mtx_
        |
        +--> Group Reserve: add active demand, compute target delta
        |
        +--> add final loaded resource and unbound overhead
        |
        +--> scale, admit, and evict if necessary
        |
        v
successful explicit reservation result
        |
        v
runtime Budget acquisition / executor execution / load
        |
        v
runtime transient resource is released or work completes
        |
        v
DList locks list_mtx_
        |
        +--> Group Release: remove active demand, compute release delta
        |
        +--> release final request-local loading reservation
        |
        +--> reprocess waiters
```

The downstream runtime ordering must be:

```text
DList / Group Reserve
  -> runtime Budget Acquire or executor work
  -> runtime Budget Release or executor task completion
  -> DList / Group Release
```

The Group does not observe actual runtime in-flight state. Its safety depends on conservative Translator estimates and this ordering contract.

### 6.2 Essential and Bonus Cells

CacheSlot first attempts the combined essential-plus-bonus reservation with a zero timeout. If that attempt fails, Group demand is rolled back with the rest of the DList transaction, and CacheSlot retries the essential cells with the caller's real timeout.

The retry is a new admission transaction. No Group demand from the failed bonus attempt remains active.

### 6.3 Failure and Exception Cleanup

After successful admission, CacheSlot installs an RAII release guard. A load exception, cancellation, or normal completion executes the paired Group/DList release exactly once.

Before successful admission, any failed capacity check or eviction attempt invokes the Group rollback callback. A failed request may then enter the waiting queue, but it does not remain part of active Group demand while waiting.

---

## 7. Binding and Lifetime

### 7.1 Construction

CacheSlot copies `Translator::Meta::loading_overhead_config` during construction. This is the actual membership registration step: it presents the DList-issued Handle back to the same DList through `BindLoadingOverheadGroups()` before construction commits.

Binding is transactional across dimensions:

1. validate memory and file bindings before mutating either Group;
2. attach the memory binding;
3. attach the file binding;
4. if the file attachment throws, roll back the memory attachment.

CacheSlot also keeps an unbind guard until all later constructor work succeeds. A throwing constructor therefore cannot leak `binding_count_` or a runtime-unit bound.

### 7.2 Stable Metadata Snapshot

Bind and Unbind must use identical metadata. CacheSlot therefore owns the original copied `LoadingOverheadConfig` instead of reading mutable Translator Meta again during destruction.

The same principle applies to queued waiters: `WaitingRequest` owns a copy of the Group configuration, including shared handles and runtime-unit metadata. It never retains a pointer into Translator Meta.

### 7.3 Destruction

CacheSlot destruction unbinds the copied configuration. Unbinding removes one runtime-unit multiset entry, recomputes the cached maximum, and decrements the binding count.

A Group is shared state and may survive with zero bindings as long as the authoritative owner retains its handle. This permits later CacheSlots to attach to the same current policy without recreating a name-based registry entry.

The application must not destroy the final binding while requests from that binding remain admitted. A zero binding count with residual active demand or booked overhead is logged as an invariant violation; Unbind is not a substitute for request Release.

---

## 8. Runtime Policy Updates

### 8.1 Consistency Model

Policy replacement is serialized with admission under `DList::list_mtx_`, but it is intentionally not a reservation transaction.

`UpdateLoadingOverheadGroup()` performs:

```text
lock DList
  -> validate replacement against current bindings
  -> replace policy
  -> reprocess queued waiters under the new policy
unlock DList
```

It does not directly change `Q` or `total_loading_size_` for already-admitted demand. Reconciliation is lazy:

- a larger target is charged by the next successful Group-aware Reserve, including a waiter retry;
- a smaller target is released by later Group-aware Release operations.

This gives the Group three relevant values after an update:

```text
active demand S
latest policy target T(S, P_new, R)
currently booked amount Q
```

`Q` may temporarily be below or above the latest target.

### 8.2 Expansion Ordering

The authoritative owner must serialize the update and apply expansion in this order:

```text
1. Read the latest configured Budget capacity or executor worker count.
2. Update the Group policy.
3. After kApplied, increase the actual Budget or executor limit.
4. The next Group-aware Reserve reconciles any positive target gap.
```

If the reconciliation Reserve cannot fit, the request fails or waits and its prospective Group mutation is rolled back. The already-published policy remains in place.

### 8.3 Tightening Ordering

Tightening uses the opposite external ordering:

```text
1. Restrict the actual Budget or configured executor concurrency first.
2. Update the Group policy.
3. Existing admitted work drains under the runtime limiter's own semantics.
4. Later Group Releases lower Q toward the new target.
```

No pending old policy or generation is retained. The first release after tightening may release more than that request originally caused DList to book because release is aggregate.

### 8.4 Update Semantics Table

| Change | External ordering | Immediate Group effect | DList reconciliation |
|---|---|---|---|
| Expansion | Group first, runtime limiter second | New larger target is visible | Next successful Reserve or waiter retry charges the positive gap |
| Tightening | Runtime limiter first, Group second | New smaller target is visible | Later Releases drain booked overhead toward the new target |
| Policy-kind replacement | Same ordering based on whether the current target grows or shrinks | Budget and Executor may replace one another | Same lazy rules |
| Incompatible bounded policy | No runtime change | Update rejected | No accounting change |

### 8.5 Explicit Limitation of Eventual Reconciliation

The ordering above reduces the unsafe window but does not provide a strict hard-limit guarantee during transition:

- after expansion, already-admitted work blocked in the external limiter may become runnable before an unrelated new Reserve charges the larger Group target;
- after tightening, pre-update work can remain in flight while a Release reconciles DList toward the smaller new target.

This POC accepts temporary divergence to avoid rejected runtime updates, pending-policy state, old-policy generations, and coupling to Budget in-flight or executor live-worker snapshots.

Therefore:

> `kApplied` means the desired policy is published, not that all existing DList and runtime state is transactionally synchronized.

Callers that require strict transition-time admission safety need a stronger protocol, such as the transactional/pending-policy alternative described in [Section 14](#14-alternatives-considered).

---

## 9. Waiting Queue Integration

A Group-aware waiter stores:

```text
loaded resource
loading overhead
copied LoadingOverheadConfig
last computed scaled requirement
deadline and cancellation state
```

The failed initial attempt has already rolled back Group state. On every retry, DList recomputes the Group transition using the latest policy, active demand, binding set, and booked amount.

### 9.1 Requirement Changes

A policy update or another request's Group activity can change a waiter's required size while it is queued. The queue therefore cannot treat the original size key as permanent.

During processing:

- the current requirement is recomputed;
- a permanently impossible request whose policy-independent minimum exceeds DList capacity fails immediately;
- aggregate-demand overflow discovered during retry fails only that waiter and does not unwind the outer DList transition;
- a capacity-blocked request is reinserted with its refreshed requirement;
- peers with exactly the same deadline are also examined so a stale or newly larger top request cannot hide a smaller request that now fits.

Finite waiters retain the exact `steady_clock::now() + timeout` deadline. They are scanned together only when those deadlines are exactly equal. Indefinite waiters use `time_point::max()`, so they share one equal-deadline scan set. Scanning that set is an intentional correctness trade-off that prevents a stale top waiter from hiding another indefinite waiter that already fits.

### 9.2 Reprocessing Triggers

Waiters are reprocessed after events that can change either available capacity or a policy-derived requirement, including:

- Group policy replacement;
- Group unbind that lowers the cached maximum runtime-unit bound;
- loading-resource Release;
- loaded-resource refund;
- capacity or watermark changes;
- eviction progress;
- timeout or cancellation cleanup.

Retired waiter objects are destroyed after releasing `list_mtx_` so cancellation callbacks cannot deadlock during destruction.

---

## 10. Concurrency and Failure Handling

### 10.1 Serialization

All supported Group state changes use the same lock:

```text
DList::list_mtx_
```

This includes:

- Bind and Unbind;
- Reserve and rollback;
- Release;
- policy replacement;
- waiter retry.

The Group itself is deliberately lock-free. There is no nested Tracker mutex and therefore no separate cross-component lock order to maintain.

### 10.2 Validation

The implementation rejects:

- a null Group binding;
- a Group whose dimension does not match the binding dimension;
- negative policy inputs;
- a negative runtime-unit bound;
- Budget/Executor binding without a runtime-unit bound;
- Budget/Executor policy replacement while any existing binding lacks that bound;
- negative loading overhead at Group-aware Reserve;
- aggregate Group demand that cannot be represented by `int64_t`.

Executor multiplication saturates at `INT64_MAX` rather than overflowing. Group aggregate additions are validated for both dimensions before either Group is mutated, so an unrepresentable Reserve is rejected transactionally. Practical resource estimates and scaling factors are still assumed to remain representable by `ResourceUsage`; checked end-to-end `ResourceUsage` arithmetic remains a follow-up.

### 10.3 Rollback Boundaries

The following operations are atomic under `list_mtx_` with respect to Group/DList accounting:

- prospective active-demand addition and DList admission;
- failed admission rollback;
- release and waiter notification;
- two-dimensional binding validation and attachment;
- policy replacement and immediate waiter re-evaluation.

Runtime Budget acquisition and executor execution occur outside this transaction. Their correctness is established by the lifecycle and owner-ordering contracts rather than a shared lock.

---

## 11. Downstream Integration Patterns

### 11.1 Transient-Memory Budget

```cpp
auto field_load_group = Manager::CreateLoadingOverheadGroup(
    LoadingOverheadDimension::kMemory,
    LoadingOverheadPolicy::Budget(load_transient_budget_bytes));

meta.loading_overhead_config = LoadingOverheadConfig{
    LoadingOverheadGroupBinding{
        field_load_group,
        max_transient_bytes_of_one_runtime_unit,
    },
    std::nullopt,
};
```

If the configured Budget is smaller than one legal acquisition, `max(capacity, max_runtime_unit)` keeps enough DList reservation for that oversized unit. If capacity is zero, the runtime Budget is disabled and the Group becomes uncapped.

### 11.2 Executor-Bounded Loading

```cpp
auto executor_group = Manager::CreateLoadingOverheadGroup(
    LoadingOverheadDimension::kMemory,
    LoadingOverheadPolicy::Executor(configured_workers));
```

The owner must ensure the supplied runtime-unit bound matches what one worker can hold concurrently. If high- and low-priority pools have independent concurrency, they should use independent Groups. If they jointly share one effective worker domain, they should share one Group and one authoritative worker-count policy.

### 11.3 Independent Memory and File Policies

Translator Meta can carry only the memory Group Handle while leaving file overhead request-local; CacheSlot then registers the memory membership from that configuration:

```cpp
LoadingOverheadConfig{
    LoadingOverheadGroupBinding{memory_group, max_memory_unit},
    std::nullopt,
};
```

The unbound file estimate still participates fully in DList admission. A memory cap must never accidentally make disk overhead disappear.

CacheSlot can also register memory and file with separate Groups and separate policy kinds. A Handle created for memory cannot be used for a file-dimension Bind.

### 11.4 Fixed and Explicit Passthrough Groups

`Fixed` is useful for a static upper bound or staged migration from the legacy tracker. `Passthrough` is useful when the owner wants stable explicit Group identity today and may switch policies later.

When no shared ownership or later replacement is needed, omitting the dimension is simpler than creating a Passthrough Group.

---

## 12. Correctness Invariants

Within one supported DList domain, the implementation maintains the following invariants:

1. **Dimension isolation**
   - memory and file Group state are independent;
   - an absent dimension remains request-local.

2. **Active-demand pairing**
   - every successful Group-aware Reserve adds its overhead exactly once;
   - every successful admission receives one paired Release;
   - failed admission leaves no active demand behind.

3. **Booked-state fidelity**
   - `Q` is the unscaled Group amount currently represented in DList;
   - the scaled Group component of `total_loading_size_` is `Scale(Q)`.

4. **Aggregate release**
   - reservation belongs to the Group, not to individual requests;
   - completion order does not change the eventual zero balance.

5. **Binding compatibility**
   - a bounded Budget/Executor policy never operates with a binding whose runtime-unit bound is unknown.

6. **Binding snapshot stability**
   - Bind, queued retries, and Unbind use copied metadata, not later mutations to Translator Meta.

7. **Serialized mutation**
   - supported Group state is mutated only while holding its DList accounting lock.

8. **Stable request metric**
   - `cache_loading_bytes` decrements exactly the value it incremented, independent of policy changes.

9. **Handle and membership separation**
   - a Handle is issued by and scoped to one DList;
   - the Policy owner may retain and distribute the Handle without creating membership;
   - membership begins at CacheSlot Bind and ends at CacheSlot Unbind;
   - all Reserve/Release operations use the same snapshotted Handle as that membership.

Policy replacement intentionally does not maintain `Q == T` at every instant. The invariant is instead that `Q` remains the amount actually booked and converges through later Reserve/Release activity.

---

## 13. Test Strategy

### 13.1 Policy and Group Tests

- negative Fixed, Budget, and Executor inputs are rejected;
- Fixed capping and aggregate release;
- independent Groups and independent dimensions;
- Budget/Executor runtime-unit requirements;
- Budget zero as uncapped behavior;
- executor multiplication saturation;
- aggregate-demand overflow rejection before either dimension mutates;
- maximum runtime-unit tracking across bind and unbind;
- Group lifetime across zero bindings;
- policy kind replacement and incompatibility checks;
- expansion and tightening lazy reconciliation;
- concurrent Reserve/Release balance.

### 13.2 DList Tests

- failed Reserve rolls back Group demand;
- expansion reconciliation failure restores the previous booked state;
- Group and request-local dimensions are scaled separately;
- fractional factor transitions remain additive;
- policy updates reprocess waiters;
- policy-dependent requirements above capacity remain queued when their policy-independent minimum can fit;
- lowering a Group bound during Unbind wakes waiters;
- growing waiter requirements are reheaped;
- shrinking non-top indefinite waiters with the same deadline are reconsidered;
- aggregate overflow during retry fails one waiter without losing deferred peers;
- binding changes serialize with admission transactions;
- zero-byte success remains distinguishable from failure.

### 13.3 CacheSlot Tests

- end-to-end Group integration;
- essential-plus-bonus fallback cleanup;
- load exception cleanup;
- invalid binding does not leak slot/cell metrics;
- constructor failure does not leak a Group binding;
- destruction uses the original binding snapshot;
- request metrics return to baseline after mid-flight policy replacement;
- unbound file overhead remains part of admission.

### 13.4 Verification Expectations

The focused unit tests should be run under both the direct `cachinglayer_test` target and the aggregate `all_tests` target. Concurrency and waiter tests should be repeated under sanitizers when the surrounding build supports them.

---

## 14. Alternatives Considered

### 14.1 Reserve Every Request's Full Overhead

This is simple and conservatively safe, but it ignores runtime concurrency limits and can reject or evict far more aggressively than actual transient usage requires. It remains the behavior for unbound dimensions.

### 14.2 Legacy String Tracker with Grow-Only Bounds

String registration is convenient for static callers, but it couples identity to CacheSlot lifetime and cannot authoritatively decrease or replace policy. Taking the maximum registration contribution also lets stale idle registrations pin the bound.

Explicit shared Group handles make ownership and policy publication unambiguous and remove hot-path registry handles.

### 14.3 Transactional Expansion and Deferred Tightening

PR #109 explored a stronger design:

- policy expansion first reserves the full positive DList target delta and rejects publication on failure;
- policy tightening retains an effective old policy plus a pending new policy until active demand safely converges;
- previous and current absolute targets are carried through every transaction;
- runtime update ordering produces a strict transition-time accounting guarantee.

That design avoids the temporary divergence described in Section 8, but requires substantially more state and protocol:

- effective and pending policy generations;
- transactional policy commit/rollback;
- crossing-policy target handling;
- update rejection and retry semantics;
- more complex waiter and metric interactions.

The current POC chooses eventual reconciliation to keep the public API and Group state small. The transactional design remains the recommended evolution if production requirements demand strict hard-limit safety across dynamic updates.

### 14.4 Group Owns the Runtime Budget or Executor

Combining admission accounting with runtime ownership would eliminate part of the external ordering contract, but it would move Milvus configuration, scheduling, and subsystem lifetime into milvus-common. The current design keeps those concerns separate and uses an explicit owner contract.

### 14.5 Lazy Policy Refresh in `CacheSlot::RunLoad()`

Request-path refresh leaves idle CacheSlots stale and allows concurrent requests to publish snapshots out of order. Policy replacement must come from one authoritative serialized owner, not from arbitrary load requests.

---

## 15. Known Limitations and Follow-Ups

### 15.1 Current Limitations

- Dynamic policy updates are eventually reconciled and can temporarily diverge from runtime in-flight usage.
- Group ownership by one DList is a documented precondition, not encoded in the handle.
- The maximum runtime-unit bound includes idle attached CacheSlots, which may over-reserve conservatively.
- Group internals are opaque and currently lack dedicated diagnostic metrics.
- `ResourceUsage` addition and floating-point scaling are not fully checked for extreme `int64_t` values.
- One request can use at most one Group per resource dimension.
- Accounting is process-local.

### 15.2 Follow-Ups

- Add Group diagnostics for policy kind, active sum, booked amount, current target, reconciliation gap, binding count, and maximum runtime unit.
- Encode the owning DList/accounting-domain identity in the Group handle or validate it explicitly.
- Add checked `ResourceUsage` arithmetic and checked factor scaling.
- Decide whether production needs transactional expansion and deferred tightening.
- Add an allocation-free fast path if runtime-unit multiset binding churn becomes measurable.
- Add request-level multi-Group demand only if a real load must be governed by more than one independent limiter in the same dimension.

---

## 16. Migration and Rollout

1. Configure the Manager/DList before creating Groups.
2. Identify each independent runtime limiter and resource dimension.
3. Ask DList to create one Group Handle per limiter/dimension and retain the control-plane copy in the authoritative owner.
4. Distribute that same Handle to the relevant Translator/CacheSlot construction path; do not synthesize or look up another identity downstream.
5. Choose the initial policy from the latest serialized Budget or executor configuration.
6. Update Translators to return separate final-resource and temporary-overhead estimates and to carry the supplied Handle in Meta.
7. Let CacheSlot register membership with `BindLoadingOverheadGroups()`, using conservative `max_runtime_unit` values for Budget/Executor policies.
8. Use the same snapshotted Handle for every Reserve/Release and Unbind it at CacheSlot destruction.
9. Apply runtime updates through one serialized owner using the expansion/tightening order in Section 8.
10. Monitor request load metrics and, once added, Group reconciliation-gap metrics.
11. If strict transition-time consistency becomes required, migrate the update protocol before relying on dynamic expansion in hard-limit scenarios.

---

## References

- [PR #110: Dynamic loading-overhead Groups](https://github.com/zilliztech/milvus-common/pull/110)
- [PR #109: Transactional loading-overhead policies](https://github.com/zilliztech/milvus-common/pull/109)
- [Milvus integration PR #51403](https://github.com/milvus-io/milvus/pull/51403)
- [`LoadingOverhead.h`](../../include/cachinglayer/LoadingOverhead.h)
- [`LoadingOverhead.cpp`](../../src/cachinglayer/LoadingOverhead.cpp)
- [`Manager.h`](../../include/cachinglayer/Manager.h)
- [`Translator.h`](../../include/cachinglayer/Translator.h)
- [`CacheSlot.h`](../../include/cachinglayer/CacheSlot.h)
- [`DList.h`](../../include/cachinglayer/lrucache/DList.h)
- [`DList.cpp`](../../src/cachinglayer/lrucache/DList.cpp)
