---
name: sglang-interface-style
description: Review or implement SGLang Python changes that cross hardware backends, cache pools, allocators, or wrappers using explicit fail-fast interface contracts. Use when shared code is tempted to bridge required fields with getattr/hasattr fallbacks, when similar capacity methods have different semantics, or when a new API must be routed through multiple SGLang implementations.
---

# SGLang Interface Style

Keep required interfaces explicit. Do not hide a missing required field or method with a defensive `getattr`, `hasattr`, or semantically different fallback merely to accommodate one backend.

## Contract-first changes

Before editing shared code:

1. Identify the semantic contract, not only the return type. For allocator code, distinguish values such as immediately available slots, schedulable capacity after compaction, allocation shortfall, and an absolute eviction quota.
2. Inspect existing direct callers and every implementation or wrapper of the interface.
3. Decide whether the capability is required or genuinely optional.

For a required capability:

- Access it directly so typos, renames, and incomplete implementations fail fast.
- Make each backend satisfy the same interface at its adapter or construction boundary.
- An identity implementation is appropriate when a backend has simpler semantics, as long as the equivalence is real and documented by the method name and tests.
- Prefer a stable alias or adapter over backend checks in shared logic.

For a genuinely optional capability, use an explicit typed branch, capability object, or documented `None` contract. Do not silently substitute another method just because its shape is compatible.

Avoid this for a required interface:

```python
allocator = getattr(req_pool, "mamba_allocator", req_pool.mamba_pool)
available = getattr(
    allocator, "schedulable_available_size", allocator.available_size
)()
```

Prefer a uniform backend contract and direct use:

```python
available = req_pool.mamba_allocator.schedulable_available_size()
```

The alternate backend should expose `mamba_allocator` and implement `schedulable_available_size()` with its correct semantics rather than relying on the shared caller to guess.

## Preserve semantic units

When introducing or routing an API, name and pass the quantity its contract expects. In particular, do not pass an absolute allocation size to an API that expects a shortfall:

```python
shortfall = max(0, required_size - schedulable_available_size)
cache.evict_for_alloc(EvictParams(mamba_num=shortfall))
```

When an upstream callback receives an absolute requested size but the downstream API expects a shortfall, convert the value once at that boundary using the allocator's semantically correct schedulable-capacity view. Merely replacing `evict()` with `evict_for_alloc()` while forwarding the same absolute size preserves the bug under a new method name.

After allocation-aware recovery, determine success from the allocator's current capacity or the allocation retry, not from a component-local eviction counter. Peer compaction and collateral frees can make capacity gain differ from the number evicted from the triggering component.

Trace the complete failure-and-retry path and audit all related call sites, including hardware backends, wrappers, load-back paths, disaggregated paths, primary pools, and hybrid extra-pool callbacks. A correct primary path does not compensate for an unchanged auxiliary path that can undo its behavior.

## Comments and tests

- Comments should record non-obvious constraints, invariants, or why two similar-looking values are not interchangeable. Do not narrate the next loop, restate an assertion, or preserve PR history.
- Prefer an expressive test name and exact assertions over a comment explaining mock mechanics. If a test docstring is still useful, state the failure mode or contract being protected, not how the mock loop works or why a particular PR introduced it.
- Test the shared contract on the primary and alternate backend when both exist.
- Pin the semantic value passed across a boundary, such as `required=10` and `available=8` producing a shortfall of `2`.
- Test that sufficient capacity skips eviction or other recovery work.
- Let missing required interface members fail naturally; do not add fallback tests that normalize structural mistakes.

## Verification

Search the full affected area for old call patterns and defensive fallbacks. Run focused unit tests for each implementation and wrapper, then run formatting and lint checks on all changed files. Keep fixes for independent review findings in separate commits when the user requests review-ready history.
