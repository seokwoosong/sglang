# Tri plan addendum — phase accounting and gate finding

The revised `tri-bounded-v2-experimental-cases.json` has all 8 cases PASS: both eager/lazy 96/8 retain95; 80/24 allocates; temporal cases retain96; strict outside KV/states survive. There is no production tri delta yet.

Two refinements requested within TRI_PLAN review:

1. Capture token walk EvictResult and subtract its `mamba_num_evicted` from the total Mamba demand-derived quota before starting the state phase. This explicitly bounds state reclaim across both phases and avoids resetting cascade credit. Each phase uses its own local tracker and remaining quota once; no duplicate victim accounting.
2. The existing FLOAT movement ignores `disagg_move_gate`: `tri_gate_probe.py` exercises legal lazy layouts, installs false gates on all three members, and counts actual fake-KV copy boundary calls through the real recovery path. Three cases in `tri-gate-cases.json` move FLOAT despite closed gates; third performs one move in ensure and a second in actual alloc. This is baseline behavior, not a candidate regression, but the planned fallback must not silently rely on it.

Request approval to guard FLOAT `make_room` and `compact_holes` at their own movement boundary using the already initialized explicit disagg gate, returning existing current-gap / zero-moves values on closed gate before settling or moving. These two are movement owners used by token/state recovery; caller-only checks would leave final alloc and state recovery able to bypass the gate. No event-order or copy implementation rewrite; apply no new host_transfer capability or generic gate abstraction. Read actual entry points and relevant callsites first; cover both token and state closed→open and immediate hole reuse without movement. If this delta should be separate, advise; retain its independent patch/test grouping and report it as baseline defect discovered during tri validation.

The geometry v4 success figures did not assert move-gate compliance, so gated successes do not establish that property. Preserve that limitation explicitly. END certificate execution still avoids FLOAT; it will be tested with movement methods forbidden as well as payload assertions.
