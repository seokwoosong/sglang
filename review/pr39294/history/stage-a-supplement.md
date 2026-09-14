# Stage A supplement (original comparison.md/hash unchanged)

1. Non-vacuous temporal-state preservation: temporal_live_probe.py constructs 5
   states, with only 3 owned by cache entries. Both eager/lazy cases pass in all
   three arms. After eviction states [3,4] remain bound and both conv and nonempty
   temporal payloads match their distinct markers. Results are in
   {pr39294,pr36729,pr39294_pure}-temporal-live-cases.json. The earlier 3-state
   temporal fixture had no surviving states after whole-cache eviction; its
   temporal survival check alone was vacuous and is superseded by this supplement.
2. tri_oracle_probe.py is an explicitly limited CPU ablation: a pure sufficient
   predicate credits allowed END-pool hole compaction in projected FULL/Mamba
   frontiers, retains the current float span and its reusable holes, and evaluates
   SWA extension on its own page grid after FULL demand. Pending reuse is not
   credited; move-gated END holes are not credited. False means unknown, not
   impossible; it does not model float relocation and is not a production design.
   With this predicate and explicit boundary preparation, all 6 independent tri
   KV/state cases pass. The lazy 96-token/8-state demand-4 case retains 95 prefixes,
   recovering the prefix lost by immediate-readiness-only ablation. Results:
   pr39294-tri-end-oracle-cases.json and tri-end-oracle-trace.json.
3. No production edits or GPU runs performed. Further tri oracle completeness,
   float-relocation, pending-reuse and page-size coverage remain unproven.
