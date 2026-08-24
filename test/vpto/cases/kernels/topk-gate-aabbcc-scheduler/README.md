# topk-gate-aabbcc-scheduler

This VPTO runtime fixture reproduces the single-group compute graph from
`mouliangyu/PTOAS#574` at its reported shape: `N=4`, `E=384`, `K=9`, and
`token_tile=4`. Each vecscope contains two independent token chains in
descriptive ABCABC source order. Scheduler OFF and ON use identical source,
data, golden, and downstream Bisheng mischeduler settings. Because both tokens
already share one vecscope and hardware issue window, Scheduler OFF is a
pair-fused ABC control, not the original CCE single-token ABCABC boundary.

## CCE to VPTO mapping

| CCE source fragment | Stage | VPTO dataflow |
|---|---|---|
| `vlds(scores[i])`, `vci(i*64)`, padding `vcmp_lt`/`vsel` | setup | Six `pto.vlds` score chunks per token, six shared `pto.vci` index vectors, then `pto.vcmp "lt"` and `pto.vsel` |
| `vdup(acc, -1e30)`, six `vmax` | A | `pto.vdup` followed by a six-node `pto.vmax` RAW chain |
| `vcmax(acc)`, `vdup(POS_LOWEST)` | B | `pto.vcmax` followed by lane-broadcast `pto.vdup {position = "LOWEST"}` |
| six max-match `vcmp_eq`/`vsel`, `vdup(INT_MAX)`, six `vmin`, `vcmin`, broadcast | C | Six `pto.vcmp "eq"`/`pto.vsel` candidates, a six-node `pto.vmin` chain, `pto.vcmin`, and lane broadcast |
| `vsts(..., ONEPT_B32)`, six winner `vcmp_eq`/`vsel` | D | One-point `pto.vsts`, then six in-register winner masks carried into the next rank |
| four staged `TLOAD`, MTE2-to-V wait, two token pairs, V-to-MTE3 wait, four `TSTORE` | outer wave | Four `pto.mte_gm_ub`, two independent pair vecscopes, four 36-byte `pto.mte_ub_gm`, and the same flag/wait/barrier structure |

For each rank the source order is `A0 B0 C0 D0 A1 B1 C1 D1`. ABCABC and
AABBCCDD are descriptive macro labels, not required scheduler output shapes.
The scheduler receives only the real SSA and memory dependencies; there are no
synthetic ordering edges or opcode-specific TopK rules. Stage two schedules
each vecscope independently and does not fuse chains across vecscope
boundaries.

The algorithmic goal is pressure-aware list scheduling over the independent
chains already exposed inside one vecscope. At low pressure it advances the
critical path to retain ready work for hardware OOO latency hiding. When one
candidate can consume the remaining headroom of a bounded pressure set, it
prefers a live-range-closing candidate inside a one-modeled-latency critical
path window. A candidate outside that window remains urgent and wins. If all
available candidates exceed a pressure limit and a dependency event is
pending, the scheduler advances to that event and fresh replay verifies the
idle. The policy is expressed entirely in critical-path and SSA pressure
properties, independent of operation names.

The Predicate limit is 7 because the validated backend allocation for this
case uses P1-P7, including the function-wide active mask. A textual modeled
peak alone is not proof of spill freedom: final device instructions and the CA
dynamic log must both contain zero predicate PLDI/PSTI and related barriers.

The deterministic inputs match the CCE artifact: descending scores, modulo-4
ties, one dominant expert, and an all-equal row. `compare.py` requires exact
equality for all 36 signed 32-bit winner indices.

## Explicit AABBCC control

The sibling case `topk-gate-aabbcc-explicit` keeps the scheduler off and
expresses the CCE pair schedule directly in VPTO source. It shares this case's
shape, input generation, ABI, golden, MTE wave, and Bisheng MISCHED setting.
Use the two cases together to diagnose fine-grained producer/consumer live
ranges. The explicit order is a no-spill control, not an order the scheduler is
required to imitate.

The sibling case `topk-gate-abc-single-token` is the fair control for the
original CCE ABCABC mode and an external performance baseline, not an input
that scheduler stage two is expected to transform into AABBCC. It uses four
independent vecscopes, one per token, so neither the compiler nor the CA issue
window can overlap vector instructions across tokens.
