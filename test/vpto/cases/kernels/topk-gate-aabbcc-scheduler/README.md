# topk-gate-aabbcc-scheduler

This VPTO runtime fixture reproduces the single-group compute graph from
`mouliangyu/PTOAS#574` at its reported shape: `N=4`, `E=384`, `K=9`, and
`token_tile=4`. Each vecscope starts in ABCABC source order for a two-token
pair; the VPTO scheduler can derive an AABBCC-style latency-hiding order from
the dependency graph. Scheduler OFF and ON use identical source, data, golden,
and downstream Bisheng mischeduler settings. Because two tokens already share
one vecscope, Scheduler OFF is a pair-fused ABC control, not the original CCE
single-token ABCABC boundary.

## CCE to VPTO mapping

| CCE source fragment | Stage | VPTO dataflow |
|---|---|---|
| `vlds(scores[i])`, `vci(i*64)`, padding `vcmp_lt`/`vsel` | setup | Six `pto.vlds` score chunks per token, six shared `pto.vci` index vectors, then `pto.vcmp "lt"` and `pto.vsel` |
| `vdup(acc, -1e30)`, six `vmax` | A | `pto.vdup` followed by a six-node `pto.vmax` RAW chain |
| `vcmax(acc)`, `vdup(POS_LOWEST)` | B | `pto.vcmax` followed by lane-broadcast `pto.vdup {position = "LOWEST"}` |
| six max-match `vcmp_eq`/`vsel`, `vdup(INT_MAX)`, six `vmin`, `vcmin`, broadcast | C | Six `pto.vcmp "eq"`/`pto.vsel` candidates, a six-node `pto.vmin` chain, `pto.vcmin`, and lane broadcast |
| `vsts(..., ONEPT_B32)`, six winner `vcmp_eq`/`vsel` | D | One-point `pto.vsts`, then six in-register winner masks carried into the next rank |
| four staged `TLOAD`, MTE2-to-V wait, two token pairs, V-to-MTE3 wait, four `TSTORE` | outer wave | Four `pto.mte_gm_ub`, two independent pair vecscopes, four 36-byte `pto.mte_ub_gm`, and the same flag/wait/barrier structure |

For each rank the source order is `A0 B0 C0 D0 A1 B1 C1 D1` (the ABCABC
baseline). The scheduler receives only the real SSA and memory dependencies;
there are no synthetic ordering edges. Its intended target order is
`A0 A1 B0 B1 C0 C1 D0 D1` (AABBCCDD).

The deterministic inputs match the CCE artifact: descending scores, modulo-4
ties, one dominant expert, and an all-equal row. `compare.py` requires exact
equality for all 36 signed 32-bit winner indices.

## Explicit AABBCC control

The sibling case `topk-gate-aabbcc-explicit` keeps the scheduler off and
expresses the CCE pair schedule directly in VPTO source. It shares this case's
shape, input generation, ABI, golden, MTE wave, and Bisheng MISCHED setting.
Use the two cases together to distinguish the quality of AABBCC itself from
the fine-grained order selected by the VPTO scheduler.

The sibling case `topk-gate-abc-single-token` is the fair control for the
original CCE ABCABC mode. It uses four independent vecscopes, one per token,
so neither the compiler nor the CA issue window can overlap vector
instructions across tokens.
