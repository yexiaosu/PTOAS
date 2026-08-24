# E384/K6 TopK `vcmp`/`vsel` scheduler fixture

This case is a semantic-equivalent VPTO runtime reproducer for
[`mouliangyu/PTOAS#508`](https://github.com/mouliangyu/PTOAS/issues/508), with
the complete compare/select semantics described by
[`mouliangyu/PTOAS#504`](https://github.com/mouliangyu/PTOAS/issues/504). The
referenced external `a5-kernel-standalone`, `pto-skills`, and
`TileKernels-Nightly` sources were not available when this fixture was created,
so this is not claimed to be a line-by-line translation of those files.

## Workload and ABI

- Shape: `N=1`, `E=384`, `K=6`.
- Scores: FP32, split into six 64-lane vectors.
- Indices: signed I32 vectors containing `0..383`.
- Inputs: `scores[384]`, `indices[384]`.
- Outputs: `winner_indices[6]`, `masked_scores[384]`.
- Tie break: larger score first; equal scores choose the smaller index.
- Feedback: each winner is replaced by `-inf` before the next TopK round.

`masked_scores` is a real semantic output. It makes the sixth-round feedback
observable and prevents dead-code elimination from removing the final six
winner-mask `vcmp`/`vsel` pairs.

The deterministic seed is 508. The injected winners are
`[0, 383, 63, 64, 192, 256]`; this covers exact-score ties, the `0/63/64/383`
boundaries, and winners in multiple 64-lane chunks.

## Issue-to-fixture mapping

| Issue #508/#504 semantic fragment | Fixture VPTO region | Expected lowered vector instructions | Predicate live range |
|---|---|---|---|
| Load 384 scores in six chunks | `score0` through `score5` | six `VLDS` | all-active mask only |
| Build lane-to-expert mapping | GM `indices=0..383`, then `index0` through `index5` | six I32 `VLDS` | all-active mask only |
| Maximum over all chunks | `max01` through `max_broadcast` | `VMAX` tree, `VCMAX`, `VDUP` | all-active mask only |
| Find equal-score candidate indices | `score_eq0/candidate0` through `score_eq5/candidate5` | six source-interleaved `VCMP, VSEL` pairs | each `score_eqN` is defined by one `VCMP` and last-used by the immediately following `VSEL` |
| Smallest matching global index | `min01` through `winner_broadcast` | `VMIN` tree, `VCMIN`, `VDUP` | all-active mask only |
| Store one winner index | predicated store to aligned `ub_winner_scratch`, reload/broadcast, then packed winner store | `PLT`, normal `VST`, `VLD`, `VDUP`, and `RPT_B32 VST` | the one-lane predicate is live from `PLT` through the scratch store; the reloaded lane is on the winner data path |
| Mask the selected winner | `winner_eq0/next0` through `winner_eq5/next5` | six source-interleaved `VCMP, VSEL` pairs | each `winner_eqN` is defined by one `VCMP` and last-used by the immediately following `VSEL` |
| Feed masked scores to the next round | six `scf.yield` values | loop-carried vector dependencies | no artificial predicate dependency |
| Preserve final feedback | six final `VSTS` plus GM writeback | six vector stores and one MTE copy | all-active mask; sixth-round compare predicates remain useful |

The source does not add fake predicates, force ordering with artificial
dependencies, or batch compares by hand. It lets the PTOAS VPTOScheduler and
the Bisheng CCE vector MI scheduler be controlled independently.

The scratch slot is `k * 8` I32 elements, so every normal predicated winner
store is 32-byte aligned. The following reload and broadcast feed both the
winner-mask phase and the packed output store; the store predicate is therefore
not a pressure-only decoration. Identical `scf.if` arms form scheduler region
boundaries around currently generic-classified `vdup`/memory helpers. The
condition is true for the complete `0 <= k < 6` loop and does not change the
runtime result.

## Validation matrix

Use `test/vpto/scripts/run_host_vpto_validation.sh` with a separate
`WORK_SPACE` for each group and `DEVICE=SIM COMPILE_ONLY=0`:

| Group | PTOAS VPTOScheduler | Bisheng vector MI scheduler |
|---|---|---|
| A: `INTERLEAVED_BASELINE` | `off` | disabled (`--enable-bisheng-vec-misched=false`) |
| B: `BISHENG_MISCHED_REPRO` | `off` | explicitly enabled (`--enable-bisheng-vec-misched`) |
| C: `ANALYZE_WITH_BISHENG_MISCHED` | `analyze` | explicitly enabled (`--enable-bisheng-vec-misched`) |
| D: `CURRENT_VPTO_SCHEDULER` | `on` with trace | disabled (`--enable-bisheng-vec-misched=false`) |

Never describe B or C as "VPTOScheduler on": the enabled scheduler in those
groups is Bisheng's downstream machine-instruction scheduler.

See `RESULTS.md` for the pre-fix A/B/C/D baseline on `04c8956fb`, including
the exact commands, provenance, instruction order, spill counts, and CA-model
measurements. See `RESULTS_LIVE_THROUGH_FIX.md` for the follow-up validation of
the live-through pressure fix from `bf7a85ca0`.
