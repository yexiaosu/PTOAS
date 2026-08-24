# CA SIM validation results

Validated on 2026-08-19. This is issue #508 semantic-equivalent VPTO runtime
coverage; the unavailable external source was not reconstructed or claimed as
a line-by-line translation.

## Source discovery and fixture identity

- Public sources: `mouliangyu/PTOAS#508` and its semantic predecessor
  `mouliangyu/PTOAS#504`, including their bodies and comments.
- Repeated GitHub repository/fork searches did not locate the referenced
  `a5-kernel-standalone`, `pto-skills`, or `TileKernels-Nightly` files.
- A server search under `/home/wanglan`, including existing PTOAS worktrees,
  old SIM results, and download/check-out locations, also found no exact
  artifact. The old `NOT COVERED` report was used only as a search lead.
- The validated `kernel.pto` SHA-256 is
  `05f99f35388729def98fc40d31a03d71d3c09493063473460b35e2e5d10e44af`.
- The fixture independently implements `N=1`, `E=384`, `K=6`, six 64-lane
  FP32 score chunks, I32 indices, six complete TopK rounds, smallest-index tie
  breaking, winner masking, GM/UB movement, a real aligned one-lane winner
  store/reload, six winner outputs, and the final masked score tensor.
- Seed 508 produces winners `[0, 383, 63, 64, 192, 256]`. Strict comparison is
  bit-exact for both the I32 winners and all 384 final FP32 scores.

The detailed issue-to-source mapping is in `README.md`. The final device log
contains 72 `VCMP` and 72 `VSEL` executions: 12 of each per TopK round for all
six rounds. This also proves that the last-round mask phase was not removed by
DCE.

## Isolated build provenance

| Item | Validated value |
|---|---|
| Local worktree | `/Users/wanglan/Desktop/wanglanl/workspace/PTOAS/worktrees/vpto-sched-issue508-sim` |
| Remote worktree | `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-issue508-sim` |
| Branch / base HEAD | `codex/vpto-sched-issue508-sim` / `04c8956fb6e0d1d59113df61bb05fc9c8721d794` |
| Remote build | `/home/wanglan/PTOAS/.worktrees/ca-sim-builds/vpto-sched-issue508-sim` |
| Remote venv / Python | `<remote-worktree>/.venv` / Python 3.12.3 |
| PTOAS | `<remote-worktree>/.venv/bin/ptoas`, version 0.60 |
| Python package | `<remote-worktree>/ptodsl/ptoas/__init__.py` |
| Native extension | `<remote-worktree>/.venv/lib/python3.12/site-packages/ptoas/_core.cpython-312-x86_64-linux-gnu.so` |
| `libPTOASCompiler.so` | the same venv under `ptoas/mlir/_mlir_libs`, confirmed by `ldd` |
| `_core` RUNPATH | `$ORIGIN:$ORIGIN/mlir/_mlir_libs:/home/wanglan/llvm-workspaces/build-vpto19/lib` |
| LLVM/MLIR | `/home/wanglan/llvm-workspaces/build-vpto19`, LLVM 19.1.7; `ldd` resolved MLIR/LLVM DSOs there |
| CANN | `/usr/local/CANN/cann-9.1.0`, version 9.1.0, timestamp `20260704_130027991` |
| Bisheng | `/usr/local/CANN/cann-9.1.0/bin/bisheng`, clang 15.0.5 build `2026-07-01T09:55:44+08:00` |
| CA model | `/usr/local/CANN/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib/libruntime_camodel.so` |
| PTO-ISA | `/home/wanglan/pto-isa`, commit `311ca0c83f5571dc165681fee0a427983c555d3c` |

`quick_install.sh -v` completed in the isolated remote build and installed an
editable package into that worktree's venv. The pre-existing dirty PTO-ISA file
`tests/script/run_st.py` was not touched. No Bisheng wrapper was used.

The macOS local native build was not used for evidence: its available CMake
binary was x86_64 while the host is arm64. All compilation, device emission,
and runtime results below come from the consistent remote Linux environment.

## Commands and scheduler matrix

Every invocation used the repository runner with these common variables:

```bash
ASCEND_HOME_PATH=/usr/local/CANN/cann-9.1.0
PTOAS_BIN=<remote-worktree>/.venv/bin/ptoas
PTO_ISA_PATH=/home/wanglan/pto-isa
CASE_NAME=kernels/topk-gate-vcmp-vsel-misched
DEVICE=SIM
SIM_LIB_DIR=/usr/local/CANN/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib
COMPILE_ONLY=0
WORK_SPACE=<authoritative-v3>/<group>/run<1..3>/work
bash <remote-worktree>/test/vpto/scripts/run_host_vpto_validation.sh
```

`strace -f -e trace=execve` wrapped every run and saved the actual Bisheng
device invocation. Only `PTOAS_FLAGS` differed:

| Group | VPTOScheduler | Bisheng vector MI scheduler | `PTOAS_FLAGS` |
|---|---|---|---|
| A | off | off | `--pto-arch=a5 --pto-backend=vpto --vpto-scheduler=off --enable-bisheng-vec-misched=false` |
| B | off | enabled/toolchain default | `--pto-arch=a5 --pto-backend=vpto --vpto-scheduler=off --enable-bisheng-vec-misched` |
| C | analyze only | enabled/toolchain default | `--pto-arch=a5 --pto-backend=vpto --vpto-scheduler=analyze --enable-bisheng-vec-misched` |
| D | on with trace | off | `--pto-arch=a5 --pto-backend=vpto --vpto-scheduler=on --vpto-scheduler-trace --enable-bisheng-vec-misched=false` |

C used `COMPILE_ONLY=1`, as its final device conditions are the same as B and
its purpose is the non-mutating scheduler analysis. A and D `execve` records
contain `-mllvm --cce-aicore-vec-misched=0`. B and C records do not contain
that option. B's different final device order directly confirms that this
Bisheng build's default vector MI scheduler is active.

The illegal `--vpto-scheduler=on --enable-bisheng-vec-misched` combination
exited 1 with `VPTO scheduler on mode conflicts with
--enable-bisheng-vec-misched.` The focused
`vpto_scheduler_cli.pto` and `bisheng_vec_misched_cli.pto` lit tests passed
2/2.

## Smoke tests

Both unrelated smoke cases used `DEVICE=SIM`, `COMPILE_ONLY=0`, scheduler off,
and Bisheng misched off:

| Case | Result | Tick | Non-empty instruction log |
|---|---|---:|---:|
| `micro-op/binary-vector/vadd` | compare passed for all 9 variants | 12971 | 752 lines |
| `micro-op/vector-load-store/vlds-post-update` | compare passed | 2426 | 111 lines |

Both logs record CA model start and stop, host execution, golden generation,
and comparison.

## VPTO and VPTOScheduler evidence

In source, both six-pair phases are strictly interleaved by chunk. The C
analyze report gives these issue-bearing core regions:

| Region | Nodes / edges | Live-ins / live-outs | Known / unknown | Original-order peak |
|---|---:|---:|---:|---:|
| reductions and loop index | 9 / 7 | 10 / 11 | 9 / 0 | vector 9, predicate 1 |
| score compare/select plus index reduction | 18 / 17 | 15 / 14 | 18 / 0 | vector 19, predicate 2 |
| winner-mask compare/select | 12 / 6 | 15 / 7 | 12 / 0 | vector 14, predicate 2 |

The height-derived critical-path lower bounds are 50 logical cycles for the
18-node phase and 10 for the 12-node phase. Analyze mode emits per-node replay
pressure but no separate aggregate original replay completion total; no CA
performance is inferred from these logical cycles.

D accepted and applied schedules to all three core regions:

| Region | Work units | Pressure idles | Scheduled peak | Last logical issue cycle |
|---|---:|---:|---:|---:|
| 9 nodes | 268 | 0 | vector 9, predicate 1 | 30 |
| 18 nodes | 753 | 0 | vector 18, predicate 7 | 50 |
| 12 nodes | 479 | 0 | vector 14, predicate 7 | 10 |

Coverage was `schedulable=70 structural=3 boundary=8 unsupported=0
unclassified=0`. The two issue-bearing compare/select regions are completely
known-classified (18/18 and 12/12) and have no unknown fallback. Surrounding
load, broadcast, and aligned scratch-store helpers are separated by control
region boundaries and retain the scheduler's conservative generic-class
ordering; they do not prevent either target region from scheduling.

The trace contains three `schedule-result` records and no failure/fallback.
In the pass implementation a `schedule-result` is printed only after schedule
verification, a fresh replay, and schedule application have each succeeded.
This is therefore evidence for verifier, replay, and apply success, in addition
to the observed changed device order. Decision reasons were 21
`deterministic-tie-break`, 5 `longer-critical-path`, and 13 `only-candidate`.

The important negative result is that pressure-driven idles and
`live-range-closing` decisions were both zero. The source-interleaved predicate
peak is 2, but the D schedule batches six compares and reports a model peak of
7, exactly the calibrated predicate limit. It therefore never enters the
over-limit/near-limit path that would close a live range. The physical device
must also retain the all-active predicate, and register allocation spills the
one-lane store predicate. This fixture exposes that remaining boundary in the
current `04c8956f` implementation; no scheduler policy was changed to hide it.

## Final device order and predicate live intervals

The first TopK iteration, ordered by CA dispatch ID, is representative and is
identical across all six iterations:

- A phase 1 and phase 2 are both
  `CMP(P3), SEL(P3)` repeated six times. Each predicate definition-to-use
  interval is one dispatch ID.
- B phase 1 is four compares, four selects, then two compares and two selects.
  The first four intervals are 4 IDs and the last two are 2 IDs. B phase 2 is
  six compares followed by six selects, with intervals 6, 6, 6, 6, 7, and 8.
- D phase 1 and phase 2 are both six compares followed by six selects. Phase-1
  intervals are 6, 6, 6, 6, 7, and 7; phase-2 intervals are 6, 6, 6, 7, 7,
  and 7.

The one-lane predicate is materialized as `RV_PLT P2`, then used by the
32-byte-aligned normal scratch `RV_VST`; the reloaded low lane is broadcast
back onto the winner data path. A keeps P2 live and uses P3 for every short
compare/select pair. B and D allocate P2-P7 to the batched comparisons and
spill/reload the store predicate.

The unique `(PC, binary)` stream hashes were stable in all three runs:

| Group | SHA-256 |
|---|---|
| A | `8d7e726c4b6b6687aee914ae0b6eb9e3768e2ef110196e8ec0c84530732249bd` |
| B | `9b61707ee77131cfb79874c4ccddfad1ec73a6608b52bbb7d35b2502c044bf0e` |
| D | `c56e7ecc1531a5871b9feb075a1307320eda36eca0bfe090c83ceac9fd63c04d` |

## Static/dynamic instruction and CA results

The core0/veccore0 executed counts are deterministic across repetitions:

| Metric | A | B | D |
|---|---:|---:|---:|
| `VCMP_EQ` / `VSEL` | 72 / 72 | 72 / 72 | 72 / 72 |
| `PSET` / `PLT` / `PGE` | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 |
| `VMAX` / `VCMAX` | 30 / 6 | 30 / 6 | 30 / 6 |
| `VMIN` / `VCMIN` | 30 / 6 | 30 / 6 | 30 / 6 |
| `VLDI` / `VLD` | 12 / 6 | 12 / 6 | 12 / 6 |
| `VST` / `VSTI` | 12 / 6 | 12 / 6 | 12 / 6 |
| `PSTI` / `PLDI` | 0 / 0 | 1 / 6 | 1 / 6 |
| `SMEM_BAR` / `MEM_BAR` | 0 / 0 | 7 / 0 | 7 / 0 |
| dispatch-log PC lines | 357 | 373 | 373 |
| retired-log PC lines | 358 | 381 | 381 |
| distinct RVECEX timestamps | 159 | 149 | 148 |
| RVECEX timestamp span | 211 | 305 | 320 |

The CA package did not emit named `simd_busy`, EXIPC, dual-issue, occupancy, or
EX-active summaries for this runner. The last two rows are explicitly derived
from the first/last and distinct `RVECEX` timestamps, not presented as those
unavailable named counters.

| Group | Strict compare | Ticks (run 1 / 2 / 3) | Min / median / max |
|---|---|---:|---:|
| A | 3/3 | 2608 / 2617 / 2614 | 2608 / 2614 / 2617 |
| B | 3/3 | 2709 / 2709 / 2710 | 2709 / 2709 / 2710 |
| D | 3/3 | 2731 / 2729 / 2723 | 2723 / 2729 / 2731 |

B's median is 3.63% above A; D's median is 4.40% above A. These measurements
are consistent with B/D's extra spill, barriers, dispatch, retired count, and
longer RVECEX span. They are CA-model results, not logical scheduler-cycle
predictions.

Each group also reports the same seven CA `vec_err_idata_inf_nan_t0`
diagnostics because `-inf` is the fixture's intentional winner-mask value.
There are no alignment/assertion diagnostics in these final runs, the model
stops normally, and both output tensors pass strict comparison.

## Logs and artifacts

- Final A/B/C/D root:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-issue508-sim/sim-results/issue508/authoritative-v3`
- D compile trace:
  `<final-root>/D-probe/compile-probe.log`
- C analyze report: `<final-root>/C/compile-analyze.log`
- Per-run runtime and direct invocation evidence:
  `<final-root>/{A,B,D}/run{1,2,3}/{run.log,execve.log,work}`
- Illegal-combination diagnostic:
  `<final-root>/logs/illegal-combination.log`
- Smoke root:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-issue508-sim/sim-results/issue508/authoritative/smoke`
- Focused lit log:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-issue508-sim/sim-results/issue508/logs/focused-lit.log`

Final conclusion: 已完成 issue #508 semantic-equivalent VPTO runtime coverage，
但未取得原始 external source。The current CANN 9.1/Bisheng toolchain reproduces
the downstream batching, predicate spill, barriers, and runtime regression.
The current VPTOScheduler also batches both phases and does not trigger its
live-range-closing preference at this exact pressure boundary.
