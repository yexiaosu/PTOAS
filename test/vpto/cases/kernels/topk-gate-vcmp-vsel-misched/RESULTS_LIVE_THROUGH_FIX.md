# Live-through pressure fix CA SIM results

Validated on 2026-08-20. This report records the follow-up validation of the
VPTO scheduler live-through pressure fix for issue #508. It does not replace
`RESULTS.md`:

- `RESULTS.md` is the pre-fix A/B/C/D baseline collected from
  `04c8956fb6e0d1d59113df61bb05fc9c8721d794`.
- This document covers the algorithm in
  `bf7a85ca00e89c43bf0ca28b8887e13e00b6b277` with the runtime fixture from
  `cfdec034298dbbd3353a81f5ffde331e36affd07`.
- The integration commit titled
  `docs(vpto): record issue 508 live-through fix SIM results` only preserves
  this report and its README link. It is not the tested algorithm revision.

## Revision identity and isolated provenance

The remote result branch did not receive a new commit. Its collection
worktree remained at Git HEAD `04c8956fb6e0d1d59113df61bb05fc9c8721d794`
with the live-through change and fixture present as working-tree files. This
is not represented here as a committed `bf7a85ca0` checkout. Instead, the
seven algorithm/test blobs were compared with `bf7a85ca0` and the seven
fixture blobs were compared with `cfdec0342`; all 14 Git blob IDs matched
exactly.

| Item | Validated value |
|---|---|
| Remote worktree | `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-live-through-fix-sim` |
| Collection worktree Git HEAD | `04c8956fb6e0d1d59113df61bb05fc9c8721d794`, with blob-identical `bf7a85ca0` and `cfdec0342` working-tree content |
| Isolated build | `/home/wanglan/PTOAS/.worktrees/ca-sim-builds/vpto-sched-live-through-fix-sim` |
| Isolated venv / Python | `<remote-worktree>/.venv` / Python 3.12 |
| PTOAS wrapper | `<remote-worktree>/.venv/bin/ptoas`, version 0.60 |
| PTOAS Python package | `<remote-worktree>/ptodsl/ptoas/__init__.py` |
| Native extension | `<remote-worktree>/.venv/lib/python3.12/site-packages/ptoas/_core.cpython-312-x86_64-linux-gnu.so` |
| `libPTOASCompiler.so` | the same venv under `ptoas/mlir/_mlir_libs`, resolved there by `ldd` |
| `_core` RUNPATH | `$ORIGIN:$ORIGIN/mlir/_mlir_libs:/home/wanglan/llvm-workspaces/build-vpto19/lib` |
| LLVM/MLIR | `/home/wanglan/llvm-workspaces/build-vpto19`, LLVM 19.1.7 |
| CANN | `/usr/local/CANN/cann-9.1.0`, version 9.1.0 |
| Bisheng | `/usr/local/CANN/cann-9.1.0/bin/bisheng`, clang 15.0.5 build `2026-07-01T09:55:44+08:00` |
| CA model | `/usr/local/CANN/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib/libruntime_camodel.so` |
| PTO-ISA | `/home/wanglan/pto-isa`, commit `311ca0c83f5571dc165681fee0a427983c555d3c` |

The runs used the repository entry point
`test/vpto/scripts/run_host_vpto_validation.sh` with `DEVICE=SIM`,
`COMPILE_ONLY=0`, and a unique workspace per run. The logs show real CA-model
initialization and shutdown, generated host/golden artifacts, kernel
instruction activity, and strict comparison. No SIM run was performed from
the later integration branch that stores this report.

## Pre-fix reference

The authoritative pre-fix root is:

```text
/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/
vpto-sched-issue508-sim/sim-results/issue508/authoritative-v3
```

The detailed A/B/C/D setup and interpretation remain in `RESULTS.md`. The
runtime measurements relevant to this follow-up are:

| Group | Configuration | Ticks | Median | Dynamic PSTI / PLDI |
|---|---|---:|---:|---:|
| A | VPTO scheduler off, Bisheng misched off | 2608 / 2617 / 2614 | 2614 | 0 / 0 |
| B | VPTO scheduler off, Bisheng misched on | 2709 / 2709 / 2710 | 2709 | 1 / 6 |
| D | pre-fix VPTO scheduler on, Bisheng misched off | 2731 / 2729 / 2723 | 2729 | 1 / 6 |

The pre-fix D trace batches all six compares before their selects. Its two
issue-bearing compare/select regions reach modeled Predicate pressure 7 but
do not pressure-idle, because live-through values were not included in the
region's initial pressure. The final device code spills the one-lane store
predicate.

## Post-fix scheduler trace

The fix accounts for pure live-through values in region liveness and includes
them in DAG live-in/live-out pressure. The first post-fix run records:

| Region | Nodes | Pressure-driven idles | Modeled Predicate peak |
|---|---:|---:|---:|
| Score compare/select plus index reduction | 18 | 1 | 7 |
| Winner-mask compare/select | 12 | 1 | 7 |

In both regions the scheduler issues five compares, reaches the pressure
boundary, and schedules a live-range-closing select before issuing the sixth
compare. The 18-node region then releases more compare predicates with
selects before continuing the reduction. Schedule-result records are emitted
only after schedule verification, fresh replay, and application succeed, so
these traces also cover all three checks.

## Post-fix device code and runtime

The authoritative post-fix root is:

```text
/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/
vpto-sched-live-through-fix-sim/sim-results
```

| Metric | Post-fix D |
|---|---:|
| Tick runs | 2606 / 2595 / 2598 |
| Tick min / median / max | 2595 / 2598 / 2606 |
| Strict exact compare | 3/3 pass |
| Dynamic PSTI / PLDI | 0 / 0 |
| Dynamic predicate/memory barriers | 0 |
| Dispatch PC records | 357 |
| Dynamic VCMP / VSEL | 72 / 72 |

The representative post-fix instruction log has the same 357 PC records and
358 total lines as A. Sorting the decoded opcode names gives the same complete
opcode-multiset SHA-256 for A and post-fix D:
`ce7831c12449e6e39c42132167d533e7a1e6df391fa9b50cb7c4a7fc05abc896`.
The difference is instruction order and Predicate allocation, not opcode
count: A repeats `CMP(P3) -> SEL(P3)`, while post-fix D first issues up to five
compares using P3-P7, releases a range with a select, and only then issues the
sixth compare. The final device code contains no predicate spill, reload, or
barrier.

Post-fix D's median is 16 ticks below A (2598 versus 2614), about 0.61%. This
small difference is treated as performance parity; it is not evidence that
post-fix D is faster than A.

## Smoke validation

Both unrelated smoke cases ran before the target fixture with real simulator
execution and comparison:

| Case | Result | Ticks |
|---|---|---:|
| `micro-op/binary-vector/vadd` | all 9 variants compare passed | 12962 |
| `micro-op/vector-load-store/vlds-post-update` | compare passed | 2417 |

## Evidence and archive

- Pre-fix raw root:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-issue508-sim/sim-results/issue508/authoritative-v3`
- Post-fix raw root:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-live-through-fix-sim/sim-results`
- Standalone evidence archive:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-evidence/vpto-sched-issue508-live-through-integration-20260820`
- SHA-256 manifest: `<archive>/SHA256SUMS`
- Manifest SHA-256:
  `6b2ffa9945cab9434f6d5f1e8f3f9b177108f6b9b40c37a1bb4754b6e9c8a139`

The archive contains 16 representative issue #508 files: A/B/old-D
instruction and `execve` evidence, C analyze and old-D probe logs, post-fix D
run-1 instructions, all three post-fix validation logs, and both smoke
validation logs. Its combined manifest also covers 15 selected issue #574
files. `sha256sum -c SHA256SUMS` passed for all 31 archived files. The source
raw result directories were not moved, deleted, or rewritten.
