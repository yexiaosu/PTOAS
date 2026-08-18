# A5 CA SIM validation

This report records the validation performed on 2026-08-18 for the VPTO
scheduler stage-two baseline `1aec0919154a8a0cba87bd848f78974d2ee2df64`.

## Provenance

- Worktree: `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-aabbcc-sim`
- Branch: `codex/vpto-sched-aabbcc-sim`
- Build: `/home/wanglan/PTOAS/.worktrees/ca-sim-builds/vpto-sched-aabbcc-sim`
- Venv Python and PTOAS: worktree `.venv/bin/python` and `.venv/bin/ptoas`
- LLVM: `/home/wanglan/llvm-workspaces/build-vpto19`
- CANN/Bisheng: `/usr/local/CANN/cann-9.1.0`, Bisheng 15.0.5
- Simulator: `/usr/local/CANN/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib`
- Read-only PTO-ISA: `/home/wanglan/pto-isa`, commit `311ca0c83f5571dc165681fee0a427983c555d3c`

The Python package, `_core`, `libPTOASCompiler.so`, and RUNPATH all resolve
inside this worktree/venv or the LLVM directory above.

The authoritative #574 CCE artifact is commit
`623c008588397063b540178d3ba0cebae3d51a38` under
`docs/issues/574-topk-gate-vf-aabbcc-misched` in the read-only checkout:

`/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched2-sim-508-574-1168-rebase-llvm19/sim-results/issues-508-574-1168/external/PTOAS-official-574`

Relevant SHA-256 values are:

- `topk_gate_vf.cpp`: `d256db92e986b9d0a07468f08845c693958f423846738d7f98c2d239e119e6f8`
- `main_vf.cpp`: `2334c29f9fb5430c6ae568ac32d643f21c15376e545af300e3d5074b6e3507da`
- `run_vf_sim.sh`: `13ef69f63af9fbaeb2c1d0449c421571768dadcd325cae6c7389aa369a7ceffc`
- `VF_AABBCC_MISCHED_REPORT.md`: `a9a34343bcfc848e2e68b48bcb9402066c5110d6c826a76b38d7663ba46b3864`

The original uses `-mllvm -cce-aicore-vec-misched=0` for `MISCHED=0`.
PTOAS does the same by default; neither control command passes
`--enable-bisheng-vec-misched`.

## Reproduction command

Both modes use the repository runner. `MODE` is `off` or `on`; ON additionally
uses `--vpto-scheduler-trace` for the saved trace compile.

```bash
export RESULT_ROOT=/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-aabbcc-sim
export WORK_SPACE="$RESULT_ROOT/aabbcc-${MODE}-run1"
export ASCEND_HOME_PATH=/usr/local/CANN/cann-9.1.0
export SIM_LIB_DIR="$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/lib"
export PTOAS_BIN="$PTO_SOURCE_DIR/.venv/bin/ptoas"
export CASE_NAME=kernels/topk-gate-aabbcc-scheduler
export DEVICE=SIM
export COMPILE_ONLY=0
export PTOAS_FLAGS="--pto-arch a5 --pto-backend=vpto --vpto-scheduler=${MODE}"
"$PTO_SOURCE_DIR/test/vpto/scripts/run_host_vpto_validation.sh"
```

## Scheduler trace

Each of the two pair regions has 757 nodes, 2544 edges, 27 live-ins/outs,
757 known classes, and zero unknown classes. Coverage is 1514 schedulable,
zero unsupported, and zero unclassified. The critical-path lower bound is
1820 model cycles. Replaying source order monotonically gives last issue 5070
and completion 5080; the scheduled result gives last issue 1820 and completion
1830. Source-order pressure peaks at Vector 26 / Predicate 7; scheduled pressure
peaks at Vector 32 / Predicate 13.

`schedule-result` is printed only after schedule verification, fresh model
replay, and IR application all succeed. Both regions emit it; no skip,
fallback, verifier failure, or apply failure is present. The result order is
AABBCCDD within each rank.

Trace log:

`/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-aabbcc-sim/aabbcc-compile-on/kernels_topk-gate-aabbcc-scheduler/validation.log`

The trace compile and SIM compile have identical extracted device `.text`
SHA-256 `5ac0f89ab5abfdcf5691ce4631d09850ec14b240ddde8d7f5296d80293694418`.

## Device instructions and CA SIM

| Metric | Scheduler OFF | Scheduler ON |
|---|---:|---:|
| Static `.text` instruction slots | 888 | 1366 |
| Static PLDI / PSTI / SMEM_BAR | 0 / 0 / 0 | 119 / 119 / 238 |
| Dynamic PLDI / PSTI / SMEM_BAR | 0 / 0 / 0 | 238 / 238 / 476 |
| Dynamic retired log lines | 1633 | 2585 |
| VMAX / VCMAX / VMIN / VCMIN | 216 / 36 / 216 / 36 | same |
| VCMP / VSEL | 420 / 432 | same |
| VCI / VDUP-family / VLDI / VSTI | 12 / 80 / 24 / 40 | same |
| Derived SIMD span | 1606 | 9033 |
| Derived EXIPC | 0.903 | 0.161 |
| Tick runs | 4052, 4053, 4056 | 11443, 11444, 11443 |
| Tick min / median / max | 4052 / 4053 / 4056 | 11443 / 11443 / 11444 |
| Strict compare | 3/3 exact pass | 3/3 exact pass |

SIMD span is retirement-log last-minus-first RV cycle plus the 26-cycle tail
used by the #574 report. EXIPC is the unchanged 1450 compute/issue operations
divided by that span. No CA summary occupancy file was produced, so no
unsupported occupancy or stall estimate is claimed.

The VPTO source contains all original padding and final winner-mask operations.
The final device mix is smaller than the raw CCE artifact because PTOAS/Bisheng
CSEs the identical pair padding predicates and removes the last post-K winner
mask, whose result has no subsequent rank. This does not change the 36 output
indices or their strict golden semantics.

The final retired sequence confirms ABCD-ABCD with Scheduler OFF and
AABBCCDD with Scheduler ON. ON then inserts the spill/barrier sequence visible
above. Stage two therefore shortens its simplified dependency-only logical
schedule but exceeds predicate pressure and regresses CA SIM median ticks by
2.82x. The result is a successful real-equivalent scheduler coverage test, not
a performance improvement.

## Saved artifacts

- All results: `/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-aabbcc-sim`
- OFF runs: `aabbcc-off-run1`, `aabbcc-off-run2`, `aabbcc-off-run3`
- ON runs: `aabbcc-on-run1`, `aabbcc-on-run2`, `aabbcc-on-run3`
- Extracted device ELFs/text: `assembly/off`, `assembly/on`
- Smoke runs: `smoke-vadd`, `smoke-vlds-post-update`

Both smoke cases started/stopped the CA model and passed strict comparison.
