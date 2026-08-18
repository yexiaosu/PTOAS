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

## Follow-up pressure-driven idle experiment

The scheduler was then changed experimentally so that it advances to the next
pending dependency event only when every currently available candidate would
exceed at least one modeled pressure limit. It does not emit a hardware NOP.
If there is no pending event, pressure remains a soft objective and scheduling
continues. Fresh model replay verifies the condition and the logical-cycle
advance before IR application.

On this fixture, each 757-node region records 17 pressure-driven advances. The
critical-path lower bound, last scheduled issue cycle, and completion remain
1820, 1820, and 1830 model cycles. Peak pressure changes from Vector 32 /
Predicate 13 to Vector 32 / Predicate 8. Both regions retain full known model
coverage and pass schedule verification, replay, and application without a
fallback.

| Metric | Original Scheduler ON | Pressure-idle Scheduler ON |
|---|---:|---:|
| Pressure-driven entries per pair region | 0 | 17 |
| Peak Vector / Predicate | 32 / 13 | 32 / 8 |
| Dynamic PLDI / PSTI / SMEM_BAR | 238 / 238 / 476 | 68 / 68 / 136 |
| Dynamic retired log lines | 2585 | 1905 |
| Tick runs | 11443, 11444, 11443 | 5548, 5558, 5546 |
| Tick min / median / max | 11443 / 11443 / 11444 | 5546 / 5548 / 5558 |
| Strict compare | 3/3 exact pass | 3/3 exact pass |

The experiment reduces each predicate spill/load count and its associated
barrier count by 71.4%, and reduces the Scheduler-ON median tick count by 51.5%
(2.06x faster). It does not close the gap to Scheduler OFF: 5548 versus 4053
baseline median ticks is still a 36.9% regression. Most importantly, the final
backend still emits predicate spills even though the scheduler's simplified
SSA pressure estimate no longer exceeds its Predicate limit of 8. The current
limit is therefore not a calibrated physical allocation boundary, and the
model does not yet capture all final register-allocation interference,
implicit temporaries, or lowering effects.

Follow-up artifacts are under the same result root:

- Trace compile: `pressure-idle-aabbcc-compile-on`
- Scheduler OFF reference: `pressure-idle-aabbcc-off-run1`
- Scheduler ON runs: `pressure-idle-aabbcc-on-run1`,
  `pressure-idle-aabbcc-on-run2`, and `pressure-idle-aabbcc-on-run3`
- Smoke runs: `pressure-idle-smoke-vadd` and
  `pressure-idle-smoke-vlds-post-update`

## Explicit hand-ordered AABBCC control

An additional `topk-gate-aabbcc-explicit` control keeps the VPTO scheduler
OFF and expresses the issue #574 hand-written AABBCC order directly in VPTO.
It has the same input, output ABI, operation counts, strict golden semantics,
MTE wave, and Bisheng `MISCHED=0` setting as the original fixture. Within each
two-token rank it uses the following fine-grained order:

```text
AA  (VMAX0, VMAX1) x 6
BB  VCMAX0, VCMAX1, broadcast0, broadcast1
CC  (VCMP0, VSEL0) x 6, (VCMP1, VSEL1) x 6,
    (VMIN0, VMIN1) x 6, VCMIN0, VCMIN1, broadcast0, broadcast1
DD  VSTI0, VSTI1,
    (VCMP0, VSEL0) x 6, (VCMP1, VSEL1) x 6
```

This differs materially from merely observing an AABBCC macro-stage label in
the scheduler result. The original scheduler emits all ready predicate
producers first and delays their consumers; pressure-driven idle reduces the
batch size but retains the same producer-batching shape. The explicit control
closes every compute predicate immediately.

| Per 757-node pair region | ABC source OFF | Explicit AABBCC OFF | Original scheduler AABBCC | Pressure-idle AABBCC |
|---|---:|---:|---:|---:|
| Model Vector / Predicate peak | 26 / 7 | 32 / 7 | 32 / 13 | 32 / 8 |
| Mean Vector live-range length | 30.09 | 32.92 | 31.28 | 32.34 |
| Mean Predicate live-range length | 4.99 | 4.82 | 16.16 | 11.49 |
| 204 compute-predicate lengths | 204 x 1 | 204 x 1 | 148 x 12; 56 x 14 | 122 x 7; 26 x 8; 48 x 9; 8 x 10 |
| Dynamic PLDI / PSTI / SMEM_BAR | 0 / 0 / 0 | 0 / 0 / 0 | 238 / 238 / 476 | 68 / 68 / 136 |

Live-range length is measured in final VPTO operation positions from SSA
definition to last use, before register allocation. The 204-row distribution
excludes setup predicates and the function-wide active mask. It exactly
captures each rank's match and winner-mask `VCMP` result. Both regions have the
same result.

The analyze-only trace for the explicit control has 757 nodes, 2544 edges,
full known model coverage, and critical-path lower bound 1820 model cycles.
Monotonic source replay gives last issue 3570 and completion 3580. This is
shorter than the ABC source replay of 5070/5080 but is not equal to real CA
ticks: it is a dependency-only model without real per-instruction resource and
latency calibration.

| Final device / CA metric | ABC source OFF | Explicit AABBCC OFF | Original scheduler AABBCC | Pressure-idle AABBCC |
|---|---:|---:|---:|---:|
| Static device instruction slots | 888 | 888 | 1366 | 1026 |
| Dynamic retired log lines | 1633 | 1633 | 2585 | 1905 |
| Dynamic VMAX / VCMAX / VMIN / VCMIN | 216 / 36 / 216 / 36 | same | same | same |
| Dynamic VCMP / VSEL | 420 / 432 | same | same | same |
| Derived SIMD span | 1606 | 1696 | 9033 | 3130 |
| Derived EXIPC | 0.903 | 0.855 | 0.161 | 0.463 |
| Tick runs | 4052, 4053, 4056 | 4132, 4138, 4132 | 11443, 11444, 11443 | 5548, 5558, 5546 |
| Tick median | 4053 | 4132 | 11443 | 5548 |
| Strict compare | 3/3 exact pass | 3/3 exact pass | 3/3 exact pass | 3/3 exact pass |

The explicit control is only 1.95% slower than the current ABC source, but is
25.5% faster than pressure-idle Scheduler ON and 63.9% faster than the
original Scheduler ON result. Most importantly, it has no predicate
spill/reload/barrier instructions. This confirms the implication of issue
#574: AABBCC itself does not inherently cause spilling. The regression comes
from the scheduler-generated fine-grained order inside AABBCC, especially
extending `VCMP` live ranges, rather than from the macro-stage order.

The current candidate comparison explains the result. Once projected pressure
is not above its soft limit, all excess scores are zero and critical-path
height is compared before pressure delta. A newly ready `VSEL` that would end
a predicate live range can therefore lose to another high-height `VCMP` that
creates one. Pressure-driven idle reacts only after every available candidate
would exceed a limit; it cannot repair the already lengthened live ranges.

The next scheduling experiment should preserve macro AABBCC parallelism while
adding a near-limit, critical-path-slack-aware consumer preference: when a
ready last-use consumer can close a Predicate range without delaying a truly
urgent chain, select it before opening another Predicate range. The preference
should be continuous before the hard excess point and should keep the explicit
control's alternating token-0/token-1 VMAX/VMIN chains. Pressure-driven idle
remains a fallback for the all-candidates-over-limit state. This is narrower
than globally moving pressure delta ahead of critical path, which could erase
useful ILP. A schedule acceptance gate remains deliberately deferred while
this fixture is used to improve the strategy itself.

Explicit-control artifacts are under the same result root:

- Analyze trace: `explicit-aabbcc-analysis`
- CA runs: `explicit-aabbcc-off-run1`, `explicit-aabbcc-off-run2`, and
  `explicit-aabbcc-off-run3`
- Smoke run: `explicit-control-smoke-vadd`

## Saved artifacts

- All results: `/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-aabbcc-sim`
- OFF runs: `aabbcc-off-run1`, `aabbcc-off-run2`, `aabbcc-off-run3`
- ON runs: `aabbcc-on-run1`, `aabbcc-on-run2`, `aabbcc-on-run3`
- Extracted device ELFs/text: `assembly/off`, `assembly/on`
- Smoke runs: `smoke-vadd`, `smoke-vlds-post-update`

Both smoke cases started/stopped the CA model and passed strict comparison.
