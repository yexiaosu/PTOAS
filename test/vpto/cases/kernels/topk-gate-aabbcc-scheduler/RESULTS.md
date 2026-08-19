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

| Per 757-node pair region | Pair-fused ABC OFF | Explicit AABBCC OFF | Original scheduler AABBCC | Pressure-idle AABBCC |
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

| Final device / CA metric | Pair-fused ABC OFF | Explicit AABBCC OFF | Original scheduler AABBCC | Pressure-idle AABBCC |
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

The explicit control is only 1.95% slower than the pair-fused ABC source, but is
25.5% faster than pressure-idle Scheduler ON and 63.9% faster than the
original Scheduler ON result. Most importantly, it has no predicate
spill/reload/barrier instructions. This confirms the implication of issue
#574: AABBCC itself does not inherently cause spilling. The regression comes
from the scheduler-generated fine-grained order inside AABBCC, especially
extending `VCMP` live ranges, rather than from the macro-stage order.

This comparison is deliberately limited to the existing two-token VPTO
fixture. It does not compare against the original CCE reproducer's
single-token ABC scope boundary. The pair-fused source already exposes two
tokens to the hardware issue window at the same time, so its Scheduler-OFF
number is an optimized control rather than the fair issue #574 ABC baseline.

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

## Single-token ABC control

The `topk-gate-abc-single-token` control restores the scope boundary that was
missing from the earlier comparison. It keeps the same `N`, `E`, `K`, input,
strict golden result, kernel ABI, outer MTE wave, and Bisheng `MISCHED=0`
setting. The only structural change from the pair-fused source is that each of
the four tokens owns a separate `PSET`/`VEC_SCOPE` and its own `VCI`, padding,
and nine-rank top-k chain. Consequently, Scheduler OFF now means the same thing
as in the original CCE ABC construction: token B cannot enter the vector issue
window until token A's scope has completed.

| Final device / CA metric | Single-token ABC OFF | Pair-fused ABC OFF | Explicit AABBCC OFF | Scheduler limit 7 |
|---|---:|---:|---:|---:|
| Dynamic PSET | 4 | 2 | 2 | 2 |
| Dynamic VCI | 24 | 12 | 12 | 12 |
| Dynamic VCMP / VSEL | 432 / 432 | 420 / 432 | 420 / 432 | 420 / 432 |
| Dynamic PLDI / PSTI | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| EX instructions | 1484 | 1450 | 1450 | 1450 |
| EX active cycles | 1032 | 998 | 798 | 974 |
| EX dual-issue cycles | 452 | 452 | 652 | 476 |
| Derived SIMD span | 2882 | 1606 | 1696 | 1660 |
| Tick runs | 5347, 5348, 5340 | 4052, 4053, 4056 | 4132, 4138, 4132 | 4105, 4102, 4108 |
| Tick median | 5347 | 4053 | 4132 | 4105 |
| Strict compare | 3/3 exact pass | 3/3 exact pass | 3/3 exact pass | 3/3 exact pass |

The fair result has the same direction as issue #574. Explicit hand-ordered
AABBCC is 22.72% faster than single-token ABC, while the pair-fused ABC source
is 24.20% faster. Predicate-limit-7 scheduling is 23.23% faster than the fair
single-token control and 0.65% faster than explicit AABBCC. None of these three
fast variants spills predicates, so the improvement over single-token ABC is
latency hiding rather than spill avoidance.

The CA log makes the hidden latency concrete. The four single-token `PSET`
instructions retire at cycles 1785, 2503, 3221, and 3939, exactly 718 cycles
apart. Its EX active-cycle gap histogram contains 72 gaps of 13 cycles and 68
gaps of 8 cycles. These are dependency bubbles repeated independently in each
token's reduction/broadcast/compare chain. Its 2882-cycle derived SIMD span is
also close to the approximately 2907-cycle SIMD interval of the original CCE
ABC report.

Pair fusion does not change Scheduler OFF into compiler reordering. Instead,
it changes what the hardware is allowed to see. With two independent tokens
inside one `VEC_SCOPE`, both instruction streams reach the CA issue window.
When token 0 is waiting for a reduction result such as
`VCMAX`/`VCMIN -> VDUP -> VCMP`, the scoreboard can issue a dependency-ready
instruction from token 1. The pair therefore takes about 808 or 769 cycles,
instead of reproducing two approximately 718-cycle single-token intervals.
The pair-fused gap histogram has only 2 gaps of 13 cycles and 12 gaps of 8
cycles. This is the precise sense in which the source had already fused two
tokens into one vecscope and obtained cross-token latency hiding before the
VPTO scheduler did any work.

Explicit AABBCC exposes the same two-token window but makes the chains more
lockstep: its 652 dual-issue cycles exceed pair-fused ABC's 452. It still has
36 gaps of 12 cycles and 34 gaps of 7 cycles, which explains why its final
4132 ticks are slightly slower than pair-fused ABC despite higher dual issue.
The predicate-limit-7 schedule reduces that fine-grained ordering cost while
preserving the no-spill property.

Single-token-control artifacts are under the same result root:

- CA runs: `single-token-abc-off-run1`, `single-token-abc-off-run2`, and
  `single-token-abc-off-run3`
- Smoke run: `single-token-control-smoke-vadd`

## Predicate-limit calibration experiment

The generic A5 Predicate pressure limit was reduced from 8 to 7 to match the
registers that the current backend actually allocates, P1-P7. The function-wide
active mask occupies P1, so only six registers remain available for the
temporary `VCMP` results. With limit 7, pressure-driven idle therefore consumes
one ready predicate after six producers instead of allowing a seventh producer.

Both 757-node pair regions still have full known model coverage. Each records
17 pressure-driven advances, and the critical-path lower bound, last scheduled
issue cycle, and completion remain 1820, 1820, and 1830 model cycles. Peak
pressure is Vector 32 / Predicate 7. The 204 compute-predicate live ranges per
region are 54 of length 6 and 150 of length 8; none of these longer positional
ranges overlap beyond the calibrated peak.

| Metric | Predicate limit 8 | Predicate limit 7 |
|---|---:|---:|
| Pressure-driven entries per pair region | 17 | 17 |
| Peak Vector / Predicate | 32 / 8 | 32 / 7 |
| Compute-predicate live-range distribution | 122 x 7; 26 x 8; 48 x 9; 8 x 10 | 54 x 6; 150 x 8 |
| Dynamic PLDI / PSTI / predicate barrier | 68 / 68 / 136 | 0 / 0 / 0 |
| Tick runs | 5548, 5558, 5546 | 4105, 4102, 4108 |
| Tick min / median / max | 5546 / 5548 / 5558 | 4102 / 4105 / 4108 |
| Strict compare | 3/3 exact pass | 3/3 exact pass |

The limit-7 result removes every predicate spill, reload, and associated
barrier seen in the CA instruction logs. Its median is 26.0% faster than the
limit-8 pressure-idle result, 23.23% faster than the fair single-token ABC
control (5347 ticks), 0.65% faster than explicit AABBCC Scheduler OFF (4132
ticks), and 1.28% slower than the pair-fused ABC control (4053 ticks). This
confirms that the physical boundary relevant to this lowering is seven
allocated Predicate registers, not eight architectural names available to
ordinary SSA values. The remaining 52-tick difference from pair-fused ABC is
fine-grained ordering cost rather than spill cost; pair-fused ABC is an
optimized latency-hiding control, not the original reproducer's baseline.

Limit-7 artifacts are under the same result root:

- Trace compile: `predicate-limit7-aabbcc-trace`
- Scheduler ON runs: `predicate-limit7-aabbcc-on-run1`,
  `predicate-limit7-aabbcc-on-run2`, and `predicate-limit7-aabbcc-on-run3`
- Smoke run: `predicate-limit7-smoke-vadd`

## Saved artifacts

- All results: `/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-aabbcc-sim`
- Single-token ABC runs: `single-token-abc-off-run1`,
  `single-token-abc-off-run2`, `single-token-abc-off-run3`
- OFF runs: `aabbcc-off-run1`, `aabbcc-off-run2`, `aabbcc-off-run3`
- ON runs: `aabbcc-on-run1`, `aabbcc-on-run2`, `aabbcc-on-run3`
- Extracted device ELFs/text: `assembly/off`, `assembly/on`
- Smoke runs: `smoke-vadd`, `smoke-vlds-post-update`

Both smoke cases started/stopped the CA model and passed strict comparison.
