# A5 CA SIM validation

This report consolidates the VPTO scheduler stage-two experiments. The current
result is the near-limit pressure-aware scheduler with the calibrated A5
Predicate limit of 7. The earlier limit-7 run is equivalent for this fixture
and is intentionally not retained as a separate configuration.

## Result summary

All current configurations ran three times with the repository validation
runner, real CA-model execution, and strict exact comparison.

| Configuration | Role | Predicate / Vector peak | Dynamic PLDI / PSTI / spill barrier | Tick runs | Min / median / max | Strict compare |
|---|---|---:|---:|---|---:|---:|
| Near-limit Scheduler ON | Current scheduler result | 7 / 32 | 0 / 0 / 0 | 4110, 4116, 4113 | 4110 / 4113 / 4116 | 3/3 pass |
| Pair-fused Scheduler OFF | Same input, exact no-op control | 7 / 26 | 0 / 0 / 0 | 4053, 4054, 4059 | 4053 / 4054 / 4059 | 3/3 pass |
| Explicit AABBCC OFF | Hand-ordered no-spill diagnostic control | 7 / 32 | 0 / 0 / 0 | 4150, 4154, 4142 | 4142 / 4150 / 4154 | 3/3 pass |
| Single-token ABC OFF | Issue #574 external-scope baseline | n/a | 0 / 0 / 0 | 5357, 5357, 5351 | 5351 / 5357 / 5357 | 3/3 pass |

The current Scheduler-ON median is 1.46% above the fresh pair-fused OFF
control, 0.89% below the explicit AABBCC control, and 23.22% below the
single-token ABC control. It removes the spill failure of the original
scheduler but does not outperform pair-fused OFF.

ABC and AABB are descriptive source/schedule labels, not optimization targets.
Pair-fused OFF already exposes two independent token chains inside one
vecscope and hardware issue window. Scheduler stage two operates within each
vecscope and does not perform cross-vecscope fusion.

## Development comparison

The experiments that materially changed the result are retained below. The
old limit-7 result is omitted because the current near-limit result has the
same pressure-idle count, modeled peaks, rank-loop predicate live ranges, and
EX active/dual-issue counts.

| Scheduler generation | Predicate peak | Dynamic PLDI / PSTI / spill barrier | Median ticks | Outcome |
|---|---:|---:|---:|---|
| Original stage-two scheduler | 13 | 238 / 238 / 476 | 11443 | Large producer batches extend Predicate ranges and spill |
| Pressure-driven idle, Predicate limit 8 | 8 | 68 / 68 / 136 | 5548 | Fewer spills, but the modeled limit is above the backend-safe boundary |
| Current near-limit policy, Predicate limit 7 | 7 | 0 / 0 / 0 | 4113 | No Predicate or Vector spill; current retained result |

The main improvement came from calibrating Predicate pressure to the backend's
P1-P7 allocation and retaining pressure-driven event advancement. The
near-limit strategy adds a generic pre-limit preference, but on this fixture it
changes only three setup positions per region and does not change the main
rank-loop schedule.

## Current algorithm result

The scheduler ranks candidates using their modeled critical-path urgency,
projected bounded-set pressure, and SSA live-range release:

1. A candidate that remains within all pressure limits beats one that exceeds
   a limit.
2. Excess growth and projected excess remain the first weighted costs.
3. When a ready producer can consume the remaining headroom, candidates enter
   near-limit ranking.
4. A one-modeled-write-latency urgency band protects a materially more urgent
   critical path.
5. Within that band, lower projected pressure and greater live-range release
   are preferred.
6. Critical path, pressure delta, and original order provide the remaining
   deterministic ordering.

If every available candidate exceeds a bounded pressure set and a dependency
event is pending, the scheduler advances to that event. It emits no hardware
NOP. Fresh replay verifies that no safe candidate existed and that the cycle
advance, pressure state, dependencies, and final order are valid. If no event
is pending, scheduling must continue rather than deadlock.

The policy uses dataflow and pressure properties only. It contains no TopK
operation-name, token-index, rank-count, or fixed ABC/AABB rule.

## Scheduler trace

Implementation validated:
`cebb779526d74e1d417e5c36d2c1db53820589cf`.

| Metric | Pair region 0 | Pair region 1 |
|---|---:|---:|
| Nodes / edges | 757 / 2544 | 757 / 2544 |
| Live-ins / live-outs | 27 / 27 | 27 / 27 |
| Known / unknown classes | 757 / 0 | 757 / 0 |
| Dependency critical-path lower bound | 1820 | 1820 |
| Source-order replay last issue / completion | 5070 / 5080 | 5070 / 5080 |
| Scheduled result last issue / completion | 1820 / 1830 | 1820 / 1830 |
| Modeled Vector / Predicate peak | 32 / 7 | 32 / 7 |
| Pressure-driven event advances | 17 | 17 |
| Compute-predicate live ranges | 54 x 6; 150 x 8 | 54 x 6; 150 x 8 |

Coverage is 1514 schedulable operations, with zero structural, boundary,
unsupported, or unclassified operations. Both regions pass schedule
verification, fresh replay, and IR application. There is no unknown class,
fallback, verifier failure, replay failure, or apply failure.

Across both regions the selected-candidate reasons are:

| Reason | Count |
|---|---:|
| `deterministic-tie-break` | 732 |
| `longer-critical-path` | 376 |
| `lower-pressure-delta` | 38 |
| `pressure-safe-candidate` | 32 |
| `near-limit-pressure-preserving` | 2 |
| `only-candidate` | 334 |

No `near-limit-live-range-closing` decision occurs in this TopK trace. Its
generic behavior is covered by focused strategy tests. The two
`near-limit-pressure-preserving` selections move two setup loads ahead of one
setup compare in each region. The 204 rank-loop compute-predicate ranges per
region remain unchanged, with mean positional length 7.47.

The 5070 source replay and 1820 scheduled result are not symmetric hardware
performance estimates. Source replay keeps a monotonically nondecreasing time
over the fixed IR order, while the ON result is created from the full DAG ready
set. The current model also lacks a calibrated finite OOO window and real
issue-width/resource model. These logical cycles are dependency diagnostics,
not evidence that ON is faster than OFF.

## Final device and CA metrics

| Metric | Near-limit Scheduler ON | Pair-fused OFF | Explicit AABBCC OFF | Single-token ABC OFF |
|---|---:|---:|---:|---:|
| Device `.text` bytes / slots | 3552 / 888 | 3552 / 888 | 3552 / 888 | 2328 / 582 |
| Static unique VLDI / VSTI in executed vector body | 12 / 20 | 12 / 20 | n/a | n/a |
| Dynamic VLDI / VSTI | 24 / 40 | 24 / 40 | 24 / 40 | 24 / 40 |
| Dynamic PLDI / PSTI / spill barrier | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| EX instructions | 1450 | 1450 | 1450 | 1484 |
| EX active / dual-issue cycles | 974 / 476 | 998 / 452 | 798 / 652 | 1032 / 452 |
| Derived SIMD span | 1660 | 1606 | 1696 | 2882 |
| Retired instruction-log entries | 1631 | 1635 | 1636 | 1694 |

`EX` denotes the CA `RVECEX` vector-compute pipeline. Active cycles contain at
least one retired `RVECEX`; dual-issue cycles contain two. The current result
packs 24 more EX pairs than pair-fused OFF, but its SIMD span is 54 cycles
longer. More simultaneously live Vector values therefore did not create more
useful ready work.

### Spill evidence

Predicate and Vector spill freedom is checked from the final CA instruction
stream, not inferred from the textual pressure peak alone.

- The current result uses V0-V31 and reaches the modeled Vector boundary of
  32 without allocating outside that physical set.
- Scheduler ON and pair-fused OFF have identical expected Vector transfers:
  12 unique VLDI and 20 unique VSTI in the compiled vector body, executed
  twice as 24/40 dynamic instructions.
- The 12 VLDI are the two tokens' six score loads. The 20 VSTI are two setup
  stores plus nine rank stores for each token. There is no additional
  save/restore pair attributable to Vector spilling.
- PLDI, PSTI, and their spill-related SMEM barrier are absent from the dynamic
  stream.
- The CA popped log covers every vector PC from the first PSET at device-text
  offset `0x200` through the final SEND at `0xddd4`. Unretired slots are in the
  scalar/MTE prelude and after the final SEND, not hidden vector spill code.
- Scheduler ON, pair-fused OFF, and explicit AABBCC all retain 888 static
  instruction slots.

The installed Bisheng `llvm-objdump` recognizes the A5 HiIPU ELF but prints
`<not available>` for its instruction mnemonics. Static opcode claims above
therefore come from complete CA vector-PC coverage and decoded retired
instructions, not from an empty grep over undecoded objdump output.

The current Scheduler-ON device `.text` SHA-256 is
`b4fe848df860b37f4d19610da570736a2d49a835626360a294c0d333e3396b4c`.
It differs from prior device text, ruling out stale artifact reuse.

## Control interpretation

### Pair-fused Scheduler OFF

The source contains two independent token chains in each vecscope. OFF does
not perform compiler scheduling, but the CA issue window can use one token's
ready work while the other token waits for a reduction or broadcast result.
Its Predicate / Vector source-order pressure peak is 7 / 26, and it is the
fastest fresh control at median 4054 ticks.

### Explicit AABBCC OFF

This case expresses a hand-ordered fine-grained pair schedule while keeping
the scheduler off. It has the same ABI, operation mix, MTE wave, golden result,
and Bisheng mischeduler setting as the main fixture. It is a diagnostic
no-spill control, not an order the generic scheduler must reproduce. Its
median is 4150 ticks.

### Single-token ABC OFF

This case places each token in its own vecscope and matches the external scope
boundary of issue #574. Later tokens cannot enter the same vector issue window
to hide the current token's dependency latency. It remains the fair external
ABC baseline at median 5357 ticks, not an input that stage two is expected to
fuse.

## Provenance

- Server worktree:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-aabbcc-sim`
- Branch: `codex/vpto-sched-aabbcc-sim`
- Validation implementation:
  `cebb779526d74e1d417e5c36d2c1db53820589cf`
- Build: `/home/wanglan/PTOAS/.worktrees/ca-sim-builds/vpto-sched-aabbcc-sim`
- Python/PTOAS: the worktree's `.venv/bin/python` and `.venv/bin/ptoas`
- LLVM/MLIR: `/home/wanglan/llvm-workspaces/build-vpto19`
- CANN/Bisheng: `/usr/local/CANN/cann-9.1.0`, Bisheng 15.0.5
- Simulator:
  `/usr/local/CANN/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib`
- Read-only PTO-ISA: `/home/wanglan/pto-isa`, commit
  `311ca0c83f5571dc165681fee0a427983c555d3c`
- Result root:
  `/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-aabbcc-sim/near-limit-cebb77952`

The editable Python package, `_core`, `libPTOASCompiler.so`, and RUNPATH were
verified to resolve inside this worktree/venv or the LLVM directory above.

## Reproduction

All cases use the repository runner. `CASE_NAME`, scheduler mode, and a unique
`WORK_SPACE` vary by configuration.

```bash
export RESULT_ROOT=/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-aabbcc-sim/near-limit-cebb77952
export WORK_SPACE="$RESULT_ROOT/<configuration>-run-<n>"
export ASCEND_HOME_PATH=/usr/local/CANN/cann-9.1.0
export SIM_LIB_DIR="$ASCEND_HOME_PATH/x86_64-linux/simulator/dav_3510/lib"
export PTOAS_BIN="$PTO_SOURCE_DIR/.venv/bin/ptoas"
export DEVICE=SIM
export COMPILE_ONLY=0
export CASE_NAME=kernels/topk-gate-aabbcc-scheduler
export PTOAS_FLAGS="--pto-arch a5 --pto-backend=vpto --vpto-scheduler=on"
"$PTO_SOURCE_DIR/test/vpto/scripts/run_host_vpto_validation.sh"
```

The saved trace compile adds `--vpto-scheduler-trace` and uses
`COMPILE_ONLY=1`. Controls use their corresponding case name and
`--vpto-scheduler=off`.

Before the target runs, both smoke cases started and stopped the real CA model
and passed strict comparison:

- `micro-op/binary-vector/vadd`: 12973 ticks
- `micro-op/vector-load-store/vlds-post-update`: 2427 ticks

## Saved artifacts

Under the result root:

- Trace compile: `main-on-trace`
- Scheduler ON: `main-on-run-{1,2,3}`
- Pair-fused OFF: `main-off-run-{1,2,3}`
- Explicit AABBCC OFF: `explicit-off-run-{1,2,3}`
- Single-token ABC OFF: `single-off-run-{1,2,3}`
- Smoke: `smoke-vadd`, `smoke-vlds-post-update`
- Extracted device text: `assembly/{main-on,main-off,explicit-off,single-off}`
