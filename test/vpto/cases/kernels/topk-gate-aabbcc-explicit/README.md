# topk-gate-aabbcc-explicit

This control uses the same `N=4`, `E=384`, `K=9`, `token_tile=4` dataflow,
inputs, output ABI, golden, MTE wave, and downstream Bisheng MISCHED setting as
`topk-gate-aabbcc-scheduler`. Its VPTO source is explicitly ordered to mirror
the hand-written CCE `USE_AABBCC=1` pair schedule from
`mouliangyu/PTOAS#574`, and `ptoas.flags` keeps the VPTO scheduler off.

For each rank of a two-token pair, the source order is:

1. `AA`: initialize both max accumulators, then alternate token 0/token 1
   `vmax` operations for chunks 5 through 0.
2. `BB`: issue both `vcmax` reductions, then both lane broadcasts.
3. `CC`: issue adjacent `vcmp`/`vsel` match pairs for token 0 and then token 1;
   alternate the two `vmin` chains; then issue both `vcmin` reductions and
   broadcasts.
4. `DD`: store both indices, then issue adjacent `vcmp`/`vsel` winner-mask
   pairs for token 0 and then token 1.

There are no synthetic dependency operations. Reversing chunk order changes
only the associative max/min reduction order and matches the CCE template
recursion. Operation counts and strict output semantics are identical to the
ABCABC scheduler fixture.

Three real A5 CA SIM runs with Scheduler OFF and Bisheng `MISCHED=0` passed
strict comparison at 4132, 4138, and 4132 ticks. The final device program has
888 instruction slots and zero predicate spill/load/barrier instructions. See
the explicit-control section of the sibling
`topk-gate-aabbcc-scheduler/RESULTS.md` for the instruction-order, live-range,
and four-way performance comparison.
