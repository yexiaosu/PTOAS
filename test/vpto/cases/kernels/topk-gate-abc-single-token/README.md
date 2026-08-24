# topk-gate-abc-single-token

This runtime case is the control that matches the `ABCABC MISCHED=0` boundary
from `mouliangyu/PTOAS#574`. It uses the same `N=4`, `E=384`, `K=9`, input,
golden result, ABI, outer four-load/four-store MTE wave, and downstream Bisheng
mischeduler setting as `topk-gate-aabbcc-scheduler` and
`topk-gate-aabbcc-explicit`.

The difference is the vector-function boundary. Each of the four tokens has
its own `pto.vecscope`, `PSET`, six index vectors, padding, and complete
nine-rank `A -> B -> C -> D` chain. No vecscope contains two tokens, so the
backend and CA issue window cannot overlap one token's vector instructions with
another token's vector instructions. Keep the VPTO scheduler off when using
this case as the original ABCABC control.

Use the three sibling cases as follows:

| Case | Vecscope boundary | Source order | Purpose |
|---|---|---|---|
| `topk-gate-abc-single-token` | four single-token scopes | ABC per token | Original CCE ABCABC control |
| `topk-gate-aabbcc-explicit` | two two-token scopes | hand AABBCCDD | Original CCE pair schedule control |
| `topk-gate-aabbcc-scheduler` | two two-token scopes | ABCD-ABCD per pair | Scheduler transformation fixture |

Run on SIM with:

```bash
WORK_SPACE=/tmp/topk-gate-abc-single-token \
CASE_NAME=kernels/topk-gate-abc-single-token \
DEVICE=SIM COMPILE_ONLY=0 \
PTOAS_FLAGS="--pto-arch a5 --pto-backend=vpto --vpto-scheduler=off" \
test/vpto/scripts/run_host_vpto_validation.sh
```
