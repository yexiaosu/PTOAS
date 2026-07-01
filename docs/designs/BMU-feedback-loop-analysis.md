# BMU 反馈环可行性分析

> 配套文档：`BMU-integration-design.md`（§4.5 反馈环、§4.7 失败处理、§4.9 反馈环）。
> 本文分析「PlanMemory 失败 → reclassify → 重跑 layout」这个可选反馈环**能不能做、有什么坑**。
> 结论先行：**能做，但当前代码有三个硬前提没满足**，且反馈环本身有一个「D 不一定缓解压力」的根本性局限。第一版不做是合理的。

分析基于当前代码（本分支 Phase 1-4 完成态）：
- `MemPlan::EmitPlanMemoryFailureInfo` / `RecordOverflowIfAny`：`lib/PTO/Transforms/PTOPlanMemory.cpp:1037`、`:1048`
- 失败信息容器 `failApplyBufferInfo`：`lib/PTO/Transforms/PTOPlanMemory.h:764`
- 失败即 `signalPassFailure`：`PTOPlanMemory.cpp` 的 `runOnOperation`（`plan()` 返回 failure 处）
- 驱动 pipeline：`tools/ptoas/ptoas.cpp` 单条 `PassManager`，`pm.run(module)` 跑一次

---

## 1. 反馈环要做什么（回顾）

设计意图：把 PlanMemory 当前的「容量超限就硬 fail」换成「软回退」——

```
pto-plan-memory 规划 S 类失败（某 scope 容量峰值超限）
  → 不 signalPassFailure，改写 module attr pto.plan_memory_failed_buffers = [...]
  → driver 检测到该 attr
  → 重跑 pto-classify-buffers（把失败 buffer reclassify 成 D）
       → pto-plan-bmu-layout（把新 D 类物化成 BMU 动态 alloc/free）
       → pto-plan-memory（S 类压力被移走，这次成功）
```

价值：让「容量峰值满足、但静态布局摆不下（碎片/生命周期交叠）」的程序，靠 BMU 的运行时 bump 分配继续跑，而不是编译期报错。

---

## 2. 能不能做：可以，但要补三层改动

### 2.1 PlanMemory：失败信号从 error 改 attr（可行，中等改动）

当前 `plan()` 失败路径是 `EmitPlanMemoryFailureInfo()`（`func_.emitError()`）+ 上层 `signalPassFailure()`。改法：

- 新增 pass option `--plan-memory-soft-fail`（默认 false，保持现状），或直接在 BMU 模式下软失败。
- 软失败时不 `emitError`，改成在 module 上写 `pto.plan_memory_failed_buffers`，`plan()` 返回 success（或返回一个可区分的软失败码，让 `runOnOperation` 不调 `signalPassFailure`）。

**风险点**：`signalPassFailure` 不止这一个调用点（还有 `applyPatternsAndFoldGreedily` 失败等）。要精确地只把「容量超限」这一种失败改成软失败，别把其它真 bug 也吞掉。

### 2.2 classify：读 attr 做 reclassify（可行，小改动）

`pto-classify-buffers` 第二遍跑时读 `pto.plan_memory_failed_buffers`，把命中的 alloc 从 S 改标 D。需要一个稳定的 buffer 标识（见 §3.1 的坑）。

### 2.3 driver：把子 pipeline 放进循环（可行，但不是 MLIR PassManager 原生能力）

`tools/ptoas/ptoas.cpp` 现在是**一条 PassManager 跑一次**。MLIR 的 `PassManager` 没有「条件重跑一段」的原语。要实现反馈环，只能：

- **方案 A（driver 编排）**：把 `classify → plan-bmu-layout → plan-memory` 抽成一个可重复调用的子 `PassManager`，driver 在一个 `while` 里跑它，每轮跑完检查 module 上有没有 `pto.plan_memory_failed_buffers`，有就清掉 attr 再跑一轮，设一个最大迭代数（比如 2，设计 §4.9 就说「重跑一次」）。
- **方案 B（单 pass 内迭代）**：做一个「fixpoint」外壳 pass，内部手动 `runPipeline`。更重，不推荐。

方案 A 清晰可控，推荐。

---

## 3. 潜在问题（重点）

### 3.1 【最大的坑】失败归因是 per-scope，不是 per-buffer

`failApplyBufferInfo` 的类型是 `std::map<pto::AddressSpace, uint64_t>`（`PTOPlanMemory.h:764`），`RecordOverflowIfAny`（`:1071`）只记录了**哪个 scope 超了、超了多少 bit**，**没有记录是哪些 buffer 摆不下**。

而反馈环要 reclassify 的是**具体 buffer**（「把失败 entry 的 alloc 反查并 reclassify 成 D」）。当前失败路径给不出这个列表。所以要先做：

- 让 PlanMemory 在超限时，把**候选 reclassify buffer 列表**也算出来并写进 attr。但「该挑哪些 buffer 移去 D」本身是个决策问题：
  - 挑最大的？挑生命周期最长的？挑造成峰值的那几个？——需要一个启发式，且挑错会导致「移了还是摆不下」→ 多跑几轮甚至不收敛。
  - `RecordOverflowIfAny` 能拿到 `StorageEntry` 的 `bitsOffset` / `alignedConstBits` / `bufferLifeVec`，理论上能算出「峰值时刻活着的 buffer 集合」，但这是**新代码**，不是白捡的。

**这是把「能不能做」从『小改动』推到『中等改动』的主因。**

### 3.2 Pass 幂等性 / 重入：第一轮已经把 IR 改烂了

反馈环要重跑 `classify → plan-bmu-layout → plan-memory`。但**第一轮 `plan-bmu-layout` 已经把 H/D 的 `memref.alloc` 换成了 `pto.bmu_alloc + pointer_cast`**，第一轮 `plan-memory` 也可能已经把部分 S 的 alloc 换成了 `pointer_cast`。第二轮再跑：

- classify 走 `memref.alloc`，但很多 alloc 已经不在了（变成 pointer_cast）→ 认不到、改不了标。
- plan-bmu-layout 若对已物化的子图再跑，会**重复物化**（再插一遍 bmu_alloc）。

**解决方向**（都要额外工作）：
- **(a) 从干净 IR 重跑**：driver 保存进 BMU 子 pipeline 之前的 module 快照（clone），每轮从快照重跑，只把「累积的 reclassify 决策」带进去。干净、好推理，代价是 clone + 重跑开销。**推荐**。
- **(b) 让每个 pass 幂等**：plan-bmu-layout 跳过已带 `pto.bmu_*` 的子图，classify 认 pointer_cast……复杂且脆弱，不推荐。

方案 (a) 意味着 driver 要持有原始 IR，reclassify 决策要以「能在原始 IR 上重放」的形式累积（比如一组稳定 buffer key）。这又回到 §3.1 的 buffer 标识问题。

### 3.3 收敛性 / 终止

- 必须设**最大迭代数**（设计说「重跑一次」，即上限 2 轮）。否则若 reclassify 后仍失败，会死循环。
- 上限内没成功怎么办？——退回硬 fail（emitError），保证不会静默出错。

### 3.4 【根本性局限】D 不一定能缓解压力：物理 SRAM 是共享的

这是**语义上最需要想清楚的一点**。把 S 类 buffer 改成 D 类，并不是把它挪到「另一块内存」——BMU 的动态段和 S 类的静态尾区**共享同一块物理 SRAM**（UB/L1/...）。反馈环真正省下的是「**静态布局的碎片浪费**」：

- **有用的情形**：容量峰值其实 ≤ 物理容量，但静态规划因为生命周期交叠/碎片/对齐摆不进去。这时把几个 buffer 交给 BMU bump 分配，能消掉碎片，挤下去。✅
- **没用的情形**：容量峰值本身就 > 物理容量（程序真的要的内存超了）。这时移多少去 D 都没用——BMU 段一样超。反馈环只会白跑几轮然后退回 fail。❌

所以反馈环的收益上界是「碎片 + 对齐浪费」那部分，不是「凭空多出内存」。文档措辞（「让容量峰值满足但布局失败的程序仍能跑」）其实已经点到——**峰值满足**是前提。实现时要在 reclassify 前判一下：如果 `failApplyBufferInfo` 报的 `required_bits` 已经超过物理容量（不只是超过静态尾区），直接硬 fail，别进反馈环。

### 3.5 S/BMU 同 scope 共存的容量划分（前置依赖）

反馈环把 S→D，前提是 plan-memory 能正确地在「BMU 段之外的剩余容量」里规划 S 类。这就是设计 §4.7 的 `InitMemSpecsFromModule` 容量替换（`tail_seg3 * slice_size`）+ S 类基址上移到 BMU 段之上。**这块目前没做**（当前测试没有 S 与 BMU 同 scope 混布的用例；pypto 现在要么全 multi-buffer-H、要么全 static）。反馈环依赖它——**没有正确的共存容量划分，反馈环即使跑通也可能把 S 摆到 BMU 段头上，物理重叠**。

所以严格说，反馈环的前置是「§4.7 的 S/BMU 物理共存」先落地，否则反馈环是空中楼阁。

### 3.6 诊断 attr 的清理与可观测性

- `pto.plan_memory_failed_buffers` 每轮要清（否则第二轮的成功 run 还带着上一轮的失败标记）。
- 反馈环触发了几轮、reclassify 了哪些 buffer，最好 `LLVM_DEBUG` / remark 出来，否则线上「为什么这个 buffer 走了 BMU」难排查。

### 3.7 与混编 / 多函数的交互

- attr 是 module 级还是 func 级？多个 func 各自可能失败，需要 per-func 的失败列表，driver 的重跑粒度也要对齐到 func。
- level3（caller 拥有本地内存、跳过 PlanMemory）不进这条路径——要 gate 掉。

---

## 4. 结论与建议

**能做**，方案 A（driver 编排 + 从干净快照重跑 + 最大 2 轮 + 超物理容量直接 fail）是清晰可行的路径。但要先补齐三个硬前提，按依赖顺序：

1. **§4.7 S/BMU 同 scope 容量划分**（前置，独立价值）——没有它反馈环会物理重叠。
2. **PlanMemory 失败归因升级到 per-buffer + 挑选启发式**（§3.1）——当前只有 per-scope。
3. **重入策略：driver 持有原始 IR、从快照重跑**（§3.2）——避免重复物化。

**根本性局限**（§3.4）决定了反馈环的收益上界是「消碎片」，不是「变出内存」；实现时必须在入口用 `required_bits vs 物理容量` 判一刀，避免无谓迭代。

**建议**：第一版**不做**反馈环（与设计 §4.9「第一版可不实现」一致）。理由：
- 收益局限在碎片场景，而 BMU 的 H/D 路径（已实现）本身就能让用户**显式**把大 buffer 交给 BMU，覆盖了大部分「静态摆不下」的实际需求。
- 前置依赖（§4.7 共存）本身是更大、更有普适价值的一块，应先做。
- 反馈环的复杂度（per-buffer 归因 + 重入 + 收敛）与它的边际收益不成比例。

真要做时，最小可用版本 = 「§4.7 共存 + 单轮反馈（上限 1 次重跑）+ 从快照重跑 + 挑最大的 N 个失败 buffer 转 D + 超物理容量直接 fail」。
