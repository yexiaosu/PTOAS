# BMU 分段策略设计（改进版）

> 目标：替换 `pto-plan-bmu-layout` 阶段 1 现在「一个 scope 汇总成一个总需求、全塞
> segment 0、不看 liveness」的保守分段，改成**按 size 类切段 + D/H 隔离 + 预留静态尾区**
> 的策略，充分利用 BMU「每段独立 bump pointer」的硬件特性，消除段内 size 不齐引发的
> fragment stall / 死锁风险。
>
> 本文是 `BMU-integration-design.md` §4.6 阶段 1「segment 划分原则:每段 size 一致」的
> 落地方案，同时实装 §11.1（段共享 ping-pong）、§11.6（D/H 段隔离），并修正 §4.7 中
> 静态尾区容量方向的问题（见 §6）。

---

## 1. 现状与问题

### 1.1 当前实现

`PTOPlanBmuLayout.cpp` 阶段 1（`scopeSlices` 计算，约 294–318 行）对每个 scope
把**所有** D/H alloc 的 slice 需求直接求和，忽略 liveness；随后（324–329 行）emit
一条 `pto.bmu_config`，四个 tail 取同一值：

```
tail_seg0 = tail_seg1 = tail_seg2 = tail_seg3 = 总需求
```

即 **segment 0 = 全部动态需求，segment 1–3 为 0 大小**。所有 `bmu_alloc` 的 `segm`
字段硬编码为 `0`（355 / 400 行）。

### 1.2 三个问题

| # | 问题 | 后果 |
|---|---|---|
| P1 | 所有 size 混在 segment 0 | 段内 alloc 出现多种 slice_count。BMU 唤醒条件是「凑够连续 N slice」（H3 已实测），bump pointer wrap 后可能凑不出连续 N → **fragment stall**；配合消费者 pipe 的 free 被下游阻塞时，构成 BMU.md §7 规则 6 的**死锁**场景 |
| P2 | segment 1–3 恒为 0 大小 | 完全没用到 BMU 的核心特性——每段是**独立 bump-pointer arena**，本可用来做 size 类隔离与 D/H 隔离 |
| P3 | 保守求和、不看 liveness、不显式预留尾区 | `tail_seg3` 随函数体膨胀，可能挤占静态尾区；极端情况 `tail_seg3 > total_slices` 触发 `BmuConfigOp::verify`（PTO.cpp:7810）报错或硬件异常。静态尾区没有被成比例预留 |

### 1.3 相关硬件事实（本设计依赖）

- **段内独立 bump pointer**：`BUF ALLOC.pipe.buf Xd, Xt, #segm` 在指定 segment 内维护
  独立的 free/allocated flag 与 bump pointer（BMU.md §4）。不同 segment 互不干扰——这是
  size 类隔离能成立的根因。
- **被 stall 的 alloc，其唤醒条件 = 段内出现「连续 N 个空闲 slice」**（H3 已实测：既不是
  任意 free 就唤醒，也不是总空闲量 ≥ N 就唤醒，而是要能凑出连续 N）。注意 stall 本身是
  正常的 bump pointer backpressure——复用边界上的 stall 正是逐轮物化用来隐式承担 reuse WAR
  的机制（§4.6 a-1），并非缺陷。真正有害的是**碎片化 stall**：段内 size 不齐时，即使总空闲
  量 ≥ N，wrap 后也可能凑不出连续 N，alloc 因此 stall；若能解锁它的 free 又被下游阻塞，就
  恶化成 BMU.md §7 规则 6 的死锁。所以「每段同 size」不是硬件层面的「必须、否则 UB」，而是
  **本设计为从构造上杜绝碎片化 stall / 死锁而采用的不变式**（配合「段容量为 slot 整数倍」，
  使空闲空间恒为整槽对齐、wrap 恒能凑出连续 N；见 §3、§7）。
- **SPR 重配代价高**（BMU.md §7 规则 7）：改 `BMU_SEGM_*` 须先 free 完该 buffer 全部
  alloc + hscb barrier。由此定义分段的**生效范围 = 单个 kernel（`func.func`）的整个函数体**：
  `pto.bmu_config` 在每个 func 入口 emit 一次（现实现即如此，PTOPlanBmuLayout.cpp:322–329），
  该 func 内所有 `bmu_alloc` 共用这套 tails，到函数 END 前全部 free。因此 carve-up 是对**整个
  函数体求一套解**——「编译期定死」指这一套解不随函数内 phase 变化中途重配（中途重配可行但
  要 free-all + barrier，代价高，不在本设计范围）。分段**不跨 kernel 共享**：SPR 是 core-local
  状态，每个 kernel 进入时用自己入口的 `bmu_config` 重建，彼此独立。
- **静态尾区在 tail_seg3 之上**（BMU.md §3.1）：`reserved = [tail_seg3·slice, buffer_size)`
  留给 `get_imm` 静态 / stack 地址；动态段占 `[0, tail_seg3)`。

---

## 2. 设计目标与约束

**硬约束**（硬件 / verifier 层面，违反即报错或 UB）：

1. 每个 buffer kind 至多 4 段；`0 ≤ tail_seg0 ≤ tail_seg1 ≤ tail_seg2 ≤ tail_seg3 ≤ total_slices`
   （`BmuConfigOp::verify`）。
2. SPR 重配需先 free-all + hscb barrier（BMU.md §7 规则 7）→ 分段编译期固定，不在运行时切换。
3. 静态尾区在 `tail_seg3` 之上：`reserved = [tail_seg3, total_slices)`（BMU.md §3.1）。

**本设计采用的不变式**（非硬件强制，但违反会引入碎片化 stall / 死锁风险，见 §1.3、§7）：

4. 同段同 size + 段容量为 slot 整数倍 → 空闲空间恒整槽对齐、wrap 恒能凑出连续 N，杜绝碎片化 stall。
5. 段容量 ≥ 该段并发峰值 → 排除「alloc 等 free、free 又被下游阻塞」的规则 6 死锁。
6. `tail_seg3 < total_slices`：预留非空静态尾区给纯静态分配。

**设计目标**：

- 按 size 类切段，把不同 slice_count 的 alloc 放进不同段（呼应「4-slice 与 5-slice 分段」）。
- D 类与 H 类隔离，避免 D 的高频 churn 把 H 的 bump pointer 推到 wrap 处（§11.6）。
- 始终预留静态尾区。
- 在「至多 4 段」预算下给出确定的**回退阶梯**（size 类数 > 4 时）。

---

## 3. 核心思想

**分段单元 = partition**，key 为二元组 `(class, sliceCount)`：

- `class ∈ {H, D}`：H 类（multi-buffer / 共生死组，生命周期长、size 固定）与 D 类
  （scratch，生命周期短、size 多样）分开。
- `sliceCount = ceil(slotBytes / bmuSliceBytes(scope))`：单次 alloc 的 slice 数。

**同一 partition 内所有 alloc 的 `slice_count` 相同** → 段内 bump pointer 永远在 uniform
粒度上前进 / wrap → 只要段容量是该粒度的整数倍且 ≥ 并发峰值，就恒能凑出连续 N →
无碎片化 stall、无规则 6 死锁（复用边界上正常的 backpressure stall 仍保留，见 §1.3）。

因为每段是独立 arena，「同 size 一段」与「D/H 隔离」是**同一套机制**：给每个 partition
分配一个独立 segment。静态尾区在 `tail_seg3` 之上，与段数正交，始终预留。

---

## 4. 算法：scope carve-up（阶段 1 重写）

对每个 BMU 管理的 scope（UB/L1/L0A/L0B/L0C/BT/FB）独立执行。

### Step A — 静态需求

```
static_bytes  = S 类 buffer 在各 program point 上的字节峰值   // 复用 MemLivenessAnalysis
static_slices = ceil(static_bytes / slice)
```

### Step B — 枚举 dynamic partition 及其容量

对每个 D/H alloc 归入 partition `p = (class, sliceCount)`，算段容量（slice 单位）：

```
slot_slices(p) = sliceCount                        // 该 partition 的 uniform 粒度
live_peak(p)   = max_over_program_points(#同时活跃于 p 的 alloc)   // liveness-aware
cap(p)         = live_peak(p) * slot_slices(p)      // 段容量，天然是 slot_slices 的整数倍
```

各来源的 `live_peak` 取法：

| 来源 | live_peak |
|---|---|
| multi-buffer 逐轮物化（§4.6 a-1） | `N`（N 深并发，每槽 1 个 `slot_slices`） |
| multi-buffer 整环物化（§4.6 a-2） | 1（整环打包成一次 alloc，`slot_slices = ceil(N·slot/slice)`） |
| 通用 H 组（§4.6 b） | 「同时活的组数」，`slot_slices = group_slices` |
| D 类 scratch | 该 size 类同时活的 alloc 数（liveness-aware；无 liveness 时回退为求和） |

> 相比现状「所有 alloc 求和」，这里按**并发峰值**而非累加算容量，通常显著更省
> （§11.1 段共享 ping-pong 的收益来源）。

### Step C — partition 优先级排序

排序键（降序）：

1. `class == H` 优先于 `D`（H 生命周期长、对 stall 敏感，先保独立段；§11.6）。
2. `cap(p)` 大者优先（大段独占收益高）。
3. `live_peak(p)` 高者优先（churn 频繁，隔离收益高）。

### Step D — 分配 ≤4 段

设排序后 partition 列表为 `P = [p0, p1, ...]`：

- **|P| ≤ 4**：`p_i` 独占 `segment i`，一一对应。
- **|P| > 4**：`p0,p1,p2` 各独占 `seg0/1/2`；`p3..` 合并进 **seg3「bin 段」**：
  - `bin_slice = max_{p in 合并组} slot_slices(p)`（统一粒度）。
  - `bin_cap   = Σ cap(p)` 向上取整到 `bin_slice` 的整数倍。
  - 合并组内每次 alloc 的 `slice_count` **向上取整到 `bin_slice`**，使 seg3 内请求也 uniform
    → 仍无 stall，代价是内部碎片（`bin_slice - sliceCount` 的浪费）。

> bin 段是「4 段装不下所有 size」的确定回退：用内部碎片换取「无外部碎片 stall」，
> 保证正确性优先。

### Step E — 计算 tails 与对齐

```
t0 = cap(seg0)
t1 = t0 + cap(seg1)
t2 = t1 + cap(seg2)
t3 = t2 + cap(seg3)            // 空段令 cap = 0，tail 等于前一个（合法 0 大小段）
```

静态尾区基址 `tail_seg3·slice` 本身已 slice 对齐，起点无对齐损耗；S 类内部布局的对齐 padding
由 §6 反馈环兜底（尾区放不下则 PlanMemory emit diagnostic → carve-up 收缩动态区或 reclassify），
不在 carve-up 里预留额外余量。

### Step F — 可行性与回退阶梯

可行条件：`t3 + static_slices ≤ total_slices`。不满足时按序回退：

1. **N 协商**（§11.4）：缩小 multi-buffer 深度 `N`，`cap` 线性下降；按 `N → 偶数 → 2`
   逐档降，emit user-level diagnostic 说明降档。
2. **加大 bin 合并**：把更多低优先级 partition 并入 seg3，减少段数带来的对齐余量。
3. **D partition 降级 S**：可静态规划的低优先级 D 降级到静态尾区（改回 `plan_class = "S"`，
   走反馈环，见 §6）。
4. **软失败**：仍不可行则 emit `pto.plan_memory_failed_buffers` diagnostic（不 `signalPassFailure`），
   交 `pto-classify-buffers` 反馈环 reclassify（§4.5 反馈环）。

### Step G — emit config + 记录映射

emit `pto.bmu_config(scope, t0, t1, t2, t3)`；记录 `partition → segm` 映射，供阶段 3 选段。

### 复杂度

partition 枚举 O(#alloc)，排序 O(#partition·log)，段分配 O(4)。满足 pypto 侧
O(N log N) 约定（PTOAS 无硬性要求，但保持一致）。

---

## 5. 阶段 3 物化改动（选段）

现状：所有 `BmuAllocOp` 硬编码 `segm = 0`（PTOPlanBmuLayout.cpp:355 / 400）。

改动：物化每个 alloc 时按其 partition 查 Step G 的映射，填入正确 `segm`；若落在 bin 段，
`slice_count` 向上取整到 `bin_slice`。`bmu_free` 的 `segm` 语义由 base 地址决定，无需改；
alloc/free 的 pipe 仍走 `PTOBmu.h` 的 `bmuAllocPipeFor` / `bmuFreePipeFor`。

**示例**（呼应需求中的例子）：4-slice alloc 与 5-slice alloc 属不同 partition →
分别落在 `seg_A` 与 `seg_B`，各自 uniform，互不推动对方的 bump pointer。

---

## 6. 静态尾区与 PlanMemory 的一致性（正确性修复）

BMU.md §3.1 明确：静态尾区 = `[tail_seg3·slice, buffer_size)`，动态段占 `[0, tail_seg3)`。
因此 S 类可用容量应为：

```
S_capacity   = (total_slices - tail_seg3) * slice_bytes
S_base_offset = tail_seg3 * slice_bytes        // S 类地址须从尾区起算
```

`BMU-integration-design.md` §4.7 早期写的是「可用容量 = tail_seg3 · slice_size」，方向相反
（那是动态段容量）。若照此实现，S 类会被规划进 `[0, tail_seg3)`，与 BMU 动态段地址空间
**重叠**。本设计据 BMU.md §3.1 采用上式（`S_capacity` / `S_base_offset`）。

**实现现状**：

- ✅ **`pto-plan-memory` 侧已落地**（早期"只 `isSkippableOp` 跳过 BMU op、未做容量/基址收缩"
  的隐患已消除）：
  - `MemPlan::InitMemSpecsFromModule`（`PlanMemory.cpp:2323–2338`）walk 每个 `pto.bmu_config`，
    以 `baseBytes = tail_seg3 · slice` 把该 scope 的可用容量覆盖为
    `S_capacity = bmuScopeTotalBytes(scope) − baseBytes`（`setScopeSpaceBits`），并把
    `baseBytes`（= `S_base_offset`，多份 config 取最大）记入 `bmuStaticTailBaseBytes`。无
    `bmu_config` 的 scope 保持 arch spec 不变（非 BMU func 零影响）。
  - `MemPlan::ApplyBmuStaticTailShift()`（`PlanMemory.cpp:2342–2358`，在 `runOnOperation` 内
    `:2506` 调用）把该 scope 每个 S buffer 的 offset 整体上移 `S_base_offset`，使静态分配落在
    尾区 `[tail_seg3·slice, total)`，不与 BMU 动态段重叠。
- ⏳ **`pto-classify-buffers` 侧仍待办**：`scope_static_quota` 现仍用 `bmuScopeTotalBytes`
  （整块 scope，`PTOClassifyBuffers.cpp:170`），尚未改用 `S_capacity`。因此 classify 会**高估**
  静态容量、可能 under-promote（本该 H 的判成 S）。这属于下面反馈环的一环，随反馈环一起落地。

**反馈环**：classify 决定 H/S 需要 `S_capacity`，而 `S_capacity` 依赖 carve-up 结果 →
存在环。解法：classify 首遍用**保守 split**（如按历史比例先预留一半给动态），carve-up
产出真实 tails 后，若 `EmitPlanMemoryFailureInfo` 触发则 reclassify 重跑一次即收敛
（§4.5 反馈环已有骨架）。

---

## 7. 安全性论证

**无 fragment stall / 无规则 6 死锁**：每段 uniform size 且 `cap` 为 `slot_slices` 整数倍
→ bump pointer wrap 后总能凑出连续 `slot_slices`；`cap ≥ live_peak·slot_slices` 保证并发
峰值内不会「alloc 等 free、free 等下游」。bin 段通过向上取整到 `bin_slice` 保持 uniform。

**D 不干扰 H**：D 与 H 落在不同 segment（独立 arena），D 的高频 alloc/free 不推动 H 段的
bump pointer（§11.6）。

**静态尾区不被侵占**：Step F 的可行性条件强制 `t3 + static_slices ≤ total_slices`。

**次序约束不变**：逐轮物化下 `set_flag_dyn` 仍须 hoist 到同 pipe `bmu_alloc` 之前
（§4.8c）；分段策略不改变该约束，只改变 alloc 落在哪个 segment。

---

## 8. 改动清单

| 位置 | 改动 |
|---|---|
| `PTOBmu.h` | 新增 helper：partition key 计算、`bin_slice` / bin 归并、`S_capacity` / `S_base_offset` |
| `PTOPlanBmuLayout.cpp` 阶段 1 | 用 §4 carve-up 替换 `scopeSlices` 求和；emit 差异化 tails；产出 `partition → segm` 映射 |
| `PTOPlanBmuLayout.cpp` 阶段 3 | `BmuAllocOp` 的 `segm` 按映射填；bin 段 `slice_count` 取整 |
| `PTOClassifyBuffers.cpp` | `scope_static_quota` 改用 `S_capacity` |
| `PTOPlanMemory.cpp` | `InitMemSpecsFromModule` 容量收缩到尾区 + 基址偏移 `S_base_offset` |
| `BmuConfigOp::verify` | 无需改（已支持任意合法 tails） |

**新增 lit 测试**（建议）：

| 用例 | 覆盖 |
|---|---|
| `bmu_segment_size_class_split.pto` | 多 size 类落不同段，config tails 差异化 |
| `bmu_segment_dh_isolation.pto` | 同 size 的 D 与 H 分落不同段 |
| `bmu_segment_bin_overflow.pto` | >4 partition 触发 seg3 bin 归并 + 取整 |
| `bmu_segment_static_tail_reserve.pto` | `tail_seg3 < total_slices`，S 类规划进尾区不与动态段重叠 |
| `bmu_segment_infeasible_fallback.pto` | 容量不足触发 N 协商 / 软失败 |

---

## 9. 示例（UB / VEC，48 slice、8KB/slice）

输入：S 静态峰值 64KB（8 slice）；H multi-buffer `N=2`、slot 24KB=3 slice（逐轮物化）；
D scratch A 4 slice、峰值并发 2；D scratch B 5 slice、峰值并发 1。

partition 与容量：

| partition | slot_slices | live_peak | cap |
|---|---|---|---|
| `(H, 3)` | 3 | 2 | 6 |
| `(D, 4)` | 4 | 2 | 8 |
| `(D, 5)` | 5 | 1 | 5 |

3 个 partition ≤ 4，各独占一段：

```
seg0 = (H,3): [0,6)     t0 = 6
seg1 = (D,4): [6,14)    t1 = 14
seg2 = (D,5): [14,19)   t2 = 19
seg3 = 空:              t3 = 19
静态尾区 = [19,48) = 29 slice ≥ 8   ✓

pto.bmu_config <vec>, tail_seg0=6, tail_seg1=14, tail_seg2=19, tail_seg3=19
```

物化：`(H,3)` → `bmu_alloc segm=0 count=3`；`(D,4)` → `segm=1 count=4`；
`(D,5)` → `segm=2 count=5`。

对比现状：`scopeSlices` 求和后 emit `bmu_config(19,19,19,19)`，三种 size 混在 seg0，
4/5 slice 请求在 wrap 后可能凑不出连续 N → stall；且未显式预留尾区。

---

## 10. 与主设计文档的关系

本设计落地 `BMU-integration-design.md` §4.6 阶段 1「每段 size 一致」，实装 §11.1（段共享
ping-pong，通过 live_peak 而非累加算容量）与 §11.6（D/H 段隔离），并据 BMU.md §3.1 修正
§4.7 的静态尾区容量方向。§4.6 阶段 3 的物化形态、pipe 选择、逐轮/整环判定均不变，本设计
只改「alloc 落在哪个 segment」与「config 的四个 tail 如何算」。
