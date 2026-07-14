# BMU 特性接入设计文档

本文档描述 PTOAS 如何接入 Ascend 960 (A6) 的 BMU（Buffer Management Unit）硬件特性，
让一部分本地内存走 BMU 运行时分配，与现有的静态内存规划共存。

---

## 1. 背景

本文档描述 PTOAS 如何接入 Ascend 960 (A6) 的 BMU（Buffer Management Unit）硬件特性，让一部分本地内存走 BMU 运行时分配，与现有的静态内存规划共存。

https://github.com/hw-native-sys/PTOAS/pull/685 显式多缓冲特性提供了 `alloc_multi_tile` / `multi_tile_get` / `slot_marker` 等 IR 构造，支持 N-way 静态布局 + slot-aware 同步（常量 slot disjoint、动态 slot `set_flag_dyn/wait_flag_dyn`）；BMU 特性基于此特性进行接入，主要目标在于对多缓冲场景下的同步优化。

---

## 2. BMU 硬件特性

以下是本设计依赖的关键事实，来源 BMU 的硬件设计文档，均已通过 camodel 验证。

### 2.1 基本参数

| Scope | Slice 大小 | 总 Slice 数 | 总容量 |
|---|---|---|---|
| VEC (UB) | 8 KB | 48 | 384 KB |
| MAT (L1) | 8 KB | 64 | 512 KB |
| LEFT (L0A) | 4 KB | 16 | 64 KB |
| RIGHT (L0B) | 4 KB | 16 | 64 KB |
| ACC (L0C) | 16 KB | 16 | 256 KB |
| BIAS (BT) | 256 B | 16 | 4 KB |
| SCALING (FB) | 16 entries | 16 | 512 entries |

### 2.2 硬件行为（已验证）

| 行为 | 描述 |
|---|---|
| Segment 划分 | 每 scope 最多 4 段，由 `BMU_SEGM_*` SPR 配置 4 个 tail（slice 单位，非递减）。`[0, tail_seg3)` 为动态段，`[tail_seg3, total)` 为静态尾区 |
| `BUF ALLOC` 选址 | 段内 bump pointer，连续扫 N 个 free slice；扫到末尾 wrap 回段起点；仍不足则 stall 指定 pipe |
| `BUF ALLOC` stall | stall 指定 pipe 的**后续指令发射**，但**已发射的指令继续 retire** |
| `BUF ALLOC` 唤醒 | 必须凑够**连续 N 个 free slice** 才唤醒（不是任意 free、不是总量 ≥ N） |
| `BUF FREE` | 非阻塞回收；在指定 pipe 上「preceding instructions retired」后执行——即**drain 该 pipe** |
| 跨 pipe 同步 | BMU **不做**跨 pipe 数据同步。alloc/free 只管自己标记的 pipe |
| SPR 重配 | 必须先 free 完 + hscb barrier，才能改 `BMU_SEGM_*`。因此段配置编译期固定，函数内不切换 |
| END 前约束 | 所有 alloc 必须被 free |

**关键推论**：
- **每段同 size 是推荐行为**，不同 size 混在一段会触发碎片化导致 stall。
- 「free drain pipe」→ 函数尾 barrier 可按 free 覆盖的 pipe 做 partial 降级，并且可以减少反向同步。

---

## 3. 设计目标

1. **S/D/H 三模式共存**：S（纯静态）/ D（纯动态）/ H（动态外壳 + 静态内核）三种规划模式，每个 buffer 走最合适的。
2. **S 优先**：S 的 liveness 字节级复用比 BMU 的 slice 量化更省空间（尤其小 tile），S 和 BMU 共享同一物理内存，S 放不下 BMU 也放不下——所以 S 先规划、BMU 拿剩余。
3. **函数尾 barrier 优化**：用 `bmu_free` 的 pipe-drain 语义做 partial 降级。
4. **减少反向同步**：依赖 `bmu_free` 的 pipe-drain 语义减少反向同步。
5. **承接显式多缓冲**：基于`alloc_multi_tile` 对多缓冲的显式声明做 BMU 特性的接入。

---

## 4. 三种规划模式

| 模式 | 谁定地址 | 物化 IR | 适用场景 |
|---|---|---|---|
| **S** 纯静态 | 编译器（liveness 复用） | `pto.pointer_cast %const_addr` | 形状固定、生命周期可静态分析的 buffer |
| **D** 纯动态 | 硬件 BMU | `pto.bmu_alloc → %ssa → pto.pointer_cast %ssa` | 数据依赖分支专用 scratch |
| **H** 混合 | 编译器（组内偏移）+ 硬件（组基址） | `pto.bmu_alloc → %base; pto.pointer_cast(%base + offset)` | 多缓冲环（`alloc_multi_tile`）、共生死数据组 |

### 分类规则

| 优先级 | 条件 | 输出 |
|---|---|---|
| 0 | A2/A3 或无 `pto.uses_bmu` | S |
| 1 | `placement=bmu` 显式指定 | H |
| 5 | `scf.if` 分支局部、生命周期不跨分支 | D |
| 8 | 默认（含 `placement=auto/static`） | S |

当前分类规则为初步方案，待实际做性能分析。

---

## 5. Pass 顺序（S-first pipeline）

```
ptoas (--pto-level=level1/2):
  pto-view-to-memref          alloc_multi_tile → memref.alloc + bind_tile + slot_marker
                              placement 属性传播（auto 不写 attr，bmu/static 写显式 attr）

  pto-classify-buffers        每 alloc 打 plan_class = S/D/H + h_group_id
                              rule 1: placement=bmu → H；rule 5: branch-local → D；其余 → S

  pto-plan-memory             S-first：只规划 S 类（跳过 H/D），liveness 地址复用
                              → 精确 S 峰值 → 算 tail_seg3 → 偏移 S 地址到尾区
                              → 导出 pto.bmu_tail_<scope> func attr
                              AllocToPointerCast 跳过 H/D alloc

  pto-plan-bmu-layout         读 tail_seg3 → 分段 carve-up → emit bmu_config
                              multi-buffer H：whole-ring（一次 alloc + N 地址 + slot_marker 保留）
                              general H 组：base+offset
                              D scratch：单地址 alloc/free

  InsertSync                  bmu ops = sync-transparent（不带 OpPipeInterface）
                              BaseMemInfo SSA-base disjoint
                              尾 barrier 条件化（partial/skip）

  pto-resolve-buffer-select   slot_marker → arith.select（BMU 未改此 pass）

  PTOToEmitC                  bmu_config → set_bmu_segm_*
                              bmu_alloc/free → __ubuf_alloc/free 等 ccec intrinsic
```

关键约束：
- **S-first**：plan-memory 在 plan-bmu-layout **之前**——S 先占、BMU 拿剩余。
- plan-memory 用 `pto.bmu_tail_<scope>` func attr 传递精确的 tail_seg3 给 plan-bmu-layout。
- `AllocToPointerCast` 跳过 `plan_class="H"/"D"` 的 alloc，保留为 `memref.alloc` 供 plan-bmu-layout 消费。
- InsertSync 在所有 layout 完成之后，能看到 bmu_alloc/free 节点。

---

## 6. BMU IR

### 6.1 新增 op

```
pto.bmu_config <scope> tails = [t0, t1, t2, t3]
  // 配置 BMU 段寄存器，函数入口 emit 一次

%base = pto.bmu_alloc <scope>, <pipe>, segm N, count %cnt : index -> i64
  // 从段 N 分配 cnt 个 slice，返回 base 地址
  // pipe = stall 哪条 pipe（不是在哪条 pipe 上执行——BMU 是独立硬件单元）

pto.bmu_free <scope>, <pipe>, base %base, count %cnt : i64, index
  // 回收 slice。无 segm 参数（硬件靠 base 地址定位段）
  // pipe = drain 哪条 pipe
```

三个 op **不带 `OpPipeInterface`**——InsertSync/GSS 看不到它们（sync-transparent by construction）。

### 6.2 属性

| 属性 | 载体 | 含义 |
|---|---|---|
| `pto.uses_bmu` | module | A5 + 启用 BMU（CLI `--pto-uses-bmu`） |
| `pto.plan_class` | memref.alloc | "S" / "D" / "H" |
| `pto.h_group_id` | memref.alloc | H 类的组 id |
| `pto.multi_buffer` | memref.alloc | 多缓冲深度 N |
| `pto.multi_buf_placement` | memref.alloc | "bmu" / "static"（auto 不写） |
| `pto.bmu_tail_<scope>` | func | plan-memory 导出的 tail_seg3（供 plan-bmu-layout 读） |

### 6.3 ccec intrinsic 映射

| PTO op | ccec 调用 | 备注 |
|---|---|---|
| `bmu_config <vec>` | `set_bmu_segm_ub(packed)` | scope → `ub/l1/l0a/l0b/l0c/bt/fb` |
| `bmu_alloc <vec>, <PIPE_MTE2>, segm 0` | `__ubuf_alloc<PIPE_MTE2, 0>(cnt)` | scope → stem `ubuf/cbuf/ca/cb/cc/bt/fbuf`；BIAS 返回 `uint64_t` 非指针 |
| `bmu_free <vec>, <PIPE_MTE3>` | `__ubuf_free<PIPE_MTE3>(base, cnt)` | BIAS 直接传 base（无 reinterpret_cast） |

---

## 7. pto-classify-buffers

对每个本地 `memref.alloc` 打 `pto.plan_class` + `pto.h_group_id`。

- rule 0：A2/A3 或无 `uses_bmu` → no-op（全 S）。
- rule 1：`placement=bmu` → H（每个 multi-buffer alloc 自成一个 H 组）；`auto/static/absent` → S。
- rule 5：`scf.if` 分支局部（`isBranchLocalToIf`），生命周期不跨分支 → D。
- rule 8：默认 → S。

---

## 8. pto-plan-bmu-layout

### 8.1 分类

按 `plan_class` 和其他属性进行分类：
- `mbHAllocs`：H + 有 `multi_buffer` → multi-buffer 环
- `hGroups`：H + 有 `h_group_id` → 通用 H 组
- `dAllocs`：D → 纯动态 scratch

### 8.2 multi-buffer H 的两种物化方式

**per-advance（a-1）**：循环体内每轮一对 `bmu_alloc/free`。每轮 alloc 返回新地址，
bump pointer 的 backpressure（alloc stall 等 free）做隐式复用序列化。

```
scf.for {
  %k = arith.remui %i, %c2
  %base = bmu_alloc <scope>, <pipe>, segm S, count perSlot {pto.multi_buffer = N}
  %cast = pointer_cast(%base)                    // 单地址
  %bt1 = bind_tile %cast ...                     // clone 原 bind
  %sm = slot_marker %bt1[%k]                     // 保留，携带 slot 下标供 sync 分析
  %bt2 = bind_tile %sm ...                       // 原 slot 下游 bind
  tload / tmov / tmatmul ... outs(%bt2)
  bmu_free <scope>, <pipe>, base %base, count perSlot
}
// 原 alloc 和原函数作用域 bind 被删除
```

slot_marker 保留为**同步元数据载体**——它本身是 identity view（ODS 保证 source 和
result 逐字节别名），不做地址选择。`ResolveBufferSelect` 在 sync 之后对单地址
source 做 identity 透传自然消除。`bmu_alloc` 挂 `pto.multi_buffer = N` attr，供
`GetEventIdNum` 追溯读取多缓冲深度。

判定条件（四道关卡全过才走 per-advance）：
1. `collectPingPongShape`：alloc → 单 bind_tile → 全 slot_marker
2. `sms.size() == 1`：恰好一个 slot_marker
3. 在一个循环体内（`enclosingLoopBody`）
4. `slotStaysInIteration`：槽的 SSA 值不跨迭代

**whole-ring（a-2）**：函数作用域一次 `bmu_alloc`，N 个持久地址，`slot_marker` 保留。

```
%base = bmu_alloc <scope>, <pipe>, segm S, count ceil(N*slotBytes/sliceBytes)
%addr0 = arith.addi %base, %c0
%addr1 = arith.addi %base, %c_slotBytes
%cast = pointer_cast(%addr0, %addr1, ..., %addrN-1)   // 多地址
%sm = slot_marker %cast[%k]                            // 保留
bmu_free <scope>, <pipe>, base %base, count ...        // return 前
```

不满足 per-advance 任一关卡 → fallback 到 whole-ring。

### 8.3 分段策略

每个 scope 独立切段。分区 key = `(planClass, slicePerSlot)`。

- 排序：H 先于 D → capSlices 大 → memberCnt 多 → slicePerSlot 大
- ≤4 分区：一一对应 segment 0-3
- \>4 分区：前 3 个独占 seg0/1/2；其余 bin 进 seg3（`binSlice = max(parts[3..].slicePerSlot)`，每个 alloc 向上取整到 binSlice）
- tail 累积：`t0 = segCap[0]`, `t1 = t0 + segCap[1]`, ...
- S-first 可行性：`t3 > bmuAvailable`（= `pto.bmu_tail_<scope>` from plan-memory）→ 报错

### 8.4 物化

| 类别 | 物化方式 |
|---|---|
| mbH per-advance | 每个 slot_marker → `bmu_alloc` + `pointer_cast` + clone bind + `bmu_free`；**slot_marker 也做保留，后续同步使用** |
| mbH whole-ring | 一次 `bmu_alloc` → N 个 `base + k*slotBytes` → 多地址 `pointer_cast`；slot_marker 保留 |
| general H 组 | 挑支配全体的 anchor → 一次 `bmu_alloc` → 每成员 `base + offset` → `pointer_cast` |
| D scratch | 一次 `bmu_alloc` → 单地址 `pointer_cast` |

---

## 9. pto-plan-memory 改动

### 9.1 S-first pipeline

plan-memory 排在 plan-bmu-layout **之前**。H/D alloc 在此阶段仍是 `memref.alloc`（未被物化），
plan-memory 跳过它们：

- `MemLivenessAnalysis` walk：检查 `plan_class` attr，H/D → `advance()`
- `AllocToPointerCast` pattern：`plan_class="H"/"D"` → `return failure()`
- `isSkippableOp`：不再跳过 BMU op（此阶段 IR 里没有 BMU op）

### 9.2 tail_seg3 计算与地址偏移

plan-memory 规划完 S 后，`ApplyBmuStaticTailShift` 从 `buffer2Offsets` 扫出 S 的精确峰值字节：

```
S_slices = ceil(S_peak_bytes / sliceBytes)
tail_seg3 = totalSlices - S_slices
base = tail_seg3 * sliceBytes
```

所有 S 地址 += base（偏移到 `[tail_seg3·slice, total)`），然后把 `tail_seg3` 写到 func attr `pto.bmu_tail_<scope>` 供 plan-bmu-layout 读。

---

## 10. InsertSync 改动

### 10.1 sync-transparent

bmu_alloc/free/config 在 ODS 中不带 `OpPipeInterface` → InsertSync 的 op walk 不匹配
→ 自动 advance → sync 分析完全看不到它们。不需要任何代码改动，是 by construction 的。

### 10.2 SSA-base+offset 地址消歧

BMU whole-ring 的 `pointer_cast` 操作数是 `arith.addi %base, %constK`（而非绝对常量）。
`BaseMemInfo` 新增 `baseSSAs` 字段；`PTOIRTranslator` 识别 `arith.addi` 模式填充；
`isBufferOverlap` 判定：

- 不同非空 SSA base → disjoint（不同 `bmu_alloc` 的返回值，地址不重叠）
- 一个 null 一个非空 → 保守 overlap
- 同 base 或都 null → 按偏移范围比较

### 10.3 函数尾 barrier 条件化

`analyzeBmuFreeCoverage` 收集 `P_used`（有数据操作的 pipe）和 `P_freed`（`bmu_free` 的 pipe）：

- `P_used ⊆ P_freed` → **skip**
- `P_freed ⊊ P_used` → **partial**（只 barrier 剩余 pipe）
- 无 `bmu_free` → **full**（保留 `barrier <PIPE_ALL>`）

### 10.4 per-advance 同步

per-advance 保留 slot_marker 作为同步元数据（§8.2），使 InsertSync 能正常推导
dyn event id 和 slot affine 消歧。具体机制：

**event id 恢复**：`GetEventIdNum` 在 `baseAddresses.size() == 1` 时，追溯
`baseBuffer → bind_tile → slot_marker → bind_tile → pointer_cast → bmu_alloc`，
读 `pto.multi_buffer` attr 得到 N。scope 过滤扩展到所有 on-chip local scope
（VEC/MAT/LEFT/RIGHT/ACC/BIAS/SCALING）。

**正向 sync 也用 dyn event id**：per-advance 每轮新地址，反向 sync 不真正序列化
（不同物理地址无 WAR）→ 正向静态 event id 可能被下一轮覆盖。对 per-advance buffer
（通过 `readMultiBufferFromBmuAlloc` 判定），正向路径也走 `GetEventIdNum` + 
`findSlotMarkerExpr` → dyn event id。whole-ring 和静态路径不受影响。

**slot_marker 生命期**：per-advance 的 slot_marker 在 InsertSync 分析后、由
`ResolveBufferSelect` 对单地址 source 做 identity 透传自然消除。

---

## 11. PTOToEmitC lowering

`bmuIntrinsicInfix` 按 scope 返回 stem（ubuf/cbuf/ca/cb/cc/bt/fbuf），`bmuSegmSetterCallee` 返回 `set_bmu_segm_<hw>`。alloc/free callee 加 `__` 前缀。BIAS scope 特殊处理：返回 `uint64_t`（无 `__bt__` 指针），free 直接传 base（无 reinterpret_cast）。
