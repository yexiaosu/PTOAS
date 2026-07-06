docs/designs/ptoas-auto-sync-design.md# BMU 集成到 pypto / PTOAS 的设计方案

> 目标：在 PTO 工具链中接入 A5 的 BMU（Buffer Management Unit）硬件特性，让一部分原本由 PTOAS `pto-plan-memory` 静态规划的本地内存改走 BMU 运行时分配；同时利用 BMU `BUF FREE` 的 pipe-drain 语义优化 InsertSync 的函数尾 barrier，减少函数间 bubble。

## 1. 背景

### 1.1 当前 PTOAS 内存规划

`pto-plan-memory` 是 ModuleOp pass，对每个 func 内的 `memref.alloc` / `memref_ext.alloc_workspace` 做静态地址规划：

- 三步算法：`MemLivenessAnalysis` 收生命周期 / gen-kill / alias / inplace pair / 语义冲突 → `MemPlan` 多级 speculative 分配（`SPEC_LEVEL_0/1/2`，含失败回滚）→ `AllocToPointerCast` 把 `memref.alloc` 改写为带常量 `addr` 的 `pto.pointer_cast`。
- Multi-buffer / ping-pong：被动接受 `buffer2MultiNum > 1`（决策来自 pypto），靠 `ExpandMultiBufferStorageEntry` + `relationPongEntry` 把 ping/pong 两份紧贴排布。
- 对齐：256B；硬件容量按 scope（VEC/MAT/L0A/L0B/L0C/BIAS/SCALING）从 module attr 读。
- **Shape 假定**：所有 `memref.alloc` 必须 shape 全静态；任一维 `ShapedType::kDynamic` 在 `getStaticTotalSize`（`Utils.cpp:195`）返回 `nullopt`，进而触发 `MemLivenessAnalysis::GetBufferInfo`（`PTOPlanMemory.cpp:847`）的 `llvm::report_fatal_error`——整个编译进程 abort。这是 pypto 端 `InitMemRef` 强制 `TileType.shape` 静态（`init_memref.cpp:99-102`）的下游表现。pypto 把运行时形状变化压到 `TileView.valid_shape` SSA 字段，物理 alloc 按上界。该约束支撑了 codegen / fusion / sync 分析 / layout 优化等一整套静态优化体系，不在本方案改动范围。

**来源**：

| 项 | 位置 |
|---|---|
| Pass ODS | `PTOAS/include/PTO/Transforms/Passes.td:157`（`def PlanMemory`） |
| Pass 主体 | `PTOAS/lib/PTO/Transforms/PTOPlanMemory.cpp`（`runOnOperation` 入口 `:2282`，`MemPlan::plan` `:1022`） |
| MemLivenessAnalysis | `PTOAS/lib/PTO/Transforms/PTOPlanMemory.h`、`PTOPlanMemory.cpp:363` |
| SPEC_LEVEL_0/1/2 | `PTOAS/lib/PTO/Transforms/PTOPlanMemory.h:57-64`（含注释） |
| AllocToPointerCast | `PTOAS/lib/PTO/Transforms/AllocToPointerCast.cpp:106` |
| ping-pong 排布 | `PTOAS/lib/PTO/Transforms/PTOPlanMemory.h`（`relationPongEntry` / `multiBufferNum`）+ `PTOPlanMemory.cpp` 的 `ExpandMultiBufferStorageEntry` / `PlanRelationPongEntryAddress` |
| pypto 端 ping-pong 决策 | `pypto/src/ir/transforms/auto_tile_matmul_l0_pass.cpp:481-483`（`double_buffer_a/b/c`） |
| 对齐 + 容量常量 | `PTOAS/lib/PTO/Transforms/PTOPlanMemory.cpp` 文件顶部常量段（`kLocalMemAlignmentBytes`、`kA5VecLocalMemBits` 等，约 `:54` 起） |
| 用户视角概念 | `PTOAS/docs/PTO_IR_manual.md`（§3.1 AddressSpace） |
| pipeline 中的位置 | `PTOAS/docs/designs/ptoas-tileop-expand-design.md:412` |

### 1.2 当前 PTOAS 自动同步

四个互斥 pass，同一时刻最多启用一个：`PTOInsertSync` / `PTOGraphSyncSolver` / `PTOBufidSync` / `PTOInjectBarrierAllSync`。可用同步原语：

| 类别 | 原语 | ODS 位置（`PTOAS/include/PTO/IR/PTOOps.td`） |
|---|---|---|
| 数据依赖事件 | `pto.set_flag` / `pto.wait_flag` | `:2473` / `:2488` |
| 同上 · 动态 event_id | `pto.set_flag_dyn` / `pto.wait_flag_dyn` | `:2503` / `:2516` |
| 高层事件（op type 端点） | `pto.record_event` / `pto.wait_event` | `:1487` / `:1504` |
| 截断 / 全清 | `pto.barrier` | `:2631` |
| 高层 barrier（SyncOpType） | `pto.barrier_sync` | `:1522` |
| 资源 token（A5 专属） | `pto.get_buf` / `pto.rls_buf` | `:2536` / `:2568` |
| 跨核点对点 | `pto.sync.set` / `pto.sync.wait` | `:2594` / `:2614` |
| 跨核全员 barrier | `pto.syncall` | `:2652` |
| 直通 TSYNC | `pto.tsync` | `:2637` |

**Pass 来源**：四个 pass 的 ODS 均在 `PTOAS/include/PTO/Transforms/Passes.td`：`def PTOInsertSync`、`def PTOInjectBarrierAllSync`、`def PTOBufidSync`、`def PTOGraphSyncSolver`。互斥关系在 `PTOAS/tools/ptoas/ptoas.cpp` 的 CLI flag 校验段（`--enable-insert-sync` / `--enable-graph-sync-solver` / `--enable-bufid_sync` / `--enable-inject-barrier-all-sync`）。

高层 op 降级路径：`pto.record_event` / `pto.wait_event` 由 `pto-lowering-sync-to-pipe` pass（`PTOAS/lib/PTO/Transforms/LoweringSyncToPipe.cpp`，ODS `Passes.td:183`）按 op-pipe 映射降级为 `pto.set_flag` / `pto.wait_flag`。

### 1.3 关键观察：函数尾 PIPE_ALL barrier

`InsertSyncAnalysis::InsertLastPipeAll()`（`InsertSyncAnalysis.cpp:694`）在每个 `func.func` 末尾**无条件**插入一条 `pto.barrier <PIPE_ALL>`，由 `SyncCodegen::AppendAutoSyncTailBarrierIfNeeded`（`SyncCodegen.cpp:314`）挪到每个 `func::ReturnOp` 之前。lower 后是 `pipe_barrier(PIPE_ALL)`——硬件级"等所有 pipe 全部 retire"，是函数间 bubble 的核心来源。

### 1.4 当前 multi-buffer 显式表达 PR baseline

已合入 PR `claude/serene-bhaskara-f0221e` 在 PTOAS 中实装了「显式 multi-buffer + 自动同步」：

- 前端：`!pto.multi_tile_buf<S, count=N>` 类型；`pto.alloc_multi_tile`、`pto.multi_tile_get %mb[%k]` 两个 op
- Lowering：`PTOViewToMemref` 把 `alloc_multi_tile` 降为 `memref.alloc {pto.multi_buffer=N}` + 内部 `pto.slot_marker`
- 规划：`PTOPlanMemory` 走 N-way 静态布局（StorageEntry / ExpandMultiBufferStorageEntry / multi-address `pto.pointer_cast`）
- 同步：InsertSync / GraphSyncSolver slot-aware；`BaseMemInfo.baseAddresses` 按 slot 收窄；常量 slot 自动 disjoint；动态 slot 通过 `set_flag_dyn / wait_flag_dyn` + N event id 实现；新增 `SlotAffineAnalysis` 三态比较（kEqual / kDisjoint / kUnknown）
- 新 pass：`PTOResolveBufferSelect` 排在 sync 之后，把 `slot_marker` 物化为单地址 `pto.pointer_cast`（常量 slot）或 `arith.select` 链（动态 slot）
- 控制流：`alloc_multi_tile` 不依赖 `scf.for`，支持完全 unroll、`scf.while`、跨 block——这指的是**表达能力**，alloc 与 use 可分布在任意控制流形态

详见 `PTOAS/docs/designs/ptoas-multi-buffer-explicit-design.md`。

**与本 BMU 设计的关系**：multi-buffer alloc 是 §4.2 H 模式最自然的输入——一个 `alloc_multi_tile` 本身就是一个共生死组，**不需要任何额外的组识别**。这与 BMU 文档原计划的「`scf.for` iter_args / yield alias 链」H 组识别是**两件独立的事**：

| H 组来源 | 识别方式 | 范围 |
|---|---|---|
| **显式** `alloc_multi_tile` | 组即 IR 节点 | 本 PR 已 ready，BMU 集成首选 |
| **隐式** 隔 N 份 `memref.alloc` + scf.for iter_args | 图分析（alias + 共生死推断） | BMU 主线 phase 4，与 multi-buffer PR 解耦 |

后续章节默认假设 multi-buffer PR 已合入，BMU 工作在其之上叠加。

## 2. BMU 硬件特性总结

详见 `BMU.md`，这里抽出本设计依赖的关键事实。

### 2.1 已确认

| 项 | 行为 |
|---|---|
| Slice 粒度 | UB 8KB、L1 8KB、L0A/B 4KB、L0C 16KB、FB 32 entries、BT 256B |
| Segment 划分 | 每 buffer 最多 4 个 segment，由 `BMU_SEGM_*` SPR 配置 4 个 tail（slice 单位）。允许某 segment tail 等于前一个 segment tail，对应 0 大小 segment（合法）。`tail_seg3 < total_slices` 部分留作静态 / stack 区、 |
| `BUF ALLOC` 选址 | 每 slice 1 bit flag（0=free / 1=allocated），segment 内有 bump pointer。从 bump pointer 起向后扫连续 n 个 free slice，找到即返回 base、advance bump、置 flag；扫到末尾不够则 wrap 回 segment 起点再扫；仍不足则按 `.pipe` stall |
| `BUF FREE` 行为 | 回收 base+size 范围；**非阻塞**；在 `.pipe` 上"preceding instructions retired"后执行（H1 已验证：camodel 上 `instr_log_cycle(BUF FREE)` ≥ `instr_popped_cycle(同 pipe 前一条 micro-instr)`） |
| 跨 pipe 同步 | BMU **不**做跨 pipe 数据同步。pipe X alloc + pipe Y free 时，Y 的 free 只等 Y 自己之前的指令；X 上的数据移动必须用 set/wait_flag 单独保护 |
| SPR 重配置 | 必须先 free 完该 buffer 的所有 alloc + 加 hscb barrier，才能改 `BMU_SEGM_*` |
| END 前约束 | 所有 alloc 必须被 free，否则未定义行为 |
| 死锁规则 | 同一段地址在两次 alloc 之间必须有 free（程序序） |

### 2.2 待确认（需 demo 验证或硬件团队确认）

| # | 问题 | 状态 | 影响 |
|---|---|---|---|
| H1 | `BUF FREE` 的"preceding instructions retired"是 pipe 上**所有**先前指令，还是仅 free 数据依赖的那些？ | **已验证：是 pipe 上所有先前指令**（camodel 跑 tbuf baseline，`instr_log_cycle(BUF FREE) ≥ instr_popped_cycle(同 pipe 前一条 micro-instr)`） | 决定能否替代函数尾 PIPE_ALL barrier。结论：可以作为同 pipe drain，§4.8(d) 尾 barrier 优化前提成立 |
| H2 | `BUF ALLOC` stall 时，该 pipe 上**之前已发射**的指令是否继续推进？ | **已验证：是**（`bmu_h2_alloc_stall` trace 顺序为 alloc_a → TLOAD → V 端 TADD → V 端 free → alloc_b retire；TLOAD 已发射并在 alloc_b stall 期间继续 retire，pipe 没被冻结） | 设计文档不需要把 BMU stall 当作"冻结整个 pipe"对待，可放心在 stall alloc 前发射其它 same-pipe 工作 |
| H3 | `BUF ALLOC` 唤醒条件：等任意 free、same-segment free、还是能凑出 N 连续 slice 的 free？ | **已验证：凑够连续 N 才唤醒**（`bmu_h3_wake_condition` 中 alloc_e 紧跟 free B 后唤醒，free C / free A 都没唤醒；与硬件方原本说法一致） | §4.6 `pto-plan-bmu-layout` 「同 size 类归段」是**硬约束**，不是 best-effort——段内 size 不齐会真触发 stall。已知坑里的"size 不齐回退到 S"必须实装 |
| H4 | alloc/free 指令本身的 cycle 成本（相对 `set_flag/wait_flag`） | 未测 | 决定热路径上是否值得用 |

**结论**：H1 / H2 / H3 三条核心假设全部成立，符合硬件方与 `BMU.md` 文档的描述。Phase 1 / 2 / 3 可以放心实装，无方案级修正。H4 是性能调优阶段的问题，不影响正确性。

## 3. 设计目标

按收益排序：

1. **三种规划模式共存**：把内存规划重新组织成 **S 静态 / D 动态 / H 混合（动态外壳 + 静态内核）** 三类（详见 §4.2）。让每个 buffer 走最合适的模式，而不是非黑即白。
2. **接住静态规划处理不了的场景**：数据依赖控制流、容量峰值满足但静态布局失败的程序——这些走 D / H 模式；S 模式不再硬 fail。
   *（注：dynamic shape 本身不是 BMU 的有效目标——pypto 的 `InitMemRef` 强制 `TileType.shape` 静态，且这套设计支撑了从 codegen 到 sync 的多层优化；改成 dynamic alloc 还需要硬件确认 valid 区域外内存访问行为，超出本方案范围。dynamic 形状的真实可变性由 pypto 的 `TileView.valid_shape` 字段承载，物理 alloc 按上界——这部分不动。）*
3. **优化函数尾 barrier**：用 `BUF FREE` 序列的 drain 语义替代 / 降级 `pipe_barrier(PIPE_ALL)`（H1 已验证），减少函数间 bubble。
4. **保持热路径性能**：tile fusion 内层 tile、reduction 累加器等可静态规划的继续走 S 模式，零 runtime 开销；不强行 BMU 化。
5. **承接 multi-buffer PR 的 H 模式自然入口**：`alloc_multi_tile`（§1.4）显式表达 + 共生死语义本身就是 H 组，BMU 集成首选这条路径。multi-buffer 的同步推导（dyn flag / affine drop）与 BMU 路径正交，零冲突复用。

非目标：

- 不追求"全部 BMU 化"。S + D + H 三模式混合是常态。
- 不替换 InsertSync 的跨 pipe 数据依赖同步。BMU 与 set/wait_flag 正交。
- **不解决跨函数 buffer 复用**。BMU §7 规则 8 要求 END 前全 free，函数边界 BMU 状态强制 clean——这是函数尾 barrier 优化的硬基础，但同时意味着跨函数复用不在本方案范围。

## 4. 设计方案

### 4.1 整体分层

> **范围约束**：本方案**只动 PTOAS**，pypto 前端不改。前端按现有路径产出标准 `memref.alloc`（无 `addr` 属性）；编译流程使用 `--pto-level=level1` 或 `level2`（让 PTOAS 跑 `pto-plan-memory`），不走 level3。
>
> pypto 的 ping-pong / double-buffer 决策（`auto_tile_matmul_l0_pass` 设的 `double_buffer_a/b/c`、`LowerPipelineLoops` 复制循环体、`multibuffer-unroll-num` attr）与 `--pto-level` 完全独立——非 level3 下仍在 PTO IR 里体现为两份 `memref.alloc` + loop attr。PTOAS 拿到这些信号后决定 ping-pong 走 S / H 哪种模式，pypto 那边不需要任何改动。

```
pypto (前端 / IR)
  └── memref.alloc 系列  (无 addr，含 pypto 已经做完的 ping-pong / pipeline 标注)
                ↓
PTOAS
  ├── 新 pass: pto-classify-buffers       (给每个 alloc 打 plan_class = S/D/H + group_id)
  ├── 新 pass: pto-plan-bmu-layout        (scope 切段 + emit bmu_config/alloc/free + H 组组内静态偏移)
  ├── pto-plan-memory                     (改：只处理 S 类与 H 组的"组内布局";D 与 H 组的基址已物化)
  ├── InsertSync                          (改:感知 bmu_alloc/free + 尾 barrier 条件化)
  └── PTOToEmitC                          (新增 bmu_config / bmu_alloc / bmu_free 的 lower)
```

### 4.2 三种规划模式

每个 buffer 按其特性归到一种模式：

| 模式 | 谁定地址 | 何时定 | 物化 IR | 适用场景 |
|---|---|---|---|---|
| **S** 纯静态 | 编译器 | 编译期 | `pto.pointer_cast %const_addr` | 形状固定、生命周期可静态分析、与同 scope 同活集合不撞容量；inner-loop tile、reduction 累加器、tile fusion 中间链 |
| **D** 纯动态 | 硬件 BMU | 运行时（bump pointer 选地址） | `pto.bmu_alloc → %ssa_addr → pto.pointer_cast %ssa_addr` | 数据依赖分支专用 scratch、容量峰值满足但静态布局失败的 buffer |
| **H** 动态外壳 + 静态内核 | 编译器（组内相对偏移）+ 硬件（组基址） | 编译期偏移 + 运行时基址 | `pto.bmu_alloc → %base; sub = pto.pointer_cast(%base + const_offset)` | 一组 buffer 共生死 + 组内相对布局有用 + 组的外部并发是动态：matmul L1 数据组（A/B/scale/bias）、phase 切换的 workspace、pypto ping-pong pair（phase 2+） |

**模式选择判据**

S 模式适用条件（三件必须同时满足）：
1. size 编译期常量
2. lifetime 编译期可解
3. 与同 scope 所有 buffer 的并发集合在每个 program point 上的字节总和 ≤ scope 容量

任一条件失败 → 退到 D 模式。

H 模式适用条件（叠加在上面之上，需要识别"组"）：
1. 一组 buffer **共享生命周期**（同时生、同时死，由 `scf.for` iter_args / yield 揭示，或 pypto multi-buffer attr 揭示）
2. 组内 buffer 的**相对布局有意义**（contiguous 访问、共享对齐）
3. 组的**外部并发是动态**（ping-pong loop、phase 切换）

H 模式是 S + D 长处的并集：外层 ping-pong 用 BMU bump pointer 隐式实现（少同步指令），内层布局静态（保留 inplace / 紧贴排布的优化）。

**与前端输入形态的对应**

| 输入形态 | 对应模式（第一版默认） | 备注 |
|---|---|---|
| `alloc_multi_tile`（显式 multi-buffer，§1.4） | `placement=auto` → **S**（S 优先：静态 liveness 复用更省空间）；`placement=bmu` → **H** | 组即 IR 节点，无需识别；S 路径保住 const-addr disjoint sync 等优化 |
| `double_buffer_a/b/c = true` + `LowerPipelineLoops` 复制（pypto 旧路径） | **S**（两份 alloc 用现有 `relationPongEntry` 排布） | 第一版保守；phase 4 才尝试用 scf.for iter_args 识别为 H |
| `double_buffer_*` 但 S 失败（容量够但布局失败） | **H** | fallback：把两份 alloc 视为一组 H |
| 容量峰值满足但 PlanMemory 静态布局失败 | **D** | 走 BMU runtime alloc 绕开布局求解 |
| 普通 alloc | **S** | 不变 |

### 4.3 规划 pipeline

把内存规划重新组织成 6 步（与 PTOAS 现有 PlanMemory 内部步骤错开理解）：

```
(1) Buffer Classification          pto-classify-buffers
    每个 alloc 打标签 S / D / H
    H 类还要识别"组"和组内位置
                ↓
(2) Co-living Analysis             (复用 MemLivenessAnalysis)
    分别算各模式下"同 scope 同时活的 buffer 集合"
    S: 按 program point 算 byte peak
    D: 按 segment 候选算 size 分布
    H: 按 group 算 group-level peak（多少个组同时活）
                ↓
(3) Scope Carve-up                 pto-plan-bmu-layout 第 1 阶段
    决定每个 scope (UB/L1/L0A/...) 切多少静态尾区、多少 BMU segment
    segment 划分原则:每段 size 一致（避免 H3 验证的 fragment stall 行为）
    emit pto.bmu_config 在 kernel 入口
                ↓
(4) Static-Class Planning          pto-plan-memory（改）+ pto-plan-bmu-layout 第 2 阶段
    S 类:在静态尾区跑现 PlanMemory（StorageEntry / MergeInplace / MultiSpecPlan）
    H 类:仅算"组内相对偏移表",不绑定基址
                ↓
(5) Dynamic-Class Placement        pto-plan-bmu-layout 第 3 阶段
    D 类:每个 alloc emit pto.bmu_alloc/free,塞到对应 segment
    H 类:通用组 / multi-buffer 回退整环 = 每组一对 pto.bmu_alloc/free（group-level）+ base+offset；
         显式 multi-buffer 首选逐轮 = 循环内每轮一对 alloc/free（§4.6 (a-1)）
                ↓
(6) Address Materialization        现有 AllocToPointerCast + 新逻辑
    S: emit pto.pointer_cast %const
    D: emit pto.pointer_cast %ssa  （ssa 来自 bmu_alloc）
    H: emit pto.pointer_cast (%base + const_offset)
                ↓
[IR with all addresses resolved]
```

与现 PTOAS 的对应：

| 现 PlanMemory 组件 | 在新 pipeline 中的位置 |
|---|---|
| `MemLivenessAnalysis` | (1) 与 (2) 复用，作为 classification 与 co-living 分析的数据源 |
| `StorageEntry` / `MergeSameScopeSE` / `MergeInplaceSE` | (4) 仅 S 类使用 |
| `relationPongEntry` / `ExpandMultiBufferStorageEntry` | (4) 仅当 pypto-ping-pong-pair 保留为 S 时使用；走 H 模式时不需要 |
| `MultiSpecPlan` / `SPEC_LEVEL_0/1/2` / `ApplyFailStrategy` | (4) 仅 S 类使用 |
| `EmitPlanMemoryFailureInfo` | 不再 emit 用户级 fail，而是把失败 buffer 反馈给 (1) reclassify 成 D（见 §4.5 反馈环） |
| `AllocToPointerCast` | (6) 仅 S 类生效；D / H 路径走另一条 pattern |

### 4.4 IR 层面

新增 3 个 PTO Dialect op（放在 `PTOOps.td` 的 "Resource & Sync" 段）：

```td
def BmuConfigOp : PTO_Op<"bmu_config"> {
  let summary = "Configure BMU segment register for a buffer kind";
  let arguments = (ins
    PTO_AddressSpaceAttr:$buffer,           // VEC/MAT/LEFT/RIGHT/ACC/BIAS
    I32Attr:$tail_seg0, I32Attr:$tail_seg1,
    I32Attr:$tail_seg2, I32Attr:$tail_seg3  // slice units
  );
}

def BmuAllocOp : PTO_Op<"bmu_alloc"> {
  let summary = "Allocate slices from a BMU segment on a given pipe";
  let arguments = (ins
    PTO_PipeAttr:$pipe,
    PTO_AddressSpaceAttr:$buffer,
    I32Attr:$segm,                          // 0..3
    Index:$slice_count
  );
  let results = (outs I64:$base_addr);
}

def BmuFreeOp : PTO_Op<"bmu_free"> {
  let summary = "Free a previously allocated BMU region on a given pipe";
  let arguments = (ins
    PTO_PipeAttr:$pipe,
    PTO_AddressSpaceAttr:$buffer,
    I64:$base_addr,
    Index:$slice_count
  );
}
```

约束 / 校验（verifier 内）：

- `slice_count > 0`，与 `BMU.md` §7 规则 7 一致。
- 同一 `(buffer, base_addr)` 配对的 alloc 与 free 必须在同函数内程序序成对出现。
- `BmuConfigOp` 在每个 buffer kind 上最多出现一次；位置必须在所有 `BmuAllocOp` 之前（kernel 入口）。
- 函数 return 前所有 alloc 必须有对应 free（与硬件 §7 规则 8 一致）。

**Lowering 路径**：非 level3 流程下，`pto.alloc_tile addr=` 不能用（PTOAS main 入口会拒绝）。动态 alloc 的结果走 `pto.pointer_cast`：

```
memref.alloc (plan_class = "D" or "H")
  → pto-plan-bmu-layout 改写为:
      %addr = pto.bmu_alloc {pipe=..., buffer=..., segm=...} : i64
      %tile = pto.pointer_cast %addr : memref<...>
  → PTOToEmitC: Tile<...> + TASSIGN(tile, %addr)
```

`pto.pointer_cast` 本来就接受 dynamic SSA 地址（见 `AllocToPointerCast.cpp:114` 的 ValueRange addrs），不需要新增 codegen。`PTOAllocTileToEmitC` 完全不涉及。

**Multi-buffer 走 H 路径时的 IR 形态**：multi-address `pto.pointer_cast` 的 operand 从 N 个常量地址变为 N 个 `arith.addi %base, %const_offsetK`（base 是 `bmu_alloc` 的 SSA 结果）。

```
原 IR (PTOViewToMemref 产出):
  %a = memref.alloc() { pto.multi_buffer = 2 : i32,
                        pto.plan_class   = "H",
                        pto.h_group_id   = #7 } : memref<16x16xf16, vec>
  %s_mem = pto.slot_marker %a [%k] : memref<...> -> memref<...>

pto-plan-bmu-layout 改写为:
  %size = arith.constant <ceil(N * sizeof(slotType) / slice_bytes)> : index
  %base = pto.bmu_alloc { pipe=<producer_pipe>, buffer="vec", segm=<chosen> } (%size) : i64
  %off0 = arith.constant 0    : i64
  %off1 = arith.constant <S>  : i64
  %a0   = arith.addi %base, %off0 : i64
  %a1   = arith.addi %base, %off1 : i64
  %a_cast = pto.pointer_cast (%a0, %a1) : memref<16x16xf16, vec>
  %s_mem  = pto.slot_marker %a_cast [%k] : memref<...> -> memref<...>
  ...
  pto.bmu_free { pipe=<last_consumer_pipe>, buffer="vec" } (%base, %size)
```

下游 `PTOResolveBufferSelect`（当前 PR 引入）逻辑不变——它读 multi-address cast 的第 k 个 operand 然后 clone 出单地址 cast，operand 是 const 还是 `arith.addi` 不重要。`SlotAffineAnalysis` 同样不变，slot SSA 比较与基址形式正交。

**类型层 multi_tile_buf 扩展**：增加可选 `placement` 字段，让前端或调试时强制路径选择：

```td
def MultiTileBufType : TypeDef<PTO_Dialect, "MultiTileBuf"> {
  let parameters = (ins
    "mlir::pto::TileBufType":$slotType,
    "uint32_t":$count,
    DefaultValuedParameter<"::mlir::pto::MultiBufPlacement",
                          "MultiBufPlacement::kAuto">:$placement
  );
}
enum class MultiBufPlacement { kAuto, kStatic, kBmu };
```

`kAuto` 是默认，由 classifier 决定；`kBmu` 在 A2/A3 上 verifier 报错。

### 4.5 新 pass：`pto-classify-buffers`

ModuleOp pass，**对应规划 pipeline §4.3 第 (1) 步**。在每个 `memref.alloc` 上打两个 attr：

- `pto.plan_class = "S" | "D" | "H"`
- `pto.h_group_id : i32` （仅当 plan_class = "H" 时存在）

判定规则（按优先级从上到下，第一个命中即定）：

| 优先级 | 命中条件 | 输出 | Phase |
|---|---|---|---|
| 0 | A2/A3，或 A5 但 module attr `pto.uses_bmu = false` | S | always |
| 1 | `MultiTileBufType.placement` 显式指定 `kStatic` / `kBmu` | 按显式值（kBmu→H，组 id 取 alloc 自身） | phase 2 |
| 2/3 | `alloc_multi_tile` 产出 (`pto.multi_buffer` attr 在场) 且 `placement=auto` | **S**（S 优先：静态 liveness 复用比 BMU slice 量化更省空间，尤其小 tile；S 和 BMU 共享同一物理内存，S 放不下 BMU 也放不下，不存在外溢。只有 `placement=bmu` 显式指定时才走 H） | phase 2 |
| 4 | 与同 scope 其它 buffer 的并发字节总和 > scope 容量（dry-run 检测） | D | phase 3 |
| 5 | 处于运行时数据依赖的 `scf.if` 分支、生命周期不跨分支 | D | phase 4 |
| 6 | 参与 pypto 的 `multibuffer-unroll-num` loop（典型 ping-pong pair） | **S**（phase 2-3 保守）或 H（phase 4+） | phase 2 默认 S |
| 7 | `scf.for` iter_args / yield 揭示的 alias 组（隐式 multi-buffer） | H（同 group_id） | phase 4 |
| 8 | 默认 | S | always |

规则 1-3 是 multi-buffer PR 集成的入口；规则 4-8 是 BMU 主线通用工作。S 优先空间预留：`plan-bmu-layout` 先保守估算 S 类字节需求（求和，无 liveness），BMU 动态段在剩余空间 `bmu_available = total - S_slices_reserved` 内 carve-up。

**反馈环**：本 pass 第一遍跑只用规则 1（不依赖 PlanMemory 实际尝试）。如果后续 `pto-plan-memory` 阶段触发 `EmitPlanMemoryFailureInfo` 内部信号（不再 emit user-level error），把失败 entry 的 alloc 反查并 reclassify 成 D，重跑 layout pipeline。这把现 PTOAS 的"硬 fail"换成"软回退"。

**关于 dynamic shape 的说明**：`memref` 类型含 `ShapedType::kDynamic` 的情况在当前 pypto 输出中**不会出现**——pypto 的 `InitMemRef` 对 TileType 强制静态 shape（理由参见 §1.1 注）。本 pass 不需要专门规则覆盖。如果未来 pypto 政策放宽允许 dynamic TileType，需要硬件方先确认「物理 alloc 小于 tile 模板声明范围时，valid_shape 之外的访问行为是否安全」——这是独立的硬件特性问题，不在本方案范围。

实现复用 `MemLivenessAnalysis` 的生命周期、alias、`buffer2MultiNum`、loop attr。

### 4.6 新 pass：`pto-plan-bmu-layout`

ModuleOp pass，**对应规划 pipeline §4.3 第 (3)(5) 步**——做 scope 切段 + 物化 D / H 类的 alloc/free op。

**第一阶段：scope carve-up**

按 §4.3 (2) 的 co-living 结果：

1. 算每个 scope 的"静态需求"`static_peak`（所有 S 类 buffer 在 program point 上的字节峰值）。
2. 算每个 scope 的"动态需求"——每段一类 size，每段容量 = 「同时活的该类 alloc 数」×「单 alloc size」（注意 H 组按 group-size 算）。
3. 决定 `BMU_SEGM_*` 寄存器值：填满 4 个 tail 给动态段；`tail_seg3 < total_slices` 剩下的留给 S 类。
4. emit `pto.bmu_config` 在 kernel 入口。

**第二阶段：D 类物化**

对每个 D 类 `memref.alloc`：

```
原 IR:
  %tile = memref.alloc() : memref<...>

改写为:
  %size = arith.constant <slice_count> : index   // 静态 size 时；动态 size 时用 SSA
  %addr = pto.bmu_alloc {pipe, buffer, segm=<chosen>} (%size) : i64
  %tile = pto.pointer_cast %addr : memref<...>
```

死亡点 emit `pto.bmu_free(%pipe, %addr, %size)`。

**第三阶段：H 组物化**

H 组有两类来源，对应不同的 IR 形态。

**(a) multi-buffer 子路径**（`alloc_multi_tile` 产出，§1.4）

输入只有一个 `memref.alloc` 带 `pto.multi_buffer = N` attr；组成员是它的 N 个 slot。

**识别信号 = 这条 `pto.multi_buffer` 属性本身**（由 `alloc_multi_tile` 经 `PTOViewToMemref` 下放，classify 已在读）。显式路径**不需要、也不应该**去检测 pypto 的 `double_buffer` / `multibuffer-unroll` / `LowerPipelineLoops` 等流水 pass 的结构——那些是**隐式** multi-buffer（两份 `memref.alloc` + loop attr，规则 6/7）的识别信号，与显式路径无关。整条 BMU 物化只在 **level1/2** 生效：level3 下 pypto 已完成内存规划、PTOAS `pto-plan-memory` 不跑（见 §4.7 / reserve_buffer 分流）。

有两种物化形态。**H 路径下首选 (a-1)**，前提不满足时回退 (a-2)。

**(a-1) 逐轮物化（per-advance，循环内 alloc/free）——首选**

适用前提：`alloc_multi_tile` 的所有 slot 使用都落在一个循环体内，且**槽的生命周期在迭代内闭合**（纯 produce→consume 的 ping-pong；不存在按 index 跨迭代持有特定数据的依赖）。这正是显式 multi-buffer 的典型形态——循环里 `multi_tile_get %mb[%k]`（k=i%N）用 index 轮转表达 ping-pong，本身就是流水化结构，无需再判"是否流水"。

物化：

1. segment 容量 = N 个**单 slot** 的 slice 数：`tail_seg3 = N * ceil(sizeof(slotType) / slice_bytes)`，保证最大并发槽数 N（峰值 N 轮流水重叠、同时占 N 个槽）。
2. 在每轮 slot 首次写入点 emit `pto.bmu_alloc <scope>, <producerPipe>, count = ceil(sizeof(slotType)/slice_bytes)`，得本轮 `%slot_base`。
3. 该 slot 的 `pto.pointer_cast` 直接用 `%slot_base`（单地址，无 index 偏移）。
4. 在该 slot 最后消费点 emit `pto.bmu_free %slot_base`（free 的 pipe 见下方约束）。
5. 绕开 index 多槽选择：per-advance 下每轮 `bmu_alloc` 即本轮唯一的槽，`slot_marker` / `PTOResolveBufferSelect` 的多地址选择不再需要。

收益（**这是 line 166 / §5「少同步指令」收益的实现路径**，整环物化拿不到）：复用边界的 reuse WAR 由下一次 `bmu_alloc` 的 backpressure stall **隐式承担**（§2.1 bump pointer + §4.8(a)），省掉 S 模式必须显式 emit 的反向 `wait_flag`；正向 disjoint 仍成立（不同轮 alloc 同源 SSA-base 判 disjoint，§10.2）。硬件按 base+size 发放、live 分配互不重叠，非重叠由硬件保证。

代价：alloc 粒度是**每槽至少 1 slice**，当 `sizeof(slotType) < slice_bytes` 时比整环打包更占 slice（例如 512B slot、8KB slice、N=2：逐轮需 2 slice，整环 `ceil(2*512/8192)=1` slice）。这是换取硬件隐式同步的固有成本。

死锁安全：alloc 进入循环后，`set_flag_dyn` 必须 hoist 到同 pipe 的 `bmu_alloc` 之前（§4.8(c)），否则 waiter 阻塞 alloc 等待的资源 → 死锁（§5 已知坑）。§4.8(c) 在整环物化下是 latent no-op，逐轮物化下变为**承重路径**。

**(a-2) 整环物化（whole-ring，函数级一次 alloc）——回退**

当 (a-1) 前提不满足（slot 值跨迭代逃逸、非循环使用、或 index 携带跨迭代状态）时回退到此形态（也是 phase 2 baseline 已实现的形态）：

1. 算 `slice_count = ceil(N * sizeof(slotType) / slice_bytes)`（slice_bytes 见 §2.1 表）。
2. 在 `alloc_multi_tile` 生命周期的最早 dominator block emit 一个 `pto.bmu_alloc` 得到 `%base`。
3. 把 `PTOPlanMemory` / `AllocToPointerCast` 原本会 emit 的 multi-address `pto.pointer_cast(%c0..%c_{N-1})` 改写为：

```
%off_k  = arith.constant <k * sizeof(slotType)> : i64   // k = 0..N-1
%addr_k = arith.addi %base, %off_k : i64
%cast   = pto.pointer_cast (%addr_0, ..., %addr_{N-1}) : memref<...>
```

4. 在覆盖所有 slot last-use 的 post-dominator emit `pto.bmu_free %base, %slice_count`。
5. `pto.slot_marker` 不动；下游 `PTOResolveBufferSelect` 仍按 slot 索引选第 k 个 operand。

整环物化不产生 per-iteration backpressure，因此**不获得**反向同步收益，只提供函数级容量分时；作用是正确性回退 + 保住 baseline 行为。

**(b) 通用 H 组子路径**（多 alloc 共生死，BMU 主线 phase 4+）

输入是 N 个独立 `memref.alloc`，由规则 7 识别成同 `h_group_id`：

1. 算组内成员的相对偏移（用现 PlanMemory 的 `StorageEntry` / `MergeInplaceSE` 在 group 范围内跑一次小规模布局，输出 `member -> offset` map）。
2. 算 group 总 size = max(member.offset + member.size)。
3. emit 一对 `pto.bmu_alloc / pto.bmu_free`，alloc 拿到 `%group_base`。
4. 每个成员 `memref.alloc` 改写为：

```
%offset_i = arith.constant <intra_group_offset_i> : i64
%addr_i   = arith.addi %group_base, %offset_i : i64
%tile_i   = pto.pointer_cast %addr_i : memref<...>
```

两条子路径产出形态相同（multi-address cast 或单 cast 的 operand 都是 `arith.addi %base, %const`），下游 sync / EmitC 不区分。

**关键约束 / 注意**

- 段内 size 一致：phase 1 强制同 size 的 D / H alloc 才能进同一段；混合 size 退回到独立段或退到 S 类。multi-buffer 内部 N 个 slot 天然同 size，自然满足；跨 multi-buffer 共享 segment 时按 slot size 分桶。
- 死亡点判定：用 `MemLivenessAnalysis` 的 `kill` 集合；H 组的 free 点 = 组内最后一个成员的 kill 时刻。
- alloc / free 的 pipe 选择：alloc 在生产者 pipe，free 在消费者 pipe。当前按每个 buffer kind 取固定 canonical pipe（`PTOBmu.h` 的 `bmuAllocPipeFor` / `bmuFreePipeFor`）：alloc vec/mat/bt→MTE2、L0A/L0B/fb→MTE1、L0C→M；free vec→MTE3、mat→MTE1、L0A/L0B→M、L0C/bt/fb→FIX。这套矩阵是 op verifier（`verifyBmuPipeForBuffer`）与本 pass 的**唯一真值源**——两者都从 `PTOBmu.h` 的 `bmuValidPipesFor(scope, isAlloc)` 取，保证不漂移（§4.4 生产者矩阵 / §5.4 消费者矩阵）。alloc 与 free 落在不同 pipe，故生产者→消费者的数据依赖仍由 InsertSync 主流程的 set/wait_flag 保护（BMU 不做跨 pipe 数据同步，§2.1 / §4.8(a)）。「按实际最后消费者 pipe 取」是后续优化（能复活 §4.8(d) 的 skip 档，见该节）。

### 4.7 `pto-plan-memory` 改动

**对应规划 pipeline §4.3 第 (4) 步**——只处理 S 类与 H 组的"组内布局"。

具体改动：

- `MemLivenessAnalysis::build` 加 attr 过滤：跳过 `pto.plan_class != "S"` 的 alloc。
- `MemPlan::InitMemSpecsFromModule` 把每个 scope 的可用容量改成 `tail_seg3 * slice_size`（来自 `pto-plan-bmu-layout` emit 的 config），而不是硬件 raw capacity。
- 失败处理：原 `EmitPlanMemoryFailureInfo + signalPassFailure` 改成「emit 一条 module-level diagnostic attr `pto.plan_memory_failed_buffers = [...]`」，不调 `signalPassFailure`。`pto-classify-buffers` 在反馈环里读这个 attr 做 reclassify。
- 组内布局子任务（§4.6 第三阶段调用）：拆出 `MemPlan` 的核心算法为可独立调用的子例程，输入是「一组 buffer + 它们的容量上界」，输出是「member → offset map」。组内允许 inplace、ping/pong 排布。

不变的部分：

- `MultiSpecPlan` / `SPEC_LEVEL_0/1/2` / `ApplyFailStrategy` 在 S 类的尾区布局内继续用。
- `AllocToPointerCast` pattern 不变（emit `pto.pointer_cast %const`），只是只对 S 类生效。

### 4.8 InsertSync 改动

四处增量：

**(a) 把 `pto.bmu_alloc` / `pto.bmu_free` 当作 sync 节点纳入分析**

- alloc 在 `.pipe` 上产生「资源 backpressure 等待点」。对 same-pipe + same-buffer 的依赖可视作隐式 wait_flag（已被 InsertSync 的依赖图覆盖时可省 set/wait）。
- free 在 `.pipe` 上产生「该 pipe drain 到此点」语义（H1 已验证）。

**(b) `BaseMemInfo.baseAddresses` 扩展同源 SSA-base + offset**

H 路径下 multi-address `pto.pointer_cast` 的 operand 不再全是常量。`UpdatePointerCastOpMemInfo` 要识别 `arith.addi(%base, const)` 形态：

```cpp
struct AddressEntry {
  Value baseSSA;            // null = 绝对常量；非空 = 同源 SSA 基址
  uint64_t valueOrOffset;   // null base 时是绝对地址；非空时是组内 offset
};
class BaseMemInfo {
  SmallVector<AddressEntry> baseAddresses;
};
```

`MemAlias::isBufferAddressRangeOverlap` 判 overlap 规则：

- 双方都 null baseSSA → 纯 const range（原路径，A2/A3 与 A5 S 模式都走这条）
- 同源 baseSSA → 按 offset+size range（H 模式组内）
- 不同源 baseSSA → 默认 disjoint（不同 bmu_alloc 拿到不同物理槽）
- 一方 null 一方非空 → unknown，保守判 overlap

multi-buffer PR 现有的「常量 slot 自动 disjoint」「affine drop forward sync」依赖 baseAddresses 的 slot 收窄；扩展后两条优化在 H 路径下仍成立（同源 base 内的不同 offset 仍 disjoint）。

**(c) dyn flag 与 stall alloc 死锁规避**

`set_flag_dyn` 与 `bmu_alloc` 在同 pipe 且 set_flag_dyn 的等待方阻塞了 bmu_alloc 释放的资源时会死锁（BMU §5 已知坑）。`SyncCodegen` emit `set_flag_dyn` 时必须检查后续 same-pipe 是否有 `bmu_alloc`，若有则把 set_flag_dyn hoist 到 alloc 之前。这是 multi-buffer 在 H 路径下引入的新约束。

**(d) 函数尾 barrier 条件化**

修改 `InsertSyncAnalysis::Run` 与 `InsertLastPipeAll`：

```
P_used  = 函数内出现指令的物理 PIPE 集合
P_freed = 函数尾（return 前）出现的 pto.bmu_free 覆盖的 PIPE 集合

if P_used ⊆ P_freed:
  skip InsertLastPipeAll               # 完全省掉尾 barrier
elif P_freed ⊂ P_used:
  emit pto.barrier <P_used \ P_freed>  # 降级为部分 pipe barrier
else:
  保持现状 pto.barrier <PIPE_ALL>
```

前提：函数内跨 pipe 数据依赖已经被 InsertSync 主流程的 set/wait_flag 保护好（这是 InsertSync 一贯的不变量）。

**skip 档在消费者 pipe free 下不可达（已实测）**：`bmu_free` 按 §5.4 落在 buffer 的**消费者** pipe（vec→MTE3、mat→MTE1……，见 PTOBmu.h `bmuFreePipeFor`），而 buffer 一定先被某个**生产者** pipe 写入（vec←MTE2 load / V compute……，`bmuAllocPipeFor`）。生产者 pipe ≠ 消费者 pipe，所以 P_freed 与生产者 pipe 恒不相交，`P_used ⊆ P_freed` **永远不成立** → skip 档对真实 buffer 不可达。实际落到 partial（load→store：残差 = 生产者 pipe，如 {MTE2}）或 full（load-only：free 的 MTE3 与 used 的 MTE2 不相交 → PIPE_ALL）。测例 `bmu_tail_barrier_skip.pto` 现记录这一 full 回退。若要保留 skip 收益，需把 free pipe 改成「实际最后消费者 pipe」（V-only 就地计算 kernel 才可能 P_used=P_freed={V}）——当前按固定 canonical pipe 实现，skip 档暂搁置。

实现位置：在 `InsertSyncAnalysis::Run`（`InsertSyncAnalysis.cpp:198`）末尾、调用 `InsertLastPipeAll` 之前加一个 query 函数 `analyzeBmuFreeCoverage(func_) -> (P_used, P_freed)`，根据结果决定 `InsertLastPipeAll` 的 mode。

对 SyncCodegen 端（`SyncCodegen.cpp:314` `AppendAutoSyncTailBarrierIfNeeded`）：根据 mode 改成支持 emit `<PIPE_ALL>` / `<某 pipe 集合>` / 完全跳过。

**Multi-buffer PR 既有的 sync 机制不变**：`SlotAffineAnalysis`、`getMultiBufferEventIdInfo`、`findSlotSSAExprForRWOp`、`set_flag_dyn`/`wait_flag_dyn` codegen 在 H 路径下零改动可用，因为 slot SSA 比较与基址形式正交。

### 4.9 Pass 顺序

```
pypto                         → 产出含 memref.alloc 的 PTO IR
                                  含 pypto 端 ping-pong 标注
                                  含 multi-buffer PR 产出的 pto.multi_buffer attr
ptoas (--pto-level=level1/2):
  pto-view-to-memref          ← 已有：alloc_multi_tile → memref.alloc{multi_buffer=N}
                                       + pto.slot_marker
                                改：根据 placement 写 plan_class 初值（可选）
  pto-infer-mem-scope
  pto-classify-buffers        ← 新：每 alloc 打 plan_class = S/D/H + group_id
                                    multi-buffer 走规则 1-3；其它 alloc 走规则 4-8
  pto-plan-bmu-layout         ← 新：scope 切段 + emit bmu_config/alloc/free
                                    multi-buffer H 组走子路径 (a)；其它 H 组走 (b)
  pto-plan-memory             ← 改：只规划 S 类；H 组内布局通过子例程调用
                                    失败不 emit fail，改写 attr 反馈给 classify
  (可选反馈环：触发 reclassify → 重跑 layout)
  InsertSync / GraphSyncSolver← 改：感知 bmu_alloc/free + BaseMemInfo SSA-base
                                    + dyn flag hoist over stall alloc
                                    + 尾 barrier 条件化
  pto-resolve-buffer-select   ← 已有 (multi-buffer PR)：把 slot_marker 物化为单地址 cast
                                改：读 multi-addr cast 的第 k 个 operand SSA
                                    （支持 const 与 arith.addi 两种形态）
  pto-lowering-sync-to-pipe
  Expand TileOp / ...
  PTOToEmitC                  ← 新增 BmuConfigOp / BmuAllocOp / BmuFreeOp 的 lower
```

关键约束：
- `pto-view-to-memref` 必须在 `pto-classify-buffers` 之前（attr 标注的载体必须先存在）。
- `pto-classify-buffers` 必须在 `pto-plan-bmu-layout` 之前（标注先于消费）。
- `pto-plan-bmu-layout` 必须在 `pto-plan-memory` 之前（静态部分要消费 segment 划分后的剩余容量）。
- InsertSync 在所有 layout 完成之后，确保它能看到 `pto.bmu_alloc` / `pto.bmu_free` 节点。
- `pto-resolve-buffer-select` 位置不变（sync 之后）；它处理的 multi-addr cast operand 可能是 const（S 路径）或 `arith.addi`（H 路径），两种都支持。
- 反馈环（可选）：若 `pto-plan-memory` 写出 `pto.plan_memory_failed_buffers` attr，driver 重新跑 `pto-classify-buffers → pto-plan-bmu-layout → pto-plan-memory` 一次。第一版可不实现反馈环。

## 5. 关键取舍与已知坑

| 取舍点 | 选择 | 理由 |
|---|---|---|
| 二分（BMU vs 静态）vs 三分（S / D / H） | **三分** | H 模式（动态外壳 + 静态内核）能拿到 S 的布局优化 + D 的灵活性，匹配 matmul ping-pong 这类「组内固定 + 组外流动」的真实模式 |
| 静态 + 动态混合 vs 全 BMU | 混合 | 热路径静态规划开销为零；BMU 指令本身有 cycle 成本 |
| `alloc_multi_tile` 默认 S 还是 H | **首选 S，容量超限自动 H** | S 保住 const-addr disjoint sync、affine drop forward 等当前 PR 的优化；只在 N×slot_size 超过 segment 静态尾区时切 H |
| pypto-决策的 ping-pong pair 默认 S 还是 H | **第一版默认 S** | 保住 pypto + `relationPongEntry` 已有的优化收益；第二版再 try H 看是否更省 sync |
| 候选分类由 PTOAS 自决 vs 由 pypto 标 | PTOAS 自决 | 不动 pypto；用 IR 启发式 + dry-run + 反馈环判定 |
| ping-pong 显式排布 vs 依赖 bump pointer | S 模式显式，H 模式依赖硬件 | bump pointer 在 size 一致 + segment 容量 ≥ 2 × 单次 alloc 时自然产生 ping/pong（适合 H）；S 模式仍用现 `relationPongEntry`。H 路径的具体物化见 §4.6 (a-1) 逐轮 / (a-2) 整环 |
| 显式 multi-buffer H 的物化：整环 vs 逐轮 | **首选逐轮 (a-1)，前提不满足回退整环 (a-2)** | 逐轮 alloc/free 才拿到 bump pointer 的隐式 reuse 同步（省反向 wait_flag，line 166 收益）；整环一次 alloc 只有函数级容量分时、拿不到该收益。见 §4.6 (a-1)/(a-2) |
| segment 编译期定 vs 运行时切 | 编译期定 | 硬件 §7 规则 7：重配 SPR 必须先 free 全部 + hscb barrier，运行时切代价过高 |
| 用 BMU sync 替换 InsertSync 跨 pipe wait | 不替换 | 硬件确认跨 pipe alloc/free 不做数据同步；只在 buffer reuse 边界用 BMU 隐式资源同步替代 wait_flag |
| 规划失败时 emit fail vs 反馈给 classifier | 反馈给 classifier | 利用 BMU 让"容量峰值满足但布局失败"的程序仍能跑；现 `EmitPlanMemoryFailureInfo` 改为写 attr 而非 emit user-level error |
| BaseMemInfo 用绝对地址 vs 同源 SSA+offset | 同源 SSA+offset | multi-buffer H 路径下 baseAddresses 是 `arith.addi` 链；扩展后既兼容 S 模式纯常量，也支持不同 BMU 组默认 disjoint |

已知坑：

- **段内 size 不一致导致 stall**（**H3 已验证为硬约束**）：BMU 的 alloc 唤醒条件是"凑够连续 N slice"，所以同段 alloc 出现多种 slice 数时，bump pointer wrap 后可能凑不出连续 N 而 stall。`pto-plan-bmu-layout` 必须实装「同 size 类归段」+ size 不齐的候选回退到 S 类。multi-buffer 内部 N 个 slot 天然同 size 自动满足；跨 multi-buffer 共享 segment 时按 slot size 分桶。
- **H 组识别准确性**：H 模式依赖正确识别"共生死的 buffer 组"。multi-buffer 用 `alloc_multi_tile` 显式表达组本身就是 IR 节点，无误识别风险。隐式 H 组用 `scf.for` iter_args / yield 揭示的 alias 链 + pypto `multibuffer-unroll-num` 推断；漏识别退回 S 模式不影响正确性。
- **dyn flag 与 stall alloc 死锁**：multi-buffer PR 的 `set_flag_dyn` 当前 emit 在 producer 计算 op 之后。如果与 H 路径下的 `bmu_alloc` 同 pipe，且 set_flag_dyn 的 waiter 阻塞了 alloc 等待的资源 → 死锁。`SyncCodegen` 必须把 set_flag_dyn hoist 到 same-pipe 的 bmu_alloc 之前。这是 multi-buffer + H 引入的新约束。
- **死锁规则（通用）**：因 BMU alloc 会 stall 同 pipe 后续指令，所有"告知 wake-up 候选者本 pipe 已就绪"的 `set_flag` 必须 issue 在会 stall 的 alloc 之前（H2 demo 设计中证实）。
- **Tile 字节数 vs slice 数必须吻合**：BMU alloc 单位是 slice（UB/L1 = 8KB, L0A/B = 4KB, L0C = 16KB），但 TLOAD/TSTORE 按 tile 实际字节数搬运。`pto-plan-bmu-layout` emit alloc 时必须用 `ceil(N * sizeof(slotType) / slice_bytes)` 作为 slice 数；否则 `mte_gdma_write_overflow`。
- **跨函数 BMU 状态**：BMU 规则 8 要求 END 前全 free；函数边界 BMU 状态强制 clean——是函数尾 barrier 优化的硬基础，但跨函数复用不在范围内。multi-buffer PR 限制 `multi_tile_buf` 不能跨 func arg / return，与该约束兼容。
- **混编场景的 ABI**：同一程序内既有走 BMU 的 kernel 又有不走的，函数尾 barrier 优化只能开启在走 BMU 的那部分；用 module attr `pto.uses_bmu` 标识。A2/A3 上该 attr 永远 false，multi-buffer PR 完全走 S 路径。
- **依赖 `--pto-level` 非 level3**：pypto 调 ptoas 当前默认 level3，本方案需要 `level1` 或 `level2`。这是 pypto 调用方一行参数改动（`pto_backend.py:850`），不属于设计变更，但部署时要确认。

## 6. 工作分解

基线假设：multi-buffer PR（§1.4）已合入。以下 phase 在此之上叠加。

### Phase 0 · 硬件特性验证（前置，已完成）

每个假设对应一个独立 demo，位置与方法学如下：

| # | 假设（简版） | 位置 | 判读方式 |
|---|---|---|---|
| H1 | `BUF FREE` 是否严格后于同 pipe 先前所有指令 | **复用 `pto-isa/tests/npu/a5/src/st/testcase/tbuf` baseline**（不改 kernel） | 跑 `case_float_pingpong_baseline_32x128`，读 device trace 里 `retire(__ubuf_free<PIPE_V>(y))` vs `retire(TADD on V)`，以及 `retire(__ubuf_free<PIPE_MTE3>(z))` vs `retire(TSTORE on MTE3)`。两组 same-pipe pair 都能直接看 |
| H2 | alloc stall 期间，同 pipe 上 **stall 之前已发射** 的 long-latency 指令是否继续 retire | 新增 testcase `pto-isa/tests/npu/a5/src/st/testcase/bmu_h2_alloc_stall/`，case `case_stall_drains_prior_work_32x128` | 1-slice segment 强制下一次 alloc 必 stall；stall 前发射一条 long-latency `TLOAD`。读 trace 比较 `retire(TLOAD into A)` 与 `retire(alloc B)`：若前者落在 stall 窗口内 → pipe 继续推进；若两者几乎同时 retire → pipe 被 stall 冻结 |
| H3 | alloc 唤醒条件：**任意 free** / **总量满足 N** / **凑够连续 N**（三选一） | 新增 testcase `pto-isa/tests/npu/a5/src/st/testcase/bmu_h3_wake_condition/`，case `case_wake_condition_32x128` | 4-slice segment + 4 × 1-slice alloc 填满 + stall 2-slice alloc；V 依次 free C、A、B，间插 spacer。读 trace 看 `retire(alloc E)` 紧跟哪次 free：C 后 → 任意 free 唤醒；A 后 → 总量满足；B 后 → 凑够连续（与硬件方说法一致） |

三个 testcase 与 tbuf 并列，独立 CMakeLists / main / kernel / gen_data；不污染 tbuf。

H1 的简化方法学：tbuf 的 `tadd_kernel_func` 末尾本来就有 `__ubuf_free<PIPE_V>(y/x)` 紧跟在 V 上的 `TADD` 之后，以及 `__ubuf_free<PIPE_MTE3>(z)` 紧跟在 MTE3 上的 `TSTORE` 之后。同 pipe 上「非 free 指令 → free」两组 pair 自然在 trace 里出现，**无需任何代码改动**。

输出：硬件行为表，回填到本文 §2.2。三条假设全部验证为成立。

### Phase 1 · BMU IR + EmitC lowering

**目标**：为后续 phase 提供基础 IR 与 codegen 能力，不改 PlanMemory / Sync。

- 新增 `BmuConfigOp` / `BmuAllocOp` / `BmuFreeOp`（ODS + verifier，§4.4）
- 新增 `PTOToEmitC` lowering pattern：`set_bmu_segm_<ub|l1|l0a|l0b|l0c|bt|fb>(...)` / `__ubuf_alloc / __cbuf_alloc / __ca_alloc / ...` / `__ubuf_free / ...`（BT 的 `__bt_alloc/__bt_free` 直接用 `uint64_t`，无 `__bt__` 指针）
- 新增 module attr `pto.uses_bmu`，CLI flag `--pto-uses-bmu` 控制；A2/A3 上永远 false
- 类型扩展：`MultiTileBufType` 增加可选 `placement` 字段（kAuto / kStatic / kBmu，§4.4），A2/A3 上 `kBmu` verifier 报错
- 测试：手写 `.pto` 文件含三种 BMU op，过 `ptoas --emit-cpp` 看输出；multi-buffer placement 字段 parse/print 测试

**验收**：BMU op 能正确 emit ccec intrinsic；multi-buffer PR 既有 18 个 lit 测试零回归。

### Phase 2 · Multi-buffer 端到端走通 H 路径（核心 phase）

**目标**：让 `alloc_multi_tile` 在 A5+`uses_bmu` 下能完整编译运行，覆盖 const slot / dyn slot / unroll / while / 跨 block 所有控制流形态。

- 新 pass `pto-classify-buffers`：实装规则 0-3（A2/A3 gate、placement 显式、multi-buffer 容量决策），规则 4-8 默认全 S
- 新 pass `pto-plan-bmu-layout`：
  - 第一阶段：scope carve-up + emit `pto.bmu_config`
  - 第三阶段子路径 (a)（multi-buffer 子路径）：把 `memref.alloc{multi_buffer=N, plan_class="H"}` 改写为 `bmu_alloc` + `arith.addi` 链 + multi-address cast
  - 第二阶段（D 类）和第三阶段子路径 (b) 留空
- `pto-plan-memory` 改造：
  - `MemLivenessAnalysis::build` 加 `plan_class != "S"` 过滤
  - `MemPlan::InitMemSpecsFromModule` 改用 `tail_seg3 * slice_size`
  - 失败仍保持 `signalPassFailure`（反馈环延后到 Phase 4）
- `BaseMemInfo` 扩展 SSA-base + offset（§4.8(b)）：`UpdatePointerCastOpMemInfo` 识别 `arith.addi(base, const)`；`MemAlias::isBufferAddressRangeOverlap` 按同源 SSA 判 overlap
- `PTOResolveBufferSelect` 小调整：读 multi-address cast 的第 k 个 operand SSA（兼容 const 与 arith.addi 两种形态）
- multi-buffer PR 既有 18 个 lit 测试加 `--pto-target-arch=a5 --pto-uses-bmu=false` 重跑，验证零回归
- 新增 multi-buffer A5+BMU 测试集（11 个）：

| 测试 | 覆盖点 |
|---|---|
| `multi_tile_a3_no_bmu_baseline.pto` | A3 上 alloc_multi_tile 走 S 路径，行为与当前 PR 一致 |
| `multi_tile_a5_bmu_off_baseline.pto` | A5 + uses_bmu=false，行为同 A3 |
| `multi_tile_a5_h_mode_const_slot.pto` | A5 + H + const slot：验证 base+offset 物化 + const-slot disjoint sync |
| `multi_tile_a5_h_mode_dyn_slot.pto` | A5 + H + iv%N：验证 N dyn flag + affine drop |
| `multi_tile_a5_h_mode_no_loop.pto` | 无 scf.for 的 unrolled multi-buffer 走 H |
| `multi_tile_a5_h_mode_while.pto` | scf.while 内 multi-buffer 走 H |
| `multi_tile_a5_h_mode_cross_block.pto` | alloc 与 get 跨 region：bmu_free 位置正确 |
| `multi_tile_a5_two_groups_disjoint_sync.pto` | 两个独立 alloc_multi_tile 不同源 base：sync 不串 |
| `multi_tile_a5_two_groups_same_segment.pto` | 同 slot size 两组共享 segment：bump pointer ping-pong |
| `multi_tile_a5_placement_kbmu_a3_errors.pto` | A3 上 placement=kBmu 编译报错 |
| `multi_tile_a5_auto_threshold_S_to_H.pto` | placement=kAuto，N×slot_size 越过阈值时切到 H |

**验收**：multi-buffer 在 A5+BMU 下能跑 cpu sim 与 NPU 端到端，11 个新测试 + 18 个回归测试全绿。

### Phase 3 · Multi-buffer sync 路径 BMU 适配

**目标**：闭合 multi-buffer 在 H 路径下的所有 sync 正确性问题，启用尾 barrier 优化。

- `InsertSync` / `GSS` 把 `BmuAllocOp` / `BmuFreeOp` 当 sync 节点（§4.8(a)）
- `SyncCodegen` 实装 dyn flag hoist over stall alloc（§4.8(c)）—— multi-buffer 引入的新约束
- `analyzeBmuFreeCoverage` + `InsertLastPipeAll` 三档条件化（§4.8(d)）
- `SyncCodegen::AppendAutoSyncTailBarrierIfNeeded` 支持 emit `<PIPE_ALL>` / `<某 pipe 集合>` / 完全跳过
- 新增测试：

| 测试 | 覆盖点 |
|---|---|
| `multi_tile_a5_dyn_flag_before_alloc.pto` | dyn flag 必须 hoist 到 bmu_alloc 之前 |
| `multi_tile_a5_tail_barrier_skip.pto` | 函数尾 PIPE_ALL barrier 被 bmu_free 覆盖时跳过 |
| `multi_tile_a5_tail_barrier_partial.pto` | 部分 pipe 被 free 覆盖：emit 子集 barrier |
| `bmu_general_tail_barrier_keep.pto` | 不走 multi-buffer 也能用尾 barrier 三档机制 |

**验收**：典型 prefetch / softmax UB ping-pong kernel 在 A5 上对比 S vs H 两种模式，trace 上看 H 模式的尾 barrier 减少；无任何死锁。

### Phase 4 · D 类支持 + 通用 H 组识别 + 反馈环（BMU 主线）

**目标**：让 BMU 覆盖 multi-buffer 之外的 alloc 场景。multi-buffer 已经在 Phase 2-3 完整可用，本 phase 与 multi-buffer 解耦。

- `pto-classify-buffers` 增加规则 4-7：
  - 规则 4（容量 dry-run overflow → D）
  - 规则 5（数据依赖 scf.if 分支 → D）
  - 规则 6（pypto multibuffer-unroll-num pair 可选 H）
  - 规则 7（scf.for iter_args / yield 隐式 H 组识别）
- `pto-plan-bmu-layout`：
  - 第二阶段（D 类物化）
  - 第三阶段子路径 (b)（通用 H 组）
- `pto-plan-memory` 反馈环：失败写 `pto.plan_memory_failed_buffers` attr，driver 重跑 classify
- 从 `MemPlan` 拆出「一组 buffer + 容量上界 → offset map」子例程，给通用 H 组用
- 选定一类典型 H 场景（如 matmul L1 A/B/scale/bias）做端到端验证

**验收**：D 类与通用 H 组都能编译运行；Phase 2-3 测试零回归。

### Phase 5 · 性能优化（详见 §11）

multi-buffer 在 BMU 下的进一步优化：跨 multi-buffer 段共享、跨 pipe slot 释放、函数中段早释放、容量自适应 N 降级、free fold 等。每条独立立项，不阻塞主功能。

> **out of scope**：pypto 端的 DSL 参数（`pl.dyn_alloc` / `dynamic=True`）、`l0_tile_chooser` 的 slice 对齐、`auto_tile_matmul_l0_pass` 的 double_buffer 调整、跨函数 multi-buffer 复用——这些被本轮明确剔除；如未来要做，单独立项。

### Phase 依赖与并行机会

```
Phase 0 (已完成)
    ↓
Phase 1 (IR + EmitC + placement) ──────────┐
    ↓                                       │
Phase 2 (multi-buffer H 端到端) ←───────────┘
    ↓
    ├── Phase 3 (sync 路径优化)
    └── Phase 4 (D + 通用 H + 反馈环)         ← 与 Phase 3 可并行
              ↓
         Phase 5 (性能优化)
```

Phase 3 与 Phase 4 在 Phase 2 完成后可并行——Phase 3 闭合 multi-buffer 自己的正确性与性能，Phase 4 把 BMU 推广到非 multi-buffer 场景。

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| BMU 指令 cycle 成本高于预期，热路径退化 | classifier 保守默认 S；只对静态规划失败 buffer 走 D |
| 硬件 free drain 语义不覆盖全 pipe（H1 否） | H1 已验证为成立，但若实际场景超出验证条件：尾 barrier 优化降级为只跳过 free 覆盖的 pipe |
| 段内 fragmentation 触发 stall | 同 size 类归段；size 不齐候选回退到 S（在 `pto-plan-bmu-layout` 内改 `plan_class` attr） |
| H 组识别错误（误把不共生死的 buffer 当成一组） | H 模式比 S 模式没有更弱的正确性；最坏后果是少优化，不会引入正确性 bug |
| 跨函数 ABI 不一致 | module attr 标识；InsertSync 只在所有调用方都 `uses_bmu` 时启用尾 barrier 优化 |
| pypto 端 tile size 与 slice 不齐 | 第一版默认 S（用 PlanMemory 现 256B 对齐）；走 BMU 的 D 类必须 size = ceil(tile_bytes / slice_bytes) |

## 8. 与现有代码的接触点速查

| 改动类型 | 文件 |
|---|---|
| 新 op 定义（`pto.bmu_config` / `pto.bmu_alloc` / `pto.bmu_free`） | `PTOAS/include/PTO/IR/PTOOps.td` |
| `MultiTileBufType` 新增 placement 字段 + verifier | `PTOAS/include/PTO/IR/PTOTypeDefs.td`、`PTOAS/lib/PTO/IR/PTOTypeDefs.cpp` |
| Module attr `pto.uses_bmu` + CLI flag | `PTOAS/include/PTO/IR/PTO.h`、`PTOAS/tools/ptoas/ptoas.cpp` |
| 新 pass `pto-classify-buffers` / `pto-plan-bmu-layout` 注册 | `PTOAS/include/PTO/Transforms/Passes.td`、`PTOAS/lib/PTO/Transforms/` |
| InsertSync 改（感知 bmu_alloc/free + dyn flag hoist + 尾 barrier 条件化） | `PTOAS/lib/PTO/Transforms/InsertSync/`（`InsertSyncAnalysis::Run` / `InsertLastPipeAll` / `SyncCodegen`） |
| GSS 改（感知 bmu_alloc/free） | `PTOAS/lib/PTO/Transforms/GraphSyncSolver/` |
| BaseMemInfo SSA-base 扩展 | `PTOAS/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp`（`UpdatePointerCastOpMemInfo`）、`PTOAS/lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp`（`isBufferAddressRangeOverlap`） |
| PlanMemory 改（过滤 + 容量 + 失败 attr + 组内 layout 子例程） | `PTOAS/lib/PTO/Transforms/PTOPlanMemory.cpp`（`MemLivenessAnalysis::build` / `MemPlan::InitMemSpecsFromModule` / `EmitPlanMemoryFailureInfo`） |
| PTOViewToMemref 写 plan_class 初值 | `PTOAS/lib/PTO/Transforms/PTOViewToMemref.cpp` |
| PTOResolveBufferSelect 适配 SSA operand | `PTOAS/lib/PTO/Transforms/PTOResolveBufferSelect.cpp` |
| EmitC lower（三个新 op 到 ccec intrinsic） | `PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp` |
| 部署时 ptoas 调用参数 | `pypto/python/pypto/backend/pto_backend.py:850`（把 `--pto-level=level3` 改为 `level1` 或 `level2`；非设计变更） |

## 9. IR 层如何统一 BMU 与非 BMU 的 multi-buffer 接口

前端到 sync 入口完全统一，差异收敛在 PlanMemory / `pto-plan-bmu-layout` 这一档：

```
前端:        pto.alloc_multi_tile + pto.multi_tile_get   (arch-agnostic)
                          ↓
ViewToMemref: memref.alloc { multi_buffer=N, plan_class }
              pto.slot_marker [%k]                       (arch-agnostic)
                          ↓
Classify:    plan_class ∈ {S, H} ← arch + uses_bmu + 容量决定
                          ↓
        ┌── S ─────────────┐    ┌── H (仅 A5+BMU) ──────────────┐
        │ PlanMemory N-way │    │ bmu_alloc → %base               │
        │ const addr       │    │ multi-addr cast = base + offset │
        └──────────────────┘    └─────────────────────────────────┘
                          ↓
        所有路径都汇成：multi-address pto.pointer_cast
                       + pto.slot_marker
                          ↓
InsertSync / GSS:  按 BaseMemInfo.baseAddresses 做 slot 收窄分析
                          ↓
PTOResolveBufferSelect:  按 slot 选一个单地址 cast (const / arith.select 链)
```

**统一接口的关键载体**：multi-address `pto.pointer_cast` 的语义不变——「这个 alloc 有 N 个候选地址」；只是地址可能是 const（S 路径）或 SSA `addi`（H 路径）。下游 sync / resolve 不需要知道哪种。

兼容性矩阵：

| 层 | A2/A3 | A5 + uses_bmu=true | A5 + uses_bmu=false |
|---|---|---|---|
| 前端 IR（`alloc_multi_tile` / `multi_tile_get`） | 不变 | 不变 | 不变 |
| `PTOViewToMemref` | 不变 | 不变 | 不变 |
| `PTOPlanMemory` N-way 静态布局 | 走 | 仅 S 类（multi-buffer 默认 S，容量超限切 H） | 走 |
| `PTOResolveBufferSelect` 单地址 cast / `arith.select` 链 | 走 | 走（base 改 SSA） | 走 |
| `bmu_alloc`/`bmu_free` op | 不出现 | 围绕每个 multi_tile_buf 组 | 不出现 |
| 函数尾 `PIPE_ALL` barrier | 保持 | 条件化降级 | 保持 |
| dyn flag（`set_flag_dyn`/`wait_flag_dyn`） | 不变 | 不变 | 不变 |

入口判定：`isTargetArchA5(module) && module->getAttrOfType<BoolAttr>("pto.uses_bmu")`（缺省 false），决定 BMU 路径是否启用。

## 10. 用 BMU 对同步有什么影响

### 10.1 不影响 multi-buffer 内部同步语义

`set_flag_dyn` / `wait_flag_dyn` 的推导路径不变——slot SSA 比较、affine disjoint 检测、N event-id 分配都建立在 `pto.slot_marker.slot` 上，与基址是 const 还是 SSA 无关。multi-buffer PR 的 `SlotAffineAnalysis` / `getMultiBufferEventIdInfo` / `findSlotSSAExprForRWOp` 在 H 模式下零改动可用。

### 10.2 MemAlias 需要扩展处理「同源 SSA-base + const offset」

详见 §4.8(b)。当前 `BaseMemInfo` 的 `baseAddresses` 是 uint64 数组，按数值范围判 overlap。H 路径下变成 `[%base+0, %base+S, ...]` 的 SSA 偏移。判 overlap 规则：
- 同 baseSSA（或都 null）→ 按 offset+size range
- 不同 baseSSA → 默认 disjoint
- 一方 null 一方非空 → 保守 overlap

这是 BMU 接入要做的主要 sync 改动；否则两个不同 H 组的 alloc 会被误判 overlap，产生多余同步。

### 10.3 函数尾 PIPE_ALL barrier 可优化

`alloc_multi_tile` 走 H 路径后，函数结束前会有一组 `pto.bmu_free`。按 §4.8(d) 三档机制：

```
P_used  ⊆ P_freed  → skip
P_freed ⊂ P_used  → emit barrier <P_used \ P_freed>
otherwise         → emit barrier <PIPE_ALL>
```

multi-buffer 是 free 来源之一；通用 D 类 alloc 也贡献 free。

### 10.4 dyn flag 与 BMU stall 的次序约束

`set_flag_dyn` 必须 issue 在可能 stall 的 `bmu_alloc` 之前（§4.8(c)）。multi-buffer PR 现在的 dyn flag emit 在 producer 计算 op 之后；如果与 multi-buffer H 路径下的 bmu_alloc 同 pipe，sync codegen 必须 hoist。**这是 BMU + multi-buffer 组合特有的次序约束**，是 BMU 接入对 sync 的唯一硬性新约束。

### 10.5 不影响跨 pipe 数据同步

BMU 硬件确认跨 pipe 的 alloc/free 不做数据同步（§2.1）。跨 pipe 的 set/wait_flag 一律由 InsertSync 主流程负责，BMU 不替换、不重叠。multi-buffer PR 的跨 pipe 同步逻辑全部保留。

## 11. BMU 在 multi-buffer 之上的额外优化机会

§4-§10 让 multi-buffer 在 BMU 下「能跑」。下面列出超出此范围的优化点，可作为 Phase 5 立项依据。

### 11.1 跨 multi-buffer 的 BMU 段共享 ping-pong

两个 lifetime 不相交（或部分相交）的 `alloc_multi_tile`，同 slot size，可共享同一 BMU segment。bump pointer 自然产生 ping-pong 排布，省掉 PlanMemory 的 inplace 合并分析。

收益：classifier 把这两个标到同一 segment，PlanMemory 完全不参与；segment 容量按"任意时刻同时活的 group 数"× group size，而不是所有 group 累加。

### 11.2 跨 pipe 的 slot 释放点优化

当前 PR 没有指定 slot lifetime 由哪个 pipe 管理。BMU bmu_free 是 same-pipe drain，可以故意把 free 放到最后消费者所在的 pipe（如 MTE3）而不是 producer pipe（如 MTE2），让 producer 提早进入下一轮工作。

收益：减少 producer pipe 的 backpressure；prefetch 流水可以多推一轮。

### 11.3 函数中段的 multi-buffer 早释放

当前 PR 的释放隐含在函数尾 PIPE_ALL barrier。BMU 下可在 last-use 后立即 `bmu_free`，让出 slot 容量给后续 alloc。

适用场景：函数前半段 prefetch（multi-buffer A），后半段 reduction（独立大 alloc B）。当前 PR 中 A、B 必须共存在静态 layout 里；BMU 下 A 提前 free，B 可占用 A 的 slice。

### 11.4 容量自适应回退（N 协商）

BMU 设计 §5.4.4 的「资源不足回退：N → 偶数 → 2 → 1」当前 PR 是设计目标但未实现。BMU 下可在编译期 N 协商：classifier 算出 segment 容量上限，若用户写的 N 超容量，自动改写为最大可放 N'，sync 推导按 N' 走，给 user-level diagnostic 提示。

收益：用户写 `count=8` 但容量只支持 `count=4`，不再硬 fail，自动降到 4 槽 + 多余 sync。

### 11.5 conditional multi_tile_get 的 slot 按需 alloc

`multi_tile_get` 在 `scf.if` 分支内：当前 PR 必须 alloc 全 N 槽（静态 layout）。BMU 下，组 size 仍按 N 算（slot 索引可能跨分支跳跃），但**释放可分支条件化**：未走的分支不计入 last-use，free 点更早。

### 11.6 multi-buffer 与 D 类 scratch 的段隔离

multi-buffer 段 size 固定、生命周期长；D 类段 size 多样、生命周期短。混在一起会让 D 类频繁 alloc 把 multi-buffer 的 bump pointer 推到 wrap 处，增加 stall 概率。`pto-plan-bmu-layout` 阶段 1 的 segment 分配策略要做隔离。

### 11.7 prefetch depth 的运行时调节

当前 PR 的 N 是编译期常量。BMU 设计的 `bmu_alloc` 接受 SSA size，理论上可让 prefetch 深度变成 kernel arg。需要让 `multi_tile_buf.count` 字段允许动态值——类型系统支持代价较大，**不建议短期做**，作为长期方向。

### 11.8 函数尾 barrier 与 multi-buffer 协同进一步降级

§10.3 的尾 barrier 优化只看 `bmu_free` 的 pipe 覆盖。multi-buffer 引入后，bmu_free 之间还有隐含的同 pipe drain 关系：若 multi-buffer A 的 free 跑在 multi-buffer B 的 free 之前且同 pipe，A 的 drain 被 B 涵盖，A 的 bmu_free 可 fold 进 B（少一条指令）。

### 11.9 跨函数 multi-buffer 复用（明确不在范围）

BMU §7 规则 8 要求 END 前全 free。如果未来放宽（比如某些 inline 函数不算独立 kernel），multi-buffer 可跨函数 carry。这是体系级改动，非本设计范围。

### 优先级建议

| 优先级 | 优化项 | 复杂度 |
|---|---|---|
| 立即可做 | 11.1（段共享）、11.2（free pipe 选择）、11.3（早释放）、11.6（段隔离）、11.8（free fold） | 低 |
| 中期 | 11.4（容量自适应 N 协商）、11.5（分支条件化） | 中 |
| 长期 | 11.7（运行时 N）、11.9（跨函数）| 高 |

---

附：本文参考自 `BMU.md`（硬件设计文档）、`pto-isa/tests/npu/a5/src/st/testcase/tbuf/tbuf_kernel.cpp`（手写 demo）、PTOAS 现有源码与 ODS、`PTOAS/docs/designs/ptoas-multi-buffer-explicit-design.md`（multi-buffer PR 设计文档）。
