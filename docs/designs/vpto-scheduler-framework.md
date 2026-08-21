# VPTO 指令调度 Pass 设计与实现

本文只描述当前代码已经实现的`pto-vpto-scheduler`的行为。尚未实现或需要实测后调整的内容统一放在最后一节。

## 这个 Pass 解决什么问题

VPTO 指令最初按照前端展开和各级转换产生的顺序排列。这个顺序能正确执行，但不一定适合后续代码生成。例如，两条互不依赖的计算链可能排列成：

```text
A0 -> B0 -> C0
A1 -> B1 -> C1

原顺序：A0 B0 C0 A1 B1 C1
```

如果 A0 和 A1 都没有前置依赖，而 B 必须等对应的 A 完成，重新排列为下面的顺序可以优化流水：

```text
新顺序：A0 A1 B0 B1 C0 C1
```

这个 Pass 的工作就是：在不改变程序语义的前提下，重新排列同一基本块中的 VPTO 指令。选择顺序时，它同时考虑：

- 哪些指令之间存在必须遵守的先后关系；
- 一条指令的结果要经过多少 latency 才能被使用；
- 当前有多少个仍有后续使用点的 VPTO SSA value；当前 A5 模型只为 `!pto.vreg` 类型的 vector value 和 `!pto.mask` 类型的 predicate value 计算这项压力；
- 当多个指令都可以选择时，需要得到稳定、可重复的结果。

当前实现只处理 A5 vector kernel 中 `pto.vecscope` 和 `pto.strict_vecscope` 内的指令。它不会把指令移动到另一个基本块、另一个 vecscope，或者越过不能安全跨越的操作。

## 从输入到新顺序

对每个函数，Pass 先执行两种模式共用的静态分析：

1. **找出可以独立排序的连续指令段。** Pass 逐个扫描 block。具有 `OpTrait::IsTerminator` 的 op 是 block 的结束标记，它不进入调度区间，并结束当前连续指令段。自身包含 region 的 op 也不进入相邻指令段；Pass 随后单独遍历它的嵌套 block，内层 vecscope 则由自己的入口处理。没有 scheduling 分类且无法证明无 memory effect 的 op，以及明确标记为 `Unsupported` 的 op，同样会结束当前指令段。
2. **建立必须保持的先后关系。** 数据使用、可能冲突的内存访问、隐式状态读写和同步屏障都会形成依赖。
3. **分析原始顺序。** 查询 A5 模型，输出依赖图、分类覆盖率，以及每条指令执行前后的 vector/predicate 存活压力。模型不认识某条指令时，这些静态分析仍然可以完成。

`analyze` 在第 3 步结束，不产生新顺序。`on` 继续执行：

4. **计算新顺序。** 从前向后选择依赖已经满足的节点，并记录 direction、逻辑周期和选择原因。
5. **检查新顺序。** 确认结果是原调度区间的完整排列，并且没有破坏 Must edge 或 SSA 顺序。
6. **重放新顺序。** 使用全新的 Boundary 和 pressure tracker，确认依赖就绪周期和压力峰值可以复现。
7. **修改 IR。** 第 4 至 6 步全部成功后才按照结果移动操作；任一步失败都保持原顺序。

## 阅读本文需要的几个概念

| 本文用语 | 代码中的名称 | 含义 |
| --- | --- | --- |
| 调度区间 | `VPTOSchedRegion` | 一个基本块内连续、且可以独立分析和排序的一段操作 |
| 调度节点 | `VPTOSUnit` | 一条进入调度区间的操作在依赖图中的表示 |
| 必须依赖 | `Must` edge | 新顺序中前驱必须仍然排在后继之前 |
| 候选节点 | `Candidate` | 所有前置依赖已经满足、当前可以选择的节点 |
| 逻辑周期 | `issueCycle` | 根据依赖延迟划分的层级，不代表真实硬件同周期发射 |
| 寄存器压力 | pressure | 当前顺序下仍有后续使用点的 vector/predicate SSA value 数量 |
| A5 调度模型 | `VPTOGenericA5SchedModel` | 指令分类、逻辑延迟和压力参数的静态表 |

## 在编译流水线中的位置

### 执行位置

在 `ptoas.cpp` 的 `prepareVPTOForEmission` 中，关键的相邻 Pass 顺序如下。代码块使用 `Passes.td` 中注册的真实 Pass 名：

```text
...
canonicalize
cse
pto-vpto-scheduler      # mode != off 时加入
pto-validate-vpto-emission-ir
```

因此，`pto-vpto-scheduler` 紧跟最后一轮 `canonicalize` 和 `cse`，并位于 `pto-validate-vpto-emission-ir` 之前。对应的 C++ 创建调用是 `createVPTOSchedulerPass`。

### 运行范围

`pto-vpto-scheduler` 运行在 `ModuleOp` 上：

- `mode=off` 时立即返回，不检查目标架构；
- `mode=analyze` 或 `mode=on` 时，当前模块或外层模块必须具有 `pto.target_arch = "a5"`；
- 如果模块明确标记为非 vector kernel，则跳过；
- 没有 kernel kind 的模块仍可用于独立 Pass 测试；
- Pass 遍历模块内的函数，但只进入函数中的 vecscope。

Pass 会先找出函数中的所有 vecscope。处理一个 vecscope 时，它可以继续进入普通嵌套区域，但不会从外层 vecscope 再进入一个内层 vecscope；内层 vecscope 会由自己的入口单独处理，因此不会重复分析。

### 三种运行模式

| 模式 | 行为 | 是否修改 IR |
| --- | --- | --- |
| `off` | 不加入调度流程 | 否 |
| `analyze` | 输出 DAG、coverage 和原始顺序 pressure，不进入调度 | 否 |
| `on` | 在和`analyze`相同的静态分析之后计算、检查、重放并应用新顺序 | 是，仅修改检查通过的调度区间 |

Pass 自身默认 `off`。`ptoas` driver 的默认行为是：

- A5 且没有显式传递 `--vpto-scheduler`：使用 `on`；
- 其他架构且没有显式传递该选项：使用 `off`；
- 显式指定 `off`、`analyze` 或 `on` 时，以用户选择为准。

`on` 默认只报告跳过调度的情况。`--vpto-scheduler-trace` 或 Pass 选项 `trace=true` 会先输出与 `analyze` 相同的静态分析报告，再输出最终顺序、逻辑周期、峰值压力和工作量计数；trace 只能配合 `on`。

`on` 与 `--enable-bisheng-vec-misched` 互斥，避免两个调度器连续重排同一组 vector 指令。`analyze` 不修改 IR，可以与 Bisheng vector MISched 同时配置。

## 整体结构

```text
Operation
  -> getVPTOSchedulingSemantics()
  -> VPTOSchedulingSemantics
       分类、隐式影响、内存访问
  -> VPTOSchedRegionBuilder
  -> VPTOSchedRegion[]
       基本块内的连续调度区间及两侧边界
  -> VPTOSchedDAGBuilder + VPTOGenericA5SchedModel
  -> VPTOSchedDAG
       VPTOSUnit[]：操作、原始位置、初始依赖计数、depth/height
       VPTOSchedEdge[]：数据、内存、隐式状态、同步、保序依赖
       live-ins / live-outs
  -> 调度前分析报告（analyze 或 on+trace）
       DAG / edge / coverage
       VPTORegPressureTracker：原始顺序 delta/current/peak
  -> mode
       analyze -> 结束，不产生 VPTOScheduleResult
       on
         -> VPTOScheduler 从 Top VPTOSchedBoundary 获取 available 节点
         -> 构造 VPTOScheduleContext + VPTOSchedCandidate[]
         -> VPTOSchedStrategy::pickCandidate()
         -> VPTOSchedDecision
         -> VPTOSchedBoundary::commit(..., VPTOSchedulingBudget)
         -> VPTOScheduleResult
         -> verifyVPTOScheduleResult()
         -> replayVPTOScheduleResult()
         -> applyVPTOScheduleResult()
```

## 如何判断一条操作能否参与调度

### 每条操作需要提供的信息

在划分调度区间之前，Pass 先回答三个问题：这条操作能否参与调度、它是否读写了没有出现在操作数中的隐式状态、它是否访问了内存。回答被统一保存在 `VPTOSchedulingSemantics` 中：

```text
VPTOSchedulingSemantics
  schedulingClass
  classificationKnown
  effects[]
  memoryBehavior
  memoryAccesses[]
```

| 字段 | 含义 |
| --- | --- |
| `schedulingClass` | 这条操作能否进入调度区间 |
| `classificationKnown` | 当前分类是否来自明确规则，而不是保守兜底 |
| `effects` | 未直接体现在 SSA 输入输出中的状态读写或同步影响 |
| `memoryBehavior` | 是否具有完整、明确的普通内存访问说明 |
| `memoryAccesses` | 规范化后的具体读写范围 |

`schedulingClass` 有四种取值：

| 分类 | 含义 |
| --- | --- |
| `Schedulable` | 进入调度区间，并作为可选择的调度节点 |
| `Structural` | 没有通过调度接口分类的辅助操作；仍与相邻指令一起进入调度区间，以保留数据关系 |
| `SchedulingBoundary` | 结束当前调度区间，自身不参与排序 |
| `Unsupported` | 已明确知道当前实现不能调度它；结束当前调度区间 |

这里有两层分类，不能混淆：`schedulingClass` 决定操作是否进入调度区间；A5 模型中的 `sched class` 决定进入后使用什么延迟和压力参数。当前 A5 模型按 operation 声明的 pipe 或 micro-op family 分配通用 sched class，不维护 opcode 白名单。防御性的 unknown 处理仍保留给将来没有 family/pipe 覆盖的模型实现；`analyze` 会继续报告这类区间的 DAG 和原始顺序压力，`on` 则保持该区间的原顺序。

### 操作分类顺序

`getVPTOSchedulingSemantics` 按以下顺序工作：

1. 基本块结束操作，或自身包含嵌套区域的操作，作为已知 `SchedulingBoundary`；
2. 实现 `VPTOSchedulingOpInterface` 的操作使用接口提供的信息；
3. 未实现接口但确认没有内存影响的操作作为已知 `Structural`；
4. 其他操作保守地作为 `SchedulingBoundary`，并在覆盖率报告中标为 unclassified。

`Unsupported` 表示“明确不支持”，不表示“尚未分类”，只能由接口显式返回。

默认接口会把具有执行 pipe、已知操作族或隐式影响的操作标为已知 `Schedulable`。

### 未体现在输入输出值上的影响

| 影响 | 当前来源 | 如何限制顺序 |
| --- | --- | --- |
| `Barrier` | `pto.mem_bar` | 屏障前的操作不能移到屏障后，反之亦然 |
| `AtomicMemory` | atomic CAS/exchange/add/sub/min/max/and/or/xor | 内存访问必须保持顺序 |
| `VolatileMemory` | `volatile` 或 `is_volatile` 属性 | 内存访问必须保持顺序 |
| `PostUpdate` | 返回更新后地址的 load/store 类操作 | 地址结果仍按普通数据依赖保序，同时记录更明确的原因 |
| `ImplicitWrite` | `pto.sprclr`、`pto.set_ctrl` | 与同一隐式状态的读写保持顺序 |
| `ImplicitRead` | `pto.sprsti`、`pto.sprsts`、`pto.get_ctrl` | 必须位于对应的最近一次隐式写之后 |

当前识别“返回更新后地址”的操作为：

```text
pto.vlds, pto.vldsx2, pto.sprsti, pto.sprsts, pto.vldus,
pto.plds, pto.pldi, pto.psti, pto.vsts, pto.psts,
pto.vsldb, pto.vsstb, pto.vstas
```

### 如何描述内存访问

每次内存访问用 `VPTOMemoryAccess` 表示。它记录地址、地址空间、可选的字节偏移和长度、读写方向、是否必须保序，以及信息是否完整。

当前规则为：

- `pto.mem_bar`、`pto.sprclr`、`pto.get_ctrl`、`pto.set_ctrl` 明确没有普通 memory access；
- 没有 `MemoryEffectOpInterface` 且不是 memory-effect-free 的 operation 生成一个无地址、ordered、unknown 的保守 write access；
- 实现 `MemoryEffectOpInterface` 时，只记录无 value 的 effect，或 value 类型为 `!pto.ptr`/memref 的 effect；
- Read 映射为 read，Write/Allocate/Free 映射为 write；
- store-like operation 即使接口只给出模糊 effect，也强制标为 write；
- atomic 或 volatile access 统一标为 ordered；
- ordered operation 没有可枚举 access 时，补一个 read+write、ordered、unknown access。

以下 operation 在 offset 为常量且元素 byte size 可计算时记录静态 byte 区间：

```text
pto.load, pto.store, pto.ldg, pto.stg, pto.ld_dev, pto.st_dev
```

区间计算使用：

```text
byteOffset = elementOffset * elementByteSize
byteSize   = elementByteSize
```

溢出、scalable vector、非整数字节元素宽度或动态 offset 会使区间保持未知。

## 如何划分调度区间

### 扫描规则

调度区间构建器 `VPTOSchedRegionBuilder` 逐个扫描基本块中的操作：

- `Schedulable` 和 `Structural` 追加到当前连续片段；
- `SchedulingBoundary` 或 `Unsupported` 结束当前片段，并成为相邻调度区间的边界；
- 到达基本块末尾时结束最后一个片段；
- 只有至少包含一个 `Schedulable` 的片段才成为调度区间；只有辅助结构操作的片段被忽略。

每个 `VPTOSchedRegion` 保存所属基本块、区间编号、连续操作列表、跨越整个区间的 live-through 值，以及前后边界和形成边界的原因。原因可能是 block 开始/结束、实现 `OpTrait::IsTerminator` 的 op、包含 nested region 的 op，或 `分类名:操作名`。

构建器在基本块内反向计算活跃值，并从函数级 liveness 引入基本块出口值。对于嵌套在 loop-like operation 中的基本块，定义于循环外但被循环区域捕获的值在回边上仍然存活，因此也作为基本块出口活跃值参与计算。一个值如果在调度区间入口和出口都活跃，就记录为该区间的 live-through；即使区间内没有直接引用它，它占用的寄存器压力也不能忽略。

### 分类覆盖率

Pass 以函数为单位统计分类覆盖率：

- 四种操作分类各有多少条；
- 每种边界原因出现多少次；
- 明确不支持的操作；
- 因没有分类规则而成为边界的操作。

`analyze` 总是输出覆盖率；`on` 只在 trace 开启时输出。

## 如何建立必须保持的先后关系

### 依赖图中的节点和边

每个调度区间中的操作对应一个依赖图节点 `VPTOSUnit`。节点的 `id` 和 `originalIndex` 当前都等于该操作在原区间中的位置。

依赖边支持以下种类：

```text
Data, Anti, Output, Memory, Control, Sync, Artificial, Cluster
```

依赖强度支持 `Must` 和 `Weak`。当前构建器生成的正确性依赖全部是 `Must`，表示新顺序必须保留这条边的前后关系。只有 `Must` 依赖参与环检查、候选节点判断和依赖链长度计算。

### 建图顺序

依赖图按固定顺序构建：

```text
SSA
  -> memory
  -> implicit state and sync
  -> defensive unknown-model fallback
  -> critical path
  -> dependency counts
```

如果 `Must` 依赖形成环，当前调度区间不能产生合法顺序，建图失败。

### 数据依赖和区间外数据

对每个输入值：

- 定义该值的操作在同一依赖图中：从定义者到使用者建立 `Data/Must` 依赖；
- 依赖延迟取定义者在 A5 模型中的 `writeLatency`；
- 定义者不在当前调度区间：该值记为入口存活值，即 live-in。

如果一个结果在当前调度区间之外仍被使用，就记为出口存活值，即 live-out。定义者位于区间外、且在区间入口和出口都活跃的 live-through 值，同时记为 live-in 和 live-out；这既包括“在区间内使用后仍继续存活”的值，也包括“区间内完全不引用但跨过区间”的值。这样正向跟踪不会遗漏或提前释放周边代码仍需保留的寄存器，反向跟踪也会从正确的出口压力开始。

更新后地址与普通 SSA 结果使用相同的数据依赖，只把原因记录为 `post-update address operand #N`，方便分析报告定位。

### 内存访问依赖

对于原顺序中的每一对“较早操作 -> 较晚操作”，Pass 判断它们的内存访问是否可能冲突：

1. 沿仓库已有的别名关系追踪到根地址；
2. 如果追踪过程中地址发生变化，原来的静态偏移和长度不再可信；
3. 两侧地址空间都已知且不同，可以确认不冲突；
4. 根地址相同且两侧都有完整静态区间时，按半开区间是否重叠判断；
5. 其他情况都保守认为可能冲突，包括同一物理空间内的不同 SSA 根地址。

两次访问可能冲突，并且满足下列任一条件时，按原顺序建立 latency 为 0 的 `Memory/Must` 依赖：

- 任一访问明确要求保序；
- 任一访问的信息不完整；
- 任一访问会写内存。

因此，只有信息完整、无需保序的纯读组合可以自由交换；其他无法证明安全的组合都保持原顺序。

### 隐式状态和同步依赖

对于同一个隐式状态名称：

- 最近一次写到后续读：`Data/Must`，latency 1；
- 最近一次写到后续写：`Output/Must`，latency 0；
- 自上次写以来的所有读到下一次写：`Anti/Must`，latency 0。

完整屏障建立两侧依赖：

- 当前调度区间内所有更早节点到屏障的 `Sync/Must` 依赖；
- 最近一个屏障到所有后续节点的 `Sync/Must` 依赖；
- latency 均为 0。

### 模型不认识某个节点时

当前 A5 模型对所有由 scheduling semantics 判定为 `Schedulable` 的 operation 提供通用 sched class。Vector、Cube、MTE 和 SIMT micro-op 按各自 family 分类；带 `OpPipeInterface` 的 operation 优先按 pipe 分类；仍无法细分的 schedulable operation 使用已知的 `generic-zero` 类。

框架仍允许其他模型返回 `known=false`。遇到这种节点时，依赖图构建器会在它与原顺序相邻节点之间增加 latency 为 0 的 `Artificial/Must` 保序依赖。这些边用于静态分析报告和依赖图完整性。

`analyze` 会把这些节点报告为 `known=false`，但不会称为调度 fallback。`on` 在真正排序前检查整个调度区间；只要存在一个模型未知节点，就列出所有未知操作并保持整个区间的原顺序，不会只重排其中一部分。

### 依赖链长度

依赖图沿 `Must` 依赖计算每个节点距离入口和出口的最长逻辑延迟：

```text
depth(successor) = max(depth(successor), depth(node) + edge.latency)
height(node)     = max(height(node), height(successor) + edge.latency)
```

`height` 表示从当前节点到区间出口还剩多长的依赖链，是候选节点的第三级选择条件；`depth` 表示从入口到当前节点的长度，用于分析报告。

## 当前 A5 调度模型

本节列出当前代码中 `VPTOGenericA5SchedModel` 会实际影响调度的静态数据。`analyze` 和 `on` 共用同一个 model instance。

### 基本参数

| 字段 | 当前值 |
| --- | --- |
| target | `a5` |
| version | `generic-a5-v2` |

当前 A5 模型中与 resource 和 hazard 有关的字段或实现都是为框架占位的 mock 值。`VPTOSchedBoundary` 仍持有相应 tracker，保留后续扩展契约，但当前调度、重放和 `analyze` 报告都不使用或展示这些数据，不能据此得出实际硬件性能分析结论。

### 逻辑延迟

当前所有 Vector micro-op，以及声明 `PIPE_V`/`PIPE_V2` 的 operation，共用 `vector-predicate` sched class，并按非零 `write latency` 处理，当前值为 10。这个延迟只用于“结果定义者 -> 同一调度区间内使用者”的数据依赖。Scalar、Cube、MTE、Control、Structural 和未细分的通用 sched class 当前不增加这类逻辑等待时间。

### vector/predicate SSA value 的压力参数

| ID | 名称 | limit | weight | spill cost | value contribution |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | vector | 32 | 1 | 1 | 每个 `!pto.vreg` 为 1 unit |
| 1 | predicate | 7 | 1 | 1 | 每个 `!pto.mask` 为 1 unit；按当前后端实际分配的 P1-P7 校准 |

其他值类型不计入压力。`limit` 不是禁止调度的硬上限；如果所有候选节点都会超限，Pass 仍会选择压力恶化最小的节点，把当前调度区间排完。

## 共同分析前缀与 `on` 调度后缀

`analyze` 和 `on` 共用调度区间划分、依赖图构建、分类覆盖率统计和原始顺序压力分析。原始顺序压力报告逐节点输出 `delta/current/peak`，不要求所有 sched class 都是 known，因此防御性的 unknown region 仍然有完整的静态分析结果。

`analyze` 在这个公共前缀结束后直接返回，不调用 Scheduler、结果检查器或 model replay，也不产生 `VPTOScheduleResult`。`on` 才继续执行调度、检查、重放和应用；启用 trace 时，它输出与 `analyze` 相同的调度前报告，并额外输出 `schedule-result`。当前两种报告都不展示占位的 resource/hazard 数据。

## `on` 模式的调度算法

### 开始排序前的检查

每个调度区间使用以下编译工作量上限：

| 预算 | 当前值 |
| --- | ---: |
| 最大节点数 | 4096 |
| 最大边数 | 262144 |
| 建图、候选评价、Boundary 提交/队列维护、结果检查、重放和应用共享的工作量计数 | `2^20` |

节点数在分配 DAG 节点前检查；每条依赖边在分配前检查边数上限。`on` 的建图、排序、结果检查、重放和应用共用同一个调度预算。建图完成后，排序入口仍会防御性地复查节点数和边数，并确认 A5 模型认识区间中的所有节点。

工作量计数在以下位置增加：

- 建图时每扫描一个 live-through 值、SSA operand 或 result user；
- 建图时每解析一个 memory access、遍历一个候选节点对或比较一对 memory access；
- 建图时每扫描一个隐式 effect、模型分类或新增一条依赖边；
- 计算关键路径时每处理一个节点或一条边；
- 每评估一次候选节点；
- Boundary 提交时每检查一条相邻依赖边，并分别计算依赖更新的验证和应用；
- available 队列的常数时间插入/删除，以及 pending 周期桶插入的对数复杂度上界；
- 每次推进周期时确定需要释放的 pending 周期桶，并预付本次 cycle 更新和节点释放工作；
- 结果检查时每扫描一个区间节点、结果项、DAG 节点、依赖边或 SSA operand；
- 重放时每处理一个结果节点；
- 重放调用同一个 Boundary commit/advance 接口，不绕过上述计数；
- 应用结果前一次性预付全部节点移动工作。

预算使用确定的计数，不使用依赖机器性能的墙钟超时。`VPTOSchedBoundary::commit` 先扫描依赖并预付本次依赖更新和队列维护的全部工作，再修改 pressure tracker、节点状态或队列；预算不足时不会出现部分提交。静态报告使用独立的报告预算；输出 baseline pressure 不会消耗 `on` 的调度预算，因此打开 trace 不会改变调度的预算起点、成功与否或最终顺序。

### 节点状态和逻辑周期

`VPTOSchedBoundary` 为当前方向独立维护依赖计数、ready cycle 和节点状态，实际存储结构如下：

```text
states / remainingDependencies / readyCycles
  SmallVector<...>，以 VPTOSUnit::id 直接索引

available
  SmallVector<VPTOSUnit *> + SmallVector<size_t> availablePositions
  无序稠密数组；按 id 位置表做 O(1) 成员检查和 swap-remove

pending
  std::map<unsigned, SmallVector<VPTOSUnit *>> 周期桶
  key 是 readyCycle；同一周期按 originalIndex 保持确定顺序
```

节点状态含义为：

| 状态 | 含义 |
| --- | --- |
| `Unavailable` | 仍有必须排在它前面的节点没有调度 |
| `Pending` | 前置节点都已调度，但依赖延迟尚未满足 |
| `Available` | 前置节点和依赖延迟都已满足，当前作为候选可以选择 |
| `Scheduled` | 已提交到结果 |

初始没有前置依赖的节点在逻辑周期 0 成为候选。提交节点后，后继节点最早可以被选择的周期更新为：

```text
readyCycle(successor) =
  max(readyCycle(successor), issueCycle(node) + edge.latency)
```

当后继节点的所有前置依赖都已处理时：

- 最早可选周期不晚于当前周期：立即成为候选；
- 否则进入等待状态。

没有候选节点时，当前逻辑周期直接跳到最早 pending 周期，不逐周期空转。同一周期的节点会在预算检查通过后整桶加入 available。

通常只要仍有候选节点，Pass 就在同一个逻辑周期继续选择。唯一例外是所有当前候选都会超过至少一个已知压力上限、并且仍有依赖等待节点时，Pass 会执行 pressure-driven idle，推进到最早的等待周期后重新评价候选。这里的周期只表示依赖层级，同一逻辑周期可以记录多个节点，不能解释为真实硬件同周期发射。

### Strategy、Candidate 和 Decision 契约

Scheduler 不直接实现候选评分策略。每轮选择使用以下公开契约：

```text
VPTOScheduleContext
  model / dag
  direction / issueCycle
  currentPressure

VPTOSchedCandidate
  unit / direction / issueCycle
  criticalPath / originalIndex
  pressure(delta/released/introduced/projected/projectedExcess)

VPTOSchedStrategy::pickCandidate(context, candidates)
  -> VPTOSchedDecision
       unit / direction / issueCycle / reason

VPTOScheduler
  -> 验证 Decision 确实来自本轮 Candidate
  -> VPTOSchedBoundary::commit(unit, issueCycle, budget)
  -> VPTOScheduleEntry
       unit / direction / issueCycle / reason
```

`VPTOScheduler` 构造 Context 和 Candidate，但不解释其中的评分信息；`VPTODefaultSchedStrategy` 负责当前 pressure、critical path 和原始位置策略。Scheduler 构造函数接受任意 `VPTOSchedStrategy`，因此增加新策略不需要修改调度主循环。Decision 的方向、逻辑周期和选择原因会随节点一起保存到 `VPTOScheduleResult`，on trace 因而可以说明节点是根据 excess、closure group、有限前瞻、critical path、pressure delta 还是确定性 tie-break 选中的。Candidate 保存即时 pressure、固定深度前瞻、是否打开新的 pressure frontier，以及是否推进当前 closure group；后续仍可在不改变 Scheduler/Boundary 职责的情况下增加真实的 resource 或 hazard 评价。

### 计算一个候选节点的压力影响

对每个候选节点，Scheduler 通过正向压力跟踪器生成 Candidate 的 pressure 评价：

```text
pressureDelta[s]
releasedPressure[s]
introducedPressure[s]
projectedPressure[s] = currentPressure[s] + pressureDelta[s]

if limit[s] is known:
  currentExcess[s]   = max(0, currentPressure[s] - limit[s])
  projectedExcess[s] = max(0, projectedPressure[s] - limit[s])
else:
  currentExcess[s]   = 0
  projectedExcess[s] = 0
```

其中：

- `pressureDelta`：选择该节点后，存活值数量增加或减少多少；
- `releasedPressure`：正向调度中该节点作为最后一次使用而结束的 live range 数量；
- `introducedPressure`：正向调度中该节点结果新建立的 live range 数量；
- `projectedPressure`：选择后的存活值数量；
- `currentExcess`：当前已经超过模型上限多少；
- `projectedExcess`：选择后会超过模型上限多少。

默认 Strategy 再汇总出 excess、普通 pressure delta，以及只对接近上限的 pressure set 生效的 projected/released 代价。具有已知上限的集合在当前压力达到上限一半时进入 near-limit 状态，达到上限三分之二时进入 high-pressure 状态：

```text
excessGrowthCost =
  sum(spillCost[s] *
      max(0, projectedExcess[s] - currentExcess[s]))

projectedExcessCost =
  sum(spillCost[s] * projectedExcess[s])

pressureDeltaCost =
  sum(weight[s] * pressureDelta[s])

nearLimitProjectedCost =
  sum(spillCost[s] * projectedPressure[s])

nearLimitReleaseCredit =
  sum(spillCost[s] * releasedPressure[s])
```

这些代价分别表示“是否/多少新增超限”“选择后总共超限多少”“临界压力下选择后的存活量和关闭的 live range 数”“普通状态下存活值总量倾向增加还是减少”。乘法和累加都会检查 `int64_t` 溢出。压力上限是可选数据：没有可信上限时，该集合不参与 excess 或 near-limit 代价，但仍以 `weight * pressureDelta` 参与普通 delta 代价，Tracker 也继续记录它的 current 和 peak。Strategy 不为这种集合虚构默认上限。权重和超限代价不能为负，否则当前调度区间按模型无效处理并保持原顺序。

### Near-limit 多用户 closure group

压力达到已知上限的一半后，Scheduler 从 normalized pressure 最高的有界集合开始处理；已经超限的集合优先。它按原始位置遍历 pressure tracker 当前的 live values，并把同一个 operation 产生、属于同一压力集合且仍存活的多个结果视为一个 target bundle。当前实现每轮只评价最早的一个 bundle；直接用户总数超过 8 的高 fan-out 值不进入该启发式，避免把接近全局的 mask 或公共值扩展成大范围调度约束。

对 bundle 的所有尚未调度用户，Scheduler 沿正向 `Data/Must` 边收集 closure core；为使 core 节点变为 ready，再沿反向 `Data/Must` 边收集必要的 support 节点。support 节点的其他无关用户不会继续扩展 group，因此共享的 mask、常量或搬运准备不会把整个 region 纳入 group。group 最多模拟 96 个节点，模拟时仍按合法依赖顺序选择最小 excess、最小加权压力和最早原始位置的节点。

只有模拟满足以下条件时，group 才会激活：所有 target value 已结束；中途峰值不超过 `max(起始压力, 压力上限)`；扣除 group 为 closure 准备、但在 target 结束时仍存活的 support-only 结果后，结束压力不高于起始压力。最后一个完成这些条件的节点作为 group target，group 在该节点实际完成前缓存。Strategy 在不增加 excess 的前提下优先选择 group 内节点，允许 group 的第一步即时压力略增，只要完整模拟已经证明不会产生新的 spill 风险并最终关闭目标 live range。group 激活后不再对其候选重复固定深度前瞻。

该启发式只依赖 SSA、DAG 边、压力集合和 live range，不识别 `vsel`、store、tile 或其他特定 opcode。bundle 收集、依赖遍历和模拟均计入共享 work-unit 预算；超过预算时当前调度区间保持原顺序。

### 多个候选节点之间如何选择

`VPTODefaultSchedStrategy` 按以下顺序逐项比较候选节点；只有上一项相同才比较下一项，不把所有指标混成一个总分：

1. 优先 pressure-safe candidate，再比较 `excessGrowthCost` 和 `projectedExcessCost`；
2. 有活动 closure group 时，先比较是否推进 group，再比较目标集合的即时 projected pressure 和 release credit；
3. high-pressure 集合再比较汇总后的 projected pressure 和 release credit；
4. near-limit 集合依次比较固定深度前瞻的 excess/risk、critical-path urgency window、是否打开新 pressure frontier、前瞻结束压力和即时 projected/released pressure；
5. 上述指标相同后，继续优先更大的 `height`，再比较更小的 `pressureDeltaCost`；
6. 最后选择更小的 `originalIndex`，并记录 `deterministic-tie-break`。

最后使用原始位置决胜，保证相同输入和模型总能得到完全相同的顺序。

Strategy 返回的 `VPTOSchedDecision` 同时记录节点、方向、逻辑周期和选择原因。Scheduler 先验证 Decision 属于本轮 Candidate，再通过 `VPTOSchedBoundary::commit` 预付预算并一次性更新压力、方向局部依赖计数和 available/pending 队列，最后把节点、方向、当前逻辑周期、选择原因以及该节点前是否发生 pressure-driven idle 追加到调度结果。Scheduler 不维护第二套 ready queue。所有节点都选完后，结果同时记录 vector/predicate 压力峰值。

### 一个完整的排序例子

对于两条独立、且每一段结果延迟都为 10 的三段链：

```text
A0 -> B0 -> C0
A1 -> B1 -> C1
```

开始时 A0/A1 都是周期 0 的候选；B0/B1 到周期 10 才能选择；C0/C1 到周期 20 才能选择。一种可能的宏观顺序是：

```text
A0 A1 B0 B1 C0 C1
```

即描述性的 AABBCC。但 ABCABC/AABBCC 不是算法目标或输出契约；实际细粒度顺序由 ready 状态、critical-path urgency 和 pressure 决定。

当一个 pressure producer 和 last-use consumer 同时可选、当前 headroom 会被一个候选耗尽、并且 consumer 仍处于 urgency window 时，consumer 会在真正超限前优先。低压力时仍优先关键链；consumer 落在 urgency window 外时，紧急关键链也仍然优先。如果消费者尚在等待依赖延迟，且至少一个当前候选不会超过压力上限，Pass 会继续选择当前候选；如果所有当前候选都会超过至少一个已知压力上限，Pass 会推进到最早的 pending 周期，让压力释放节点参与下一轮候选评价。没有 pending 节点时，上限仍是软约束，Pass 必须继续取得进展。

## 如何检查并应用新顺序

### 先生成结果，不立即移动操作

`VPTOScheduler::schedule` 只返回：

```text
VPTOScheduleResult
  entries[] = {unit, direction, issueCycle, reason, pressureDrivenIdle}
  peakPressure[]
```

在调度结果产生、正确性检查和独立重放完成之前，调度区间中的操作不会移动。

### 检查结果是否仍然正确

结果检查器确认：

- 原调度区间仍在同一个基本块中，前后边界和连续性没有变化；
- 结果节点数等于原区间节点数；
- 每个结果项都指向当前基本块中的有效节点，且没有重复；
- 原区间中的每个节点恰好出现一次；
- 每条 `Must` 依赖的前驱仍位于后继之前；
- 每个区间内定义的 SSA 值仍然先定义、后使用。

### 用全新状态再执行一遍结果

重放阶段重新创建一个完整的 Top `VPTOSchedBoundary`，不复用第一次排序时已经改变的依赖、队列或压力状态。Boundary 只读共享 DAG，因此重放不会受到首次调度的状态污染。结果项同时包含普通依赖等待和 pressure-driven idle 时，重放先推进 available 为空所强制要求的 pending 依赖事件，再验证剩余的压力等待。它逐个检查结果中的节点：

- 工作量预算仍可用；
- 没有候选时，确实可以推进到最早的等待周期；
- `pressureDrivenIdle=true` 时，当前 available 候选确实都会增加已超限或临界集合的压力风险，并且可以沿 pending 事件推进到结果记录的逻辑周期；
- 当前节点通过 Boundary 的 O(1) `isAvailable` 检查，并且结果记录的逻辑周期一致；
- Boundary 的压力状态和 available 队列都能接受该节点；
- 重放结束后所有节点都已处理；
- 重放得到的压力峰值与结果记录完全一致。

第一次排序、结果检查和独立重放共享同一个工作量上限。

### 修改 IR

结果检查和独立重放都通过后，应用阶段先从共享预算中一次性预付全部节点移动工作，然后按结果顺序把操作移动到调度区间后边界之前；区间位于基本块末尾时，就移动到基本块末尾。应用阶段不重复执行完整结果检查，因为从检查、重放到应用之间没有其他 IR 修改。

当前 MLIR `moveBefore` 操作不会返回失败，因此代码先完成所有可能失败的检查，再开始移动操作。当前实现没有额外保存原顺序，也没有事务式回滚；由于所有可报告失败都发生在移动前，失败时调度区间仍保持原顺序。

## 失败时如何处理

失败类别为：

```text
Budget, InvalidModel, Scheduling,
SemanticVerification, ModelReplay, Apply
```

处理规则：

- `on` 遇到 A5 模型不认识的操作：为当前调度区间中的每个未知操作发出 remark，包含基本块、区间编号、原始位置和操作名；`analyze` 只在报告中标记 `known=false`；
- 节点数、依赖边数或工作量超限，以及模型参数无效：remark；
- 无法继续选择节点、依赖图存在环、正确性检查失败、独立重放失败或应用阶段不一致：warning；
- 命令行模式、目标架构、trace 组合或 Bisheng 配置冲突：error，停止编译。

普通调度失败只影响当前调度区间，Pass 会继续处理后面的区间。`on` 未开启 trace 时，成功区间不输出报告；`analyze` 只输出调度前静态分析，不产生调度结果。

## 当前保证与明确限制

当前实现保证：

- `off` 和 `analyze` 不修改 IR；
- `analyze` 不依赖 sched class 完整性，也不调用 Scheduler、结果检查器或 model replay；
- `on` 只修改 sched class 完整、正确性检查和独立重放都通过的调度区间；
- 调度区间不会跨越基本块、vecscope 或明确边界；
- SSA 数据、保守内存冲突、隐式状态和完整屏障都由 `Must` 依赖保序；
- 相同 IR、A5 模型和选项会得到相同的调度顺序及跳过原因；
- 模型未知、预算超限和内部检查失败只影响当前调度区间；
- 静态报告使用独立预算，打开报告不会改变 `on` 的调度预算和结果；
- 逻辑周期、压力代价和预算计算都有整数溢出或上限检查。

当前实现限制：

- 逻辑周期只来自 `Must` 依赖延迟，不是硬件发射时间线；
- A5 的所有 Vector micro-op 和 `PIPE_V`/`PIPE_V2` operation 共用当前 vector/predicate 非零延迟；
- Scalar、Cube、MTE、Control、Structural 和未细分的 generic 类使用零调度成本；
- 只统计 vector 和 predicate 两类压力；
- pressure-driven idle 只在所有当前候选都会超过已知压力上限时触发，仍依赖当前未校准的逻辑 latency，不代表真实硬件空转周期；
- 多用户 closure group 当前每轮只评价最早的一个低 fan-out bundle，并使用 8 个直接用户和 96 个模拟节点的固定上限；
- 不支持跨基本块调度、双向调度、指令捆绑/配对、bank conflict、NOP、软件流水或 Cube kernel 调度；
- 修改 IR 时没有独立事务回滚，因为当前移动操作不可失败；
- 不设置额外的收益门槛；新顺序合法且通过重放就会应用，即使新旧顺序相同也允许执行移动流程。

## 测试覆盖

当前测试覆盖：

| 测试 | 主要内容 |
| --- | --- |
| `vpto_scheduler_analyze.pto` | off/analyze IR 不变、原顺序压力、analyze 无 schedule-result、on trace、非法 pass mode |
| `vpto_scheduler_cli.pto` | A5 默认 on、显式模式、trace 约束、target 约束、Bisheng 冲突 |
| `vpto_scheduler_coverage.pto` | boundary reason 和 unclassified coverage |
| `vpto_scheduler_dependencies.pto` | SSA、memory range、volatile、unknown memory、post-update、SPR/CTRL、barrier |
| `vpto_scheduler_on.pto` | analyze 只做静态分析、on trace 双链示例、pressure tie-break、pressure-driven idle、报告开关不改变 IR |
| `vpto_scheduler_generic_op_coverage.pto` | vcvt/vmul/vdiv/vexp/vmula/vcadd 等通用 Vector micro-op 使用统一 sched class，on 不因 opcode 未登记而跳过 region |
| `vpto_scheduler_multi_user_closure.pto` | near-limit 多用户 predicate closure group 可接受安全的瞬时压力增加，并在打开无关 producer 前关闭完整 fan-out live range |
| `vpto_scheduler_trackers.pto` | live-through 与无上限 pressure、near-limit/closure-group/低压力/紧急 critical-path/tie-break 策略、Predicate limit 7、无 pending 进展、非法 idle replay、独立 top/bottom Boundary、fan-out commit 原子预算、pending cycle buckets、verify/replay、随机 DAG differential test |
| `bisheng_vec_misched_cli.pto` | Bisheng vector MISched 选项存在性 |

随机 DAG differential test 使用 8 个固定 seed，覆盖完整 permutation、Must edge、ready cycle、独立 pressure oracle、decision metadata、非法结果拒绝、精确/不足预算以及最终 apply 顺序。

## 后续可能的迭代目标

以下项目不是当前行为，实施前需要分别补充模型依据、正确性测试和性能验收：

1. **校准 A5 模型数据**：根据 hardware RA、CA 和 NPU 数据校准 vector/predicate limit、contribution、weight、spill cost 和 operation latency，避免把当前静态初值当作最终硬件事实。
2. **细化 operation 模型**：在通用 pipe/family 覆盖之上，根据可信硬件数据为确有差异的 operation 增加可审计的 latency 或 resource 参数；没有数据时继续使用通用类。
3. **建立代表性性能 gate**：完善 compare/select、mask-or+scatter 和双链 AABBCC fixture，记录静态 pressure、最终 assembly spill/barrier 指标、CA/NPU ticks、EXIPC、编译时间和内存增量。
4. **扩展调度范围与策略**：在 correctness semantics 和模型充分后评估 bidirectional scheduling、跨 block调度、bundle/pair、software pipelining 及 Cube kernel scheduling。
5. **模型声明机制**：在静态 C++ 模型稳定后评估 TableGen 或生成式描述，同时保持 `VPTOSchedModel` 只读接口不变。
6. **可失败 apply 的事务保护**：如果以后 apply 引入可能失败的 bundle 构造、跨 block移动或其他 mutation，在移动前保存完整原顺序并实现显式 rollback。
7. **预算再校准**：基于真实 region 的 node、edge、Candidate evaluation 和 replay 分布调整当前常量，但继续使用确定性计数而非墙钟超时。
