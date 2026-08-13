# VPTO 调度框架设计与实现

本文描述当前 VPTO 调度框架的完整实现。原阶段一的语义归一化、region、依赖 DAG、目标模型和分析 tracker，与原阶段二的确定性 list scheduler、结果验证、model replay 和 IR 重排已经组成同一条流水线，因此本文按当前组件关系统一说明，不再分别维护两套设计。

本文只陈述当前代码已经实现的行为。尚未实现或需要基于实测数据调整的内容集中列在最后一节。

## 1. 目标与作用范围

VPTO scheduler 在 emission-ready VPTO 上工作，目标是：

- 用统一的 operation-local scheduling semantics 构建 block-local region；
- 为 SSA、memory、隐式状态和同步建立 correctness dependency DAG；
- 在 `analyze` 模式下输出可复现的原顺序分析；
- 在 `on` 模式下执行确定性的 top-down list scheduling；
- 在修改 IR 前验证调度结果，并用全新状态重放结果；
- 当模型不完整、预算超限或内部检查失败时，只跳过当前 region。

当前实现只支持 A5 Vector kernel module，只处理 `pto.vecscope` 和 `pto.strict_vecscope` 内的 operation。它不跨 block、vecscope 或 scheduling boundary 调度。

## 2. 流水线集成与模式

### 2.1 集成位置

在 `ptoas` 的 VPTO emission preparation 中，scheduler 位于最后一轮 canonicalization/CSE 之后、emission IR verifier 之前：

```text
wrapper expansion / vecscope inference / loop normalization
  -> canonicalize
  -> CSE
  -> VPTO scheduler
  -> emission IR validation
```

因此 scheduler 看到的是已经完成主要结构化 lowering、但尚未交给下游代码生成的 VPTO。

### 2.2 Pass 约束

`pto-vpto-scheduler` 是 `ModuleOp` pass：

- `mode=off` 时立即返回，不要求 target 属性；
- `mode=analyze` 或 `mode=on` 时，当前 module 或祖先 module 必须具有 `pto.target_arch = "a5"`；
- 存在 kernel kind 且不是 Vector 时跳过该 module；
- 未标注 kernel kind 的 module 可用于独立 pass 测试；
- pass 遍历 module 内的 `func.func`，每个函数只处理其 vecscope 内部。

函数中的所有 vecscope 会先被收集。处理某个 vecscope 时，遍历其 region 中的 block 和普通嵌套 region，但不递归进入另一个 vecscope；嵌套 vecscope 由它自己的处理入口负责，避免重复分析。

### 2.3 模式行为

| 模式 | 行为 | 是否修改 IR |
| --- | --- | --- |
| `off` | 不运行调度 pass | 否 |
| `analyze` | 构建 region/DAG，按原顺序运行分析 tracker，输出完整报告 | 否 |
| `on` | 构建 region/DAG，调度、验证、重放并应用结果 | 是，仅限验证通过的 region |

Pass 自身的默认模式是 `off`。`ptoas` driver 根据有效架构计算默认值：

- A5 且用户没有显式传递 `--vpto-scheduler`：`on`；
- 其他架构且用户没有显式传递该选项：`off`；
- 用户显式指定 `off`、`analyze` 或 `on` 时，以显式值为准。

`on` 默认不打印成功 region 的详细信息，只报告 fallback。`--vpto-scheduler-trace` 或 pass option `trace=true` 会输出结果顺序、logical cycle、peak pressure 和 work-unit 用量；trace 只允许与 `on` 同时使用。

`on` 与 `--enable-bisheng-vec-misched` 互斥，driver 会直接报错。`analyze` 不修改 IR，因此可以与 Bisheng vector MISched 配置共存。

## 3. 统一组件与数据流

```text
VPTOSchedulingOpInterface / conservative fallback
  -> VPTOSchedulingSemantics
  -> VPTOSchedRegionBuilder
  -> VPTOSchedDAGBuilder
       -> SSA edges
       -> memory edges
       -> implicit-state and sync edges
       -> unknown-model fallback edges
       -> critical depth / height
  -> analyze
       -> top/bottom Boundary
       -> ResourceTracker
       -> RegPressureTracker
       -> NullHazardRecognizer
       -> original-order report
  -> on
       -> ReadyQueue + top RegPressureTracker
       -> deterministic Candidate strategy
       -> VPTOScheduleResult
       -> semantic verification
       -> fresh model replay
       -> region apply
```

主要数据对象如下：

| 对象 | 当前职责 |
| --- | --- |
| `VPTOSchedulingSemantics` | operation-local 分类、非 SSA effect 和规范化 memory access |
| `VPTOSchedRegion` | 一个 block 内连续且可独立处理的 operation 序列及两侧 boundary |
| `VPTOSUnit` | DAG 节点，保存原始位置、语义、依赖计数、depth 和 height |
| `VPTOSchedEdge` | 带 kind、strength、latency 和 reason 的有向边 |
| `VPTOSchedModel` | machine、resource、sched class 和 pressure set 的只读目标契约 |
| `VPTOSchedBoundary` | 分析框架中的方向局部 ready/pending、resource、pressure 和 hazard 状态 |
| `VPTOScheduleResult` | 调度后的完整节点序列、每个节点的 logical issue cycle 和 peak pressure |

## 4. Operation scheduling semantics

### 4.1 语义记录

每个 operation 被归一化为：

```text
VPTOSchedulingSemantics
  schedulingClass
  classificationKnown
  effects[]
  memoryBehavior
  memoryAccesses[]
```

四种 scheduling class 的含义是：

| 分类 | 含义 |
| --- | --- |
| `Schedulable` | 进入 scheduling region，并作为可调度节点 |
| `Structural` | 与相邻 schedulable operation 一起进入 region，以保留 SSA 结构 |
| `SchedulingBoundary` | 切断 region，自身不进入 region |
| `Unsupported` | 显式声明不支持调度，切断 region且自身不进入 region |

这里的 semantic class 与目标模型的 sched class 是两个层次：前者决定 operation 是否进入 region，后者决定进入 region 后的 latency、resource 和 pressure 模型是否完整。

### 4.2 分类优先级

`getVPTOSchedulingSemantics` 按以下顺序工作：

1. terminator 或包含 region 的 operation 返回已知 `SchedulingBoundary`；
2. 实现 `VPTOSchedulingOpInterface` 的 operation 使用接口返回的语义；
3. 未实现接口但 `isMemoryEffectFree` 的 operation 返回已知 `Structural`，且普通 memory behavior 为 `None`；
4. 其他 operation 返回未知分类的 `SchedulingBoundary`，由 coverage 报告为 unclassified。

`Unsupported` 不用于未知 fallback，只能由 scheduling interface 显式返回。

默认 scheduling interface 会把具有执行 pipe/family 或具有非空 scheduling effect 的 operation 标为已知 `Schedulable`。

### 4.3 当前非 SSA effect

| effect | 当前来源 | DAG 用途 |
| --- | --- | --- |
| `Barrier` | `pto.mem_bar` | barrier 前后建立完整顺序 |
| `AtomicMemory` | atomic CAS/exchange/add/sub/min/max/and/or/xor | 将 memory access 标为 ordered |
| `VolatileMemory` | `volatile` 或 `is_volatile` 属性 | 将 memory access 标为 ordered |
| `PostUpdate` | 带 updated-base result 的 load/store 类 operation | 标记对应 SSA edge 的原因 |
| `ImplicitWrite` | `pto.sprclr`、`pto.set_ctrl` | 建立隐式 output/anti dependency |
| `ImplicitRead` | `pto.sprsti`、`pto.sprsts`、`pto.get_ctrl` | 建立隐式 data dependency |

当前识别 post-update address 的 operation 为：

```text
pto.vlds, pto.vldsx2, pto.sprsti, pto.sprsts, pto.vldus,
pto.plds, pto.pldi, pto.psti, pto.vsts, pto.psts,
pto.vsldb, pto.vsstb, pto.vstas
```

### 4.4 Memory access 归一化

`VPTOMemoryAccess` 保存 address、address space、可选 byte offset/size、read/write、ordered 和 unknown。

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

## 5. Scheduling region

### 5.1 Region 形成

`VPTOSchedRegionBuilder` 逐 block 扫描：

- `Schedulable` 和 `Structural` 追加到当前片段；
- `SchedulingBoundary` 或 `Unsupported` 结束当前片段，并成为相邻 region 的 boundary；
- block 结尾刷新最后一个片段；
- 只有至少包含一个 `Schedulable` 的片段才生成 region，纯 `Structural` 片段被忽略。

每个 region 保存：所属 block、block 内 region index、连续 operation 列表、前后 boundary 指针及其原因。boundary 原因为 block start/end、terminator、contains-regions，或 `分类名:operation名`。

### 5.2 Coverage

coverage 以函数为单位累计：

- 四种 semantic class 的数量；
- boundary reason；
- 显式 unsupported operation；
- 未知分类而形成 boundary 的 unclassified operation。

coverage 由 `analyze` 总是输出，由 `on` 仅在 trace 开启时输出。

## 6. Dependency DAG

### 6.1 节点与边

每个 region operation 对应一个 `VPTOSUnit`，其 `id` 和 `originalIndex` 当前都等于原 region 中的位置。

edge kind 支持：

```text
Data, Anti, Output, Memory, Control, Sync, Artificial, Cluster
```

edge strength 支持 `Must` 和 `Weak`。当前 DAGBuilder 生成的 correctness edge 全部为 `Must`；只有 Must edge 参与拓扑合法性、ready 状态和 critical path。

### 6.2 构建顺序

DAG 按固定顺序构建：

```text
SSA
  -> memory
  -> implicit state and sync
  -> unknown-model fallback
  -> critical path
  -> dependency counts
```

如果 Must-edge 图存在环，DAG 构建失败。

### 6.3 SSA、live-in 与 live-out

对每个 operand：

- defining operation 在同一 DAG 中时，建立 `Data/Must` edge；
- edge latency 取 producer 的目标 sched class `writeLatency`；
- defining operation 不在 DAG 中时，operand 记为 live-in。

对每个 result，只要存在 DAG 外 user，就记为 live-out。

post-update result 与普通 SSA result 使用相同 correctness edge，只是 reason 记录为 `post-update address operand #N`。

### 6.4 Alias 与 memory edge

memory dependency 保持冲突 access 的原始先后顺序。对每一对 `earlier -> later` operation：

1. 沿 repository alias helper 追踪 address root；
2. alias root 发生变化时丢弃原 access 的静态 offset/size；
3. 两侧 address space 都已知且不同，则判定不 alias；
4. 同一 root 且两侧都有完整静态区间时，用半开区间重叠判断；
5. 其他情况保守认为 may-alias，包括同一物理空间内的不同 SSA root。

may-alias 后满足任一条件就建立 `Memory/Must`、latency 0 的 edge：

- 任一 access 为 ordered；
- 任一 access 为 unknown；
- 任一 access 为 write。

因此，已知且无序的 read-read 可以重排；不能证明不冲突的其他组合保持原顺序。

### 6.5 隐式状态与同步

对于同名隐式 resource：

- last write 到 read：`Data/Must`，latency 1；
- last write 到新 write：`Output/Must`，latency 0；
- write 前所有自上次 write 以来的 read：`Anti/Must`，latency 0。

完整 barrier 建立：

- region 内所有更早节点到 barrier 的 `Sync/Must` edge；
- 最近 barrier 到所有后续节点的 `Sync/Must` edge；
- latency 均为 0。

### 6.6 Unknown model fallback edge

如果 operation 已进入 region，但 A5 model 返回 `known=false`，DAGBuilder 会给它与原顺序相邻节点增加 `Artificial/Must`、latency 0 的保序 edge。该 edge用于 `analyze` 输出和 DAG 完整性。

`on` 在进入 scheduler 前会检测整个 region 的 model class；只要存在一个 unknown class，就报告所有 unknown operation 并跳过整个 region，因此不会依靠 artificial edge 对该 region 做部分重排。

### 6.7 Critical path

DAG 使用 Must edge 做拓扑遍历：

```text
depth(successor) = max(depth(successor), depth(node) + edge.latency)
height(node)     = max(height(node), height(successor) + edge.latency)
```

`height` 作为 `on` 模式的 Candidate 第三级比较项，`depth` 用于分析报告。

## 7. 当前 A5 目标模型数据

本节列出当前代码中 `VPTOGenericA5SchedModel` 的完整静态数据。`analyze` 和 `on` 共用同一个 model instance。

### 7.1 Machine model

| 字段 | 当前值 |
| --- | --- |
| target | `a5` |
| version | `generic-a5-v2` |
| issue width | 1 |
| micro-op buffer size | 0 |

### 7.2 Resources

| ID | 名称 | units | buffer size | group members |
| ---: | --- | ---: | ---: | --- |
| 0 | scalar | 1 | 0 | 无 |
| 1 | vector | 1 | 0 | 无 |
| 2 | mte | 1 | 0 | 无 |
| 3 | cube | 1 | 0 | 无 |
| 4 | control | 1 | 0 | 无 |
| 5 | unknown | 1 | 0 | 无 |

这些 resource 当前用于 `analyze` 的 ResourceTracker timeline。`on` 的 ReadyQueue 和 Candidate strategy 不查询 ResourceTracker，因此 issue width 和 reservation 当前不构成真实调度的 hard legality。

### 7.3 Sched classes

| ID | 名称 | known | micro-ops | write latency | resource reservation | read advance |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 0 | structural | true | 0 | 0 | 无 | 无 |
| 1 | scalar-zero | true | 0 | 0 | 无 | 无 |
| 2 | vector-predicate | true | 1 | 10 | vector：acquire 0、duration 1、units 1 | 无 |
| 3 | mte-zero | true | 0 | 0 | 无 | 无 |
| 4 | control-zero | true | 0 | 0 | 无 | 无 |
| 5 | unknown | false | 0 | 0 | 无 | 无 |

write latency 只在 producer 到同 region consumer 的 SSA Data edge 上生效。隐式状态、memory、sync 和 artificial edge 使用 DAGBuilder 自己定义的 latency。

### 7.4 显式 Vector/Predicate operation 表

只有以下 operation 映射为已知 `vector-predicate`：

| 类别 | Operation |
| --- | --- |
| Predicate 生成/范围 | `pto.pset_b8`, `pto.pset_b16`, `pto.pset_b32`, `pto.pge_b8`, `pto.pge_b16`, `pto.pge_b32` |
| Compare | `pto.vcmp`, `pto.vcmps` |
| Predicate 逻辑 | `pto.pand`, `pto.por`, `pto.pxor`, `pto.pnot` |
| Predicate 消费 | `pto.vsel`, `pto.vscatter` |
| Vector 计算 | `pto.vmax`, `pto.vmin`, `pto.vcmax`, `pto.vcmin`, `pto.vadd`, `pto.vabs` |

### 7.5 Operation 到 sched class 的解析

解析按以下优先级执行：

1. null operation 或 semantic class 为 `Structural`：`structural`；
2. 命中上一节显式表：`vector-predicate`；
3. semantics 含 `ImplicitRead`、`ImplicitWrite` 或 `Barrier`：`control-zero`；
4. `OpPipeInterface`：
   - `PIPE_S` -> `scalar-zero`；
   - `PIPE_V`/`PIPE_V2` -> `unknown`，除非已经命中显式表；
   - `PIPE_M` -> `unknown`；
   - `PIPE_MTE1` 至 `PIPE_MTE5`、`PIPE_FIX`、两个 virtual MTE2 pipe -> `mte-zero`；
   - `PIPE_ALL` -> `control-zero`；
   - `PIPE_NUM`/`PIPE_UNASSIGNED` -> 继续后续 family fallback；
5. `VectorMicroOpInterface` -> `unknown`；
6. `MteOpInterface` -> `mte-zero`；
7. `CubeMicroOpInterface` -> `unknown`；
8. `SimtOpInterface` -> `scalar-zero`；
9. 仍具有其他非空 scheduling effect -> `control-zero`；
10. 其他 -> `unknown`。

这意味着未显式登记的 Vector operation 和所有 Cube operation 当前不会被 `on` 重排；包含它们的整个 region 会 fallback。

### 7.6 Pressure sets

| ID | 名称 | limit | weight | spill cost | value contribution |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | vector | 32 | 1 | 1 | 每个 `!pto.vreg` 为 1 unit |
| 1 | predicate | 8 | 1 | 1 | 每个 `!pto.mask` 为 1 unit |

其他 value type 不贡献 pressure。limit 不是 hard legality；当所有 Candidate 都会超限时，scheduler 仍选择字典序损害最小的合法节点完成调度。

## 8. 分析 tracker

### 8.1 Boundary

`VPTOSchedBoundary` 支持 Top 和 Bottom 两个方向，每个方向独立拥有：

- available 和 pending 队列；
- current cycle；
- scheduled set；
- ResourceTracker；
- RegPressureTracker；
- HazardRecognizer。

Top 使用剩余 predecessor 数，Bottom 使用剩余 successor 数。队列以 `originalIndex` 保持确定顺序。Boundary 提供 dependency-ready 邻居释放和手工 defer/advance 接口。

当前 `on` scheduler 使用独立的 ReadyQueue，不直接使用 Boundary。Boundary 的主要当前用途是 `analyze` 的 top/bottom ready 统计和 tracker 容器。

### 8.2 ResourceTracker

ResourceTracker 保存每个 cycle 的 issue occupancy 和 resource occupancy。对一个 sched class，它检查：

- `microOps <= issueWidth`；
- 当前 cycle 的 issue occupancy 不超过 issue width；
- resource ID 存在；
- 请求 units 不超过 resource units；
- `[cycle + acquireAt, cycle + acquireAt + duration)` 内均有容量。

`evaluate` 从 requested cycle 开始搜索最早可用 cycle，搜索上限为 `2^20` 次；`commit` 只接受给定 cycle 正好合法的 reservation。

### 8.3 HazardRecognizer

接口支持 target-specific pair、spacing 和 issue restriction。当前实例始终为 `VPTONullHazardRecognizer`，所有节点都合法且 earliest cycle 等于请求 cycle，commit 不记录状态。

### 8.4 RegPressureTracker

Tracker 为每个 pressure set 保存 `current`、`peak` 和 live value 状态，并支持提交前 evaluate。

Top 初始化时把 live-in 加入 current，并统计 region 内每个 operand 的剩余 use。evaluate 一个 Candidate 时：

- live operand 在该节点耗尽全部剩余 use，且不是 live-out：减少 pressure；
- result 是 live-out 或存在 DAG 内 user，且尚未 live：增加 pressure；
- 计算 delta、projected 和 `max(0, projected-limit)`。

commit 拒绝任一 projected pressure 小于 0 的状态，随后更新 current/peak、剩余 use 和 live set。

Bottom 初始化为 live-out。逆向 evaluate 时移除当前节点的 live result，并加入此前未 live 的唯一 operand。Bottom tracker 当前用于分析和单元测试，不参与 `on` 的选择。

### 8.5 Analyze 原顺序模拟

`analyze` 为每个 region 输出：

- boundary、node/edge/live-in/live-out 数量；
- top-ready、bottom-ready；
- known/unknown sched class 数；
- 每个节点的 semantic class、sched class、depth 和 height；
- 每条 edge 的 kind、strength、latency 和 reason；
- 按原始节点顺序运行 top ResourceTracker、RegPressureTracker 和 HazardRecognizer 的 issue 记录；
- issue/resource timeline；
- 函数 coverage。

这条原顺序模拟使用 ResourceTracker 的 issue width 和 reservation，但不代表 `on` 的 logical issue cycle。

## 9. Top-down list scheduler

### 9.1 入口检查与预算

每个 region 使用默认限制：

| 预算 | 当前值 |
| --- | ---: |
| 最大节点数 | 4096 |
| 最大边数 | 262144 |
| scheduler、pending release 与 replay 共享 work units | `2^20` |

节点数和边数在调度前检查。随后再次防御性检查所有节点的 sched class 都为 known。

work unit 在以下位置消耗：

- 每次 Candidate evaluation；
- Pending 节点到达 current cycle 并释放为 Candidate；
- replay 中处理每个结果节点；
- replay 中释放 Pending 节点。

预算使用确定的计数，不使用依赖机器性能的墙钟超时。

### 9.2 Ready 状态与 logical cycle

ReadyQueue 为每个节点维护：

| 状态 | 含义 |
| --- | --- |
| `Unavailable` | 仍有 Must predecessor 未调度 |
| `Pending` | Must predecessor 已全部调度，但 ready cycle 晚于 current cycle |
| `Candidate` | Must predecessor 已全部调度，且 ready cycle 不晚于 current cycle |
| `Scheduled` | 已提交到结果 |

初始无 Must predecessor 的节点在 cycle 0 成为 Candidate。提交节点后，successor 的 ready cycle 更新为：

```text
readyCycle(successor) =
  max(readyCycle(successor), issueCycle(node) + edge.latency)
```

当 successor 的剩余 Must predecessor 归零时：

- ready cycle 不晚于 current cycle：立即进入 Candidate；
- 否则进入 Pending。

没有 Candidate 时，current cycle 直接跳到全局最早 Pending ready cycle，不逐 cycle 空转。

只要仍有 Candidate，scheduler 就在同一 current cycle 继续选择。这里的 cycle 只表示 dependency level，不执行 issue-width、resource 或 hazard 限制，因此同一 cycle 可以有多个结果节点，不能解释为真实硬件同周期发射。

### 9.3 Candidate pressure evaluation

对每个 Candidate，Top RegPressureTracker 计算：

```text
pressureDelta[s]
projectedPressure[s] = currentPressure[s] + pressureDelta[s]
currentExcess[s]     = max(0, currentPressure[s] - limit[s])
projectedExcess[s]   = max(0, projectedPressure[s] - limit[s])
```

然后计算三个整数 cost：

```text
excessGrowthCost =
  sum(spillCost[s] *
      max(0, projectedExcess[s] - currentExcess[s]))

projectedExcessCost =
  sum(spillCost[s] * projectedExcess[s])

pressureDeltaCost =
  sum(weight[s] * pressureDelta[s])
```

乘法和累加均检查 `int64_t` 溢出。pressure set 必须具有 limit，且 weight/spill cost 非负，否则当前 region 以 invalid-model 失败。

### 9.4 五级确定性选择

Candidate 使用以下字典序选择，不把指标混合成单一总分：

1. 更小的 `excessGrowthCost`；
2. 更小的 `projectedExcessCost`；
3. 更大的 DAG `height`；
4. 更小的 `pressureDeltaCost`；
5. 更小的 `originalIndex`。

前两项控制实际超限，第三项优先推进较长 dependency chain，第四项在前述条件相同时优先降低当前 live pressure，第五项保证完全确定的最终决胜。

选中节点后，先提交 pressure tracker，再提交 ReadyQueue，并把 `{unit, currentCycle}` 追加到 `VPTOScheduleResult`。所有节点提交后，结果记录最终 peak pressure。

### 9.5 当前可观察的顺序特征

对于两条独立且每条具有统一 latency 的三段链：

```text
A0 -> B0 -> C0
A1 -> B1 -> C1
```

初始 A0/A1 同为 cycle 0 Candidate，B0/B1 在 cycle 10 ready，C0/C1 在 cycle 20 ready，因此当前算法形成：

```text
A0 A1 B0 B1 C0 C1
```

即 AABBCC。每一层内部再由 pressure、height 和 original index 决胜。

当 producer 和 consumer 同时为 Candidate，消费并释放 Predicate 的节点可以通过更小的 pressure delta 优先于继续产生 Predicate 的节点。如果 consumer 仍处于 Pending，scheduler 不等待它，而会继续提交当前已有 Candidate。

## 10. Result 验证、重放与应用

### 10.1 调度阶段不修改 IR

`VPTOScheduler::schedule` 只返回：

```text
VPTOScheduleResult
  entries[] = {unit, issueCycle}
  peakPressure[]
```

在 result 产生、语义验证和 model replay 完成之前，region 中的 operation 不会移动。

### 10.2 Semantic verification

verifier 检查：

- 原 region 仍在同一个 block，前后 boundary 和连续性未变化；
- result 节点数等于 DAG 节点数；
- 每个 entry 非空、属于当前 block且没有重复；
- DAG 中每个节点恰好出现一次；
- 所有 Must edge 的 predecessor 位于 successor 之前；
- region 内每个 SSA defining unit 位于其 user 之前。

### 10.3 Fresh model replay

replay 创建新的 ReadyQueue 和 Top RegPressureTracker，不复用 scheduler 的可变状态。它逐 entry 检查：

- work-unit 预算仍可用；
- 必要时可以推进到最早 Pending cycle；
- entry 在记录的 cycle 确实是 Candidate；
- pressure commit 成功；
- ReadyQueue commit 成功；
- replay 结束后所有节点均已调度；
- replay peak pressure 与 result 完全一致。

replay 与 scheduler 共享同一个 work-unit budget。

### 10.4 Apply

apply 会再次运行 semantic verifier。验证通过后，按照 result 顺序把每个 operation 移到 following boundary 之前；没有 following boundary 时移到 block 末尾。

当前 MLIR `moveBefore` 操作没有失败返回值，因此代码在所有可失败检查完成后才开始移动，并没有保存原顺序或实现额外的事务式 rollback。所有当前可报告失败都发生在 apply 移动之前，因而失败时 region 保持原顺序。

## 11. 诊断与失败隔离

失败类别为：

```text
Budget, InvalidModel, Scheduling,
SemanticVerification, ModelReplay, Apply
```

处理规则：

- unknown sched class：为当前 region 中每个 unknown operation 发出 remark，包含 block、region、original index 和 operation name；
- node/edge/work-unit budget 或 invalid model：remark；
- scheduling、DAG cycle、semantic verification、model replay 或 apply 内部不一致：warning；
- CLI mode、target、trace 或 Bisheng 冲突：error，停止编译。

region 级失败只跳过当前 region，pass 继续处理后续 region。`on` 未开启 trace 时，成功 region 不输出报告。

## 12. 当前保证与限制

当前实现保证：

- `off` 和 `analyze` 不修改 IR；
- `on` 只对 model 完整且通过 verify/replay 的 region 应用完整 permutation；
- region 不跨 block、vecscope 或 semantic boundary；
- SSA、保守 memory、隐式状态和完整 barrier 通过 Must edge 保序；
- 相同 IR、model 和选项得到确定的调度顺序及 fallback 原因；
- unknown class、预算和内部验证失败隔离在当前 region；
- logical cycle、pressure score 和预算计算均有整数溢出或上限检查。

当前实现限制：

- `on` 不使用 ResourceTracker、issue width、resource reservation 或 HazardRecognizer；
- logical cycle 只来自 Must dependency latency，不是硬件发射 timeline；
- A5 只有显式表中的 Vector/Predicate operation 为非零 latency 的已知计算类；
- Scalar、MTE、Control 和 Structural 使用零调度成本；Cube 和未登记 Vector 为 unknown；
- 只统计 Vector 和 Predicate pressure；
- 不支持跨 block调度、bidirectional scheduling、bundle/pair、bank conflict、NOP、software pipelining 或 Cube kernel scheduling；
- apply 没有独立事务 rollback，因为当前移动操作不可失败；
- 不设置收益门槛，合法且通过 replay 的结果即应用，identity permutation 也允许执行 apply。

## 13. 回归覆盖

当前测试覆盖：

| 测试 | 主要内容 |
| --- | --- |
| `vpto_scheduler_analyze.pto` | off/analyze 不变、on trace、模型报告、非法 pass mode |
| `vpto_scheduler_cli.pto` | A5 默认 on、显式模式、trace 约束、target 约束、Bisheng 冲突 |
| `vpto_scheduler_coverage.pto` | boundary reason 和 unclassified coverage |
| `vpto_scheduler_dependencies.pto` | SSA、memory range、volatile、unknown memory、post-update、SPR/CTRL、barrier |
| `vpto_scheduler_on.pto` | AABBCC、logical cycle、pressure tie-break、确定性和默认静默 |
| `vpto_scheduler_unknown_fallback.pto` | unknown region 跳过，后续 region 继续调度 |
| `vpto_scheduler_trackers.pto` | resource、pressure、bottom tracker、verify/replay、语义拒绝、共享预算 |
| `bisheng_vec_misched_cli.pto` | Bisheng vector MISched 选项存在性 |

## 14. 后续可能的迭代目标

以下项目不是当前行为，实施前需要分别补充模型依据、正确性测试和性能验收：

1. **校准 A5 模型数据**：根据 hardware RA、CA 和 NPU 数据校准 Vector/Predicate limit、contribution、weight、spill cost 和 operation latency，避免把当前静态初值当作最终硬件事实。
2. **扩大显式模型覆盖**：从真实 emission-ready VPTO fixture 提取 operation 闭包，为更多 Vector/Predicate operation 建立可审计的 sched class；保持语义不完整的 operation 为 unknown。
3. **让 `on` 使用 resource 与 hazard**：把 issue width、resource reservation、pair/spacing、bank conflict 和 read advance 纳入 Candidate hard legality及 cycle 推进，而不只用于 analyze timeline。
4. **建立代表性性能 gate**：完善 compare/select、mask-or+scatter 和双链 AABBCC fixture，记录静态 pressure、最终 assembly spill/barrier 指标、CA/NPU ticks、EXIPC、编译时间和内存增量。
5. **加强差分验证**：增加随机 DAG differential test，覆盖 permutation、Must edge、ready cycle、pressure replay、预算边界和 fallback 稳定性。
6. **扩展调度范围与策略**：在 correctness semantics 和模型充分后评估 bidirectional scheduling、跨 block调度、bundle/pair、software pipelining 及 Cube kernel scheduling。
7. **模型声明机制**：在静态 C++ 模型稳定后评估 TableGen 或生成式描述，同时保持 `VPTOSchedModel` 只读接口不变。
8. **可失败 apply 的事务保护**：如果以后 apply 引入可能失败的 bundle 构造、跨 block移动或其他 mutation，在移动前保存完整原顺序并实现显式 rollback。
9. **预算再校准**：基于真实 region 的 node、edge、Candidate evaluation 和 replay 分布调整当前常量，但继续使用确定性计数而非墙钟超时。
