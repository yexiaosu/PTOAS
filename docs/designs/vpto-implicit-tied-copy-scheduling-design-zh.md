# VPTO tied operand 隐式 copy 调度设计

## 背景与目标

`vmula`、`vmadd` 等 destructive operation 的结果与某个 accumulator operand 绑定到同一物理寄存器。多个 consumer 共享同一 accumulator 时，后端 TwoAddress/RegisterCoalescing 会为非 owner consumer 生成物理 VMOV；scheduler 如果只分析原始 SSA operation，就会漏掉这些 copy 的依赖、资源、延迟和压力。

本设计在 scheduler 内部把所需 copy 建模为虚拟事件，但不向 VPTO IR 插入 `pto.vmov`。scheduler 只重排输入中已经存在的 operation，调度后 destructive consumer 仍引用原始共享 accumulator，最终物理 copy 继续由 Bisheng 的既有 TwoAddress/RegisterCoalescing 流程生成。

## 为什么不在 IR 中显式插入 VMOV

同源的显式 VMOV intrinsic 可能被 EarlyCSE、SelectionDAG CSE 或 MachineCSE 合并，使后端得到从第一个临时寄存器继续复制的链式形态，而不是所有 consumer 直接从原 accumulator fanout。用 masked VMOV、不同元素类型、inline asm、`nomerge` 或关闭 CSE 保护 copy 都会把后端优化细节泄漏到 VPTO IR，并且不能形成稳定的语义契约。

因此隐式 copy 是 scheduler 的内部语义：它参与依赖图、资源时间线、关键路径和压力评价，但不成为 MLIR operation，也不改变 pass 顺序或 Bisheng。

## Tied operand 接口

具有 destructive 约束的 operation 实现统一的 tied operand interface，接口返回 tied operand index 和 tied result index。DAG builder 只查询接口，不按 operation 名称分支；新增 tied operation 时必须先由其 ODS、emitter 和机器指令约束确认绑定关系，再实现同一接口。

当前已确认并接入的 operation 包括 `chistv2`、`dhistv2`、`vmadd`、`vmula`、`vusqz` 和 `vaxpy`。其中 `vaxpy` 的 tied operand 是 operand 1，其余当前 operation 的 tied operand 是 operand 0；它们的 tied result 均为 result 0。

## Physical root

Physical root 表示一组共享同一物理寄存器的 SSA view。直接值与穿过零开销 `pto.vbitcast`、`pto.pbitcast` 的值归并到同一 root；DAG tied-copy analysis 和 register-pressure tracker 共用同一个 root 工具，避免对 view 规则产生不同解释。

隐式 copy 的结果是新的物理 root，但它不是 MLIR SSA Value。本实现用 scheduling unit 上的显式 copy contract 表达这个临时 root，不伪造 SSA Value，也不插入伪 operation。

## Owner 与 copy consumer

DAG builder 按 physical root 收集普通 reader、tied consumer、region live-out，以及 cross-region 或 cross-block material use。只有 root 在 region 之后不再需要存活时，才能选择一个 tied consumer 作为 owner；确定性策略选择原始位置最靠后的 tied consumer。

Owner 直接复用并覆盖原 root，不需要 copy。其他 tied consumer 都读取同一个 physical root 并需要隐式 copy；普通 reader、每个隐式 copy 对 root 的读取以及非 owner tied consumer 都通过 Must anti-dependence 排在 owner 之前，因此正确性不依赖原 operation 顺序。

Builder 在加入 owner anti-dependence 前检查 DAG 可达性。如果任一 reader 到候选 owner 的边会形成环，则放弃 owner，并对该 root 的所有 tied consumer 使用保守 copy 模型。root 为 live-out、存在 region 外 material use 或跨 block use 时同样不选择 owner。

## 复合调度单元

需要 copy 的 tied consumer 被建模成不可拆分的复合单元：

```text
cycle N:                 implicit VMOV issue
cycle N + copyLatency:   destructive consumer issue
```

当前 A5 copy event 使用一次 micro-op、一次 vector resource reservation 和 2 cycle 的 copy latency；consumer event 保持实际 operation 对应的 sched class、micro-op、resource、write latency、read advance 和 hazard。资源 tracker 分别检查并提交两个内部事件，hazard recognizer 也分别接收 copy root 与 consumer 事件。

复合单元提交后，Boundary 的当前周期至少推进到 consumer issue cycle，因此其他 operation 不能插入 copy 与 consumer 之间。copy 不是把 consumer 的 `microOps` 简单加一，也不是在同一 cycle 同时占用两次资源。

数据边记录 successor 的实际 read offset。对于 tied accumulator，copy 在复合单元起始周期读取 root；consumer 的其他 operand 在内部 consumer 周期读取。关键路径和 ready cycle 使用 `max(0, producerLatency - successorReadOffset)`，从而既保证 copy 等待 accumulator 就绪，也避免把 consumer 的内部 offset 错加到其他 operand 的依赖上。

## Transient register pressure

普通 pressure delta 仍表示复合单元执行完成后的 SSA live state。需要 copy 的单元另外声明一个 vector register 的 transient delta：copy 时原 root 仍然 live，新临时 accumulator 增加一个物理 vector register；临时值保持到 destructive consumer；consumer 完成后临时 accumulator 被结果接管，不再把临时值与结果重复计数。

候选的 projected peak 取完成后 projected pressure 与 `current + transientDelta` 的较大值。超限代价、near-limit/high-pressure 策略、有限前瞻和最终 peak 都使用这个瞬时峰值；owner 的 transient delta 为零。root 自身通过 reader、copy 与 owner 的依赖保持存活到最后一次真实读取或覆盖。

## Analyze 与 IR 契约

`mode=analyze` 建立同一个 DAG，并在模型完整时执行 schedule、semantic verify 和 model replay，以报告 owner、copy consumer、implicit VMOV 数量、root dependency、owner anti-dependence、copy cycle、consumer cycle、transient delta 和 peak；它不应用结果，因此输出 IR 与 `mode=off` 一致。

`mode=on` 在同样的分析和验证之后只移动原 operation。输出中不会新增 `pto.vmov`、VMOV intrinsic、inline asm 或 CSE workaround。Bisheng 必须从仍共享原 accumulator 的 tied operand 关系物化最终 copy；机器级验证应确认每个 copy 直接从原 root fanout，并紧邻对应 destructive consumer。

## 当前限制

当前一个 `!pto.vreg` 按一个物理 vector register 计数，所以一次隐式 copy 表示一次完整寄存器复制，与逻辑 mask 长度无关。未来若引入一个 SSA value 对应多个物理寄存器的 aggregate，必须扩展 copy 数量、资源和 transient pressure contract，不能沿用固定一条 VMOV 的假设。

当前复合单元有意禁止把 VMOV 提前并在 VMOV 与 consumer 之间穿插其他指令。这与当前 Bisheng materialization 形态一致，但也限制了 scheduler 对 copy latency 的隐藏空间；如果后端契约未来允许分离 copy 与 consumer，需要先建立可验证的临时 root 生命周期和机器级对应关系，再改变该约束。

## 测试与机器级验证

focused lit 覆盖三个共享 accumulator 的 `vmula`、单 consumer、普通 reader、bitcast root、live-out、owner cycle rejection、masked consumer、非 `vmula` tied operation、analyze IR 不变、无显式 `pto.vmov`、同 root fanout、owner anti-dependence、内部 issue cycle 和 transient pressure。tracker test 直接验证 copy/consumer 的两个 issue slot、两个 vector resource reservation、额外一个 vector pressure unit，以及复合单元不可穿插。

机器级 CA model 验证必须开启 PTOAS scheduler、关闭 Bisheng vector scheduler、不使用关闭 MachineCSE 的 workaround，并执行 strict compare。最终指令需要表现为所有 VMOV 直接读取原 accumulator，每条 VMOV 紧邻其 consumer，且数量与 analyze 报告一致；链式 `root -> temporary -> temporary` copy 不满足本设计契约。
