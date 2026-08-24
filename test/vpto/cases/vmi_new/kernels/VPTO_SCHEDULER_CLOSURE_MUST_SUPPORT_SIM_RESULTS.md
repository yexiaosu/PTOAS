<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# VPTO Scheduler Closure Must-support 修复后 CA SIM 结果

## 结论

本报告归档 `2ad5c08d4 fix(vpto): include must support in pressure closures` 之后，#508、#574 和六个 TileKernels fixture 在同一隔离环境中的完整 Scheduler OFF/ON 对比。48/48 次目标运行都完成真实 CA model start/stop 和 strict compare。

#574 的算法缺口已修复：Scheduler ON 的模型 peak 从修复前 38/8 降到 23/5，vector/predicate spill 和 36 个 barrier 全部消失，median ticks 从 4220 降到 4001，并略优于本轮 OFF 的 4046。其余七个 fixture 没有压力回归；三个 stress fixture 继续消除全部可消除 spill。MHC natural 仍有 26/26 次物理 vector spill/reload，这与修复前一致，原因是 VPTO 模型 peak 只描述当前 VPTO SSA live range，不等价于后端物理寄存器分配峰值。

修复前快照见 [VPTO_SCHEDULER_FRONTIER_FIX_SIM_RESULTS.md](VPTO_SCHEDULER_FRONTIER_FIX_SIM_RESULTS.md)。

## Revision 与 provenance

| 项目 | 值 |
| --- | --- |
| 本地算法分支 / 修复提交 | `vpto-sched-2` / `2ad5c08d4bd087011d0156cc711be260a597c4b5` |
| 服务器验证分支 / HEAD | `codex/vpto-sched-closure-must-support-sim-ca-20260824` / `b23032dc358a59eea48dcefeef98d1485360969d` |
| 验证 HEAD 组成 | `2ad5c08d4` 加两个 issue fixture 和六个 TileKernels fixture；`2ad5c08d4` 是其祖先 |
| 服务器 | `wanglan@115.175.35.144`，host `ecs-1030-cba0` |
| CANN / CA model | CANN 9.1.0，A5 `dav_3510` SIM |
| Bisheng | clang 15.0.5，build 2026-07-01 |
| LLVM | 19.1.7，`/home/wanglan/llvm-workspaces/build-vpto19` |
| PTOAS | 0.60 |
| PTO-ISA | `311ca0c83f5571dc165681fee0a427983c555d3c`，`/home/wanglan/pto-isa` |
| Runner | `test/vpto/scripts/run_host_vpto_validation.sh` |

服务器使用隔离 worktree `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-closure-must-support-sim-20260824`、隔离 venv `<worktree>/.venv` 和隔离 build `/home/wanglan/PTOAS/.worktrees/ca-sim-builds/vpto-sched-closure-must-support-sim-20260824`。`ptoas.__file__` 来自该 worktree，`_core` 和运行时解析到的 `libPTOASCompiler.so` 来自该 venv，LLVM/MLIR 动态依赖来自上述 LLVM 19 build。验证未复用任何现有 SIM worktree 的 build 或 editable install。

每次运行使用 `DEVICE=SIM`、`COMPILE_ONLY=0`、strict compare 和独立 `WORK_SPACE`。OFF flags 为 `--pto-arch a5 --pto-backend=vpto --vpto-scheduler=off`；ON flags 为 `--pto-arch a5 --pto-backend=vpto --vpto-scheduler=on --vpto-scheduler-trace`。

## 完整 OFF/ON 矩阵

| Case | Scheduler | 模型 V/P peak | Stack VSTI/VLDI | Stack PSTI/PLDI | `SMEM_BAR` | ticks（3 次；median） | ON 相对 OFF | Strict compare |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| #574 `topk-gate-aabbcc-scheduler` | OFF | 原始顺序 | 0/0 | 0/0 | 0 | 4046, 4049, 4042; **4046** | - | 3/3 pass |
| #574 `topk-gate-aabbcc-scheduler` | ON | 23/5 | 0/0 | 0/0 | 0 | 4001, 4000, 4004; **4001** | -1.11% | 3/3 pass |
| #508 `topk-gate-vcmp-vsel-misched` | OFF | 原始顺序 | 0/0 | 0/0 | 0 | 2609, 2604, 2605; **2605** | - | 3/3 pass |
| #508 `topk-gate-vcmp-vsel-misched` | ON | 最大 region 18/4 | 0/0 | 0/0 | 0 | 2595, 2599, 2599; **2599** | -0.23% | 3/3 pass |
| per-token natural | OFF | 原始 14/4 | 0/0 | 0/0 | 0 | 3277, 3264, 3268; **3268** | - | 3/3 pass |
| per-token natural | ON | 28/4 | 0/0 | 0/0 | 0 | 3285, 3273, 3275; **3275** | +0.21% | 3/3 pass |
| per-token stress | OFF | 原始 44/4 | 11/11 | 0/0 | 22 | 3297, 3312, 3314; **3312** | - | 3/3 pass |
| per-token stress | ON | 30/4 | 0/0 | 0/0 | 0 | 3272, 3282, 3274; **3274** | -1.15% | 3/3 pass |
| SwiGLU natural | OFF | 原始 8/5 | 0/0 | 0/0 | 0 | 2830, 2831, 2828; **2830** | - | 3/3 pass |
| SwiGLU natural | ON | 24/5 | 0/0 | 0/0 | 0 | 2810, 2816, 2807; **2810** | -0.71% | 3/3 pass |
| SwiGLU stress | OFF | 原始 38/4 | 6/6 | 0/0 | 12 | 2894, 2889, 2899; **2894** | - | 3/3 pass |
| SwiGLU stress | ON | 24/6 | 0/0 | 0/0 | 0 | 2800, 2810, 2808; **2808** | -2.97% | 3/3 pass |
| SwiGLU predicate-stress | OFF | 原始 14/11 | 0/0 | 19/19 | 38 | 3340, 3337, 3334; **3337** | - | 3/3 pass |
| SwiGLU predicate-stress | ON | 24/5 | 0/0 | 0/0 | 0 | 2807, 2800, 2808; **2807** | -15.88% | 3/3 pass |
| MHC natural | OFF | 原始 29/3 | 45/45 | 0/0 | 106 | 4736, 4740, 4745; **4740** | - | 3/3 pass |
| MHC natural | ON | 23/4 | 26/26 | 0/0 | 52 | 3496, 3491, 3497; **3496** | -26.24% | 3/3 pass |

## #574 根因与修复证据

修复前，closure group 从待关闭 SSA 值的用户沿正向 `Data/Must` 边建立 core，但为了让 core ready，反向也只收集 `Data/Must` 前置。#574 的 result store 同时消费待关闭的 vector/predicate 值，并受 initial zero-store 的 `Memory/Must` 依赖约束；memory 前置不进入模拟，使 result store 永远无法在 closure 模拟中 ready，group 因此无法激活。普通排序随后把两个 initial zero-store 和 result stores 一起拖到 757-node region 尾部，造成 peak 38/8 和物理 spill。

修复后，反向 readiness completion 收集所有类型的 Must 前置作为 support；正向 closure core 仍只沿 `Data/Must` 扩展，非 Data support 的无关后继不会被拉入 group。core 与 support 合计仍受 96 节点上限约束，超过时放弃该 group，因此没有扩大正常 memory DAG 或取消共享 work-unit 预算。

ON run1 trace 中，两个 initial zero-store 从修复前 region 最后 20 个位置移到 result position 41/42：node 2 以 `advance-pressure-closure` 选择，node 3 以 `closure-pressure-preserving` 选择。结果为 peak 23/5、0 spill、0 barrier，证明修复确实补齐了 closure 的合法依赖路径，而不是依靠 acceptance fallback 隐藏回归。

## MHC peak 23/4 为什么仍有 spill

`23/4` 是 VPTO scheduler 在当前 IR 层根据 SSA 值、live-in/live-out 和已知 pressure contribution replay 得到的峰值。它回答的是“这个 VPTO 顺序同时有多少个模型化 vector/predicate live units”，不是“Bisheng 最终寄存器分配需要多少个可分配物理寄存器”。

MHC 的后续 lowering、指令选择和物理寄存器分配还会引入或延长模型里没有逐项表达的物理 live range，包括后端临时值、同一时刻的 operand/result 干涉、固定或保留寄存器约束以及不能完成的 coalescing。因此 VPTO 模型 peak 低于 32 只能说明 scheduler 没有在其建模层直接制造超限，不能证明最终机器码零 spill。

本轮机器码证据与该边界一致：OFF 原始模型 peak 29/3 时有 45/45 vector spill/reload；ON 模型 peak 23/4 时仍有 26/26，但已经减少 19 对 spill/reload、54 个 barrier，并使 median ticks 降低 26.24%。这说明当前压力模型对 MHC 仍有方向性价值，但其数值不是物理分配的充分条件。closure Must-support 修复前后 MHC 都是 peak 23/4 和 26/26，说明残余 spill 不是本次 #574 修复引入的回归。

## DAG 与分析复杂度

| Case | Nodes / edges | Live-in / live-out | ON peak | Work units | Pressure idles |
| --- | ---: | ---: | ---: | ---: | ---: |
| #574（每个主 region） | 757 / 2132 | 27 / 27 | 23/5 | 183641、183599 | 378 |
| #508（最大 pressure region） | 18 / 17 | 24 / 24 | 18/4 | 784 | 3 |
| per-token natural | 891 / 2023 | 39 / 2 | 28/4 | 360721 | 388 |
| per-token stress | 891 / 2023 | 39 / 2 | 30/4 | 326369 | 337 |
| SwiGLU natural | 471 / 1139 | 39 / 3 | 24/5 | 169904 | 176 |
| SwiGLU stress | 471 / 1139 | 39 / 3 | 24/6 | 152983 | 173 |
| SwiGLU predicate-stress | 471 / 1139 | 39 / 3 | 24/5 | 140643 | 132 |
| MHC natural | 597 / 1331 | 43 / 9 | 23/4 | 428547 | 125 |

所有 op 均有已知 sched class；没有 max-edges、max-work-units、analysis timeout 或 skipped compute region。#574 修复后的 work units 明显低于默认上限，六个 TileKernels fixture 的复杂度没有相对修复前扩大。

## Smoke 与本地回归

| 检查 | 结果 |
| --- | --- |
| `micro-op/binary-vector/vadd` | 12955 ticks；9 variants strict compare passed |
| `micro-op/vector-load-store/vlds-post-update` | 2418 ticks；strict compare passed |
| focused scheduler lit | 10/10 passed |
| tracker tests | 全部通过，包括 random-DAG differential |
| 完整 `check-pto` | 1810 passed，1 unsupported，0 failed |
| changed-code compliance | 1 个 changed source file，0 errors，0 warnings |

## Raw evidence

统一 artifact 根目录：

`/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-closure-must-support-sim-20260824`

- provenance：`provenance.log`
- smoke：`smoke-vadd/`、`smoke-vlds-post-update/`
- 48 次目标运行：`runtime-matrix/<case>/<off|on>/run{1,2,3}/`
- 每次运行保留 `driver.log`、case `validation.log`、host/golden/output 和 CA instruction dumps。
- `SHA256SUMS` 覆盖 21728 个文件；manifest SHA-256 为 `f73bf8fc038adad2a60130489cc5bb4a20790d12308011efff070d1378d5a867`。
- 归档大小为 63 MiB；raw logs 和生成二进制未加入 Git。

验证 worktree 最终只包含 manager 生成且未跟踪的 `.ptoas-workspace.json` 和 `env.sh`，没有源代码修改。
