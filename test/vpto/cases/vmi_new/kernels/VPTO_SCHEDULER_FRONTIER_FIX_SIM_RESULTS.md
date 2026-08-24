<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# VPTO Scheduler Memory-frontier 修复后 CA SIM 结果

## 结论

本报告归档 memory-frontier partial-overlap 正确性修复和 one-point `vsts` footprint 修正之后、closure-support 修复之前的完整 A5 CA-model 对比结果。#508 保持无 spill；TileKernels 的三个 stress fixture 仍能由 Scheduler ON 消除可消除的 vector 或 predicate spill；#574 在恢复正确 memory dependency 后 strict compare 通过，但暴露出 closure group 不会补齐非 data must-predecessor 的独立调度缺口；MHC 仍有残余 vector spill。

本报告是后续 closure-support 算法修改前的历史快照，不应用其中的 #574 或 MHC ON 结果代表更新后的算法。

## Revision 与 provenance

| 项目 | 值 |
| --- | --- |
| 本地算法分支 / HEAD | `vpto-sched-2` / `8346fde413458649fee0242c54665f459d71ad0b` |
| Partial-overlap 修复 | `96f6fd2f2e60affb11b7cc3a6a549a20d4277ede` |
| One-point store range 修复 | `8346fde413458649fee0242c54665f459d71ad0b` |
| 服务器验证分支 | `codex/vpto-sched-frontier-fix-sim-ca-20260824` |
| #508/#574 实际验证 revision | `07b4b024e1cd6d8a6792e53ce17ddf700f2d5d51` |
| TileKernels 实际验证 revision | `93654245c45999ada7266de61242a61537552bbb` |
| Revision 差异 | `93654245c` 只在相同算法上追加六个 TileKernels runtime fixtures 及 host-input 修正 |
| 服务器 | `wanglan@115.175.35.144`，host `ecs-1030-cba0` |
| CANN / CA model | CANN 9.1.0，A5 `dav_3510` SIM |
| Bisheng | clang 15.0.5，build 2026-07-01 |
| LLVM | 19.1.7，`/home/wanglan/llvm-workspaces/build-vpto19` |
| PTOAS | 0.60 |
| PTO-ISA | `311ca0c83f5571dc165681fee0a427983c555d3c` |
| Runner | `test/vpto/scripts/run_host_vpto_validation.sh` |

服务器使用隔离 worktree `/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-frontier-fix-sim-20260824`、隔离 venv `<worktree>/.venv` 和隔离 build `/home/wanglan/PTOAS/.worktrees/ca-sim-builds/vpto-sched-frontier-fix-sim-20260824`。`ptoas.__file__` 来自该 worktree，`_core` 和 `libPTOASCompiler.so` 来自该 venv，LLVM/MLIR 动态依赖解析到 `/home/wanglan/llvm-workspaces/build-vpto19/lib`。

每项 runtime 使用 `DEVICE=SIM`、`COMPILE_ONLY=0`、strict compare 和独立 `WORK_SPACE`。OFF flags 为 `--pto-arch a5 --pto-backend=vpto --vpto-scheduler=off`；ON flags 额外使用 `--vpto-scheduler=on --vpto-scheduler-trace`。日志确认 CA model start/stop、host executable、golden、kernel instruction activity 和 strict compare。

## #508 与 #574

#508 和 #574 的 OFF 行沿用相同 fixture、相同 Scheduler-OFF/Bisheng 配置的已有权威三次基线；frontier 修复不参与 Scheduler OFF 的 IR 重排。ON 行是在上述 frontier-fix 隔离 worktree 中重新执行的三次 CA-model 结果。

| Case | Scheduler | 模型 peak | Stack VSTI/VLDI | Stack PSTI/PLDI | `SMEM_BAR` | ticks（3 次；median） | Strict compare |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| #508 `topk-gate-vcmp-vsel-misched` | OFF reference | 原始顺序 | 0/0 | 0/0 | 0 | 2603, 2613, 2614; **2613** | 3/3 pass |
| #508 `topk-gate-vcmp-vsel-misched` | ON | 最大 region 18/4 | 0/0 | 0/0 | 0 | 2589, 2590, 2603; **2590** | 3/3 pass |
| #574 `topk-gate-aabbcc-scheduler` | OFF reference | 原始顺序 | 0/0 | 0/0 | 0 | 4052, 4057, 4059; **4057** | 3/3 pass |
| #574 `topk-gate-aabbcc-scheduler` | ON | 38/8 | 12/12 | 6/6 | 36 | 4220, 4222, 4217; **4220** | 3/3 pass |

#574 的 memory frontier 修复恢复了 initial zero-store 到后续 one-point result-store 的必要 WAW edges，因而修复了旧 ON 顺序的功能错误。当前 ON 把 initial zero-store 和全部 result-store 排到 757-node region 的最后 20 个位置，使结果 live range 集中存活，产生 vector 和 predicate spill；这不是 memory frontier 再次漏边，而是 closure group 无法补齐非 data must-predecessor 的算法缺口。

## TileKernels 六 fixture

六个 case、两种 scheduler 模式各运行三次，共 36/36 次完成 CA model start/stop 和 strict compare。

| Case | Scheduler | 模型 V/P peak | Stack VSTI/VLDI | Stack PSTI/PLDI | `SMEM_BAR` | ticks（3 次；median） | ON 相对 OFF | Strict compare |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| per-token natural | OFF | 原始 14/4 | 0/0 | 0/0 | 0 | 3279, 3276, 3277; **3277** | - | 3/3 pass |
| per-token natural | ON | 28/4 | 0/0 | 0/0 | 0 | 3282, 3286, 3287; **3286** | +0.27% | 3/3 pass |
| per-token stress | OFF | 原始 44/4 | 11/11 | 0/0 | 22 | 3311, 3315, 3314; **3314** | - | 3/3 pass |
| per-token stress | ON | 30/4 | 0/0 | 0/0 | 0 | 3280, 3282, 3280; **3280** | -1.03% | 3/3 pass |
| SwiGLU natural | OFF | 原始 8/5 | 0/0 | 0/0 | 0 | 2838, 2843, 2836; **2838** | - | 3/3 pass |
| SwiGLU natural | ON | 24/5 | 0/0 | 0/0 | 0 | 2821, 2813, 2817; **2817** | -0.74% | 3/3 pass |
| SwiGLU stress | OFF | 原始 38/4 | 6/6 | 0/0 | 12 | 2898, 2903, 2906; **2903** | - | 3/3 pass |
| SwiGLU stress | ON | 24/6 | 0/0 | 0/0 | 0 | 2810, 2807, 2811; **2810** | -3.20% | 3/3 pass |
| SwiGLU predicate-stress | OFF | 原始 14/11 | 0/0 | 19/19 | 38 | 3334, 3338, 3333; **3334** | - | 3/3 pass |
| SwiGLU predicate-stress | ON | 24/5 | 0/0 | 0/0 | 0 | 2802, 2807, 2805; **2805** | -15.87% | 3/3 pass |
| MHC natural | OFF | 原始 29/3 | 45/45 | 0/0 | 106 | 4736, 4741, 4741; **4741** | - | 3/3 pass |
| MHC natural | ON | 23/4 | 26/26 | 0/0 | 52 | 3494, 3502, 3504; **3502** | -26.13% | 3/3 pass |

模型 peak 是 VPTO SSA/live-range pressure，不是物理寄存器分配结果。MHC 的调度模型 peak 为 23/4，但 lowering、寄存器干涉、分配约束和后端生成的临时值不完全包含在当前模型中，因此最终机器码仍有 26/26 次 vector spill/reload。当前模型仍提供了有效方向性信号：相对 OFF 的 45/45，ON 减少了 19 对 spill/reload 和 54 个 barrier，并降低约 26% median ticks；但不能用 peak 23 直接推导“物理寄存器一定不 spill”。

## DAG 与分析复杂度

| Case | Nodes / edges | Live-in / live-out | ON peak | Work units | Pressure idles |
| --- | ---: | ---: | ---: | ---: | ---: |
| per-token natural | 891 / 2023 | 39 / 2 | 28/4 | 360721 | 388 |
| per-token stress | 891 / 2023 | 39 / 2 | 30/4 | 326369 | 337 |
| SwiGLU natural | 471 / 1139 | 39 / 3 | 24/5 | 169904 | 176 |
| SwiGLU stress | 471 / 1139 | 39 / 3 | 24/6 | 153648 | 170 |
| SwiGLU predicate-stress | 471 / 1139 | 39 / 3 | 24/5 | 159589 | 161 |
| MHC natural | 597 / 1331 | 43 / 9 | 23/4 | 428547 | 125 |

所有 op 均有已知 sched class，没有 max-edges、max-work-units、analysis timeout 或 skipped region。相对 frontier 修复前同算法，per-token DAG 不变，SwiGLU 增加 2 条必要 edge，MHC 因 one-point footprint 精化从 1346 减少到 1331 edges；未观察到 frontier 修复导致的分析复杂度爆炸。

## Smoke 与本地回归

| 检查 | 结果 |
| --- | --- |
| `micro-op/binary-vector/vadd` | 12970 ticks；9 variants strict compare passed |
| `micro-op/vector-load-store/vlds-post-update` | 2424 ticks；strict compare passed |
| focused scheduler lit | 9/9 passed |
| tracker tests | 全部通过，包括 random-DAG differential |
| 完整 `check-pto` | 1809 passed，1 unsupported，0 failed |
| changed-code compliance | 0 errors，0 warnings |

## Raw evidence

统一 artifact 根目录：

`/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-frontier-fix-sim-20260824`

- #508 final ON：`issue508/scheduler-on-onepoint/run{1,2,3}/`
- #574 final ON：`issue574/scheduler-on-onepoint/run{1,2,3}/`
- Final-revision smoke：`smoke-vadd-onepoint/`、`smoke-vlds-post-update-onepoint/`
- TileKernels：`tilekernels-final-93654245c/runtime-matrix/<case>/<off|on>/run{1,2,3}/`
- TileKernels smoke：`tilekernels-final-93654245c/smoke-vadd/`、`tilekernels-final-93654245c/smoke-vlds-post-update/`

TileKernels 子归档的 `SHA256SUMS` 覆盖 16528 个文件，manifest SHA-256 为 `d1cc13687d9ce66b890a03f0792439877f9c7f96ad59cf7194283b83c41cf67d`。此前 #508/#574 artifact 根 manifest SHA-256 为 `a9b53f81d1bf760cb85e86d81443d99dd717d4a69f1c2c075d9ec7096dba79ef`；该 manifest 在追加 TileKernels 子目录前生成，因此 TileKernels 使用独立 manifest。
