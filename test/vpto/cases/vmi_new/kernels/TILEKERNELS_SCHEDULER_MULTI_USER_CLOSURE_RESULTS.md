<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# TileKernels VPTO Scheduler Multi-user Closure 验证结果

## 结论

本报告记录 `acbd18ae5617525640c339a41c8ee96f31d41c48` 上最新的
Scheduler OFF/ON A5 CA-model SIM 对比。六个 case、两种 scheduler 模式各运行
三次，共 36/36 次完成 CA model start/stop 和 strict compare。

当前版本的整体表现可以接受：

- per-token stress 的 11/11 次 vector spill/reload 被完全消除，median ticks
  从 3312 降到 3284（-0.85%）。
- SwiGLU stress 的 6/6 次 vector spill/reload 被完全消除，median ticks 从
  2903 降到 2816（-3.00%）。
- SwiGLU predicate-stress 的 19/19 次 predicate spill/reload 被完全消除，
  median ticks 从 3331 降到 2800（-15.94%）。
- MHC 的 vector spill/reload 从 45/45 降到 24/24，median ticks 从 4743
  降到 3412（-28.06%）；这是显著改善，但尚未完全消除 spill。
- 两个 natural case 在 OFF 和 ON 下都无 spill。per-token natural 的 ON median
  比 OFF 高 6 ticks（+0.18%），可视为基本持平；SwiGLU natural 低 21 ticks
  （-0.74%）。

因此，本版本已经修复此前 scheduler 会在 vector stress case 中扩大压力、导致
编译失败的问题，也保留了 predicate-stress 的正向结果。CA-model ticks 仍只作为
观测数据，不作为 scheduler acceptance gate。

历史负向矩阵和 fixture 构造说明保留在
[TILEKERNELS_SCHEDULER_PRESSURE_RESULTS.md](TILEKERNELS_SCHEDULER_PRESSURE_RESULTS.md)；
该报告记录的是更早 revision 的当时结果，不应使用其中的 ON 数据代表当前算法。

## Revision 与执行环境

| 项目 | 值 |
| --- | --- |
| 当前实际验证 revision | `acbd18ae5617525640c339a41c8ee96f31d41c48` |
| 直接父版本 | `4fe22d64c07d87f29f230b1c6aedc7c3ec7c0bc9` |
| 分支 | `codex/vpto-sched-multi-user-closure-sim` |
| 服务器 | `wanglan@115.175.35.144`，host `ecs-1030-cba0` |
| CANN / CA model | CANN 9.1.0，A5 `dav_3510` SIM |
| Bisheng | clang 15.0.5，build 2026-07-01 |
| LLVM | 19.1.7，`/home/wanglan/llvm-workspaces/build-vpto19` |
| PTOAS | 0.60 |
| PTO-ISA | `311ca0c83f5571dc165681fee0a427983c555d3c` |
| Runner | `test/vpto/scripts/run_host_vpto_validation.sh` |
| Runtime 配置 | `DEVICE=SIM`、`COMPILE_ONLY=0`、strict compare |

服务器使用独立 worktree、venv 和 build：

- Worktree：`/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/vpto-sched-multi-user-closure-sim`
- Venv：上述 worktree 的 `.venv`
- Build：`/home/wanglan/PTOAS/.worktrees/ca-sim-builds/vpto-sched-multi-user-closure-sim`
- `ptoas.__file__`：上述 worktree 的 `ptodsl/ptoas/__init__.py`
- `_core`：上述 venv 的
  `lib/python3.12/site-packages/ptoas/_core.cpython-312-x86_64-linux-gnu.so`
- `libPTOASCompiler.so`：上述 venv 的
  `lib/python3.12/site-packages/ptoas/mlir/_mlir_libs/libPTOASCompiler.so`
- Build 和安装后的 `libPTOASCompiler.so` ELF Build ID 均为
  `a3334042d17f5be65d1528bf014d5fa4f0ee6259`。

Scheduler flags：

- OFF：`--pto-arch a5 --pto-backend=vpto --vpto-scheduler=off`
- ON：`--pto-arch a5 --pto-backend=vpto --vpto-scheduler=on --vpto-scheduler-trace`

## 当前 revision 的 ON/OFF 对比

`VSTI/VLDI` 和 `PSTI/PLDI` 只统计 instruction log 中使用 stack base
`Sn[95]=0x40000` 的指令，避免把算法自身的普通访存计为 spill。表中 spill 和
barrier 数量在每个 case 的三次运行中完全一致。

| Case | Scheduler | 模型 V/P peak | Stack VSTI/VLDI | Stack PSTI/PLDI | `SMEM_BAR` | ticks（3 次；median） | ON 相对 OFF | Strict compare |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| per-token natural | OFF | 原始 14/4 | 0/0 | 0/0 | 0 | 3280, 3270, 3276; **3276** | - | 3/3 pass |
| per-token natural | ON | 28/4 | 0/0 | 0/0 | 0 | 3288, 3279, 3282; **3282** | +0.18% | 3/3 pass |
| per-token stress | OFF | 原始 44/4 | 11/11 | 0/0 | 22 | 3308, 3312, 3315; **3312** | - | 3/3 pass |
| per-token stress | ON | 30/4 | 0/0 | 0/0 | 0 | 3284, 3285, 3278; **3284** | -0.85% | 3/3 pass |
| SwiGLU natural | OFF | 原始 8/5 | 0/0 | 0/0 | 0 | 2836, 2838, 2837; **2837** | - | 3/3 pass |
| SwiGLU natural | ON | 24/5 | 0/0 | 0/0 | 0 | 2813, 2816, 2824; **2816** | -0.74% | 3/3 pass |
| SwiGLU stress | OFF | 原始 38/4 | 6/6 | 0/0 | 12 | 2906, 2903, 2898; **2903** | - | 3/3 pass |
| SwiGLU stress | ON | 24/6 | 0/0 | 0/0 | 0 | 2807, 2816, 2816; **2816** | -3.00% | 3/3 pass |
| SwiGLU predicate-stress | OFF | 原始 14/11 | 0/0 | 19/19 | 38 | 3337, 3331, 3331; **3331** | - | 3/3 pass |
| SwiGLU predicate-stress | ON | 24/5 | 0/0 | 0/0 | 0 | 2798, 2804, 2800; **2800** | -15.94% | 3/3 pass |
| MHC natural | OFF | 原始 29/3 | 45/45 | 0/0 | 106 | 4743, 4744, 4741; **4743** | - | 3/3 pass |
| MHC natural | ON | 23/4 | 24/24 | 0/0 | 48 | 3412, 3415, 3412; **3412** | -28.06% | 3/3 pass |

对压力结果的解释：

- natural case 的原始顺序本来就不 spill。ON 的模型 peak 高于原始 peak，表示
  scheduler 为重排引入了更长的 live range，但仍在物理结果上保持无 spill。
- 三个合成 stress case 都证明存在有意义的正向调度结果：原始顺序产生真实
  vector 或 predicate spill，ON 后全部归零，且 strict compare 保持通过。
- MHC 的 ON 模型 peak 为 23/4，但最终机器码仍有 24/24 次 vector spill/reload。
  当前压力模型足以指导算法显著降压，但还不能把其峰值直接等同于最终物理寄存器
  分配结果。

## Scheduler trace 与分析复杂度

六个 case 均形成一个可调度 region，所有 op 都有已知 sched class，unknown 为 0。

| Case | Nodes / edges | Live-in / live-out | Scheduled V/P | Work units | Pressure idles |
| --- | ---: | ---: | ---: | ---: | ---: |
| per-token natural | 891 / 2023 | 39 / 2 | 28/4 | 360721 | 388 |
| per-token stress | 891 / 2023 | 39 / 2 | 30/4 | 326369 | 337 |
| SwiGLU natural | 471 / 1137 | 39 / 3 | 24/5 | 168852 | 176 |
| SwiGLU stress | 471 / 1137 | 39 / 3 | 24/6 | 153201 | 170 |
| SwiGLU predicate-stress | 471 / 1137 | 39 / 3 | 24/5 | 158651 | 161 |
| MHC natural | 597 / 1346 | 43 / 9 | 23/4 | 424162 | 135 |

最大的 891-node region 和 MHC region 都完成了 DAG 构建、调度和 runtime 编译，
没有触发 max-edges、max-work-units、analysis timeout 或跳过 region。MHC 的 memory
DAG 已从历史报告中的 41023 条边压缩到 1346 条边。

## 相对直接父版本 Scheduler ON

这一节用于区分“当前 ON 相对 OFF 的效果”和 multi-user closure 对算法本身带来的
增量变化。父版本 raw evidence 来自同一组 fixture 的
`4fe22d64c07d87f29f230b1c6aedc7c3ec7c0bc9` Scheduler ON 运行。

| Case | 父版本 ON | 当前 ON | 变化 |
| --- | --- | --- | --- |
| per-token natural | peak 39/4，12/12 spill，median 3446 | peak 28/4，0/0 spill，median 3282 | spill 消除，-4.76% |
| per-token stress | peak 39/4，12/12 spill，median 3443 | peak 30/4，0/0 spill，median 3284 | spill 消除，-4.62% |
| SwiGLU natural | peak 26/6，无 spill，median 2771 | peak 24/5，无 spill，median 2816 | 压力更低，ticks +1.62% |
| SwiGLU stress | peak 26/6，无 spill，median 2767 | peak 24/6，无 spill，median 2816 | vector peak 更低，ticks +1.77% |
| SwiGLU predicate-stress | peak 26/6，无 spill，median 2767 | peak 24/5，无 spill，median 2800 | 压力更低，ticks +1.19% |
| MHC natural | peak 23/4，30/30 spill，median 3742 | peak 23/4，24/24 spill，median 3412 | spill 减少，-8.82% |

当前 closure group 明显修复了 per-token 的 canonical schedule 问题，并进一步
改善 MHC；代价是三个本来已经无 spill 的 SwiGLU 父版本 ON schedule 慢约
1.2%--1.8%。这属于后续性能排序可以继续研究的 trade-off，不影响本轮“避免
stress 压力爆炸并消除可消除 spill”的主要结论。

## Smoke 与仓库回归

| 检查 | 结果 |
| --- | --- |
| `micro-op/binary-vector/vadd` Scheduler OFF | 12962 ticks；9 variants compare pass |
| `micro-op/vector-load-store/vlds-post-update` Scheduler ON | 2421 ticks；compare pass |
| focused scheduler lit | 4/4 pass |
| tracker tests | 全部通过 |
| 完整 `check-pto` | 1809 pass，1 unsupported，0 fail |
| changed-code compliance | 0 error，0 warning |

算法提交后的本地回归结果沿用该 revision 已完成的验证。本次补测只增加 CA-model
OFF runtime evidence 和结果文档，没有修改 scheduler 实现或测试 fixture。

## Raw evidence

Artifact 根目录：

`/home/wanglan/PTOAS/.worktrees/ca-sim-results/vpto-sched-multi-user-closure-sim`

- 当前 revision OFF：`runtime-off/<case>/run{1,2,3}/`
- 当前 revision ON：`runtime-on/<case>/run{1,2,3}/`
- Smoke：`smoke-vadd/`、`smoke-vlds-post-update/`
- 工具链与动态库 provenance：`provenance.txt`
- 结果摘要：`summary.txt`
- SHA-256 manifest：`SHA256SUMS`
- Manifest 自身 SHA-256：
  `648b0b1e31adb2e4ad8da79ba98e9cadde28b15c761d09f435206c407820634e`
  （记录于 `SHA256SUMS.sha256`，覆盖 16529 个文件）

每个 runtime 目录均保留 runner log、host/compile 产物和 CA-model instruction
dump。Raw evidence 不加入源码 Git 提交。

## 已知限制

- MHC 仍有 24/24 次 vector spill/reload，当前结果是显著改善而非完全解决。
- per-token natural 的 ON/OFF median 差异只有 6 ticks（+0.18%），不足以宣称
  明确性能回退或提升，应视为基本持平。
- CA-model ticks 反映当前单核模型与生成机器码的观测结果；当前 scheduler 尚未
  使用这些 ticks 自动接受或回退某个 schedule。
- 模型 peak 是 SSA/live-range 层面的调度指标，不等价于最终物理寄存器分配的
  spill 数量；报告同时记录两者，避免只凭模型 peak 下结论。
