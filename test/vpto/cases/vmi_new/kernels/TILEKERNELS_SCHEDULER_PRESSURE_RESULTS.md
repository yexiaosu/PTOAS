<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# TileKernels VMI Scheduler Pressure 验证结果

## 结论

本轮在 A5 CA model 上真实执行了 natural/stress、Scheduler OFF/ON 矩阵。两个
stress case 的 OFF 版本都满足“建模 Vector peak > 32、最终指令出现真实 vector
spill/reload、strict compare 通过”；但 ON 版本的建模压力反而增大，并在进入
CA model 前因 vector stack 超过 6144 bytes 而编译失败。因此，这两个 case 是
可信的 scheduler 负向结果，不能称为“ON 降压并消除 spill”的正向验证。

natural case 均能在 Scheduler OFF/ON 下完成 CA-model SIM 和 strict compare。
ticks 仅作观测，不作为 schedule acceptance gate。

后续新增的 SwiGLU predicate-stress case 给出了独立的正向结果：原始顺序的
Vector/Predicate peak 为 14/11，Scheduler ON 后为 28/7。OFF 的最终指令包含
19/19 次真实 predicate spill/reload 和 38 次 `SMEM_BAR`，ON 全部归零；两种
模式各三次 CA-model SIM 均 strict compare 通过，median ticks 从 3337 降至
2845（-14.7438%）。该 case 没有 `Sn[95]` vector spill，因此可以隔离验证
predicate pressure 调度。

## Revision 与执行环境

| 项目 | 值 |
| --- | --- |
| TileKernels-vmi 来源 | `learning-chip/TileKernels-vmi@ab0b018b60f7c9057909acd1c75f2adf5f40aeb5` |
| PTOAS 集成 base | `d9d460bcf357110c476617f11fb821b266b04cef` |
| 原始矩阵实际验证 revision | `8be6592e6669736c85a7af60ce0b1f30159b137b` |
| Predicate-stress 实际验证 revision | `81f6f94005fee276bd3d5e6b2dd273844c554bdf` |
| 分支 | `codex/tilekernels-vmi-scheduler-pressure-v3` |
| 服务器 | `wanglan@115.175.35.144`，host `ecs-1030-cba0` |
| CANN / CA model | `/usr/local/CANN/cann-9.1.0`，A5 `dav_3510` SIM |
| Bisheng | `/usr/local/CANN/cann-9.1.0/bin/bisheng`，clang 15.0.5，build 2026-07-01 |
| LLVM | `/home/wanglan/llvm-workspaces/build-vpto19`，LLVM 19.1.7 |
| PTO-ISA | `/home/wanglan/pto-isa@311ca0c83f5571dc165681fee0a427983c555d3c` |
| SIM library | `/usr/local/CANN/cann-9.1.0/x86_64-linux/simulator/dav_3510/lib` |
| 仓库 runner | `test/vpto/scripts/run_host_vpto_validation.sh` |

服务器使用从 Git fetch/pull 得到的独立 worktree、venv 和 build：

- Worktree：`/home/wanglan/PTOAS/.worktrees/ca-sim-workspaces/tilekernels-vmi-scheduler-pressure-v3-sim`
- Python：上述 worktree 的 `.venv/bin/python`
- PTOAS CLI：上述 worktree 的 `.venv/bin/ptoas`
- `ptoas.__file__`：上述 worktree 的 `ptodsl/ptoas/__init__.py`
- `_core.__file__`：上述 worktree 的
  `.venv/lib/python3.12/site-packages/ptoas/_core.cpython-312-x86_64-linux-gnu.so`
- `libPTOASCompiler.so`：上述 worktree 的
  `.venv/lib/python3.12/site-packages/ptoas/mlir/_mlir_libs/libPTOASCompiler.so`
- Build：`/home/wanglan/PTOAS/.worktrees/ca-sim-builds/tilekernels-vmi-scheduler-pressure-v3-sim`

`libPTOASCompiler.so` 的 LLVM/MLIR 动态依赖均解析到
`/home/wanglan/llvm-workspaces/build-vpto19/lib`。每次 runtime 执行均设置
`DEVICE=SIM`、`COMPILE_ONLY=0`、`COMPARE_STRICT=1`，日志确认 CA model
start/stop、host executable 执行和真实 kernel instruction activity。

## 算法映射与 stress 构造

| TileKernels-vmi 来源 | PTOAS fixture | shape / dtype | 构造 |
| --- | --- | --- | --- |
| `quant/per_block_cast_vmi.py` | 复用 `block-quant-bf16-fp8-4x128` smoke | BF16 -> E4M3，4x128 | 现有 case 已覆盖等价 cast/scale 语义，未复制新 fixture |
| `quant/per_token_cast_and_cast_back_vmi.py` | `tilekernels-per-token-roundtrip-bf16-e4m3-16x256-{natural,stress}` | BF16 -> E4M3 -> BF16 | natural 逐 tile 完成 quant/scale/cast-back；stress 仅把独立 tile producer 集中、延迟真实 cast-back consumer/store |
| `quant/swiglu_backward_vmi.py` | `tilekernels-swiglu-backward-bf16-2x512-{natural,stress,predicate-stress}` | BF16，N=2，hidden=512，`with_weights=false` | natural 逐 chunk 完成梯度链；stress 延迟 gradient convert/store；predicate-stress 每四个 chunk 集中产生八个真实 clamp mask 和 raw gradient，再执行参与最终输出的 `vsel`/convert/store |
| `mhc/post_vmi.py::mhc_post_bwd_vmi` | `tilekernels-mhc-post-backward-bf16-n1-mhc4-h256-natural` | BF16，N=1，mhc=4，hidden=256 | 保留 reduction 与 load/store 依赖下的自然数据流；本 revision 未增加合成 stress 依赖 |

8x256 是 per-token 的初始筛选规模；最终 runtime 压力矩阵使用最小幅度扩大后的
16x256。每个 fixture 的 OFF/ON 使用相同输入、golden 与有效计算，所有中间
结果均参与最终 store；没有 dummy value、volatile、伪依赖、额外 barrier 或
依赖 DCE 的构造。

Predicate-stress 的 golden 还将 205 个 lanes 设为 `x=4,y=±4`。两个 clamp mask
在这些 lanes 都应选择精确零；任一 predicate spill/reload 恢复错误都会暴露
非零 raw gradient，因而 strict compare 不会因输入恰好为零而漏检。

没有选择 `topk_gate_vmi.py`（与 #508/#574 场景重复）、
`swiglu_forward_vmi.py` 和 `cast_back_vmi.py`（数据流线性）、
`per_channel_cast_vmi.py`（首批 layout/MTE 因素过多）、`pre_apply_mix_bwd`
（优先级低于 MHC post）以及 `multilayer_recompute_vmi.py`（控制流和跨层状态复杂）。

## Region 与压力分析

所有目标 case 均 lowering 为一个从 block start 到 block end 的可调度 region，
unknown class 为 0。

| Case | Regions | Nodes / edges | Live-in / live-out | Known / unknown | Original V/P | Scheduled V/P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per-token natural 16x256 | 1 | 891 / 2399 | 39 / 2 | 891 / 0 | 14 / 4 | 14 / 4 |
| per-token stress 16x256 | 1 | 891 / 2399 | 39 / 2 | 891 / 0 | 44 / 4 | 97 / 4 |
| SwiGLU natural | 1 | 471 / 3140 | 39 / 3 | 471 / 0 | 8 / 5 | 9 / 5 |
| SwiGLU stress | 1 | 471 / 3140 | 39 / 3 | 471 / 0 | 38 / 4 | 68 / 19 |
| SwiGLU predicate-stress | 1 | 471 / 3140 | 39 / 3 | 471 / 0 | 14 / 11 | 28 / 7 |
| MHC natural | 1 | 597 / 41023 | 43 / 9 | 597 / 0 | 29 / 3 | 29 / 4 |

per-token stress 的 ON trace 记录 468 个 pressure idle；SwiGLU stress 记录 269
个。两者均未把可结束 live range 的 consumer 调度到足以降低峰值的位置。

## A5 CA-model runtime 矩阵

`VSTI/VLDI` 计数只统计 instruction log 中使用 stack base `Sn[95]=0x40000`
的 `RV_VSTI`/`RV_VLDI`，以排除算法本身的普通 load/store。Barrier 和有效计算
opcode 使用每组 run1 的代表性 instruction log 统计。

| Case | Scheduler | Compile | Original -> scheduled V/P | Stack VSTI/VLDI | Barrier | 有效计算 ops | ticks（3 次；median） | Strict compare |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| per-token natural | OFF | pass | 14/4 -> - | 0/0 | 1 | 726 | 3280, 3278, 3270; **3278** | 3/3 pass |
| per-token natural | ON | pass | 14/4 -> 14/4 | 0/0 | 1 | 726 | 3278, 3273, 3278; **3278** | 3/3 pass |
| per-token stress | OFF | pass | 44/4 -> - | 11/11 | 23 | 732 | 3314, 3308, 3312; **3312** | 3/3 pass |
| per-token stress | ON | **fail** | 44/4 -> 97/4 | - | - | - | 无 CA run | 未执行 |
| SwiGLU natural | OFF | pass | 8/5 -> - | 0/0 | 1 | 388 | 2840, 2832, 2842; **2840** | 3/3 pass |
| SwiGLU natural | ON | pass | 8/5 -> 9/5 | 0/0 | 1 | 388 | 2834, 2826, 2822; **2826** | 3/3 pass |
| SwiGLU stress | OFF | pass | 38/4 -> - | 6/6 | 13 | 390 | 2905, 2906, 2899; **2905** | 3/3 pass |
| SwiGLU stress | ON | **fail** | 38/4 -> 68/19 | - | - | - | 无 CA run | 未执行 |
| MHC natural | OFF | pass | 29/3 -> - | 45/45 | 107 | 258 | 4738, 4745, 4742; **4742** | 3/3 pass |
| MHC natural | ON | pass | 29/3 -> 29/4 | 45/45 | 113 | 263 | 4800, 4800, 4803; **4800** | 3/3 pass |

两组 stress ON 各重复编译三次，均在 device LLVM 阶段失败，未启动 CA model：

- per-token stress ON：Vector Slots 83，Total Spilled Byte Size 21504，超过
  VF stack 6144 bytes。
- SwiGLU stress ON：Vector Slots 43，Total Spilled Byte Size 11456，超过
  VF stack 6144 bytes。

因此整个矩阵共尝试 30 项：24 项完成真实 CA-model run 和 strict compare，6 项
在 CA model 前按预期重复确认同一编译失败。MHC 的建模 peak 低于 Vector limit
32，但最终指令仍有 45/45 stack store/load；ON 没有减少该数量，median ticks
由 4742 增至 4800。本任务只记录这个观测，不据此修改 pressure model 或 scheduler。

### Predicate-stress 扩展矩阵

Predicate spill/reload 同样只统计 stack base `Sn[95]=0x40000`。最终固定矩阵显式
设置 `PTO_ISA_PATH=/home/wanglan/pto-isa`，每项均确认 CA model start/stop、
instruction activity 和 strict compare。

| Scheduler | Compile | Original -> scheduled V/P | Stack PSTI/PLDI | Stack VSTI/VLDI | SMEM_BAR | ticks（3 次；median） | Strict compare |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| OFF | pass | 14/11 -> - | 19/19 | 0/0 | 38 | 3338, 3337, 3335; **3337** | 3/3 pass |
| ON | pass | 14/11 -> 28/7 | 0/0 | 0/0 | 0 | 2843, 2848, 2845; **2845** | 3/3 pass |

ON trace 记录 32 个 pressure idle，并把 predicate peak 压到硬件限制 7。median
减少 492 ticks（14.7438%）；更关键的 acceptance evidence 是 19/19
`PSTI/PLDI` 和 38 个 barrier 被完全消除，同时 vector stack spill 始终为零。

## 仓库回归验证

| 检查 | 结果 |
| --- | --- |
| `llvm-lit --show-tests` | 新增 `vpto_scheduler_vector_pressure.pto` 可发现；tracker 与 live-through 用例仍可发现 |
| focused scheduler lit | vector pressure、tracker、live-through 共 3/3 pass |
| 完整 `check-pto` | 1810 discovered，1809 pass，1 unsupported，0 fail |
| changed-code compliance | base `d9d460bcf357110c476617f11fb821b266b04cef`；20 files，0 error，0 warning |
| Predicate fixture changed-code compliance | base `82b7a4c37f0df25938371a32e379ad4ced51a420`；4 files，0 error，0 warning |
| `git diff --check` | pass |

## Smoke

在目标矩阵前运行了三个仓库既有 case，均确认 CA model start/stop、host 执行、
kernel instruction activity 和 strict compare：

| Case | ticks | Compare / activity |
| --- | ---: | --- |
| `micro-op/binary-vector/vadd` | 12966 | pass；`RV_VADD` |
| `micro-op/vector-load-store/vlds-post-update` | 2421 | pass |
| `vmi_new/kernels/block-quant-bf16-fp8-4x128` | 2519 | pass |

Predicate-stress revision 在固定矩阵前另行复跑 `micro-op/binary-vector/vadd`，
得到 12973 ticks，并确认真实 CA activity 与 strict compare。

## Raw evidence

Artifact 根目录：

`/home/wanglan/PTOAS/.worktrees/ca-sim-results/tilekernels-vmi-scheduler-pressure-v3-sim`

- 三次矩阵日志和 instruction dump：`runtime-matrix-fixed/<case>/<off|on>/run{1,2,3}/`
- Compiler/scheduler trace：`scheduler-analysis/*.trace`
- Smoke：`smoke-vadd/`、`smoke-postupdate/`、`smoke-quant/`
- 完整 native provenance：`provenance.txt`
- SHA-256 manifest：`SHA256SUMS`
- Manifest 自身 SHA-256：`075ae65bf2aff82d358466e913a763eadefce182e664c2a43645df7632c84cfd`

Predicate-stress 扩展 artifact：

`/home/wanglan/PTOAS/.worktrees/ca-sim-results/tilekernels-vmi-scheduler-pressure-v3-sim/predicate-stress-81f6f9400`

- 最终固定矩阵：`runtime-matrix-fixed/predicate-stress/<off|on>/run{1,2,3}/`
- Compiler/scheduler trace：`scheduler-analysis/*.trace`
- Smoke：`smoke-vadd/`
- Native provenance 与结果摘要：`provenance.txt`、`summary.txt`
- SHA-256 manifest：`SHA256SUMS`
- Manifest 自身 SHA-256：`3831d1f8e11d36661454ff2218f0d622f65bd509046d72c3d5cc648cdfd965c8`

## 已知限制与后续候选

- 当前两个 stress case 都证明 OFF 有真实 spill，但没有证明 scheduler ON 能降压；
  它们应保留为负向诊断输入，而不是正向性能回归基准。
- SwiGLU predicate-stress 是独立的正向 predicate-pressure 基准；它不替代上述
  vector stress 的负向结论。
- MHC 仅有 natural fixture；没有为追求压力数字而增加伪依赖或额外 barrier。
- 本轮未增加 `round_sf=true`、SwiGLU `with_weights=true` 或 MHC hidden=512，
  应先解决当前 ON 顺序使压力恶化的问题，再扩展矩阵。
- CA SIM ticks 是单核局部 scheduler 的直接观测；不需要依赖 Vector Core
  任务分配、persistent 调度或全芯片带宽来解释本轮结论。
