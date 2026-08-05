# 给 SIM 执行 Agent 的 Prompt

你要验证 A5 `vldus` / `vstus` Post-Update 的真实语义。只做探针与证据整理，
不要实现 PTOAS 功能，也不要根据函数签名猜语义。

## 隔离与同步（必须先做）

1. 不得在现有 checkout 直接工作。先在 PTOAS 仓库执行 `git fetch origin`，
   确认并记录最新 `origin/codex/vmi-post-update-4-impl` commit。
2. 使用仓库的 `ptoas-workspace-manager` 创建独立 worktree、独立 `.venv` 和
   独立 build dir；新分支建议命名
   `codex/vldus-vstus-postupdate-sim-probe`，base ref 必须是刚同步的
   `origin/codex/vmi-post-update-4-impl`，不能使用脏主工作区或陈旧本地分支。
3. 如果需要修改 `pto-isa` ST，也必须为 `pto-isa` 创建独立 worktree，并先
   fetch/sync 它的相应远端默认分支。不要复用别的任务的 build 目录。
4. 在报告首部写明仓库绝对路径、worktree 路径、两个 base commit（若使用
   两个仓库）、CANN 版本、编译器版本和 simulator target。

## 必读与目标

在 PTOAS worktree 中阅读：

- `docs/designs/vldus-vstus-postupdate-analysis-zh.md`（若远端尚无该本地分析
  文件，就以本 prompt 的测试要求为准）；
- `test/cann_probes/vldus_vstus_post/README.md`（同上）；
- `docs/designs/vpto-soft-postupdate-design-zh.md` 中 Soft/Auto 分支区别；
- 当前 `VldusOp/VstusOp` ODS、verifier 与两套 LLVM emitter。

已确认但仍需在你的环境复核的 ABI：

- `vldus.post.v*`: `{vector, align, ptr}(ptr, align, i32 inc)`；
- `vstus.post.v*`: `{align, ptr}(vector, ptr, i32 offset, align)`。

你需要用 A5 CA model / CANN SIM 回答：

1. `vldus inc` 是字节、元素还是其他单位；load 使用更新前还是更新后 base；
   `base_out` 公式；一次 `vldas` 后是否可同时穿针 align 与 base。
2. `vstus offset` 的单位与 `base_out` 公式；store 使用更新前还是更新后
   base；post chain 最终应以原 base、`base_out` 还是其他地址执行
   `vstas/vstar` flush。
3. `NO_POST_UPDATE` reference overload 是否与普通 overload 等价且 base
   保持不变。

## 探针要求

- 输入使用逐字节唯一 ramp，输出使用 `0xA5` guard；base 取 `region + 3`。
- 同时测试 `u8` 和 `f32`，必须包含 `f32 inc/offset = 1`，以区分 1 byte 与
  1 element。
- 每次调用把输入/输出 UB 指针转换成整数地址，记录**字节差**到 GM meta；
  不要用 typed pointer subtraction作为最终证据。
- Load：一次 `vldas` 后连续两次 post `vldus`，保存两个 vector 和两次 base
  delta；至少跑 `u8 inc=3/32`、`f32 inc=1`。
- Store：单步 post `vstus` 记录 base delta；再用独立 region 对比四个 flush
  候选：normal 基线、`original_base + original offset`、
  `base_out + zero offset`、`base_out + original offset`；若均不匹配，增加
  `vstar` 候选。再跑 offset 3/5 的两步 chain。
- 分别运行 POST_UPDATE 与 NO_POST_UPDATE 阴性对照。
- 使用 `camodel-isa-verification` 工作流运行 A5 ST。保留并打包 LLVM IR、
  编译日志、`core0.veccore0.instr_log.dump`、UB read/write dump、输入输出和
  pointer-delta meta。日志中抽取 `VLDUS|VSTUS|VSTAS|VSTAR` 相关行。
- 如果某个 wrapper 在当前 CANN/arch 不可用，先给出最小 compile probe 与
  完整诊断，不要把“编译失败”写成硬件语义结论。

## 输出格式

提交一份中文报告，至少包含：

1. 环境、同步后的 commit 与精确命令；
2. 每个 probe case 的输入参数、base 字节差、数据匹配区间、flush 匹配结果；
3. 对每个问题标记“已证实 / 已证伪 / 仍不确定”，并链接原始日志文件；
4. 给出可直接写回 PTOAS 的公式，例如
   `base_out = base_in + inc_bytes`，同时注明单位；
5. 明确判断 `vstus` 能否进入 generic Auto table，还是必须使用 stateful
   专用 rewrite；
6. 保留所有 probe 源码，不要在本任务中修改 ODS/emitter/pass 功能代码。

遇到 SIM 不支持或结果冲突时，缩小 case 并继续收集证据；不要用 C++ 参数名、
旧 release spec 或其他指令的行为填补结论。
