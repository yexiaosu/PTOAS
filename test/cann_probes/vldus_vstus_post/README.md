# A5 `vldus` / `vstus` Post-Update SIM 探针

## 目标

本探针只回答 Post-Update 语义，不实现 PTOAS 功能。使用原始 CCE wrapper
或 PTO-ISA ST 在 A5 CA model 上运行，并同时保存 LLVM IR、指令日志和 UB/GM
结果。禁止根据参数名补写预期值；每个 case 都要让互斥解释产生不同输出。

## 公共设置

- 输入为逐字节唯一 ramp，例如 `input[i] = (17 * i + 3) % 251`。
- 输出先填 `0xA5` guard；每个 case 使用独立 1 KiB 区域，避免相互覆盖。
- load/store 起点使用 `region + 3`，强制走 unaligned 路径。
- 将每次 post 调用前后的 UB 指针转为整数地址，以**字节差**写入独立 meta
  区；不要使用 typed pointer subtraction作为最终记录，因为它会自动按元素
  缩放。
- 先跑 `u8`，再跑 `f32`。`f32` 的 `inc/offset = 1` 可区分 1-byte 与
  1-element(4-byte) 更新。
- 每个 case 保留 `core0.veccore0.instr_log.dump`、UB read/write dump、kernel
  输出、编译生成的 `.ll` 与完整命令行。

## Load cases

### L0：ABI 阳性对照

编译下列 wrapper，并确认调用：

```cpp
vldus(dst, align, base, inc, POST_UPDATE);
```

LLVM 必须出现 `llvm.hivm.vldus.post.v*`，返回 `{vector, align, ptr}`，实参为
`(base, align, i32 inc)`。若该项失败，后续运行结果无效，应先排查 CANN 版本
或 A5 target。

### L1：updated base 单位

分别对 `u8` 与 `f32` 执行一次：

```text
A0 = vldas(B0)
(V0, A1, B1) = vldus.post(B0, A0, inc)
record_bytes(B1 - B0)
```

矩阵：`inc = 0, 1, 3, 4, 31, 32, 64`。至少 `f32/inc=1` 必跑。

判读：

- `delta_bytes == inc`：字节单位；
- `delta_bytes == inc * sizeof(T)`：元素单位；
- 其他结果：记录精确公式，不归入上面两类。

### L2：更新前/更新后地址与双状态链

只调用一次 `vldas`，随后连续两次 post `vldus`：

```text
A0 = vldas(B0)
(V0, A1, B1) = vldus.post(B0, A0, inc)
(V1, A2, B2) = vldus.post(B1, A1, inc)
store V0 and V1 to disjoint observable regions
record B1-B0 and B2-B0 in bytes
```

输入 ramp 使以下解释互斥：

- load-before-update：`V0` 从 `B0` 开始，`V1` 从 `B1` 开始；
- load-after-update：`V0` 已从 `B1` 开始；
- align/base 不能共同穿针：第二个 vector 与任一连续切片都不匹配。

对 `u8/inc=3`、`u8/inc=32`、`f32/inc=1` 各跑一次。

### L3：NO_POST_UPDATE 阴性对照

对 reference-base overload 传 `NO_POST_UPDATE`，与普通 overload 比较：

- base 字节差必须记录为 0，或明确报告实际非零值；
- vector 与 align 后续行为必须和普通 overload 一致；
- LLVM/指令日志必须能区分传入的 post mode。

## Store cases

### S0：ABI 阳性对照

编译：

```cpp
vstus(align, offset, src, base, POST_UPDATE);
```

LLVM 必须出现 `llvm.hivm.vstus.post.v*`，返回 `{align, ptr}`，实参为
`(vector, base, i32 offset, align)`。

### S1：updated base 单位

从已知 vector 与 fresh align 开始，执行一次 post `vstus`，立即记录
`base_out - base_in` 的字节差。矩阵同 L1，至少覆盖：

- `u8/offset = 1, 3, 31, 32`；
- `f32/offset = 1, 3, 4`。

这一步只裁决 base 公式，不用未 flush 的内存内容裁决 store 语义。

### S2：写地址与 flush 地址

对同一输入 vector、base 与 offset，在互不重叠的 region 中构造四组：

1. 普通基线：normal `vstus`，再按当前普通序列使用 normal `vstas`；
2. post `vstus`，用原始 base 与普通 offset flush；
3. post `vstus`，用 `base_out` 与 zero offset flush；
4. post `vstus`，用 `base_out` 与原 offset flush。

将四个完整 region 拷回 GM。只有与普通基线逐字节相同的候选才可作为 flush
规则；若都不匹配，追加 `vstar(align_out, base_candidate)` 候选并记录结果，
不要任选最接近者。

至少运行 `u8/offset=3` 与 `f32/offset=1`。输出比较必须包含整个 guard 区，
以识别更新前/更新后地址和越界多写。

### S3：两步链

使用两个不同的 256-byte vector pattern：

```text
(A1, B1) = vstus.post(A0, 3, V0, B0)
(A2, B2) = vstus.post(A1, 5, V1, B1)
flush(A2, candidate)
record B1-B0, B2-B0
```

与显式 normal 序列对比，normal 第二步分别尝试 `B0 + observed_delta1` 与
原 base；以完整内存结果确认：

- offset 是否同时控制 align stream 与 base update；
- 第二步是否从第一次返回的 cursor 写入；
- 最终 flush cursor 是否等于 `B2`。

### S4：NO_POST_UPDATE 阴性对照

reference-base overload 传 `NO_POST_UPDATE`，要求记录 base delta、align 输出和
完整 flush 后内存，并与普通 overload 对比。

## 日志判读与完成门槛

从 instruction log 抽取包含 `VLDUS|VSTUS|VSTAS|VSTAR` 的行；从 UB dump
抽取每个 case 的读写地址。最终报告必须逐项给出 L1-L4/S1-S4 的结论、原始
地址差和匹配切片，不能只写“测试通过”。

只有下列三点都确定后，才能更新 PTOAS 设计：

1. `vldus inc` 与 `vstus offset` 的硬件单位；
2. 两条指令的 update-before/after 规则与 `base_out` 公式；
3. `vstus` post chain 的合法 flush 地址。
