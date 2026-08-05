# `vldus` / `vstus` Post-Update 补充设计分析

## 1. 范围与结论

本文补充 `docs/designs/vpto-soft-postupdate-design-zh.md` 未覆盖的
`pto.vldus` 与 `pto.vstus`。本轮只确定接口、变换约束和待验证语义，不在
SIM 证据缺失时猜测硬件行为。

结论如下：

1. 两版已盘点的 CANN（9.0.0-beta.1 与 9.0.0）都存在独立的
   `vldus.post` / `vstus.post` intrinsic，且 LLVM ABI 一致。
2. `vldus` 不能直接加入现有 `PostUpdateTable`：普通形式没有 offset，post
   形式却新增 `inc`，属于设计文档所说的 Soft 分支特例。
3. `vstus` 虽已有 `offset`，但它是 unaligned-store align stream 的推进量，
   当前文档和运行用例都不把它解释为“在 `base + offset` 起始写一个完整
   vector”。因此不能把它当成普通 Auto 指令直接套用
   `effective_address = base + offset` 的公式。
4. 两条指令的 post 形式都应显式暴露 `updated_base`；`updated_align` 的现有
   线性 state chain 仍需保留，base chain 与 align chain 必须同步演化。
5. `inc/offset` 的硬件单位、`updated_base` 的精确增量，以及 `vstus` 尾部
   flush 应使用哪个地址，不能仅凭 C++ 参数名或 LLVM `i32` 类型确定，必须
   由 SIM 探针裁决。

## 2. 已确认的静态事实

### 2.1 当前 PTO IR

普通 `vldus`：

```mlir
%vec, %align1 = pto.vldus %base, %align0
    : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align
```

普通 `vstus`：

```mlir
%align1 = pto.vstus %align0, %offset, %vec, %base
    : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align
```

当前两套 emitter 的行为相同：

- `vldus` 调用非 post `llvm.hivm.vldus.v*`。该 intrinsic 内部返回
  `{vector, align, ptr}`，但当前已安装的 no-post 接口中第 3 个字段不是
  PTO 用户可依赖的更新地址，PTOAS 丢弃它。
- `vstus` 先用 `convertElementOffsetToBytes` 将 PTO 的元素 offset 转成字节，
  再调用非 post `llvm.hivm.vstus`，只返回 align。
- 已有运行用例对 packed-byte 类型执行 `offset = 3` 的
  `vstus + vstas`，只观察到 3 个显式字节，和上述 lowering 一致。

历史 release spec 中曾出现带 `base_out` 的 `vldus/vstus` 草案，但它们与
当前 ODS、emitter 和主 ISA 文档不一致，不能作为硬件语义证据。

### 2.2 已确认的 post ABI

| PTO op | C++ A5 post wrapper（代表类型） | LLVM intrinsic |
|---|---|---|
| `vldus` | `vldus(dst, align, T *&base, uint32_t inc, POST_UPDATE)` | `{vector, align, ptr} @llvm.hivm.vldus.post.v*(ptr, align, i32)` |
| `vstus` | `vstus(align, uint32_t offset, src, T *&base, POST_UPDATE)` | `{align, ptr} @llvm.hivm.vstus.post.v*(vector, ptr, i32, align)` |

两版清单都覆盖常见整数、浮点和低精度类型；本轮语义探针只需先使用 `u8`
和 `f32`。`u8` 便于查看逐字节结果，`f32` 用于区分“字节单位”和“元素单位”。

## 3. 建议的 PTO IR 形状

### 3.1 `vldus`

普通语法保持不变；post 形式新增 `i32 increment` 与最后一个
`updated_base` 结果：

```mlir
%vec, %align1, %base1 = pto.vldus %base0, %align0, %increment
    : !pto.ptr<T, ub>, !pto.align, i32
      -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>
```

ODS 可表达为成对出现的 `Optional<I32> increment` 与
`Optional<PTO_BufferType> updated_base`。verifier 必须要求：

- 两者同时存在或同时不存在；
- `updated_base` 与 `source` 类型完全相同；
- post 形式仍只接受 UB pointer；
- align 输入仍来自同一条合法 `vldas/vldus` load-state chain。

不要把 no-post intrinsic 的隐藏第 3 个返回字段直接暴露为 post 结果；post
形式必须选择独立的 `llvm.hivm.vldus.post.v*`，并显式传入 increment。

### 3.2 `vstus`

操作数保持不变，最后追加可选 base 结果：

```mlir
%align1, %base1 = pto.vstus %align0, %offset, %vec, %base0
    : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub>
      -> !pto.align, !pto.ptr<T, ub>
```

verifier 必须要求 `base_out` 与 `base` 类型相同，并继续执行现有 store align
chain 检查。lowering 根据 `base_out` 是否存在选择 `vstus` 或
`vstus.post.v*`；post 返回 aggregate 的第 0/1 项分别映射到
`align_out/base_out`。

PTO 层是否继续将 `%offset` 定义为“元素数”、再由 emitter 乘元素字节数，
取决于探针结果。为了保持普通/post 形式的 PTO 合同一致，除非 SIM 证明两种
wrapper 的单位不同，否则不应只为 post 形式改变 PTO offset 单位。

## 4. Soft-Postupdate 变换语义

### 4.1 `vldus`：从 base 递推产生新 increment

设普通程序第 `i` 次访问的显式地址为 `B_i`：

```text
(V_i, A_{i+1}) = vldus.no_post(B_i, A_i)
```

期望的 post 链是：

```text
(V_i, A_{i+1}, P_{i+1}) = vldus.post(P_i, A_i, I_i)
P_0 = B_0
```

只有 SIM 证明 `P_{i+1}` 的公式后，才能把 `B_{i+1} - B_i` 换算成 `I_i`。
变换必须遵守：

- 不允许在分析失败时用 vector 长度兜底；
- 循环路径把新增 base 与既有 align 一起作为 loop-carried state；
- 顺序路径除 root base 外还要按 load-align root/state chain 分桶，不能把两条
  独立的 `vldas/vldus` stream 串成同一条 base chain；
- generic descriptor 需要支持“原 op 无 stride operand、post op 插入一个
  stride operand”，不能继续假设只替换现有 operand。

若探针确认 increment 是字节，则 `addptr` 元素 delta `D` 对应
`I = elemBytes * D`；若确认是元素，则 `I = D`。在结论出来前该
`StrideUnit` 必须保持未决。

### 4.2 `vstus`：offset 不能按普通有效地址项处理

当前合同是 stateful unaligned store：`offset` 控制该步 stream 推进与残留
align state，不等价于“从 `base + offset` 写完整 vector”。因此 generic
Auto 公式

```text
effective_address = base + offset
stride_new = delta(base) + delta(offset)
```

对 `vstus` 没有成立依据。

若 SIM 确认 post 返回 `P_{i+1} = P_i + advance(offset_i)`，正确改写应保留
原来的 `offset_i`，并仅在下一次普通显式 base 正好等于该返回地址时建立
base chain：

```text
(A_{i+1}, P_{i+1}) = vstus.post(A_i, offset_i, V_i, P_i)
要求下一步原地址 B_{i+1} == P_{i+1}
```

这与 `vlds/vsts/vsstb` 的“把地址公差替换进 offset operand”机制不同，建议
给 `vstus` 单独的 stateful rewrite descriptor/路径。若 offset 是循环变量，
它可以原样保留；关键是证明 base recurrence 与硬件返回地址一致。

最终 `vstas/vstar` flush 地址也必须随 base chain 一起定义。probe 未完成前，
不能默认使用原 base、`base_out`、`base_out + offset` 中的任何一个。

## 5. 必须由 SIM 裁决的问题

| 编号 | 问题 | 为什么静态 ABI 不够 |
|---|---|---|
| L1 | `vldus.post` 的 `inc` 是字节、元素还是其他单位？ | LLVM 中仅为 `i32`，typed pointer reference 不证明硬件单位。 |
| L2 | load 是否始终发生在更新前的 base？ | 需要用第二次 chained load 的数据位置判定。 |
| L3 | `base_out` 是否精确等于 `base + inc`？ | 返回 `ptr` 只证明有结果，不证明公式。 |
| L4 | 一次 `vldas` 后连续 post `vldus` 是否可同时穿针 align/base？ | 需要比较数据与 align stream。 |
| S1 | `vstus.post` 的 offset 单位及 base 增量公式是什么？ | no-post lowering 的单位不能自动外推到 post wrapper。 |
| S2 | post `vstus` 的写地址使用更新前还是更新后 base？ | stateful store 不能按普通 store 地址推断。 |
| S3 | post chain 的最后一次 flush 应用哪个 base/offset？ | align 中可能仍有未提交 tail。 |
| S4 | `NO_POST_UPDATE` reference overload 是否与普通 overload 数据/align 等价且 base 不变？ | 用作 post 探针的阴性对照。 |

具体测试矩阵和判读规则见
`test/cann_probes/vldus_vstus_post/README.md`。

## 6. 探针完成后的实现层清单

探针通过并写回结论后，功能实现至少需要同步：

1. `include/PTO/IR/VPTOOps.td`：可选 operand/result 与 assembly form；
2. `lib/PTO/IR/VPTO.cpp`：类型、成对出现、align/base chain verifier；
3. 两套 VPTO LLVM emitter：typed `.post` callee、aggregate 结果、单位换算；
4. `VPTOSoftPostUpdate.cpp`：`vldus` 无原 stride 的 Soft 路径，以及
   `vstus` stateful base recurrence 路径；
5. lit：parse/print、invalid verifier、两版 LLVM ABI、loop/sequential 正反例；
6. SIM/NPU runtime case：数据、base delta、flush 完整性；
7. 主设计文档、micro ISA 文档以及打包 release 文档。
