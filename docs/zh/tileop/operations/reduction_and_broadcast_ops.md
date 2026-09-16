# 轴规约与广播操作

本节描述了 PTO ISA 中沿行或沿列进行规约（reduction）和广播（broadcast）的全部操作。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用"目标传递风格"（Destination-Passing Style, DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。全部操作执行在 **Vector 流水线**（`PIPE_V`）上。

这一类操作通常具有如下装配形式：

```mlir
pto.op ins(%src : !pto.tile_buf<...>)
       outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 所有 tile 必须位于 `loc=vec`（VEC/UB 存储空间）
- 输入 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
- 输入与输出元素类型一致（`argmax`/`argmin` 除外，其输出为整数索引类型）

---

## 目录

- [`pto.tcolexpand` — 列广播](#ptotcolexpand--列广播)
- [`pto.tcolmax` — 列最大值规约](#ptotcolmax--列最大值规约)
- [`pto.tcolargmax` — 列最大值索引规约](#ptotcolargmax--列最大值索引规约)
- [`pto.tcolmin` — 列最小值规约](#ptotcolmin--列最小值规约)
- [`pto.tcolargmin` — 列最小值索引规约](#ptotcolargmin--列最小值索引规约)
- [`pto.tcolsum` — 列求和规约](#ptotcolsum--列求和规约)
- [`pto.trowexpand` — 行广播](#ptotrowexpand--行广播)
- [`pto.trowmax` — 行最大值规约](#ptotrowmax--行最大值规约)
- [`pto.trowargmax` — 行最大值索引规约](#ptotrowargmax--行最大值索引规约)
- [`pto.trowmin` — 行最小值规约](#ptotrowmin--行最小值规约)
- [`pto.trowargmin` — 行最小值索引规约](#ptotrowargmin--行最小值索引规约)
- [`pto.trowsum` — 行求和规约](#ptotrowsum--行求和规约)
- [`pto.tcolprod` — 列乘积规约](#ptotcolprod--列乘积规约)
- [`pto.trowprod` — 行乘积规约](#ptotrowprod--行乘积规约)
- [`pto.trowexpandsub` — 行广播减法](#ptotrowexpandsub--行广播减法)
- [`pto.trowexpandmul` — 行广播乘法](#ptotrowexpandmul--行广播乘法)
- [`pto.trowexpanddiv` — 行广播除法](#ptotrowexpanddiv--行广播除法)
- [`pto.tcolexpandmax` — 列广播取最大值](#ptotcolexpandmax--列广播取最大值)
- [`pto.tcolexpandmin` — 列广播取最小值](#ptotcolexpandmin--列广播取最小值)
- [`pto.tcolexpandmul` — 列广播乘法](#ptotcolexpandmul--列广播乘法)
- [`pto.tcolexpandadd` — 列广播加法](#ptotcolexpandadd--列广播加法)
- [`pto.tcolexpanddiv` — 列广播除法](#ptotcolexpanddiv--列广播除法)
- [`pto.tcolexpandexpdif` — 列广播指数差](#ptotcolexpandexpdif--列广播指数差)
- [`pto.tcolexpandsub` — 列广播减法](#ptotcolexpandsub--列广播减法)
- [`pto.trowexpandadd` — 行广播加法](#ptotrowexpandadd--行广播加法)
- [`pto.trowexpandexpdif` — 行广播指数差](#ptotrowexpandexpdif--行广播指数差)
- [`pto.trowexpandmax` — 行广播取最大值](#ptotrowexpandmax--行广播取最大值)
- [`pto.trowexpandmin` — 行广播取最小值](#ptotrowexpandmin--行广播取最小值)

---

## 操作详解

### `pto.tcolexpand` — 列广播

```mlir
pto.tcolexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile，行向量，每列携带一个逻辑标量 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 和 `dst` 必须使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 元素类型一致：`dst_type == src_type`
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                   v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolmax` — 列最大值规约

```mlir
pto.tcolmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```text
For each column j:
    dst[0, j] = max over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的最大值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查（A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolargmax` — 列最大值索引规约

```mlir
pto.tcolargmax ins(<src>[, <tmp>] : <src_type>[, <tmp_type>])
               outs(<dst> : <dst_type>)
```

**语义：**

```text
For each column j:
    dst[0, j] = argmax over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 可选临时缓冲区，容量和类型要求见下文 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每列最大值的行索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **公共约束（A2A3/A5）**
  - `src`、`dst` 必须使用 `loc=vec` 和 ND-style 布局（`blayout=row_major`、`slayout=none_box`）。
  - `src` 支持 8、16、32 位整数以及 `f16`、`bf16`、`f32`；`dst` 为 32 位整数索引，例如 `i32` 或 `ui32`。
  - `src` 的有效行数、有效列数必须非零；`dst valid_shape[0] == 1`，源和目标的有效列数相等。
  - `tmp` 可省略。显式提供时必须是 `loc=vec`、`blayout=row_major` 的 tile。
  - 动态有效维度在运行时满足相同条件。

- **显式临时缓冲区（A2A3）**
  - `tmp` 与 `src` 的元素类型相同，容量至少为 32 字节；不要求物理 shape 相同。
  - 若二者的 valid shape 完全相同，满足上述容量条件即可。
  - 否则，`tmp` 的有效行数至少为 1。设源有效列数为 `C`，每个元素占 `E` 字节，
    `R = ceil(C / (256 / E))`，`B = 32 / E`，则临时缓冲区的有效列数至少为
    `(ceil(2 * R / B) + ceil(R / B)) * B`。动态 `C` 的缓冲区大小需由调用方保证。

- **显式临时缓冲区（A5）**
  - `tmp` 保留为可选缓冲区参数，不要求与 `src` 的物理 shape、valid shape 或元素类型相同。

**示例：**

```mlir
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolmin` — 列最小值规约

```mlir
pto.tcolmin ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```text
For each column j:
    dst[0, j] = min over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的最小值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查（A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolmin ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolargmin` — 列最小值索引规约

```mlir
pto.tcolargmin ins(<src>[, <tmp>] : <src_type>[, <tmp_type>])
               outs(<dst> : <dst_type>)
```

**语义：**

```text
For each column j:
    dst[0, j] = argmin over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 可选临时缓冲区，容量和类型要求见下文 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每列最小值的行索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **公共约束（A2A3/A5）**
  - `src`、`dst` 必须使用 `loc=vec` 和 ND-style 布局（`blayout=row_major`、`slayout=none_box`）。
  - `src` 支持 8、16、32 位整数以及 `f16`、`bf16`、`f32`；`dst` 为 32 位整数索引，例如 `i32` 或 `ui32`。
  - `src` 的有效行数、有效列数必须非零；`dst valid_shape[0] == 1`，源和目标的有效列数相等。
  - `tmp` 可省略。显式提供时必须是 `loc=vec`、`blayout=row_major` 的 tile。
  - 动态有效维度在运行时满足相同条件。

- **显式临时缓冲区（A2A3）**
  - `tmp` 与 `src` 的元素类型相同，容量至少为 32 字节；不要求物理 shape 相同。
  - 若二者的 valid shape 完全相同，满足上述容量条件即可。
  - 否则，`tmp` 的有效行数至少为 1。设源有效列数为 `C`，每个元素占 `E` 字节，
    `R = ceil(C / (256 / E))`，`B = 32 / E`，则临时缓冲区的有效列数至少为
    `(ceil(2 * R / B) + ceil(R / B)) * B`。动态 `C` 的缓冲区大小需由调用方保证。

- **显式临时缓冲区（A5）**
  - `tmp` 保留为可选缓冲区参数，不要求与 `src` 的物理 shape、valid shape 或元素类型相同。

**示例：**

```mlir
pto.tcolargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolsum` — 列求和规约

```mlir
// 不提供临时缓冲区
pto.tcolsum ins(<src> : <src_type>) outs(<dst> : <dst_type>)

// 显式临时缓冲区；属性字典位于 ins 内、类型列表之前
pto.tcolsum ins(<src>, <tmp> {isBinary = <bool>} : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**语义：**

```text
For each column j:
    dst[0, j] = sum over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 可选临时缓冲区，用于中间计算 |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的求和结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `isBinary` — 是否使用二叉规约树。默认值为 `false`。
  - `true` — 使用二叉规约树
  - `false` — 使用默认规约方式

显式临时缓冲区形式中可写 `{isBinary = true}` 或 `{isBinary = false}`；省略属性时采用默认值 `false`。
使用二叉规约形式时应同时提供 `tmp` 和 `{isBinary = true}`。

**约束：**

- **实现检查（A2A3）**
  - `src`、`dst` 和显式提供的 `tmp` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`；有 `tmp` 时其元素类型也必须相同
  - `src valid column == dst valid column`

- **实现检查（A5）**
  - `src`、`dst` 和显式提供的 `tmp` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`；有 `tmp` 时其元素类型也必须相同
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

- **二叉规约临时缓冲区**
  - 源 valid shape 和元素字节大小必须静态已知。
  - `tmp` 的物理列数至少为源有效列数，容量至少为
    `ceil(src valid_shape[0] / 2) * src valid_shape[1] * sizeof(dtype)` 字节。

**示例：**

```mlir
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowexpand` — 行广播

```mlir
pto.trowexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile，列向量，每行携带一个逻辑标量 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 必须使用 `slayout=none_box`
  - `dst` 必须使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 元素类型一致：`dst_type == src_type`
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `src valid row == dst valid row`
  - `src valid row != 0` 且 `src valid column != 0` 且 `dst valid row != 0` 且 `dst valid column != 0`

**示例：**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowmax` — 行最大值规约

```mlir
pto.trowmax ins(<src>[, <tmp>] : <src_type>[, <tmp_type>]) outs(<dst> : <dst_type>)
```

**语义：**

```text
For each row i:
    dst[i, 0] = max over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 可选临时缓冲区；A2A3 要求与源元素类型相同且容量至少为 32 字节 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的最大值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`
  - 非空目标满足 `dst valid_shape[1] == 1`，与物理列数及行/列主序无关
  - 完全为空的 `dst valid_shape = [0, 0]` 表示不写入任何元素，此时跳过源非空、有效行相等及目标有效列为 1 的约束；只有一个有效维度为 0 不属于此情况
  - 提供 `tmp` 时必须使用 `loc=vec`；A2A3 要求其元素类型与 `src` 相同、容量至少为 32 字节；A5 不要求其 shape、valid shape 或元素类型与源相同

**示例：**

```mlir
pto.trowmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowargmax` — 行最大值索引规约

```mlir
pto.trowargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```text
For each row i:
    dst[i, 0] = argmax over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每行最大值的列索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `dst` 使用 `slayout=none_box`，且为 DN-style 列向量（`blayout=col_major`, `cols=1`）或 ND-style tile（`valid column == 1`）
  - `src` 元素类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 元素类型：`i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**示例：**

```mlir
pto.trowargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowmin` — 行最小值规约

```mlir
pto.trowmin ins(<src>[, <tmp>] : <src_type>[, <tmp_type>])
            outs(<dst> : <dst_type>)
```

**语义：**

```text
For each row i:
    dst[i, 0] = min over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 可选临时缓冲区；A2A3 要求与源元素类型相同且容量至少为 32 字节 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的最小值 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`
  - 非空目标满足 `dst valid_shape[1] == 1`，与物理列数及行/列主序无关
  - 完全为空的 `dst valid_shape = [0, 0]` 表示不写入任何元素，此时跳过源非空、有效行相等及目标有效列为 1 的约束；只有一个有效维度为 0 不属于此情况
  - 提供 `tmp` 时必须使用 `loc=vec`；A2A3 要求其元素类型与 `src` 相同、容量至少为 32 字节；A5 不要求其 shape、valid shape 或元素类型与源相同

**示例：**

```mlir
pto.trowmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowargmin` — 行最小值索引规约

```mlir
pto.trowargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```text
For each row i:
    dst[i, 0] = argmin over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每行最小值的列索引 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `dst` 使用 `slayout=none_box`，且为 DN-style 列向量（`blayout=col_major`, `cols=1`）或 ND-style tile（`valid column == 1`）
  - `src` 元素类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 元素类型：`i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**示例：**

```mlir
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowsum` — 行求和规约

```mlir
pto.trowsum ins(<src>[, <tmp>] : <src_type>[, <tmp_type>]) outs(<dst> : <dst_type>)
```

**语义：**

```text
For each row i:
    dst[i, 0] = sum over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 可选临时缓冲区；A2A3 要求与源元素类型相同且容量至少为 32 字节 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的求和结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`
  - 非空目标满足 `dst valid_shape[1] == 1`，与物理列数及行/列主序无关
  - 完全为空的 `dst valid_shape = [0, 0]` 表示不写入任何元素，此时跳过源非空、有效行相等及目标有效列为 1 的约束；只有一个有效维度为 0 不属于此情况
  - 提供 `tmp` 时必须使用 `loc=vec`；A2A3 要求其元素类型与 `src` 相同、容量至少为 32 字节；A5 不要求其 shape、valid shape 或元素类型与源相同

**示例：**

```mlir
pto.trowsum ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolprod` — 列乘积规约

```mlir
pto.tcolprod ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```text
For each column j:
    dst[0, j] = product over i of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的乘积结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`ui16`、`i32`、`ui32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查（A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i16`、`ui16`、`i32`、`ui32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

**示例：**

```mlir
pto.tcolprod ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                 v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.trowprod` — 行乘积规约

```mlir
pto.trowprod ins(<src>[, <tmp>] : <src_type>[, <tmp_type>])
             outs(<dst> : <dst_type>)
```

**语义：**

```text
For each row i:
    dst[i, 0] = product over j of src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 可选临时缓冲区；A2A3 要求与源元素类型相同且容量至少为 32 字节 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的乘积结果 |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`
  - 非空目标满足 `dst valid_shape[1] == 1`，与物理列数及行/列主序无关
  - 完全为空的 `dst valid_shape = [0, 0]` 表示不写入任何元素，此时跳过源非空、有效行相等及目标有效列为 1 的约束；只有一个有效维度为 0 不属于此情况
  - 提供 `tmp` 时必须使用 `loc=vec`；A2A3 要求其元素类型与 `src` 相同、容量至少为 32 字节；A5 不要求其 shape、valid shape 或元素类型与源相同

**示例：**

```mlir
pto.trowprod ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=1,
                 v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.trowexpandsub` — 行广播减法

```mlir
pto.trowexpandsub ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（被减数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **使用约束（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - 可选 `tmp`：A2A3 的显式临时缓冲区形式使用列主序、每行一个标量的广播源；缓冲区容量要求见下文

- **使用约束（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - 可选 `tmp` 可保留；不要求其 shape 与源相同

A2A3 显式提供 `tmp` 时，它必须位于 `vec`，与目标元素类型相同。设目标有效行数为 `M`，
所需容量为：`M < 256` 时 `ceil(M / 8) * 256` 字节，否则为 `7680` 字节；
动态 `M` 按 `8192` 字节预留。

**示例：**

```mlir
pto.trowexpandsub ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmul` — 行广播乘法

```mlir
pto.trowexpandmul ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（乘数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- `src0`、`src1`、`dst` 元素类型一致，`dst` 使用 `blayout=row_major`。
- A2A3 支持 `i16`、`i32`、`f16`、`f32`；A5 还支持 `i8`。
- `tmp` 可选；A2A3 显式提供时采用列主序、每行一个标量的广播源，临时缓冲区位于 `vec`，
  与目标元素类型相同，容量要求与 `pto.trowexpandsub` 相同。A5 不要求 `tmp` 与源同 shape。

**示例：**

```mlir
pto.trowexpandmul ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpanddiv` — 行广播除法

```mlir
// 默认精度
pto.trowexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)

// 高精度（需要 tmp）
pto.trowexpanddiv ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = #pto<div_precision high_precision>}
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile（被除数） |
| `src1` | `pto.tile_buf` | 每行标量载体（除数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 除法精度模式。默认值为 `#pto<div_precision default>`。
  - `#pto<div_precision default>` — 标准精度除法
  - `#pto<div_precision high_precision>` — 高精度除法，需要浮点元素类型和额外的 `tmp` 操作数

**约束：**

- **目标约束（A2A3/A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - A2A3 支持 `f16`、`f32`；A5 的默认精度模式还支持 `i8`、`i16`、`i32`
  - `dst` 使用 `blayout=row_major`
  - 高精度模式用于 `f16`、`f32`，且必须提供 `tmp`
  - A2A3 显式提供 `tmp` 时，广播源采用列主序、每行一个标量；缓冲区类型和容量要求与 `pto.trowexpandsub` 相同

**示例：**

```mlir
// 默认精度
pto.trowexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)

// 高精度
pto.trowexpanddiv ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  {precisionType = #pto<div_precision high_precision>}
```

---

### `pto.tcolexpandmax` — 列广播取最大值

```mlir
pto.tcolexpandmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[0, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmin` — 列广播取最小值

```mlir
pto.tcolexpandmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[0, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmul` — 列广播乘法

```mlir
pto.tcolexpandmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandadd` — 列广播加法

```mlir
pto.tcolexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpanddiv` — 列广播除法

```mlir
pto.tcolexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = <precision>}
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile（被除数） |
| `src1` | `pto.tile_buf` | 每列标量载体（除数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 除法精度模式。默认值为 `#pto<div_precision default>`。
  - `#pto<div_precision default>` — 标准精度除法
  - `#pto<div_precision high_precision>` — 高精度除法，仅当元素类型为 `f16` 或 `f32` 时合法

**约束：**

- **实现检查（A2A3/A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - A2A3 支持 `f16`、`f32`；A5 的默认精度模式还支持 `i8`、`i16`、`i32`；高精度模式用于 `f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandexpdif` — 列广播指数差

```mlir
pto.tcolexpandexpdif ins(<src0>, <src1> : <src0_type>, <src1_type>)
                     outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[0, j])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体（指数差中的减数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                         v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.tcolexpandsub` — 列广播减法

```mlir
pto.tcolexpandsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[0, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体（被减数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例：**

```mlir
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandadd` — 行广播加法

```mlir
pto.trowexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, 0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - `src1` 为 row_major 时：`src1 valid_shape[1] == 32 / sizeof(dtype)`；否则：`src1 valid_shape[1] == 1`

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - `src1` 为 row_major 时：`src1 valid_shape[1] == 32 / sizeof(dtype)`；否则：`src1 valid_shape[1] == 1`

**示例：**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandexpdif` — 行广播指数差

```mlir
pto.trowexpandexpdif ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                     outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[i, 0])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（指数差中的减数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 必须与目标元素类型相同；A5 接受该参数，不要求它与源具有相同的 shape 或 valid shape

广播时，一个输入的有效区域与目标相同并使用行主序，另一个输入与目标有效行数相同。
广播输入使用行主序时有效列数为 `32 / sizeof(dtype)`，使用列主序时有效列数为 1。
A2A3 的显式 `tmp` 形式使用列主序广播输入。目标有效区域为 `[0, 0]` 时不写入数据。

**示例：**

```mlir
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                         v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.trowexpandmax` — 行广播取最大值

```mlir
pto.trowexpandmax ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, 0])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 位于 `vec`，与目标元素类型相同；使用列主序的标量广播源，容量要求与 `pto.trowexpandsub` 相同

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 必须与目标元素类型相同；A5 接受该参数，不要求它与源具有相同的 shape 或 valid shape

广播时，一个输入的有效区域与目标相同并使用行主序，另一个输入与目标有效行数相同。
广播输入使用行主序时有效列数为 `32 / sizeof(dtype)`，使用列主序时有效列数为 1。
A2A3 的显式 `tmp` 形式使用列主序广播输入。目标有效区域为 `[0, 0]` 时不写入数据。

**示例：**

```mlir
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmin` — 行广播取最小值

```mlir
pto.trowexpandmin ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, 0])
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 位于 `vec`，与目标元素类型相同；使用列主序的标量广播源，容量要求与 `pto.trowexpandsub` 相同

- **实现检查（A5）**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 必须与目标元素类型相同；A5 接受该参数，不要求它与源具有相同的 shape 或 valid shape

广播时，一个输入的有效区域与目标相同并使用行主序，另一个输入与目标有效行数相同。
广播输入使用行主序时有效列数为 `32 / sizeof(dtype)`，使用列主序时有效列数为 1。
A2A3 的显式 `tmp` 形式使用列主序广播输入。目标有效区域为 `[0, 0]` 时不写入数据。

**示例：**

```mlir
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```
