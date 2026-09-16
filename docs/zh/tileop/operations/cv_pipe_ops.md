# 核内 CV Pipe 通信操作

本节描述 Cube（AIC）和 Vector（AIV）核心之间的 FIFO 数据交换接口。这些操作支持 MPMD（多程序多数据）执行模型，允许两端通过管道异步交换本地 tile。`aic_initialize_pipe`、`aiv_initialize_pipe` 通过编译时属性 `id` 与对应的数据操作关联；`initialize_l2g2l_pipe`、`initialize_l2l_pipe` 则返回用于后续操作的管线句柄。

这一类操作的通用特性包括：

- **dir_mask** 属性：控制 Pipe 通信方向，1 = C2V（Cube 到 Vector），2 = V2C（Vector 到 Cube），3 = 双向
- **id** 属性：编译时整数常量，绑定初始化操作与生产/消费/释放操作
- **slot_size** 属性：单个逻辑 Pipe 条目的大小，单位为字节
- **slot_num** 属性：可选，控制 FIFO 深度（默认 dir_mask=1/2 时为 8，dir_mask=3 时为 4）
- **split** 属性：编译时属性，取值为 `0`、`1`、`2`、`3`、`4`；奇数尺寸分割的适用条件见 `pto.tpush_to_aiv`
- **tpop 操作返回值**：`tpop_from_aic` 和 `tpop_from_aiv` 是有返回值的操作
- **Pipe 条目类型**：本地条目（`!pto.tile_buf`）

---

## 目录

- [`pto.aic_initialize_pipe` — Cube 侧 Pipe 初始化](#ptoaic_initialize_pipe--cube-侧-pipe-初始化)
- [`pto.aiv_initialize_pipe` — Vector 侧 Pipe 初始化](#ptoaiv_initialize_pipe--vector-侧-pipe-初始化)
- [`pto.tpush_to_aiv` — C2V 生产者推送](#ptotpush_to_aiv--c2v-生产者推送)
- [`pto.tpush_to_aic` — V2C 生产者推送](#ptotpush_to_aic--v2c-生产者推送)
- [`pto.tpop_from_aic` — C2V 消费者弹出](#ptotpop_from_aic--c2v-消费者弹出)
- [`pto.tpop_from_aiv` — V2C 消费者弹出](#ptotpop_from_aiv--v2c-消费者弹出)
- [`pto.tfree_from_aic` — C2V 消费者释放](#ptotfree_from_aic--c2v-消费者释放)
- [`pto.tfree_from_aiv` — V2C 消费者释放](#ptotfree_from_aiv--v2c-消费者释放)
- [`pto.reserve_buffer` — 预留本地消费者缓冲区](#ptoreserve_buffer--预留本地消费者缓冲区)
- [`pto.import_reserved_buffer` — 导入对端预留缓冲区](#ptoimport_reserved_buffer--导入对端预留缓冲区)
- [`pto.section.cube` — Cube 核代码区域](#ptosectioncube--cube-核代码区域)
- [`pto.section.vector` — Vector 核代码区域](#ptosectionvector--vector-核代码区域)
- [`pto.initialize_l2g2l_pipe` — 初始化 L2G2L 管线](#ptoinitialize_l2g2l_pipe--初始化-l2g2l-管线)
- [`pto.initialize_l2l_pipe` — 初始化 L2L 管线](#ptoinitialize_l2l_pipe--初始化-l2l-管线)

---

## 操作详解

### `pto.aic_initialize_pipe` — Cube 侧 Pipe 初始化

```mlir
pto.aic_initialize_pipe {[id = <id>,] dir_mask = <dir>, slot_size = <size>
                         [, slot_num = <num>] [, local_slot_num = <local_num>]
                         [, nosplit = <bool>]}
    ([gm_slot_buffer = <buf> : <ptr_type>]
     [, gm_slot_tensor = <tensor> : <tensor_view_type>]
     [, c2v_consumer_buf = <c2v> : i32]
     [, v2c_consumer_buf = <v2c> : i32])
```

方括号表示可选项；操作数列表只在实际条目之间写逗号。属性放在 `{}` 中，SSA 操作数及其类型放在随后的 `()` 中。
`slot_num` 可以直接写在前一组属性中，例如 `slot_num = 2`。

**语义：**

在 Cube 核中初始化 Pipe 通道，为双向或单向的本地 tile 数据交换建立共享的 FIFO 缓冲区。该操作不产生返回值，仅完成初始化配置。

**属性：**

- `id` — Pipe 标识符。非负 `i32` 常量，默认 `0`，用于绑定此初始化操作与后续的 `tpush`/`tpop`/`tfree` 操作。必须在同一内核中唯一。

- `dir_mask` — 通信方向掩码。决定 Pipe 的通信模式：
  - `1` — C2V（Cube 到 Vector），仅允许 Cube 端推送，Vector 端消费
  - `2` — V2C（Vector 到 Cube），仅允许 Vector 端推送，Cube 端消费
  - `3` — 双向（Bidirectional），两端均可推送与消费

- `slot_size` — 必需，Pipe 条目的大小（单位：字节）。必须大于 0。

- `slot_num` — 可选，FIFO 深度（FIFO 中的条目数）。默认值：`dir_mask` 为 1 或 2 时为 8，为 3 时为 4。必须大于 0。

- `local_slot_num` — 可选，仅限 A2/A3。本地 Tile 缓冲区的 FIFO 深度。必须大于 0 且不超过 `slot_num`。A5 上必须省略此属性。

- `nosplit` — 可选布尔属性，省略时不禁用分割。若为 `true`，禁用 Tile 分割，要求所有绑定的 `tpush`/`tpop`/`tfree` 操作的 `split` 属性为 0（`TILE_NO_SPLIT`）。

- `gm_slot_buffer` — 可选，类型为 `!pto.ptr<T>`。全局内存 FIFO 插槽缓冲区指针，用于本地 tile 的 GM 中转存储。

- `gm_slot_tensor` — 可选，类型为 `!pto.tensor_view<...>`。描述本地 tile FIFO 的 GM 中转存储，与消费者缓冲区组合使用。

- `c2v_consumer_buf` — 可选，类型为 `i32`。C2V 方向 Vector 消费者本地缓冲区的地址或导入地址值，单位为字节。

- `v2c_consumer_buf` — 可选，类型为 `i32`。V2C 方向 Cube 消费者本地缓冲区的地址或导入地址值，单位为字节。

**约束：**

- **实现检查（A2A3）**
  - 必须出现在 Cube 内核中（`pto.kernel_kind = #pto.kernel_kind<cube>`）。
  - `id` 必须在该内核中唯一。
  - `slot_num` 必须大于 0。
  - 若指定 `local_slot_num`，则必须大于 0 且 `<= slot_num`。
  - `dir_mask` 必须为 1、2 或 3。
  - `slot_size` 必须大于 0。
  - 若 `nosplit` 为 `true`，则所有使用此 `id` 的 `tpush`/`tpop`/`tfree` 操作的 `split` 属性必须为 0。

- **实现检查（A5）**
  - 必须出现在 Cube 内核中。
  - `local_slot_num` 必须省略（A5 不支持）。
  - 其他约束同 A2A3。

- **缓冲区组合**
  - `gm_slot_tensor` 与 `gm_slot_buffer` 互斥。
  - 本地 tile FIFO 必须提供方向对应的消费者缓冲区：C2V 提供 `c2v_consumer_buf`，V2C 提供 `v2c_consumer_buf`，双向同时提供两者。
  - `gm_slot_tensor` 与消费者缓冲区组合时，各目标都支持单向 C2V；V2C 和双向组合仅支持 A2A3。

**示例：**

```mlir
// A2/A3: 使用 GM 缓冲区和本地槽数配置的 C2V 初始化
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
    (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
     c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)

// A5: 无本地槽配置的 C2V 初始化
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
    (c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)
```

---

### `pto.aiv_initialize_pipe` — Vector 侧 Pipe 初始化

```mlir
pto.aiv_initialize_pipe {[id = <id>,] dir_mask = <dir>, slot_size = <size>
                         [, slot_num = <num>] [, local_slot_num = <local_num>]
                         [, nosplit = <bool>]}
    ([gm_slot_buffer = <buf> : <ptr_type>]
     [, gm_slot_tensor = <tensor> : <tensor_view_type>]
     [, c2v_consumer_buf = <c2v> : i32]
     [, v2c_consumer_buf = <v2c> : i32])
```

**语义：**

在 Vector 核中初始化 Pipe 通道，与 `pto.aic_initialize_pipe` 结构完全相同。该操作为 Vector 端配置 FIFO 管道，使其能够作为消费者（C2V 模式）或生产者（V2C 模式）参与数据交换。

**属性：**

属性定义与 `pto.aic_initialize_pipe` 完全相同。

**约束：**

- **实现检查（A2A3）**
  - 必须出现在 Vector 内核中（`pto.kernel_kind = #pto.kernel_kind<vector>`）。
  - 所有约束同 `pto.aic_initialize_pipe`。

- **实现检查（A5）**
  - 必须出现在 Vector 内核中。
  - `local_slot_num` 必须省略。

**示例：**

```mlir
// A2/A3: Vector 侧 C2V 消费者初始化
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
    (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
     c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)

// A5: Vector 侧初始化
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
    (c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)
```

---

### `pto.tpush_to_aiv` — C2V 生产者推送

```mlir
pto.tpush_to_aiv(<pipe_entry> : <pipe_entry_type>)
    {[id = <id>,] split = <split>}
```

**语义：**

从 Cube 核中推送一个本地 tile 条目到 C2V FIFO，执行 tile 传输。该操作不产生返回值。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `pipe_entry` | `!pto.tile_buf` | 要推送的本地 tile 条目 |

**返回值：** 无。操作提交条目至 FIFO 后返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Cube 内核中的 `pto.aic_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。编译时属性，决定条目如何分割：
  - `0` — `TILE_NO_SPLIT`，不分割
  - `1` — `TILE_UP_DOWN`，按行分割
  - `2` — `TILE_LEFT_RIGHT`，按列分割
  - `3` — 按奇数有效行数分割
  - `4` — 按奇数有效列数分割

**约束：**

- **实现检查（A2A3/A5）**
  - 位于 Cube 内核或 `pto.section.cube` 中；`id` 为非负整数，默认 `0`。
  - `id` 必须匹配一个使用 `dir_mask=1` 或 `dir_mask=3` 的 `pto.aic_initialize_pipe` 操作。
  - 输入类型不包括裸指针、`memref` 或 `partition_tensor_view`。本地 tile 的存储位置与传输方向匹配；Cube 累加结果通常使用 `loc=acc`。
  - 对完整二维 tile，`split=1/2` 对应的有效行数/列数必须为偶数；`split=3/4` 则要求为奇数。
  - `split=3/4` 仅用于经 GM 中转的本地 tile FIFO，初始化必须启用 C2V 并提供 `c2v_consumer_buf`。
  - 当初始化配置 `acc_push_epilogue` 时，源为 `loc=acc` 的 tile，元素类型符合所选转换模式，且 `split=0`。
  - 若初始化操作的 `nosplit` 为 `true`，则 `split` 必须为 0。

**示例：**

```mlir
// Tile buffer 条目推送（按行分割）
pto.tpush_to_aiv(%tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
    {id = 0, split = 1}
```

---

### `pto.tpush_to_aic` — V2C 生产者推送

```mlir
pto.tpush_to_aic(<pipe_entry> : <pipe_entry_type>)
    {[id = <id>,] split = <split>} [aiv_subblockid(<subblock_id>)]
```

**语义：**

从 Vector 核中推送一个 V2C Pipe 条目到 FIFO。结构与 `pto.tpush_to_aiv` 相同，但方向相反（Vector 生产，Cube 消费）。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `pipe_entry` | `!pto.tile_buf` | 要推送的本地 tile 条目 |

`aiv_subblockid` 是可选 `i64` 操作数，用于提供 Vector 子块编号；仅适用于 `loc=vec` 的本地条目且 `split != 0`。

**返回值：** 无。操作提交条目至 FIFO 后返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Vector 内核中的 `pto.aiv_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式（定义同 `pto.tpush_to_aiv`）。

**约束：**

- **实现检查（A2A3/A5）**
  - 位于 Vector 内核或 `pto.section.vector` 中；`id` 为非负整数，默认 `0`。
  - `id` 必须匹配一个使用 `dir_mask=2` 或 `dir_mask=3` 的 `pto.aiv_initialize_pipe` 操作。
  - `split` 取 `0`、`1`、`2`、`3`、`4`；`nosplit=true` 时为 `0`。
  - `split=3/4` 的 V2C 传输仅支持 A2A3，要求经 GM 中转的本地 tile FIFO、启用 V2C 且提供 `v2c_consumer_buf`；A5 V2C 不支持这两种模式。
  - 输入为 Vector 生产者条目；分割时它表示本子块提供的部分，不能套用 Cube 完整输入 tile 的尺寸奇偶约束。

**示例：**

```mlir
// Vector 侧 V2C 推送
pto.tpush_to_aic(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    {id = 0, split = 1}
```

---

### `pto.tpop_from_aic` — C2V 消费者弹出

```mlir
%entry = pto.tpop_from_aic {id = <id>, split = <split>}
    -> <pipe_entry_type>
```

**语义：**

从 FIFO 中弹出一个 C2V Pipe 条目在 Vector 核中消费。该操作返回一个本地 tile，供后续操作使用。

**返回值：** 一个本地 tile 条目，类型为 `!pto.tile_buf<...>`。

**属性：**

- `id` — Pipe 标识符。必须与同一 Vector 内核中的 `pto.aiv_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式（定义同 `pto.tpush_to_aiv`）。

**约束：**

- **实现检查（A2A3/A5）**
  - 必须出现在 Vector 内核中。
  - `id` 必须匹配一个使用 `dir_mask=1` 或 `dir_mask=3` 的 `pto.aiv_initialize_pipe` 操作。
  - 返回类型必须与初始化配置兼容。

**示例：**

```mlir
// Tile buffer 返回值
%tile = pto.tpop_from_aic {id = 0, split = 1}
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>
```

---

### `pto.tpop_from_aiv` — V2C 消费者弹出

```mlir
%entry = pto.tpop_from_aiv {id = <id>, split = <split>}
    -> <pipe_entry_type>
```

**语义：**

从 FIFO 中弹出一个 V2C Pipe 条目在 Cube 核中消费。结构与 `pto.tpop_from_aic` 相同，但方向相反（Cube 消费来自 Vector 的数据）。

**返回值：** 一个本地 tile 条目，类型为 `!pto.tile_buf<...>`。

**属性：**

- `id` — Pipe 标识符。必须与同一 Cube 内核中的 `pto.aic_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。

**约束：**

- **实现检查（A2A3/A5）**
  - 必须出现在 Cube 内核中。
  - `id` 必须匹配一个使用 `dir_mask=2` 或 `dir_mask=3` 的 `pto.aic_initialize_pipe` 操作。

**示例：**

```mlir
// Cube 侧 V2C 消费
%tile = pto.tpop_from_aiv {id = 0, split = 1}
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>
```

---

### `pto.tfree_from_aic` — C2V 消费者释放

```mlir
pto.tfree_from_aic {id = <id>, split = <split>}
```

**语义：**

在 Vector 核中释放当前 C2V FIFO 消费者槽位。该操作推进消费者指针，使下一个条目可用。

**参数：** 无操作数。

**返回值：** 无。操作执行释放并返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Vector 内核中的 `pto.aiv_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。

**约束：**

- **实现检查（A2A3/A5）**
  - 必须出现在 Vector 内核中。
  - `id` 必须匹配一个使用 `dir_mask=1` 或 `dir_mask=3` 的 `pto.aiv_initialize_pipe` 操作。
  - Tile buffer 释放使用无操作数形式。

**示例：**

```mlir
// Tile buffer 条目释放（无操作数）
pto.tfree_from_aic {id = 0, split = 1}
```

---

### `pto.tfree_from_aiv` — V2C 消费者释放

```mlir
pto.tfree_from_aiv {id = <id>, split = <split>}
```

**语义：**

释放当前 V2C FIFO 消费者槽位在 Cube 核中。结构与 `pto.tfree_from_aic` 相同，但方向相反。

**参数：** 无操作数。

**返回值：** 无。操作执行释放并返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Cube 内核中的 `pto.aic_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。

**约束：**

- **实现检查（A2A3/A5）**
  - 必须出现在 Cube 内核中。
  - `id` 必须匹配一个使用 `dir_mask=2` 或 `dir_mask=3` 的 `pto.aic_initialize_pipe` 操作。

**示例：**

```mlir
// Cube 侧 V2C 释放（无操作数）
pto.tfree_from_aiv {id = 0, split = 1}
```

---

### `pto.reserve_buffer` — 预留本地消费者缓冲区

```mlir
pto.reserve_buffer {name = <name>, size = <size>,
                    location = <location>, auto = <autoAlloc>
                    (, base = <base>)?} -> i32
```

**语义：**

```text
addr = reserve_local_buffer(name, size, location, autoAlloc, base?)
// 在当前函数中预留一块本地消费者槽位缓冲区，
// 返回其地址供 CV Pipe 初始化或对端导入使用
```

**参数：** 无操作数。

**返回值：** `i32` — 预留缓冲区的地址。

**属性：**

- `name` — 缓冲区名称（字符串），在函数内必须唯一。
- `size` — 缓冲区大小（`i32`），必须大于 0。
- `location` — 地址空间，必须为 `#pto.address_space<vec>` 或 `#pto.address_space<mat>`。
- `autoAlloc` — 是否自动分配（`bool`）。当为 `false` 时必须提供 `base`。
- `base` — 可选的固定基地址（`i32`），当 `autoAlloc = false` 时必须提供，且为非负整数。

**约束：**

- **实现检查（A2A3/A5）**
  - 必须嵌套在 `func.func` 内部。
  - `size` 必须大于 0。
  - `location` 必须为 `vec` 或 `mat`。
  - 当 `autoAlloc = false` 时必须提供 `base`。
  - `name` 在所属函数内必须唯一。

**示例：**

```mlir
%buf = pto.reserve_buffer
    {name = "c2v_slot_buffer", size = 131072,
     location = #pto.address_space<vec>,
     auto = false, base = 0} -> i32
```

---

### `pto.import_reserved_buffer` — 导入对端预留缓冲区

```mlir
pto.import_reserved_buffer {name = <name>, peer_func = <peer_func>} -> i32
```

**语义：**

```text
addr = import_peer_buffer(name, peer_func)
// 导入由对端函数通过 pto.reserve_buffer 预留的缓冲区地址，
// 用于两侧共享同一块 CV Pipe 相关本地缓冲区
```

**参数：** 无操作数。

**返回值：** `i32` — 对端预留缓冲区的地址。

**属性：**

- `name` — 缓冲区名称（字符串），必须与对端函数中的 `pto.reserve_buffer` 名称匹配。
- `peer_func` — 对端函数引用（`FlatSymbolRefAttr`），必须指向已存在的 `func.func`。

**约束：**

- **实现检查（A2A3/A5）**
  - 必须嵌套在 `func.func` 内部。
  - `peer_func` 引用的函数必须存在。
  - 对端函数中必须存在名称匹配的 `pto.reserve_buffer`。
  - `(name, peer_func)` 组合在所属函数内必须唯一。

**示例：**

```mlir
%import_buf = pto.import_reserved_buffer
    {name = "c2v_slot_buffer",
     peer_func = @vector_kernel} -> i32
```

---

### `pto.section.cube` — Cube 核代码区域

```mlir
pto.section.cube {
  <body>
}
```

**语义：**

```text
#if defined(MACRO_CUBE)
  <body>
#endif
// 降低为条件编译宏保护的代码区域，仅在 Cube 核上编译和执行
```

**参数：** 无。包含一个单块区域（body）。

**返回值：** 无。

**约束：**

- **实现检查（A2A3/A5）**
  - 区域为单块、无终止符。
  - body 中的操作仅在 Cube 核（AIC）上下文中有效。

**示例：**

```mlir
pto.section.cube {
  %tile = pto.alloc_tile addr = %c0_i64
      : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                      v_row=16, v_col=256, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>
  pto.tmatmul ins(%a, %b : ...) outs(%tile : ...)
}
```

---

### `pto.section.vector` — Vector 核代码区域

```mlir
pto.section.vector {
  <body>
}
```

**语义：**

```text
#if defined(MACRO_VECTOR)
  <body>
#endif
// 降低为条件编译宏保护的代码区域，仅在 Vector 核上编译和执行
```

**参数：** 无。包含一个单块区域（body）。

**返回值：** 无。

**约束：**

- **实现检查（A2A3/A5）**
  - 区域为单块、无终止符。
  - body 中的操作仅在 Vector 核（AIV）上下文中有效。

**示例：**

```mlir
pto.section.vector {
  pto.tload ins(%view : !pto.partition_tensor_view<16x16xf16>)
            outs(%dst : !pto.tile_buf<...>)
  pto.tadd ins(%a, %b : ...) outs(%c : ...)
}
```

---

### `pto.initialize_l2g2l_pipe` — 初始化 L2G2L 管线

```mlir
pto.initialize_l2g2l_pipe {dir_mask = <N>, slot_size = <S>, slot_num = <M>
                           (, local_slot_num = <L>)? (, flag_base = <F>)?
                           (, nosplit = <B>)?}
                          (<gm_addr> : <gm_type> (, <local_addr> : <l_type>)?
                           (, <peer_local_addr> : <p_type>)?)
                          -> !pto.pipe
```

**语义：**

```text
pipe = init_l2g2l_pipe(dir_mask, slot_size, slot_num, gm_addr, ...)
// 初始化一个 Local→Global→Local 管线句柄，用于通过 GM 中转的跨核数据传输
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `gm_addr` | 地址值或 GM 视图 | GM 中转存储的地址，可使用 `!pto.ptr`、GM `memref` 或相应地址表示 |
| `local_addr` | 可选，本地缓冲区地址 | 本地 DMA 缓冲区（优化路径） |
| `peer_local_addr` | 可选，对端本地缓冲区地址 | 对端核的本地缓冲区 |

**返回值：** `!pto.pipe` — 管线句柄。

**属性：**

- `dir_mask` — 方向掩码（`i8`）。指定 C2V/V2C 方向。
- `slot_size` — 每个 FIFO 槽位大小（`i32`，字节）。
- `slot_num` — FIFO 槽位数量（`i32`）。
- `local_slot_num` — 可选，本地槽位数量（`i32`）。
- `flag_base` — 可选，同步 flag 基地址（`i32`）。
- `nosplit` — 可选，禁用 split 模式（`bool`）。

**约束：**

- `dir_mask` 必须为 `1`（C2V）、`2`（V2C）或 `3`（双向）。
- `slot_size`、`slot_num` 均为正 `i32` 整数，前者以字节计。
- 若指定 `flag_base`，它必须非负；单向管线最多为 `14`，双向管线最多为 `12`。
  多条管线使用的同步资源范围不能重叠。
- 没有 `local_addr` 时，必须同时省略 `peer_local_addr` 和 `local_slot_num`。
- 有 `local_addr` 时，`local_slot_num` 若提供必须满足 `1 <= local_slot_num <= slot_num`。
- 有 `local_addr` 且 `dir_mask=3` 时必须提供 `peer_local_addr`；单向管线必须省略它。
- 地址操作数的 IR 类型不被限定为单一指针类型；调用方应使 GM、本地和对端地址分别指向对应存储空间，
  并保证底层缓冲区覆盖所配置的槽位。

**示例：**

```mlir
%pipe = pto.initialize_l2g2l_pipe
    {dir_mask = 1, slot_size = 16384, slot_num = 2}
    (%gm_ptr : !pto.ptr<f32>) -> !pto.pipe
```

---

### `pto.initialize_l2l_pipe` — 初始化 L2L 管线

```mlir
pto.initialize_l2l_pipe {dir_mask = <N>, slot_size = <S>, slot_num = <M>
                         (, flag_base = <F>)? (, nosplit = <B>)?}
                        (<local_addr> : <l_type>
                         (, <peer_local_addr> : <p_type>)?)
                        -> !pto.pipe
```

**语义：**

```text
pipe = init_l2l_pipe(dir_mask, slot_size, slot_num, local_addr, ...)
// 初始化一个 Local→Local 管线句柄，用于直接本地内存的跨核数据传输（无 GM 中转）
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `local_addr` | 本地缓冲区地址 | 本地 DMA 缓冲区 |
| `peer_local_addr` | 可选，对端本地缓冲区地址 | 对端核的本地缓冲区 |

**返回值：** `!pto.pipe` — 管线句柄。

**属性：**

- `dir_mask` — 方向掩码（`i8`）。
- `slot_size` — 每个 FIFO 槽位大小（`i32`，字节）。
- `slot_num` — FIFO 槽位数量（`i32`）。
- `flag_base` — 可选，同步 flag 基地址（`i32`）。
- `nosplit` — 可选，禁用 split 模式（`bool`）。

**约束：**

- `dir_mask` 必须为 `1`（C2V）、`2`（V2C）或 `3`（双向）。
- `slot_size`、`slot_num` 均为正 `i32` 整数，前者以字节计。
- 若指定 `flag_base`，它必须非负；单向管线最多为 `14`，双向管线最多为 `12`。
  多条管线使用的同步资源范围不能重叠。
- `local_addr` 必须提供；`dir_mask=3` 时还必须提供 `peer_local_addr`，单向管线必须省略对端地址。
- 地址操作数可采用整数地址值等表示，不限于携带地址空间的类型；调用方应保证它们指向正确的本地存储，
  且缓冲区覆盖所配置的槽位。

**示例：**

```mlir
%pipe = pto.initialize_l2l_pipe
    {dir_mask = 1, slot_size = 8192, slot_num = 2}
    (%local_buf : i64) -> !pto.pipe
```
