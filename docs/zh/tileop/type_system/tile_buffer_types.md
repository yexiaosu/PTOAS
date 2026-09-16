# Tile Buffer 类型

## 概述

`!pto.tile_buf<...>` 是当前 `ptoas` 中最核心的局部存储类型。它直接把局部 tile 计算所需的关键元信息编码进类型本身。

## 语法

```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0, compact=1>
```

也支持紧凑语法；物理尺寸和元素类型写在位置之后，其余字段按需指定：

```mlir
!pto.tile_buf<vec, 1x64xi32>
!pto.tile_buf<vec, 16x32xf16, valid=8x24, compact=1>
!pto.tile_buf<vec, 16x32xf16, valid=?x?, blayout=row_major, slayout=none_box, fractal=512, pad=1>
```

紧凑语法中的 `valid=行数x列数` 对应 `v_row` / `v_col`，省略时等于物理尺寸。
省略布局配置时使用 `blayout=row_major`、`slayout=none_box`、`fractal=512`、`pad=0`、
`compact=0`。显式 key-value 语法中的可选 `compact` 位于 `pad` 之后。

## 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `loc` | 关键字 | 局部位置，如 `vec`、`mat`、`left`、`right`、`acc`、`bias`、`scaling` |
| `dtype` | 元素类型 | tile 中元素的数据类型 |
| `rows` | `int64` | 物理行数 |
| `cols` | `int64` | 物理列数 |
| `v_row` | `int64` 或 `?` | 有效行数 |
| `v_col` | `int64` 或 `?` | 有效列数 |
| `blayout` | 布局助记符 | 基础布局：`row_major` 或 `col_major` |
| `slayout` | 布局助记符 | 次级布局：`none_box`、`row_major` 或 `col_major` |
| `fractal` | 整数 | 分形大小，单位为字节；支持 `32`、`512`、`1024` |
| `pad` | 整数枚举 | `0`（`null`）表示不指定填充，`1`（`zero`）表示零，`2`（`max`）表示最大值，`3`（`min`）表示最小值；类型语法中填写整数 |
| `compact` | 可选整数枚举 | `0`（`null`）关闭紧凑模式，`1`（`normal`）使用普通紧凑模式，`2`（`row_plus_one`）使用行数加一的紧凑模式；默认 `0` |

## 类型承载的信息

`tile_buf` 同时表达：

- tile 位于哪一类本地存储位置
- tile 的元素类型
- tile 的物理尺寸
- tile 的有效区域
- tile 的布局和 padding 语义
- tile 的紧凑存储模式

这使很多位置、布局和有效区域相关检查能够更早在类型层面完成。

## 常见构造路径

- `pto.alloc_tile`
- `pto.bind_tile`
- `pto.materialize_tile`
- `pto.declare_tile`

其中最常见的是 `pto.alloc_tile`。

## 特殊说明

对于 `dtype=!pto.f4E1M2x2` 和 `dtype=!pto.f4E2M1x2`：

- `rows` / `cols` 描述的是物理打包 extent
- `v_row` / `v_col` 描述的也是物理有效 extent
- 它们不是逻辑标量 FP4 元素个数

## Constraints

- `loc`、`dtype`、布局和尺寸组合必须满足后端支持边界
- `v_row` / `v_col` 不应超过对应物理尺寸
- 具体操作还会进一步限制输入输出 `tile_buf` 的位置和布局组合

## Example

```mlir
%tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```
