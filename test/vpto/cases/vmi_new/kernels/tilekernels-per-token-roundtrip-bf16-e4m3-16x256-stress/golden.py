#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

import numpy as np


ROWS = 16
COLS = 256


def bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) >> np.uint32(16)).astype(np.uint16)


def main() -> None:
    output_dir = Path(".")
    group = np.array(
        [
            -1.75, -1.5, -1.25, -1.0, -0.875, -0.75, -0.625, -0.5,
            -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5,
            0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, -1.75,
            1.5, -1.25, 1.0, -0.75, 0.5, -0.25, 0.125, -0.125,
        ],
        dtype=np.float32,
    )
    row = np.tile(group, COLS // group.size)
    values = np.stack(
        [row if index % 2 == 0 else row[::-1] for index in range(ROWS)]
    )
    source = bf16_bits(values)
    sentinel = np.full((ROWS, COLS), 0xA5A5, dtype=np.uint16)

    source.tofile(output_dir / "v1.bin")
    sentinel.tofile(output_dir / "v2.bin")
    source.tofile(output_dir / "golden_v2.bin")


if __name__ == "__main__":
   main()
