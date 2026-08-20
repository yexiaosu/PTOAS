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
N = 2
HIDDEN = 512
def bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) >> np.uint32(16)).astype(np.uint16)
def main() -> None:
    lane = np.arange(N * HIDDEN, dtype=np.int32)
    x = np.zeros((N, HIDDEN), dtype=np.float32)
    y_values = np.array([-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0,
                         2.0, 3.0, 4.0], dtype=np.float32)
    g_values = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    y = y_values[lane % y_values.size].reshape(N, HIDDEN)
    g = g_values[lane % g_values.size].reshape(N, HIDDEN)
    packed_x = np.concatenate([x, y], axis=1)
    x_grad = g * np.float32(0.5) * np.clip(y, -3.0, 3.0)
    y_grad = np.zeros_like(x_grad)
    expected = np.concatenate([x_grad, y_grad], axis=1)
    output_dir = Path(".")
    bf16_bits(packed_x).tofile(output_dir / "v1.bin")
    bf16_bits(g).tofile(output_dir / "v2.bin")
    np.full((N, HIDDEN * 2), 0xA5A5, dtype=np.uint16).tofile(output_dir / "v3.bin")
    bf16_bits(expected).tofile(output_dir / "golden_v3.bin")
if __name__ == "__main__":
   main()
