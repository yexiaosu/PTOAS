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


ELEMENT_COUNT = 1024
OUTPUT_COUNT = 3
SEED = 1327


def main() -> None:
    rng = np.random.default_rng(SEED)
    input_values = rng.uniform(-4.0, 4.0, size=ELEMENT_COUNT).astype(np.float32)
    expected = (input_values + np.abs(input_values) * np.abs(input_values)).astype(np.float32)
    output_values = np.zeros(ELEMENT_COUNT * OUTPUT_COUNT, dtype=np.float32)
    golden_values = np.tile(expected, OUTPUT_COUNT)

    output_dir = Path.cwd()
    input_values.tofile(output_dir / "input.bin")
    output_values.tofile(output_dir / "output.bin")
    golden_values.tofile(output_dir / "golden_output.bin")


if __name__ == "__main__":
    main()
