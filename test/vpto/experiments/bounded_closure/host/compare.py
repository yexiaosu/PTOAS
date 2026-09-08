#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import numpy as np


def main() -> None:
    golden = np.fromfile("golden.bin", dtype=np.int32)
    output = np.fromfile("output.bin", dtype=np.int32)
    if golden.shape != (36,) or output.shape != golden.shape:
        print(f"[ERROR] shape mismatch: golden={golden.shape}, output={output.shape}")
        sys.exit(2)
    mismatch = np.flatnonzero(golden != output)
    if mismatch.size != 0:
        index = int(mismatch[0])
        print(
            f"[ERROR] mismatch at token={index // 9}, rank={index % 9}: "
            f"golden={int(golden[index])}, output={int(output[index])}"
        )
        sys.exit(2)
    print("[INFO] exact int32 compare passed")


if __name__ == "__main__":
    main()
