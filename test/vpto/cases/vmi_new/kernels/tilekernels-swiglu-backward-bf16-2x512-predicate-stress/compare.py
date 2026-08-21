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
    golden = np.fromfile("golden_v3.bin", dtype=np.uint16)
    output = np.fromfile("v3.bin", dtype=np.uint16)
    if golden.shape != output.shape or not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0] if golden.shape == output.shape else []
        index = int(diff[0]) if len(diff) else -1
        print(f"[ERROR] strict BF16 compare failed idx={index} "
              f"golden={int(golden[index]) if index >= 0 else 'n/a'} "
              f"output={int(output[index]) if index >= 0 else 'n/a'}")
        sys.exit(2)
    print("[INFO] strict BF16 compare passed")
if __name__ == "__main__":
   main()
