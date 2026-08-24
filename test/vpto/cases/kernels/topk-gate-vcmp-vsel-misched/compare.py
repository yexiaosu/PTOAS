#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
from pathlib import Path

import numpy as np


def compare_exact(golden_path: Path, output_path: Path, dtype: np.dtype) -> bool:
    if not golden_path.is_file() or not output_path.is_file():
        print(f"[ERROR] missing {golden_path} or {output_path}")
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(f"[ERROR] shape mismatch for {output_path}: {golden.shape} vs {output.shape}")
        return False
    if np.array_equal(golden, output):
        return True
    mismatches = np.flatnonzero(golden != output)
    first = int(mismatches[0])
    print(
        f"[ERROR] mismatch in {output_path} at {first}: "
        f"golden={golden[first]} output={output[first]}"
    )
    return False


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    checks = (
        compare_exact(Path("golden_winner_indices.bin"), Path("winner_indices.bin"), np.int32),
        compare_exact(Path("golden_masked_scores.bin"), Path("masked_scores.bin"), np.float32),
    )
    if all(checks):
        print("[INFO] strict compare passed: winner indices and final masked scores")
        return
    if strict:
        sys.exit(2)


if __name__ == "__main__":
    main()
