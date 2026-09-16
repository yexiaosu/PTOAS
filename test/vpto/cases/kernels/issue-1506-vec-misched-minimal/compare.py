# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Require every output to satisfy the declared BF16 attention tolerance."""

import json
from pathlib import Path

import numpy as np

from golden import decode


def main():
    actual_bits = np.fromfile("output.bin", dtype=np.uint16)
    actual = decode(actual_bits)
    expected = decode(np.fromfile("golden.bin", dtype=np.uint16))
    if actual.size != 301056 or actual.shape != expected.shape:
        raise ValueError(f"unexpected output sizes: {actual.shape}, {expected.shape}")
    active = np.zeros((2, 196, 12, 64), dtype=bool)
    active[0, :128, 0, :] = True
    active = active.reshape(-1)
    untouched = bool(np.all(actual_bits[~active] == 0x7FC1))
    actual, expected = actual[active], expected[active]
    difference = np.abs(actual - expected)
    # Allows accumulated probability BF16 rounding and final BF16 conversion.
    valid = np.isfinite(actual) & (difference <= 0.001 + 0.02 * np.abs(expected))
    result = {
        "elements": int(actual.size), "failed": int(np.count_nonzero(~valid)),
        "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean()),
        "rtol": 0.02, "atol": 0.001, "pass": bool(valid.all()) and untouched,
        "untouched_sentinel_pass": untouched, "untouched_elements": int(np.count_nonzero(~active)),
    }
    Path("compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not result["pass"]:
        raise RuntimeError("strict attention comparison failed")


if __name__ == "__main__":
    main()
