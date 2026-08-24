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


N = 4
E = 384
K = 9


def main() -> None:
    scores = np.full((N, E), -1.0e30, dtype=np.float32)
    experts = np.arange(E, dtype=np.float32)
    scores[0] = E - experts
    scores[1] = np.mod(experts, 4)
    scores[2] = experts
    scores[2, 0] = 100.0
    scores[3] = 1.0

    working = scores.copy()
    golden = np.empty((N, K), dtype=np.int32)
    for token in range(N):
        for rank in range(K):
            winner = int(np.argmax(working[token]))
            golden[token, rank] = winner
            working[token, winner] = -1.0e30

    output_dir = Path(".")
    scores.tofile(output_dir / "scores.bin")
    np.full((N, K), -1, dtype=np.int32).tofile(output_dir / "output.bin")
    golden.tofile(output_dir / "golden.bin")


if __name__ == "__main__":
    main()
