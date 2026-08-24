#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np


EXPERTS = 384
TOP_K = 6
SEED = 508
EXPECTED_WINNERS = np.array([0, 383, 63, 64, 192, 256], dtype=np.int32)


def topk_with_index_tie_break(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    working = scores.copy()
    winners = np.empty((TOP_K,), dtype=np.int32)
    for rank in range(TOP_K):
        maximum = np.max(working)
        candidates = np.flatnonzero(working == maximum)
        if candidates.size == 0:
            raise RuntimeError(f"no TopK candidate at rank {rank}")
        winner = int(candidates[0])
        winners[rank] = winner
        working[winner] = np.float32(-np.inf)
    return winners, working


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = (rng.integers(-800, 801, size=EXPERTS).astype(np.float32) / 8.0).astype(np.float32)

    # Boundary indices and two equal-score pairs exercise stable, smallest-index tie breaking.
    scores[0] = np.float32(1000.0)
    scores[383] = np.float32(1000.0)
    scores[63] = np.float32(900.0)
    scores[64] = np.float32(900.0)
    scores[192] = np.float32(800.0)
    scores[256] = np.float32(700.0)

    indices = np.arange(EXPERTS, dtype=np.int32)
    winners, masked_scores = topk_with_index_tie_break(scores)
    if not np.array_equal(winners, EXPECTED_WINNERS):
        raise RuntimeError(f"unexpected generated winners: {winners.tolist()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.tofile(output_dir / "scores.bin")
    indices.tofile(output_dir / "indices.bin")
    np.zeros((TOP_K,), dtype=np.int32).tofile(output_dir / "winner_indices.bin")
    np.zeros((EXPERTS,), dtype=np.float32).tofile(output_dir / "masked_scores.bin")
    winners.tofile(output_dir / "golden_winner_indices.bin")
    masked_scores.tofile(output_dir / "golden_masked_scores.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic E384/K6 TopK inputs and strict golden outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
