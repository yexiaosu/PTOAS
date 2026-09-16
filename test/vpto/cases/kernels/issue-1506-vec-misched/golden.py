# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic reconstructed workload for the unmodified issue #1506 IR."""

import numpy as np


def bf16_bits(values):
    """Round float32 to nearest-even bfloat16 and return storage bits."""
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((words + np.uint32(0x7FFF) + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def decode(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def main():
    rng = np.random.default_rng(1506)
    shape = (2, 196, 12, 64)
    tensors = [bf16_bits(rng.normal(0, 0.25, shape)) for _ in range(3)]
    k, q, v = tensors
    rh = bf16_bits(rng.normal(0, 0.125, (2, 12, 196, 16)))
    rw = bf16_bits(rng.normal(0, 0.125, (2, 12, 196, 16)))
    initial = np.full(shape, 0x7FC1, dtype=np.uint16)
    for index, values in enumerate((k, initial, q, rh, rw, v)):
        values.tofile(f"input_{index}.bin")
    kh, qh, vh = [decode(x).transpose(0, 2, 1, 3) for x in (k, q, v)]
    logits = (qh @ kh.swapaxes(-1, -2)) * np.float32(0.125)
    positions = np.arange(196)
    logits += decode(rh)[..., positions // 14]
    logits += decode(rw)[..., positions % 14]
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected = (weights @ vh) / weights.sum(axis=-1, keepdims=True)
    bf16_bits(expected.transpose(0, 2, 1, 3)).tofile("golden.bin")
    print("golden: seed=1506, B=2 N=196 H=12 D=64, bias stride=16")


if __name__ == "__main__":
    main()
