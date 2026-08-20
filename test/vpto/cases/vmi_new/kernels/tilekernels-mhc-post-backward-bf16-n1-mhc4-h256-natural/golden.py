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
def bf16_bits(v):
    bits=v.astype(np.float32).view(np.uint32); bias=np.uint32(0x7FFF)+((bits>>np.uint32(16))&np.uint32(1))
    return ((bits+bias)>>np.uint32(16)).astype(np.uint16)
def main():
    lane=np.arange(256,dtype=np.int32)
    vals=np.array([-1.0,-0.5,0.5,1.0],dtype=np.float32)
    d=np.stack([np.roll(vals[lane%4],m) for m in range(4)])
    comb = np.array(
        [
            [0.5, -0.25, 1.0, 0.75],
            [-0.5, 1.0, 0.25, -0.75],
            [1.0, 0.5, -0.5, 0.25],
            [0.25, -1.0, 0.75, 0.5],
        ],
        dtype=np.float32,
    )
    post=np.zeros(8,dtype=np.float32); post[:4]=np.array([0.5,-0.25,1.0,0.75],dtype=np.float32)
    zero_res=np.zeros((4,256),dtype=np.float32); zero_x=np.zeros(256,dtype=np.float32)
    dx=np.einsum("mh,m->h",d,post[:4],optimize=False).astype(np.float32)
    dres=np.einsum("mh,im->ih",d,comb,optimize=False).astype(np.float32)
    p=Path("."); bf16_bits(d).tofile(p/"v1.bin"); comb.tofile(p/"v2.bin"); bf16_bits(zero_res).tofile(p/"v3.bin")
    post.tofile(p/"v4.bin"); bf16_bits(zero_x).tofile(p/"v5.bin")
    np.full(16,np.float32(np.nan),dtype=np.float32).tofile(p/"v6.bin")
    np.full((4,256),0xA5A5,dtype=np.uint16).tofile(p/"v7.bin")
    np.full(8,np.float32(np.nan),dtype=np.float32).tofile(p/"v8.bin")
    np.full(256,0xA5A5,dtype=np.uint16).tofile(p/"v9.bin")
    np.zeros(16,dtype=np.float32).tofile(p/"golden_v6.bin"); bf16_bits(dres).tofile(p/"golden_v7.bin")
    np.zeros(4, dtype=np.float32).tofile(p / "golden_v8.bin")
    bf16_bits(dx).tofile(p / "golden_v9.bin")
if __name__=="__main__": main()
