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
def main():
    for n,dtype in [(6,np.float32),(7,np.uint16),(8,np.float32),(9,np.uint16)]:
        g=np.fromfile(f"golden_v{n}.bin",dtype=dtype); o=np.fromfile(f"v{n}.bin",dtype=dtype)
        if g.shape!=o.shape or not np.array_equal(g,o):
            diff=np.nonzero(g!=o)[0] if g.shape==o.shape else []; i=int(diff[0]) if len(diff) else -1
            print(f"[ERROR] strict compare v{n} failed idx={i}"); sys.exit(2)
    print("[INFO] strict MHC outputs compare passed")
if __name__=="__main__": main()
