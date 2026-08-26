// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <cstddef>

extern "C" __global__ [aicore] void vmula_shared_acc_kernel(
    __gm__ float *input, __gm__ float *output0, __gm__ float *output1, __gm__ float *output2);

void LaunchVmulaSharedAcc(float *input, float *output, void *stream) {
    constexpr std::size_t kElementCount = 1024;
    vmula_shared_acc_kernel<<<1, nullptr, stream>>>(
        (__gm__ float *)input, (__gm__ float *)output, (__gm__ float *)(output + kElementCount),
        (__gm__ float *)(output + 2 * kElementCount));
}
