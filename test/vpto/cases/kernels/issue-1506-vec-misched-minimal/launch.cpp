// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>
#include "acl/acl.h"

extern "C" __global__ [aicore] void _attn_rel_h_rel_w_kernel(
    __gm__ bfloat16_t *k, __gm__ bfloat16_t *out, __gm__ bfloat16_t *q,
    __gm__ bfloat16_t *rh, __gm__ bfloat16_t *rw, __gm__ bfloat16_t *v,
    int32_t batch, int32_t kb, int32_t kh, int32_t kn,
    int32_t qb, int32_t qh, int32_t qn,
    int32_t rhb, int32_t rhh, int32_t rhn,
    int32_t rwb, int32_t rwh, int32_t rwn,
    int32_t vb, int32_t vh, int32_t vn);

void LaunchIssue1506(void *k, void *out, void *q, void *rh, void *rw,
                     void *v, void *stream) {
  // Q/K/V: [2,196,12,64]. Relative bias: [2,12,196,16], 14 useful lanes.
  // Minimal reproduction: one AIC block and its two AIV subblocks.
  _attn_rel_h_rel_w_kernel<<<1, nullptr, stream>>>(
      (__gm__ bfloat16_t *)k, (__gm__ bfloat16_t *)out,
      (__gm__ bfloat16_t *)q, (__gm__ bfloat16_t *)rh,
      (__gm__ bfloat16_t *)rw, (__gm__ bfloat16_t *)v,
      2, 150528, 64, 768, 150528, 64, 768,
      37632, 3136, 16, 37632, 3136, 16, 150528, 64, 768);
}
