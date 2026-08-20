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
#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <cstdint>
#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif
extern "C" __global__ [aicore] void vmi_tilekernels_mhc_post_backward_bf16_n1_mhc4_h256_natural_kernel(
  __gm__ bfloat16_t*, __gm__ float*, __gm__ bfloat16_t*, __gm__ float*, __gm__ bfloat16_t*,
  __gm__ float*, __gm__ bfloat16_t*, __gm__ float*, __gm__ bfloat16_t*);
void LaunchVmi_tilekernels_mhc_post_backward_bf16_n1_mhc4_h256_natural_kernel(
  uint16_t *d, float *comb, uint16_t *res, float *post, uint16_t *x,
  float *dcomb, uint16_t *dres, float *dpost, uint16_t *dx, void *stream) {
  vmi_tilekernels_mhc_post_backward_bf16_n1_mhc4_h256_natural_kernel<<<1, nullptr, stream>>>(
    (__gm__ bfloat16_t*)d, (__gm__ float*)comb, (__gm__ bfloat16_t*)res, (__gm__ float*)post,
    (__gm__ bfloat16_t*)x, (__gm__ float*)dcomb, (__gm__ bfloat16_t*)dres,
    (__gm__ float*)dpost, (__gm__ bfloat16_t*)dx);
}
