// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
using namespace PtoTestCommon;
#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError _ret = (expr);                                              \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   (int)_ret, __FILE__, __LINE__);                             \
      rc = 1;                                                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)
void LaunchVmi_tilekernels_swiglu_backward_bf16_2x512_stress_kernel(
    uint16_t *x, uint16_t *g, uint16_t *out, void *stream);
int main() {
  constexpr size_t kXBytes = 2 * 1024 * sizeof(uint16_t);
  constexpr size_t kGBytes = 2 * 512 * sizeof(uint16_t);
  uint16_t *xHost = nullptr;
  uint16_t *gHost = nullptr;
  uint16_t *outHost = nullptr;
  uint16_t *xDevice = nullptr;
  uint16_t *gDevice = nullptr;
  uint16_t *outDevice = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
    char *end = nullptr;
    const long parsedDevice = std::strtol(envDevice, &end, 10);
    if (*envDevice == '\0' || *end != '\0' || parsedDevice < 0 ||
        parsedDevice > INT_MAX) {
      std::fprintf(stderr, "[ERROR] invalid ACL_DEVICE_ID: %s\n", envDevice);
      return 1;
    }
    deviceId = static_cast<int>(parsedDevice);
  }
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&xHost), kXBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&gHost), kGBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&outHost), kXBytes));
  ACL_CHECK(aclrtMalloc((void **)&xDevice, kXBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&gDevice, kGBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, kXBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin", kXBytes, xHost, kXBytes);
  ReadFile("./v2.bin", kGBytes, gHost, kGBytes);
  ReadFile("./v3.bin", kXBytes, outHost, kXBytes);
  ACL_CHECK(aclrtMemcpy(xDevice, kXBytes, xHost, kXBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(gDevice, kGBytes, gHost, kGBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, kXBytes, outHost, kXBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_tilekernels_swiglu_backward_bf16_2x512_stress_kernel(xDevice, gDevice, outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outHost, kXBytes, outDevice, kXBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", outHost, kXBytes);
  aclrtFree(xDevice);
  aclrtFree(gDevice);
  aclrtFree(outDevice);
  aclrtFreeHost(xHost);
  aclrtFreeHost(gHost);
  aclrtFreeHost(outHost);
  if (stream) {
    aclrtDestroyStream(stream);
  }
  if (deviceSet) {
    aclrtResetDevice(deviceId);
  }
  if (aclInited) {
    aclFinalize();
  }
  return rc;
}
