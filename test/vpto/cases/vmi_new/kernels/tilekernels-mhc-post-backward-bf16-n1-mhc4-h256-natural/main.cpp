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
      return 1;                                                                \
    }                                                                          \
  } while (0)

void LaunchVmi_tilekernels_mhc_post_backward_bf16_n1_mhc4_h256_natural_kernel(
    uint16_t *, float *, uint16_t *, float *, uint16_t *, float *, uint16_t *,
    float *, uint16_t *, void *);

int main() {
  constexpr size_t kDOutBytes = 2048;
  constexpr size_t kCombBytes = 64;
  constexpr size_t kResidualBytes = 2048;
  constexpr size_t kPostBytes = 32;
  constexpr size_t kPostOutputBytes = 16;
  constexpr size_t kXBytes = 512;
  uint16_t *dOutHost = nullptr;
  uint16_t *residualHost = nullptr;
  uint16_t *xHost = nullptr;
  uint16_t *dResidualHost = nullptr;
  uint16_t *dXHost = nullptr;
  float *combHost = nullptr;
  float *postHost = nullptr;
  float *dCombHost = nullptr;
  float *dPostHost = nullptr;
  uint16_t *dOutDevice = nullptr;
  uint16_t *residualDevice = nullptr;
  uint16_t *xDevice = nullptr;
  uint16_t *dResidualDevice = nullptr;
  uint16_t *dXDevice = nullptr;
  float *combDevice = nullptr;
  float *postDevice = nullptr;
  float *dCombDevice = nullptr;
  float *dPostDevice = nullptr;
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

  ACL_CHECK(aclrtMallocHost((void **)(&dOutHost), kDOutBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&combHost), kCombBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&residualHost), kResidualBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&postHost), kPostBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&xHost), kXBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dCombHost), kCombBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dResidualHost), kResidualBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dPostHost), kPostBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dXHost), kXBytes));

  ACL_CHECK(aclrtMalloc((void **)&dOutDevice, kDOutBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&combDevice, kCombBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&residualDevice, kResidualBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&postDevice, kPostBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&xDevice, kXBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dCombDevice, kCombBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dResidualDevice, kResidualBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dPostDevice, kPostBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dXDevice, kXBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  size_t dOutFileBytes = kDOutBytes;
  size_t combFileBytes = kCombBytes;
  size_t residualFileBytes = kResidualBytes;
  size_t postFileBytes = kPostBytes;
  size_t xFileBytes = kXBytes;
  size_t dCombFileBytes = kCombBytes;
  size_t dResidualFileBytes = kResidualBytes;
  size_t dPostFileBytes = kPostBytes;
  size_t dXFileBytes = kXBytes;
  ReadFile("./v1.bin", dOutFileBytes, dOutHost, kDOutBytes);
  ReadFile("./v2.bin", combFileBytes, combHost, kCombBytes);
  ReadFile("./v3.bin", residualFileBytes, residualHost, kResidualBytes);
  ReadFile("./v4.bin", postFileBytes, postHost, kPostBytes);
  ReadFile("./v5.bin", xFileBytes, xHost, kXBytes);
  ReadFile("./v6.bin", dCombFileBytes, dCombHost, kCombBytes);
  ReadFile("./v7.bin", dResidualFileBytes, dResidualHost, kResidualBytes);
  ReadFile("./v8.bin", dPostFileBytes, dPostHost, kPostBytes);
  ReadFile("./v9.bin", dXFileBytes, dXHost, kXBytes);

  ACL_CHECK(aclrtMemcpy(dOutDevice, kDOutBytes, dOutHost, kDOutBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(combDevice, kCombBytes, combHost, kCombBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(residualDevice, kResidualBytes, residualHost,
                       kResidualBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(postDevice, kPostBytes, postHost, kPostBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(xDevice, kXBytes, xHost, kXBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dCombDevice, kCombBytes, dCombHost, kCombBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dResidualDevice, kResidualBytes, dResidualHost,
                       kResidualBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dPostDevice, kPostBytes, dPostHost, kPostBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dXDevice, kXBytes, dXHost, kXBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchVmi_tilekernels_mhc_post_backward_bf16_n1_mhc4_h256_natural_kernel(
      dOutDevice, combDevice, residualDevice, postDevice, xDevice, dCombDevice,
      dResidualDevice, dPostDevice, dXDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dCombHost, kCombBytes, dCombDevice, kCombBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(dResidualHost, kResidualBytes, dResidualDevice,
                       kResidualBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(dPostHost, kPostBytes, dPostDevice, kPostBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(dXHost, kXBytes, dXDevice, kXBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v6.bin", dCombHost, kCombBytes);
  WriteFile("./v7.bin", dResidualHost, kResidualBytes);
  WriteFile("./v8.bin", dPostHost, kPostOutputBytes);
  WriteFile("./v9.bin", dXHost, kXBytes);

  aclrtFree(dOutDevice);
  aclrtFree(combDevice);
  aclrtFree(residualDevice);
  aclrtFree(postDevice);
  aclrtFree(xDevice);
  aclrtFree(dCombDevice);
  aclrtFree(dResidualDevice);
  aclrtFree(dPostDevice);
  aclrtFree(dXDevice);
  aclrtFreeHost(dOutHost);
  aclrtFreeHost(combHost);
  aclrtFreeHost(residualHost);
  aclrtFreeHost(postHost);
  aclrtFreeHost(xHost);
  aclrtFreeHost(dCombHost);
  aclrtFreeHost(dResidualHost);
  aclrtFreeHost(dPostHost);
  aclrtFreeHost(dXHost);
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
