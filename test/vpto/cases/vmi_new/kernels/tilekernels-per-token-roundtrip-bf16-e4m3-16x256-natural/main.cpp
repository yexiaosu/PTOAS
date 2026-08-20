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

void
LaunchVmi_tilekernels_per_token_roundtrip_bf16_e4m3_16x256_natural_kernel(
    uint16_t *src, uint16_t *dst, void *stream);

int main() {
  constexpr size_t kRows = 16;
  constexpr size_t kCols = 256;
  constexpr size_t kBytes = kRows * kCols * sizeof(uint16_t);
  uint16_t *srcHost = nullptr;
  uint16_t *dstHost = nullptr;
  uint16_t *srcDevice = nullptr;
  uint16_t *dstDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), kBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dstHost), kBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  size_t srcFileBytes = kBytes;
  size_t dstFileBytes = kBytes;
  ReadFile("./v1.bin", srcFileBytes, srcHost, kBytes);
  ReadFile("./v2.bin", dstFileBytes, dstHost, kBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, kBytes, srcHost, kBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstDevice, kBytes, dstHost, kBytes,
                       ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_tilekernels_per_token_roundtrip_bf16_e4m3_16x256_natural_kernel(srcDevice, dstDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstHost, kBytes, dstDevice, kBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", dstHost, kBytes);

  aclrtFree(srcDevice);
  aclrtFree(dstDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(dstHost);
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
