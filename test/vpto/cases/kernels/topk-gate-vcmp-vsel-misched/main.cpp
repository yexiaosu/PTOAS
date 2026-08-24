// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <system_error>

using namespace PtoTestCommon;

void LaunchTopkGateVcmpVselMisched(float *scores, int32_t *indices,
                                   int32_t *winnerIndices,
                                   float *maskedScores, void *stream);

namespace {
constexpr size_t kExpertCount = 384;
constexpr size_t kWinnerCount = 6;
constexpr size_t kScoresBytes = kExpertCount * sizeof(float);
constexpr size_t kIndicesBytes = kExpertCount * sizeof(int32_t);
constexpr size_t kWinnerBytes = kWinnerCount * sizeof(int32_t);
constexpr size_t kWinnerTransferBytes = 32;
} // namespace

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError ret = (expr);                                               \
    if (ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   static_cast<int>(ret), __FILE__, __LINE__);                \
      const char *recent = aclGetRecentErrMsg();                               \
      if (recent != nullptr && recent[0] != '\0') {                            \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", recent);           \
      }                                                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define FILE_CHECK(expr, path)                                                 \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",    \
                   path, __FILE__, __LINE__);                                  \
      return false;                                                            \
    }                                                                          \
  } while (0)

int main() {
  float *scoresHost = nullptr;
  int32_t *indicesHost = nullptr;
  int32_t *winnersHost = nullptr;
  float *maskedScoresHost = nullptr;
  float *scoresDevice = nullptr;
  int32_t *indicesDevice = nullptr;
  int32_t *winnersDevice = nullptr;
  float *maskedScoresDevice = nullptr;
  void *scoresHostAllocation = nullptr;
  void *indicesHostAllocation = nullptr;
  void *winnersHostAllocation = nullptr;
  void *maskedScoresHostAllocation = nullptr;
  void *scoresDeviceAllocation = nullptr;
  void *indicesDeviceAllocation = nullptr;
  void *winnersDeviceAllocation = nullptr;
  void *maskedScoresDeviceAllocation = nullptr;
  aclrtStream stream = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  size_t fileSize = 0;

  const auto execute = [&]() -> bool {
    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
      const std::string_view deviceText(envDevice);
      const auto parseResult = std::from_chars(
          deviceText.data(), deviceText.data() + deviceText.size(), deviceId);
      if (parseResult.ec != std::errc{} ||
          parseResult.ptr != deviceText.data() + deviceText.size() ||
          deviceId < 0) {
        std::fprintf(stderr, "[ERROR] invalid ACL_DEVICE_ID: %s\n", envDevice);
        return false;
      }
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost(&scoresHostAllocation, kScoresBytes));
    ACL_CHECK(aclrtMallocHost(&indicesHostAllocation, kIndicesBytes));
    ACL_CHECK(aclrtMallocHost(&winnersHostAllocation, kWinnerTransferBytes));
    ACL_CHECK(aclrtMallocHost(&maskedScoresHostAllocation, kScoresBytes));
    ACL_CHECK(aclrtMalloc(&scoresDeviceAllocation, kScoresBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&indicesDeviceAllocation, kIndicesBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&winnersDeviceAllocation, kWinnerTransferBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&maskedScoresDeviceAllocation, kScoresBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));

    scoresHost = static_cast<float *>(scoresHostAllocation);
    indicesHost = static_cast<int32_t *>(indicesHostAllocation);
    winnersHost = static_cast<int32_t *>(winnersHostAllocation);
    maskedScoresHost = static_cast<float *>(maskedScoresHostAllocation);
    scoresDevice = static_cast<float *>(scoresDeviceAllocation);
    indicesDevice = static_cast<int32_t *>(indicesDeviceAllocation);
    winnersDevice = static_cast<int32_t *>(winnersDeviceAllocation);
    maskedScoresDevice = static_cast<float *>(maskedScoresDeviceAllocation);

    fileSize = kScoresBytes;
    FILE_CHECK(ReadFile("./scores.bin", fileSize, scoresHost, kScoresBytes) &&
                   fileSize == kScoresBytes,
               "./scores.bin");
    fileSize = kIndicesBytes;
    FILE_CHECK(ReadFile("./indices.bin", fileSize, indicesHost, kIndicesBytes) &&
                   fileSize == kIndicesBytes,
               "./indices.bin");
    fileSize = kWinnerBytes;
    FILE_CHECK(ReadFile("./winner_indices.bin", fileSize, winnersHost,
                        kWinnerTransferBytes) &&
                   fileSize == kWinnerBytes,
               "./winner_indices.bin");
    fileSize = kScoresBytes;
    FILE_CHECK(ReadFile("./masked_scores.bin", fileSize, maskedScoresHost,
                        kScoresBytes) &&
                   fileSize == kScoresBytes,
               "./masked_scores.bin");

    ACL_CHECK(aclrtMemcpy(scoresDevice, kScoresBytes, scoresHost, kScoresBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(indicesDevice, kIndicesBytes, indicesHost,
                          kIndicesBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(winnersDevice, kWinnerTransferBytes, winnersHost,
                          kWinnerBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(maskedScoresDevice, kScoresBytes, maskedScoresHost,
                          kScoresBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchTopkGateVcmpVselMisched(scoresDevice, indicesDevice, winnersDevice,
                                  maskedScoresDevice, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(winnersHost, kWinnerBytes, winnersDevice, kWinnerBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(maskedScoresHost, kScoresBytes, maskedScoresDevice,
                          kScoresBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    FILE_CHECK(WriteFile("./winner_indices.bin", winnersHost, kWinnerBytes),
               "./winner_indices.bin");
    FILE_CHECK(WriteFile("./masked_scores.bin", maskedScoresHost, kScoresBytes),
               "./masked_scores.bin");
    return true;
  };

  rc = execute() ? 0 : 1;
  aclrtFree(scoresDeviceAllocation);
  aclrtFree(indicesDeviceAllocation);
  aclrtFree(winnersDeviceAllocation);
  aclrtFree(maskedScoresDeviceAllocation);
  aclrtFreeHost(scoresHostAllocation);
  aclrtFreeHost(indicesHostAllocation);
  aclrtFreeHost(winnersHostAllocation);
  aclrtFreeHost(maskedScoresHostAllocation);
  if (stream != nullptr) {
    const aclError ret = aclrtDestroyStream(stream);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "[ERROR] aclrtDestroyStream failed: %d\n",
                   static_cast<int>(ret));
      rc = 1;
    }
  }
  if (deviceSet) {
    const aclError ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "[ERROR] aclrtResetDevice failed: %d\n",
                   static_cast<int>(ret));
      rc = 1;
    }
  }
  if (aclInited) {
    const aclError ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "[ERROR] aclFinalize failed: %d\n",
                   static_cast<int>(ret));
      rc = 1;
    }
  }
  return rc;
}
