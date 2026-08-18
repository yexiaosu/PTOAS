// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>

using namespace PtoTestCommon;

void LaunchTopkGateAabbccScheduler(float *scores, int32_t *topkIdx,
                                   void *stream);

namespace {
constexpr size_t kScoresBytes = 4U * 384U * sizeof(float);
constexpr size_t kOutputBytes = 4U * 9U * sizeof(int32_t);

bool CheckAcl(aclError result, const char *expression)
{
    if (result == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", expression,
                 static_cast<int>(result));
    return false;
}
} // namespace

int main()
{
    int result = 1;
    bool aclInitialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    float *scoresHost = nullptr;
    int32_t *outputHost = nullptr;
    float *scoresDevice = nullptr;
    int32_t *outputDevice = nullptr;
    size_t scoresFileSize = kScoresBytes;
    size_t outputFileSize = kOutputBytes;

    do {
        if (!CheckAcl(aclInit(nullptr), "aclInit")) {
            break;
        }
        aclInitialized = true;
        if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) {
            break;
        }
        deviceSet = true;

        const bool resourcesReady =
            CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") &&
            CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&scoresHost),
                                     kScoresBytes),
                     "aclrtMallocHost(scores)") &&
            CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost),
                                     kOutputBytes),
                     "aclrtMallocHost(output)") &&
            CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&scoresDevice),
                                 kScoresBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                     "aclrtMalloc(scores)") &&
            CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice),
                                 kOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                     "aclrtMalloc(output)");
        if (!resourcesReady) {
            break;
        }

        const bool inputsReady =
            ReadFile("./scores.bin", scoresFileSize, scoresHost, kScoresBytes) &&
            scoresFileSize == kScoresBytes &&
            ReadFile("./output.bin", outputFileSize, outputHost, kOutputBytes) &&
            outputFileSize == kOutputBytes;
        if (!inputsReady) {
            std::fprintf(stderr, "[ERROR] failed to read exact-size input files\n");
            break;
        }
        const bool inputsCopied =
            CheckAcl(aclrtMemcpy(scoresDevice, kScoresBytes, scoresHost,
                                 kScoresBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                     "aclrtMemcpy(scores H2D)") &&
            CheckAcl(aclrtMemcpy(outputDevice, kOutputBytes, outputHost,
                                 kOutputBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                     "aclrtMemcpy(output H2D)");
        if (!inputsCopied) {
            break;
        }

        LaunchTopkGateAabbccScheduler(scoresDevice, outputDevice, stream);
        const bool outputReady =
            CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") &&
            CheckAcl(aclrtMemcpy(outputHost, kOutputBytes, outputDevice,
                                 kOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                     "aclrtMemcpy(output D2H)");
        if (!outputReady) {
            break;
        }
        if (!WriteFile("./output.bin", outputHost, kOutputBytes)) {
            std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
            break;
        }
        result = 0;
    } while (false);

    if (outputDevice != nullptr &&
        !CheckAcl(aclrtFree(outputDevice), "aclrtFree(output)")) {
        result = 1;
    }
    if (scoresDevice != nullptr &&
        !CheckAcl(aclrtFree(scoresDevice), "aclrtFree(scores)")) {
        result = 1;
    }
    if (outputHost != nullptr &&
        !CheckAcl(aclrtFreeHost(outputHost), "aclrtFreeHost(output)")) {
        result = 1;
    }
    if (scoresHost != nullptr &&
        !CheckAcl(aclrtFreeHost(scoresHost), "aclrtFreeHost(scores)")) {
        result = 1;
    }
    if (stream != nullptr &&
        !CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream")) {
        result = 1;
    }
    if (deviceSet &&
        !CheckAcl(aclrtResetDevice(0), "aclrtResetDevice")) {
        result = 1;
    }
    if (aclInitialized && !CheckAcl(aclFinalize(), "aclFinalize")) {
        result = 1;
    }
    return result;
}
