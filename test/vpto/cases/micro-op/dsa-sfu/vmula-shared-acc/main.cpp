// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstddef>
#include <cstdio>

using namespace PtoTestCommon;

void LaunchVmulaSharedAcc(float *input, float *output, void *stream);

namespace {

bool CheckAcl(aclError result, const char *operation) {
    if (result == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", operation, static_cast<int>(result));
    return false;
}

struct AclResources {
    void *inputHost = nullptr;
    void *outputHost = nullptr;
    void *inputDevice = nullptr;
    void *outputDevice = nullptr;
    aclrtStream stream = nullptr;
    bool aclInitialized = false;
    bool deviceSet = false;

    ~AclResources() {
        if (inputDevice != nullptr) {
            (void)aclrtFree(inputDevice);
        }
        if (outputDevice != nullptr) {
            (void)aclrtFree(outputDevice);
        }
        if (inputHost != nullptr) {
            (void)aclrtFreeHost(inputHost);
        }
        if (outputHost != nullptr) {
            (void)aclrtFreeHost(outputHost);
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            (void)aclrtResetDevice(0);
        }
        if (aclInitialized) {
            (void)aclFinalize();
        }
    }
};

} // namespace

int main() {
    constexpr std::size_t kElementCount = 1024;
    constexpr std::size_t kOutputCount = 3;
    constexpr std::size_t kInputBytes = kElementCount * sizeof(float);
    constexpr std::size_t kOutputBytes = kElementCount * kOutputCount * sizeof(float);

    AclResources resources;

    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.aclInitialized = true;
    if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }
    if (!CheckAcl(aclrtMallocHost(&resources.inputHost, kInputBytes), "aclrtMallocHost(input)")) {
        return 1;
    }
    if (!CheckAcl(aclrtMallocHost(&resources.outputHost, kOutputBytes), "aclrtMallocHost(output)")) {
        return 1;
    }
    if (!CheckAcl(aclrtMalloc(&resources.inputDevice, kInputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(input)")) {
        return 1;
    }
    if (!CheckAcl(aclrtMalloc(&resources.outputDevice, kOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(output)")) {
        return 1;
    }

    ReadFile("./input.bin", kInputBytes, resources.inputHost, kInputBytes);
    ReadFile("./output.bin", kOutputBytes, resources.outputHost, kOutputBytes);
    if (!CheckAcl(aclrtMemcpy(resources.inputDevice, kInputBytes, resources.inputHost, kInputBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(input)")) {
        return 1;
    }
    if (!CheckAcl(aclrtMemcpy(resources.outputDevice, kOutputBytes, resources.outputHost, kOutputBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(output)")) {
        return 1;
    }

    LaunchVmulaSharedAcc(static_cast<float *>(resources.inputDevice), static_cast<float *>(resources.outputDevice),
                         resources.stream);
    if (!CheckAcl(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }
    if (!CheckAcl(aclrtMemcpy(resources.outputHost, kOutputBytes, resources.outputDevice, kOutputBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(result)")) {
        return 1;
    }
    WriteFile("./output.bin", resources.outputHost, kOutputBytes);
    return 0;
}
