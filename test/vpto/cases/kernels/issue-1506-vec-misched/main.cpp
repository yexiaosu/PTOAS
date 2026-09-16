// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void LaunchIssue1506(void *k, void *out, void *q, void *rh, void *rw,
                     void *v, void *stream);

namespace {
void check(aclError result, const char *operation) {
  if (result != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
  }
}

class Runtime {
public:
  bool initialized = false;
  bool deviceSet = false;
  aclrtStream stream = nullptr;
  std::array<void *, 6> buffers{};

  Runtime() = default;
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  ~Runtime() {
    for (void *buffer : buffers) {
      if (buffer != nullptr) {
        report(aclrtFree(buffer), "aclrtFree");
      }
    }
    if (stream != nullptr) {
      report(aclrtDestroyStream(stream), "aclrtDestroyStream");
    }
    if (deviceSet) {
      report(aclrtResetDevice(0), "aclrtResetDevice");
    }
    if (initialized) {
      report(aclFinalize(), "aclFinalize");
    }
  }

private:
  static void report(aclError result, const char *operation) {
    if (result != ACL_SUCCESS) {
      std::fprintf(stderr, "cleanup %s failed: %d\n", operation, static_cast<int>(result));
    }
  }
};

std::vector<uint16_t> readInput(const std::string &path, size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::streamsize>(count * sizeof(uint16_t));
  const auto fileSize = input.tellg();
  if (!input || fileSize != bytes) {
    throw std::runtime_error("invalid input size: " + path);
  }
  input.seekg(0);
  std::vector<uint16_t> result(count);
  if (!input.read(reinterpret_cast<char *>(result.data()), bytes)) {
    throw std::runtime_error("failed reading " + path);
  }
  return result;
}
} // namespace

int main() {
  try {
    constexpr size_t tensorCount = 2 * 196 * 12 * 64;
    constexpr size_t biasCount = 2 * 12 * 196 * 16;
    constexpr std::array<size_t, 6> counts{
        tensorCount, tensorCount, tensorCount, biasCount, biasCount, tensorCount};
    Runtime runtime;
    check(aclInit(nullptr), "aclInit");
    runtime.initialized = true;
    check(aclrtSetDevice(0), "aclrtSetDevice");
    runtime.deviceSet = true;
    check(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
    for (size_t i = 0; i < counts.size(); ++i) {
      const size_t bytes = counts[i] * sizeof(uint16_t);
      auto input = readInput("input_" + std::to_string(i) + ".bin", counts[i]);
      check(aclrtMalloc(&runtime.buffers[i], bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
      check(aclrtMemcpy(runtime.buffers[i], bytes, input.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    }
    LaunchIssue1506(runtime.buffers[0], runtime.buffers[1], runtime.buffers[2],
                    runtime.buffers[3], runtime.buffers[4], runtime.buffers[5], runtime.stream);
    check(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
    std::vector<uint16_t> output(tensorCount);
    const size_t bytes = output.size() * sizeof(uint16_t);
    check(aclrtMemcpy(output.data(), bytes, runtime.buffers[1], bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
    std::ofstream file("output.bin", std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(bytes))) {
      throw std::runtime_error("failed writing output.bin");
    }
    file.close();
    if (!file) {
      throw std::runtime_error("failed closing output.bin");
    }
    std::puts("issue1506 host completed: batch storage=2, N=196, heads=12, D=64");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
