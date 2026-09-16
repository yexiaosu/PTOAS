// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "BishengScheduler.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using mlir::pto::BishengSchedulerMode;

namespace {
std::string report(llvm::StringRef name, llvm::StringRef bytes)
{
    return "[BISHENG] SIMD VF Function properties for " + name.str() + ": Stack size: " + bytes.str() +
           " bytes, VReg number: 16, MaskReg(PReg) number: 7\n";
}

bool writeFile(llvm::StringRef path, llvm::StringRef contents)
{
    std::error_code error;
    llvm::raw_fd_ostream stream(path, error);
    if (error) {
        return false;
    }
    stream << contents;
    stream.close();
    return !stream.has_error();
}

struct Scenario {
    const char* name;
    std::string on;
    std::string off;
    unsigned expectedCalls;
    bool selectOff = false;
    bool offFails = false;
    bool onFails = false;
    BishengSchedulerMode mode = BishengSchedulerMode::Auto;
};

bool runScenario(const Scenario& scenario)
{
    llvm::SmallString<128> objectPath;
    llvm::SmallString<128> logPath;
    if (llvm::sys::fs::createTemporaryFile("bisheng-test", "o", objectPath)) {
        return false;
    }
    llvm::FileRemover objectCleanup(objectPath);
    if (llvm::sys::fs::createTemporaryFile("bisheng-test", "log", logPath)) {
        return false;
    }
    llvm::FileRemover logCleanup(logPath);
    unsigned calls = 0;
    bool validArguments = true;
    std::string retryObject;
    auto compile = [&scenario, &calls, &validArguments, &retryObject, objectPath](
                       bool enabled, bool reportUsage, llvm::StringRef object, llvm::StringRef log,
                       llvm::raw_ostream&) {
        ++calls;
        bool automatic = scenario.mode == BishengSchedulerMode::Auto;
        bool expectedOn = scenario.mode != BishengSchedulerMode::Off && calls == 1;
        validArguments &= enabled == expectedOn && reportUsage == automatic;
        if (calls > 1) {
            validArguments &= object != objectPath;
            retryObject = object.str();
        }
        bool written = writeFile(object, enabled ? "on object" : "off object");
        written &= writeFile(log, enabled ? scenario.on : scenario.off);
        return written && !(enabled ? scenario.onFails : scenario.offFails);
    };
    std::string diagnostics;
    llvm::raw_string_ostream stream(diagnostics);
    bool result = mlir::pto::compileWithBishengScheduler(scenario.mode, objectPath, logPath, compile, stream);
    auto object = llvm::MemoryBuffer::getFile(objectPath);
    llvm::StringRef expected = scenario.selectOff ? "off object" : "on object";
    bool cleaned = retryObject.empty() || !llvm::sys::fs::exists(retryObject);
    bool passed = result != scenario.onFails && validArguments && cleaned && calls == scenario.expectedCalls &&
                  object && object.get()->getBuffer() == expected;
    if (!passed) {
        llvm::errs() << "FAIL: " << scenario.name << "\n" << diagnostics;
    }
    return passed;
}
} // namespace

int main()
{
    std::string zero = report("kernel.vector.thread", "0");
    std::string spill = report("kernel.vector.thread", "128");
    std::string larger = report("kernel.vector.thread", "256");
    std::string scalar = "[BISHENG] Function properties for outer: Stack size: "
                         "4096 bytes, Used register number: 20\n";
    Scenario scenarios[] = {
        {"zero skips retry", zero + scalar, spill, 1},
        {"smaller selects off", spill, zero, 2, true},
        {"equal keeps on", spill, spill, 2},
        {"larger keeps on", spill, larger, 2},
        {"deduplicate reports", spill + spill, spill, 2},
        {"sum unique VFs", spill + report("other", "256"), larger + report("other", "0"), 2, true},
        {"different VF set", spill, report("different", "0"), 2},
        {"missing off VF", spill + report("other", "256"), zero, 2},
        {"missing on report", scalar, zero, 1},
        {"missing off report", spill, scalar, 2},
        {"conflicting duplicate", spill + zero, zero, 1},
        {"malformed on", report("kernel", "unknown"), zero, 1},
        {"malformed off", spill, report("kernel.vector.thread", "-1"), 2},
        {"value overflow", report("kernel", "18446744073709551616"), zero, 1},
        {"sum overflow", report("kernel", "18446744073709551615") + spill, zero, 1},
        {"retry failure keeps on", spill, zero, 2, false, true},
        {"first failure propagates", spill, zero, 1, false, false, true},
        {"explicit on", spill, zero, 1, false, false, false, BishengSchedulerMode::On},
        {"explicit off", spill, zero, 1, true, false, false, BishengSchedulerMode::Off},
    };
    bool passed = true;
    for (const Scenario& scenario : scenarios) {
        passed &= runScenario(scenario);
    }
    if (!passed) {
        return 1;
    }
    llvm::outs() << "Bisheng scheduler selection tests passed\n";
    return 0;
}
