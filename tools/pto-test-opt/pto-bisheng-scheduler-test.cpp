// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// These fake-compile tests cover decision branches, report parsing and cleanup.
// They do not validate the real Bisheng command line or reporting contract;
// test/vpto/scripts/check_bisheng_scheduler.py exercises the real SIM runner.
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
    bool expectedSuccess = true;
};

bool checkDiagnostics(const Scenario& scenario, llvm::StringRef diagnostics)
{
    bool recovered = scenario.onFails && scenario.expectedSuccess;
    bool onError = scenario.onFails;
    bool expectedOffError = scenario.offFails;
    bool matched = diagnostics.contains("simulated on failure") == onError;
    matched &= diagnostics.contains("simulated off failure") == expectedOffError;
    if (recovered) {
        matched &= diagnostics.contains("on compilation failed; retrying off");
        matched &= diagnostics.contains("on compilation failed; selected off");
        matched &= diagnostics.contains("recovered Bisheng on compilation failure:");
    }
    if (scenario.onFails && scenario.offFails) {
        matched &= diagnostics.contains("both on and off compilation failed");
    }
    // Auto keeps the on object whenever the reports cannot be compared. That
    // degraded decision must stay visible as a warning: a run that never printed
    // a compared stack total must not silently look like a successful selection.
    bool automatic = scenario.mode == BishengSchedulerMode::Auto;
    bool comparedReports = diagnostics.contains("SIMD VF stack bytes on=");
    if (automatic && !scenario.onFails && !comparedReports) {
        matched &= diagnostics.contains("Warning:");
    }
    return matched;
}

// Records the compile callback arguments of one scenario run and writes the requested object and log files.
class ScenarioCompiler {
public:
    ScenarioCompiler(const Scenario& scenario, llvm::StringRef objectPath, llvm::StringRef logPath)
        : scenario(&scenario), originalObject(objectPath), originalLog(logPath)
    {
    }

    bool operator()(bool enabled, bool reportUsage, llvm::StringRef object, llvm::StringRef log,
                    llvm::raw_ostream& diagnostics)
    {
        ++calls;
        checkArguments(enabled, reportUsage, object, log);
        bool fails = enabled ? scenario->onFails : scenario->offFails;
        bool written = writeVariant(enabled, object, log, fails, diagnostics);
        return written && !fails;
    }

    unsigned callCount() const
    {
        return calls;
    }

    bool argumentsValid() const
    {
        return validArguments;
    }

    bool removedRetryFiles() const
    {
        bool removed = retryObject.empty() || !llvm::sys::fs::exists(retryObject);
        removed &= retryLog.empty() || !llvm::sys::fs::exists(retryLog);
        return removed;
    }

private:
    void checkArguments(bool enabled, bool reportUsage, llvm::StringRef object, llvm::StringRef log)
    {
        bool automatic = scenario->mode == BishengSchedulerMode::Auto;
        bool expectedOn = scenario->mode != BishengSchedulerMode::Off && calls == 1;
        bool expectedUsage = automatic && (calls == 1 || !scenario->onFails);
        validArguments &= enabled == expectedOn && reportUsage == expectedUsage;
        if (calls > 1) {
            validArguments &= object != originalObject && log != originalLog;
            retryObject = object.str();
            retryLog = log.str();
        }
    }

    bool writeVariant(bool enabled, llvm::StringRef object, llvm::StringRef log, bool fails,
                      llvm::raw_ostream& diagnostics) const
    {
        std::string contents = fails ? "partial " : "";
        contents += enabled ? "on object" : "off object";
        bool written = writeFile(object, contents);
        written &= writeFile(log, enabled ? scenario->on : scenario->off);
        if (fails) {
            diagnostics << (enabled ? "simulated on failure\n" : "simulated off failure\n");
        }
        return written;
    }

    const Scenario* scenario;
    llvm::StringRef originalObject;
    llvm::StringRef originalLog;
    unsigned calls = 0;
    bool validArguments = true;
    std::string retryObject;
    std::string retryLog;
};

bool verifyScenario(const Scenario& scenario, const ScenarioCompiler& compiler, llvm::StringRef objectPath,
                    llvm::StringRef diagnostics, bool result)
{
    auto object = llvm::MemoryBuffer::getFile(objectPath);
    std::string expected = scenario.expectedSuccess ? "" : "partial ";
    expected += scenario.selectOff ? "off object" : "on object";
    bool messages = checkDiagnostics(scenario, diagnostics);
    return result == scenario.expectedSuccess && compiler.argumentsValid() && compiler.removedRetryFiles() &&
           messages && compiler.callCount() == scenario.expectedCalls && object &&
           object.get()->getBuffer() == expected;
}

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
    ScenarioCompiler compiler(scenario, objectPath, logPath);
    std::string diagnostics;
    llvm::raw_string_ostream stream(diagnostics);
    bool result = mlir::pto::compileWithBishengScheduler(scenario.mode, objectPath, logPath, compiler, stream);
    bool passed = verifyScenario(scenario, compiler, objectPath, diagnostics, result);
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
    std::string minimal = "[BISHENG] SIMD VF Function properties for kernel.vector.thread: Stack size: 0 bytes";
    Scenario scenarios[] = {
        {"stack-only report", minimal, spill, 1},
        {"stack-only off report", spill, minimal, 2, true},
        {"unknown trailing fields", minimal + ", FutureField: 42\n", spill, 1},
        {"missing bytes unit", "[BISHENG] SIMD VF Function properties for kernel: Stack size: 0", zero, 1},
        {"invalid bytes suffix", minimal + "garbage", zero, 1},
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
        {"on failure selects off", spill, zero, 2, true, false, true},
        {"on failure ignores larger off stack", zero, larger, 2, true, false, true},
        {"on failure needs no off report", "", "", 2, true, false, true},
        {"on failure ignores malformed reports", "invalid", "invalid", 2, true, false, true},
        {"both failures propagate", spill, zero, 2, false, true, true, BishengSchedulerMode::Auto, false},
        {"explicit on", spill, zero, 1, false, false, false, BishengSchedulerMode::On},
        {"explicit off", spill, zero, 1, true, false, false, BishengSchedulerMode::Off},
        {"explicit on failure does not retry", spill, zero, 1, false, false, true, BishengSchedulerMode::On, false},
        {"explicit off failure does not retry", spill, zero, 1, true, true, false, BishengSchedulerMode::Off, false},
    };
    bool passed = true;
    for (const Scenario& scenario : scenarios) {
        passed &= runScenario(scenario);
    }
    for (auto mode : {BishengSchedulerMode::Auto, BishengSchedulerMode::On, BishengSchedulerMode::Off}) {
        auto cubeMode = mode == BishengSchedulerMode::On ? mode : BishengSchedulerMode::Off;
        passed &= mlir::pto::getBishengSchedulerModeForTarget(mode, true) == mode;
        passed &= mlir::pto::getBishengSchedulerModeForTarget(mode, false) == cubeMode;
    }
    if (!passed) {
        return 1;
    }
    llvm::outs() << "Bisheng scheduler selection tests passed\n";
    return 0;
}
