// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_BISHENG_SCHEDULER_H
#define PTOAS_BISHENG_SCHEDULER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::pto {

enum class BishengSchedulerMode { Auto, On, Off };

// The callback compiles the same device LLVM IR, changing only scheduling,
// resource reporting and output paths. It must not run lowering or host builds.
using CompileBishengVariant = llvm::function_ref<bool(
    bool enabled, bool reportUsage, llvm::StringRef objectPath, llvm::StringRef logPath,
    llvm::raw_ostream& diagnostics)>;

// Auto compares the sum of stack bytes across unique SIMD VF functions.
// A failed retry or an unusable report leaves the successful on object intact.
bool compileWithBishengScheduler(
    BishengSchedulerMode mode, llvm::StringRef objectPath, llvm::StringRef logPath, CompileBishengVariant compile,
    llvm::raw_ostream& diagnostics);

} // namespace mlir::pto

#endif
