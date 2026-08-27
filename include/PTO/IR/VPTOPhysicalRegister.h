// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOPhysicalRegister.h - VPTO physical register views ---*- C++ -*-===//
//
// Scheduler analyses use this shared contract to identify SSA values that are
// zero-cost views of one physical VPTO register.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOPHYSICALREGISTER_H
#define MLIR_DIALECT_PTO_IR_VPTOPHYSICALREGISTER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::pto {

bool isPhysicalRegisterView(Operation *op);

Value getPhysicalRegisterViewRoot(Value value);

struct PhysicalRegisterRootUseAnalysis {
  bool allUsesInBlock = true;
  SmallVector<Operation *, 8> materialUsers;
};

PhysicalRegisterRootUseAnalysis
analyzePhysicalRegisterRootUses(Value root, Block *expectedBlock);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_IR_VPTOPHYSICALREGISTER_H
