// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOPhysicalRegister.cpp - VPTO physical register views -----------===//

#include "PTO/IR/VPTOPhysicalRegister.h"

#include "PTO/IR/PTO.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::pto;

bool mlir::pto::isPhysicalRegisterView(Operation *op) {
  return isa_and_nonnull<VbitcastOp, PbitcastOp>(op);
}

Value mlir::pto::getPhysicalRegisterViewRoot(Value value) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (auto result = dyn_cast<OpResult>(value)) {
    Operation *owner = result.getOwner();
    if (!visited.insert(owner).second) {
      break;
    }
    if (auto view = dyn_cast<VbitcastOp>(owner)) {
      value = view.getInput();
      continue;
    }
    if (auto view = dyn_cast<PbitcastOp>(owner)) {
      value = view.getInput();
      continue;
    }
    break;
  }
  return value;
}

PhysicalRegisterRootUseAnalysis
mlir::pto::analyzePhysicalRegisterRootUses(Value root, Block *expectedBlock) {
  PhysicalRegisterRootUseAnalysis analysis;
  SmallVector<Value, 8> worklist{root};
  llvm::DenseSet<Value> visitedValues;
  llvm::SmallPtrSet<Operation *, 8> visitedMaterialUsers;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visitedValues.insert(value).second) {
      continue;
    }
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      bool isViewOperand =
          isPhysicalRegisterView(owner) && use.getOperandNumber() == 0;
      if (isViewOperand) {
        analysis.allUsesInBlock &= owner->getBlock() == expectedBlock;
        worklist.append(owner->getResults().begin(), owner->getResults().end());
        continue;
      }
      analysis.allUsesInBlock &= owner->getBlock() == expectedBlock;
      if (visitedMaterialUsers.insert(owner).second) {
        analysis.materialUsers.push_back(owner);
      }
    }
  }
  return analysis;
}

Operation *mlir::pto::findLastPhysicalRegisterMaterialUser(
    const PhysicalRegisterRootUseAnalysis &analysis, Block *block) {
  if (!analysis.allUsesInBlock) {
    return nullptr;
  }
  Operation *last = nullptr;
  for (Operation *user : analysis.materialUsers) {
    bool isOutsideBlock = user->getBlock() != block;
    if (isOutsideBlock) {
      return nullptr;
    }
    if (!last || last->isBeforeInBlock(user)) {
      last = user;
    }
  }
  return last;
}
