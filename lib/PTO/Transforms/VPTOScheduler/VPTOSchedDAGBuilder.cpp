// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAGBuilder.cpp - VPTO scheduling DAG builder -------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAGBuilder.h"

#include "PTO/IR/PTO.h"

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

#include "../Utils.h"

using namespace mlir;
using namespace mlir::pto;

FailureOr<std::unique_ptr<VPTOSchedDAG>>
VPTOSchedDAGBuilder::build(const VPTOSchedRegion &region) const {
  VPTOScheduleFailure failure;
  return build(region, failure);
}

FailureOr<std::unique_ptr<VPTOSchedDAG>> VPTOSchedDAGBuilder::build(
    const VPTOSchedRegion &region, VPTOScheduleFailure &failure) const {
  uint64_t nodeCount = region.operations.size();
  if (limits && nodeCount > limits->maxNodes) {
    failure = {VPTOScheduleFailureKind::Budget, "nodes", nodeCount,
               limits->maxNodes, {}};
    return mlir::failure();
  }

  auto dag = std::make_unique<VPTOSchedDAG>(region);
  if (failed(buildSSAEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildMemoryEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildImplicitAndSyncEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(buildModelFallbackEdges(*dag, failure))) {
    return mlir::failure();
  }
  if (failed(consumeWork(failure, dag->getUnits().size()))) {
    return mlir::failure();
  }
  if (failed(consumeWork(failure, dag->getEdges().size()))) {
    return mlir::failure();
  }
  if (failed(dag->computeCriticalPaths())) {
    failure.kind = VPTOScheduleFailureKind::Scheduling;
    failure.detail = "dependency DAG contains a Must-edge cycle";
    return mlir::failure();
  }
  dag->resetDependencyCounts();
  return std::move(dag);
}

LogicalResult
VPTOSchedDAGBuilder::consumeWork(VPTOScheduleFailure &failure,
                                 uint64_t amount) const {
  if (!budget || budget->consume(amount)) {
    return success();
  }
  uint64_t limit = budget->getLimit();
  uint64_t actual = limit == std::numeric_limits<uint64_t>::max()
                        ? limit
                        : limit + 1;
  failure = {VPTOScheduleFailureKind::Budget, "work-units", actual, limit, {}};
  return mlir::failure();
}

LogicalResult VPTOSchedDAGBuilder::addEdge(
    VPTOSchedDAG &dag, VPTOSUnit &predecessor, VPTOSUnit &successor,
    VPTOSchedEdgeKind kind, VPTOSchedEdgeStrength strength, unsigned latency,
    Twine reason, VPTOScheduleFailure &failure) const {
  uint64_t edgeCount = dag.getEdges().size();
  if (limits && edgeCount >= limits->maxEdges) {
    uint64_t actual = edgeCount == std::numeric_limits<uint64_t>::max()
                          ? edgeCount
                          : edgeCount + 1;
    failure = {VPTOScheduleFailureKind::Budget, "edges", actual,
               limits->maxEdges, {}};
    return mlir::failure();
  }
  if (failed(consumeWork(failure))) {
    return mlir::failure();
  }
  dag.addEdge(predecessor, successor, kind, strength, latency, reason);
  return success();
}

namespace {
struct ResolvedMemoryAccess {
  VPTOMemoryAccess semantics;
  Value aliasRoot;
  std::optional<int64_t> absoluteByteOffset;
};

static std::optional<int64_t> getConstantInteger(Value value) {
  IntegerAttr constant;
  bool invalidConstant = !value ||
                         !matchPattern(value, m_Constant(&constant)) ||
                         !constant.getValue().isSignedIntN(64);
  if (invalidConstant) {
    return std::nullopt;
  }
  return constant.getValue().getSExtValue();
}

/// Resolve the byte address represented by the pointer-producing operations
/// that are explicit at the VPTO scheduling boundary. Keep all other pointer
/// transforms conservative: their layout may not be expressible as one
/// constant byte displacement.
static std::optional<int64_t> getConstantPointerAddress(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  int64_t displacement = 0;
  while (Operation *definingOp = value.getDefiningOp()) {
    if (!visited.insert(definingOp).second) {
      return std::nullopt;
    }
    if (auto cast = dyn_cast<CastPtrOp>(definingOp)) {
      Value input = cast.getInput();
      if (std::optional<int64_t> address = getConstantInteger(input)) {
        int64_t absoluteAddress;
        if (llvm::AddOverflow(*address, displacement, absoluteAddress)) {
          return std::nullopt;
        }
        return absoluteAddress;
      }
      if (!isa<PtrType>(input.getType())) {
        return std::nullopt;
      }
      value = input;
      continue;
    }
    if (auto intToPtr = dyn_cast<IntToPtrOp>(definingOp)) {
      std::optional<int64_t> address = getConstantInteger(intToPtr.getAddr());
      if (!address) {
        return std::nullopt;
      }
      int64_t absoluteAddress;
      if (llvm::AddOverflow(*address, displacement, absoluteAddress)) {
        return std::nullopt;
      }
      return absoluteAddress;
    }
    if (auto addPtr = dyn_cast<AddPtrOp>(definingOp)) {
      std::optional<int64_t> elementOffset =
          getConstantInteger(addPtr.getOffset());
      auto pointerType = dyn_cast<PtrType>(addPtr.getPtr().getType());
      if (!elementOffset || !pointerType) {
        return std::nullopt;
      }
      Type elementType = pointerType.getElementType();
      bool invalidElementType = !elementType.isIntOrFloat() ||
                                elementType.getIntOrFloatBitWidth() % 8 != 0;
      if (invalidElementType) {
        return std::nullopt;
      }
      int64_t byteOffset;
      if (llvm::MulOverflow(
              *elementOffset,
              static_cast<int64_t>(elementType.getIntOrFloatBitWidth() / 8),
              byteOffset) ||
          llvm::AddOverflow(displacement, byteOffset, displacement)) {
        return std::nullopt;
      }
      value = addPtr.getPtr();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static Value getAliasRoot(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  while (Operation *definingOp = value.getDefiningOp()) {
    if (!visited.insert(definingOp).second)
      break;
    std::optional<std::pair<Value, Value>> alias =
        getOperationAliasInfo(definingOp);
    if (!alias || alias->first != value || !alias->second)
      break;
    value = alias->second;
  }
  return value;
}

static SmallVector<ResolvedMemoryAccess>
resolveMemoryAccesses(const VPTOSchedulingSemantics &semantics) {
  SmallVector<ResolvedMemoryAccess> accesses;
  accesses.reserve(semantics.memoryAccesses.size());
  for (const VPTOMemoryAccess &memoryAccess : semantics.memoryAccesses) {
    ResolvedMemoryAccess access{memoryAccess, {}, std::nullopt};
    if (access.semantics.address) {
      std::optional<int64_t> pointerAddress =
          getConstantPointerAddress(access.semantics.address);
      if (pointerAddress && access.semantics.byteOffset) {
        int64_t absoluteByteOffset;
        if (!llvm::AddOverflow(*pointerAddress,
                               *access.semantics.byteOffset,
                               absoluteByteOffset))
          access.absoluteByteOffset = absoluteByteOffset;
      }
      access.aliasRoot = getAliasRoot(access.semantics.address);
      if (access.aliasRoot != access.semantics.address) {
        access.semantics.byteOffset.reset();
      }
    }
    accesses.push_back(std::move(access));
  }
  return accesses;
}

static bool mayAlias(const ResolvedMemoryAccess &lhs,
                     const ResolvedMemoryAccess &rhs) {
  if (lhs.semantics.addressSpace && rhs.semantics.addressSpace &&
      lhs.semantics.addressSpace != rhs.semantics.addressSpace)
    return false;
  if (!lhs.semantics.address || !rhs.semantics.address)
    return true;
  if (lhs.absoluteByteOffset && lhs.semantics.byteSize &&
      rhs.absoluteByteOffset && rhs.semantics.byteSize) {
    int64_t lhsEnd;
    int64_t rhsEnd;
    if (!llvm::AddOverflow(*lhs.absoluteByteOffset, *lhs.semantics.byteSize,
                           lhsEnd) &&
        !llvm::AddOverflow(*rhs.absoluteByteOffset, *rhs.semantics.byteSize,
                           rhsEnd))
      return *lhs.absoluteByteOffset < rhsEnd &&
             *rhs.absoluteByteOffset < lhsEnd;
  }
  if (lhs.aliasRoot == rhs.aliasRoot && lhs.semantics.byteOffset &&
      lhs.semantics.byteSize && rhs.semantics.byteOffset &&
      rhs.semantics.byteSize) {
    int64_t lhsEnd;
    int64_t rhsEnd;
    if (!llvm::AddOverflow(*lhs.semantics.byteOffset, *lhs.semantics.byteSize,
                           lhsEnd) &&
        !llvm::AddOverflow(*rhs.semantics.byteOffset, *rhs.semantics.byteSize,
                           rhsEnd))
      return *lhs.semantics.byteOffset < rhsEnd &&
             *rhs.semantics.byteOffset < lhsEnd;
  }
  // Different roots in the same physical space remain conservative: memory
  // planning may have assigned overlapping ranges to distinct SSA roots.
  return true;
}

static bool needsMemoryOrder(const ResolvedMemoryAccess &lhs,
                             const ResolvedMemoryAccess &rhs) {
  if (!mayAlias(lhs, rhs))
    return false;
  if (lhs.semantics.ordered || rhs.semantics.ordered || lhs.semantics.unknown ||
      rhs.semantics.unknown)
    return true;
  return lhs.semantics.writes || rhs.semantics.writes;
}

static bool isPostUpdateAddress(const VPTOSUnit &producer, Value value) {
  return llvm::any_of(
      producer.getSemantics().effects, [&](const VPTOSchedulingEffect &effect) {
        return effect.kind == VPTOSchedulingEffectKind::PostUpdate &&
               effect.value == value;
      });
}

} // namespace

LogicalResult VPTOSchedDAGBuilder::buildMemoryEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  SmallVector<SmallVector<ResolvedMemoryAccess>> accesses;
  accesses.reserve(dag.getUnits().size());
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (failed(consumeWork(failure,
                           unit->getSemantics().memoryAccesses.size()))) {
      return mlir::failure();
    }
    accesses.push_back(resolveMemoryAccesses(unit->getSemantics()));
  }

  struct FrontierAccess {
    VPTOSUnit *unit = nullptr;
    ResolvedMemoryAccess access;
  };
  SmallVector<FrontierAccess> frontier;
  for (auto indexedAccesses : llvm::enumerate(accesses)) {
    size_t unitIndex = indexedAccesses.index();
    ArrayRef<ResolvedMemoryAccess> currentAccesses = indexedAccesses.value();
    if (currentAccesses.empty()) {
      continue;
    }
    VPTOSUnit *unit = dag.getUnits()[unitIndex].get();
    SmallPtrSet<VPTOSUnit *, 8> predecessors;
    for (const FrontierAccess &prior : frontier) {
      for (const ResolvedMemoryAccess &current : currentAccesses) {
        if (failed(consumeWork(failure))) {
          return mlir::failure();
        }
        if (needsMemoryOrder(prior.access, current)) {
          predecessors.insert(prior.unit);
          break;
        }
      }
    }
    for (VPTOSUnit *predecessor : predecessors) {
      if (failed(addEdge(dag, *predecessor, *unit,
                         VPTOSchedEdgeKind::Memory,
                         VPTOSchedEdgeStrength::Must,
                         /*latency=*/0,
                         "may-alias memory frontier in original order",
                         failure))) {
        return mlir::failure();
      }
    }

    // A write (or ordered/unknown access) subsumes every may-alias frontier
    // entry because the edges above already preserve those earlier accesses
    // through this unit. Read-only accesses remain side by side until a later
    // write joins them, preserving WAR while avoiding transitive edges.
    llvm::erase_if(frontier, [&](const FrontierAccess &prior) {
      return llvm::any_of(currentAccesses,
                          [&](const ResolvedMemoryAccess &current) {
                            bool closesFrontier =
                                current.semantics.writes ||
                                current.semantics.ordered ||
                                current.semantics.unknown;
                            return closesFrontier &&
                                   mayAlias(prior.access, current);
                          });
    });
    for (const ResolvedMemoryAccess &current : currentAccesses) {
      frontier.push_back({unit, current});
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildImplicitAndSyncEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  llvm::StringMap<VPTOSUnit *> lastWrite;
  llvm::StringMap<SmallVector<VPTOSUnit *>> readsSinceWrite;
  VPTOSUnit *lastBarrier = nullptr;

  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    if (lastBarrier && lastBarrier != &unit &&
        failed(addEdge(dag, *lastBarrier, unit, VPTOSchedEdgeKind::Sync,
                       VPTOSchedEdgeStrength::Must, 0,
                       "after scheduling barrier", failure))) {
      return mlir::failure();
    }

    for (const VPTOSchedulingEffect &effect : unit.getSemantics().effects) {
      if (failed(consumeWork(failure))) {
        return mlir::failure();
      }
      if (effect.kind == VPTOSchedulingEffectKind::Barrier) {
        for (const std::unique_ptr<VPTOSUnit> &prior : dag.getUnits()) {
          if (prior->getOriginalIndex() >= unit.getOriginalIndex())
            break;
          if (failed(addEdge(dag, *prior, unit, VPTOSchedEdgeKind::Sync,
                             VPTOSchedEdgeStrength::Must, 0,
                             "before scheduling barrier", failure))) {
            return mlir::failure();
          }
        }
        lastBarrier = &unit;
        continue;
      }
      if (effect.resource.empty())
        continue;
      if (effect.kind == VPTOSchedulingEffectKind::ImplicitRead) {
        if (VPTOSUnit *writer = lastWrite.lookup(effect.resource)) {
          if (failed(addEdge(dag, *writer, unit, VPTOSchedEdgeKind::Data,
                             VPTOSchedEdgeStrength::Must, 1,
                             Twine("implicit read of ") + effect.resource,
                             failure))) {
            return mlir::failure();
          }
        }
        readsSinceWrite[effect.resource].push_back(&unit);
        continue;
      }
      if (effect.kind != VPTOSchedulingEffectKind::ImplicitWrite)
        continue;
      if (VPTOSUnit *writer = lastWrite.lookup(effect.resource)) {
        if (failed(addEdge(dag, *writer, unit, VPTOSchedEdgeKind::Output,
                           VPTOSchedEdgeStrength::Must, 0,
                           Twine("implicit write of ") + effect.resource,
                           failure))) {
          return mlir::failure();
        }
      }
      for (VPTOSUnit *reader : readsSinceWrite[effect.resource]) {
        if (failed(addEdge(dag, *reader, unit, VPTOSchedEdgeKind::Anti,
                           VPTOSchedEdgeStrength::Must, 0,
                           Twine("implicit anti-dependence on ") +
                               effect.resource,
                           failure))) {
          return mlir::failure();
        }
      }
      readsSinceWrite[effect.resource].clear();
      lastWrite[effect.resource] = &unit;
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildSSAEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  for (Value value : dag.getRegion().liveThroughs) {
    if (failed(consumeWork(failure))) {
      return mlir::failure();
    }
    dag.addLiveIn(value);
    dag.addLiveOut(value);
  }

  DenseSet<Value> checkedLiveIns;
  Operation *lastRegionOperation = dag.getRegion().operations.back();
  for (const std::unique_ptr<VPTOSUnit> &unitOwner : dag.getUnits()) {
    VPTOSUnit &unit = *unitOwner;
    Operation *op = unit.getOperation();

    for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
      if (failed(consumeWork(failure))) {
        return mlir::failure();
      }
      Operation *definingOp = operand.getDefiningOp();
      VPTOSUnit *predecessor = definingOp ? dag.lookup(definingOp) : nullptr;
      if (!predecessor) {
        dag.addLiveIn(operand);
        if (!checkedLiveIns.insert(operand).second) {
          continue;
        }
        for (Operation *user : operand.getUsers()) {
          if (failed(consumeWork(failure))) {
            return mlir::failure();
          }
          if (dag.lookup(user)) {
            continue;
          }
          bool isInAnotherBlock = user->getBlock() != dag.getRegion().block;
          bool isAfterRegion = !isInAnotherBlock &&
                               lastRegionOperation->isBeforeInBlock(user);
          if (isInAnotherBlock || isAfterRegion) {
            dag.addLiveOut(operand);
            break;
          }
        }
        continue;
      }
      unsigned latency =
          model ? model->getSchedClass(predecessor->getOperation()).writeLatency
                : 1;
      std::string reason =
          (isPostUpdateAddress(*predecessor, operand)
               ? Twine("post-update address operand #") + Twine(operandIndex)
               : Twine("ssa operand #") + Twine(operandIndex))
              .str();
      if (failed(addEdge(dag, *predecessor, unit, VPTOSchedEdgeKind::Data,
                         VPTOSchedEdgeStrength::Must, latency, reason,
                         failure))) {
        return mlir::failure();
      }
    }

    for (Value result : op->getResults()) {
      bool hasExternalUser = false;
      for (Operation *user : result.getUsers()) {
        if (failed(consumeWork(failure))) {
          return mlir::failure();
        }
        if (!dag.lookup(user)) {
          hasExternalUser = true;
          break;
        }
      }
      if (hasExternalUser) {
        dag.addLiveOut(result);
      }
    }
  }
  return success();
}

LogicalResult VPTOSchedDAGBuilder::buildModelFallbackEdges(
    VPTOSchedDAG &dag, VPTOScheduleFailure &failure) const {
  if (!model)
    return success();
  ArrayRef<std::unique_ptr<VPTOSUnit>> units = dag.getUnits();
  for (size_t index = 0; index < units.size(); ++index) {
    if (failed(consumeWork(failure))) {
      return mlir::failure();
    }
    VPTOSUnit &unit = *units[index];
    if (model->getSchedClass(unit.getOperation()).known)
      continue;
    if (index != 0 &&
        failed(addEdge(dag, *units[index - 1], unit,
                       VPTOSchedEdgeKind::Artificial,
                       VPTOSchedEdgeStrength::Must, 0,
                       "unknown sched class preserves predecessor order",
                       failure))) {
      return mlir::failure();
    }
    if (index + 1 != units.size()) {
      if (failed(addEdge(dag, unit, *units[index + 1],
                         VPTOSchedEdgeKind::Artificial,
                         VPTOSchedEdgeStrength::Must, 0,
                         "unknown sched class preserves successor order",
                         failure))) {
        return mlir::failure();
      }
    }
  }
  return success();
}
