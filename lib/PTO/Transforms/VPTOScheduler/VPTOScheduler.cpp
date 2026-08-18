// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOScheduler.cpp - VPTO list scheduler ---------------------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOScheduler.h"

#include "PTO/Transforms/VPTOScheduler/VPTORegPressureTracker.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedBoundary.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

static VPTOSchedCandidate buildCandidate(VPTOSUnit &unit,
                                         const VPTOSchedBoundary &boundary) {
  VPTOSchedDirection direction = boundary.getDirection();
  unsigned criticalPath =
      direction == VPTOSchedDirection::Top ? unit.getHeight() : unit.getDepth();
  return {&unit,
          direction,
          boundary.getCurrentCycle(),
          criticalPath,
          unit.getOriginalIndex(),
          boundary.getPressureTracker().evaluate(unit)};
}

static bool regionIsStillContiguous(const VPTOSchedRegion &region) {
  if (!region.block || region.operations.empty()) {
    return false;
  }
  Operation *current = region.precedingBoundary
                           ? region.precedingBoundary->getNextNode()
                           : &region.block->front();
  for (Operation *expected : region.operations) {
    if (!current || current != expected ||
        current->getBlock() != region.block) {
      return false;
    }
    current = current->getNextNode();
  }
  return current == region.followingBoundary;
}

static void setFailure(VPTOScheduleFailure &failure,
                       VPTOScheduleFailureKind kind, StringRef detail) {
  failure.kind = kind;
  failure.detail = detail.str();
}

static void setWorkBudgetFailure(VPTOScheduleFailure &failure,
                                 const VPTOSchedulingBudget &budget) {
  uint64_t actual = budget.getUsed();
  if (actual != std::numeric_limits<uint64_t>::max()) {
    ++actual;
  }
  failure = {VPTOScheduleFailureKind::Budget,
             "work-units",
             actual,
             budget.getLimit(),
             {}};
}

static FailureOr<SmallVector<VPTOSchedCandidate>>
buildCandidates(const VPTOSchedBoundary &boundary,
                VPTOSchedulingBudget &budget,
                VPTOScheduleFailure &failure) {
  SmallVector<VPTOSchedCandidate> candidates;
  candidates.reserve(boundary.getAvailable().size());
  for (VPTOSUnit *unit : boundary.getAvailable()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    candidates.push_back(buildCandidate(*unit, boundary));
  }
  return candidates;
}

static bool exceedsPressureLimit(const VPTOSchedCandidate &candidate) {
  return llvm::any_of(candidate.pressure.projectedExcess,
                      [](int64_t excess) { return excess > 0; });
}

static bool allCandidatesExceedPressureLimits(
    ArrayRef<VPTOSchedCandidate> candidates) {
  return !candidates.empty() && llvm::all_of(candidates, exceedsPressureLimit);
}

static FailureOr<bool>
advanceForPressure(VPTOSchedBoundary &boundary,
                   ArrayRef<VPTOSchedCandidate> candidates,
                   VPTOSchedulingBudget &budget,
                   VPTOScheduleFailure &failure) {
  if (!allCandidatesExceedPressureLimits(candidates)) {
    return false;
  }
  if (!boundary.getNextPending()) {
    return false;
  }
  FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
  if (failed(advanced)) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  return *advanced;
}

static LogicalResult replayPressureDrivenIdle(
    const VPTOScheduleEntry &entry, VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  if (!entry.pressureDrivenIdle) {
    return success();
  }

  bool advancedAtLeastOnce = false;
  while (true) {
    const uint64_t currentCycle = boundary.getCurrentCycle();
    if (currentCycle >= entry.issueCycle) {
      break;
    }
    FailureOr<SmallVector<VPTOSchedCandidate>> candidates =
        buildCandidates(boundary, budget, failure);
    if (failed(candidates)) {
      return mlir::failure();
    }
    if (!allCandidatesExceedPressureLimits(*candidates)) {
      setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
                 "pressure idle occurred while a non-exceeding candidate was "
                 "available");
      return mlir::failure();
    }
    FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
    if (failed(advanced)) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    if (!*advanced) {
      setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
                 "pressure idle has no pending dependency event");
      return mlir::failure();
    }
    advancedAtLeastOnce = true;
  }
  if (!advancedAtLeastOnce) {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
               "pressure idle does not advance the logical cycle");
    return mlir::failure();
  }
  return success();
}

} // namespace

bool VPTOSchedulingBudget::consume(uint64_t amount) {
  if (amount > limit - std::min(used, limit)) {
    used = limit;
    exceeded = true;
    return false;
  }
  used += amount;
  return true;
}

FailureOr<VPTOScheduleResult>
VPTOScheduler::schedule(VPTOScheduleFailure &failure) const {
  uint64_t nodeCount = dag.getUnits().size();
  uint64_t edgeCount = dag.getEdges().size();
  if (nodeCount > limits.maxNodes) {
    failure = {VPTOScheduleFailureKind::Budget,
               "nodes",
               nodeCount,
               limits.maxNodes,
               {}};
    return mlir::failure();
  }
  if (edgeCount > limits.maxEdges) {
    failure = {VPTOScheduleFailureKind::Budget,
               "edges",
               edgeCount,
               limits.maxEdges,
               {}};
    return mlir::failure();
  }

  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (!model.getSchedClass(unit->getOperation()).known) {
      setFailure(failure, VPTOScheduleFailureKind::InvalidModel,
                 "region contains an unknown scheduling class");
      return mlir::failure();
    }
  }

  VPTOSchedBoundary boundary(dag, model, VPTOSchedDirection::Top);
  VPTOScheduleResult result;
  result.entries.reserve(dag.getUnits().size());

  while (!boundary.isComplete()) {
    bool pressureDrivenIdle = false;
    SmallVector<VPTOSchedCandidate> candidates;
    while (true) {
      if (boundary.getAvailable().empty()) {
        FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
        if (failed(advanced)) {
          setWorkBudgetFailure(failure, budget);
          return mlir::failure();
        }
        if (!*advanced) {
          setFailure(failure, VPTOScheduleFailureKind::Scheduling,
                     "no candidate or pending dependency event remains");
          return mlir::failure();
        }
      }

      FailureOr<SmallVector<VPTOSchedCandidate>> builtCandidates =
          buildCandidates(boundary, budget, failure);
      if (failed(builtCandidates)) {
        return mlir::failure();
      }
      candidates = std::move(*builtCandidates);
      FailureOr<bool> advanced =
          advanceForPressure(boundary, candidates, budget, failure);
      if (failed(advanced)) {
        return mlir::failure();
      }
      if (!*advanced) {
        break;
      }
      pressureDrivenIdle = true;
    }

    VPTOScheduleContext context{model, dag, boundary.getDirection(),
                                boundary.getCurrentCycle(),
                                boundary.getPressureTracker().getCurrent()};
    std::string detail;
    FailureOr<VPTOSchedDecision> decision =
        strategy.pickCandidate(context, candidates, detail);
    if (failed(decision)) {
      setFailure(failure, VPTOScheduleFailureKind::InvalidModel, detail);
      return mlir::failure();
    }
    bool selectsCandidate =
        llvm::any_of(candidates, [&](const VPTOSchedCandidate &candidate) {
          return candidate.unit == decision->unit;
        });
    bool matchesContext = decision->direction == context.direction &&
                          decision->issueCycle == context.issueCycle;
    if (!selectsCandidate || !matchesContext || decision->reason.empty()) {
      setFailure(failure, VPTOScheduleFailureKind::Scheduling,
                 "strategy returned an invalid scheduling decision");
      return mlir::failure();
    }
    if (failed(boundary.commit(*decision->unit, decision->issueCycle, budget,
                               detail))) {
      if (budget.hasExceeded()) {
        setWorkBudgetFailure(failure, budget);
        return mlir::failure();
      }
      setFailure(failure, VPTOScheduleFailureKind::Scheduling, detail);
      return mlir::failure();
    }
    result.entries.push_back({decision->unit, decision->direction,
                              decision->issueCycle, decision->reason,
                              pressureDrivenIdle});
  }

  ArrayRef<int64_t> peakPressure = boundary.getPressureTracker().getPeak();
  result.peakPressure.assign(peakPressure.begin(), peakPressure.end());
  return result;
}

LogicalResult mlir::pto::verifyVPTOScheduleResult(
    const VPTOSchedDAG &dag, const VPTOScheduleResult &result,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  const VPTOSchedRegion &region = dag.getRegion();
  if (!budget.consume(region.operations.size())) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  if (!regionIsStillContiguous(region)) {
    setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
               "region ownership or boundaries changed before apply");
    return mlir::failure();
  }
  size_t resultSize = result.entries.size();
  size_t unitCount = dag.getUnits().size();
  if (resultSize != unitCount) {
    setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
               "schedule is not a complete region permutation");
    return mlir::failure();
  }

  DenseMap<const VPTOSUnit *, unsigned> positions;
  for (auto [index, entry] : llvm::enumerate(result.entries)) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    bool belongsToRegion =
        entry.unit && entry.unit->getOperation()->getBlock() == region.block;
    bool inserted =
        entry.unit && positions.try_emplace(entry.unit, index).second;
    if (!belongsToRegion || !inserted) {
      setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                 "schedule contains a null, foreign, or duplicate node");
      return mlir::failure();
    }
  }
  for (const std::unique_ptr<VPTOSUnit> &unit : dag.getUnits()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    if (!positions.count(unit.get())) {
      setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                 "schedule omits a region node");
      return mlir::failure();
    }
  }
  for (const std::unique_ptr<VPTOSchedEdge> &edge : dag.getEdges()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    bool isMust = edge->isMust();
    unsigned predecessorPosition = positions.lookup(edge->getPredecessor());
    unsigned successorPosition = positions.lookup(edge->getSuccessor());
    if (isMust && predecessorPosition >= successorPosition) {
      setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                 "schedule violates a Must dependency");
      return mlir::failure();
    }
  }
  for (const VPTOScheduleEntry &entry : result.entries) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    for (Value operand : entry.unit->getOperation()->getOperands()) {
      if (!budget.consume()) {
        setWorkBudgetFailure(failure, budget);
        return mlir::failure();
      }
      Operation *definingOp = operand.getDefiningOp();
      VPTOSUnit *definingUnit = definingOp ? dag.lookup(definingOp) : nullptr;
      if (definingUnit &&
          positions.lookup(definingUnit) >= positions.lookup(entry.unit)) {
        setFailure(failure, VPTOScheduleFailureKind::SemanticVerification,
                   "schedule violates SSA dominance");
        return mlir::failure();
      }
    }
  }
  return success();
}

LogicalResult mlir::pto::replayVPTOScheduleResult(
    const VPTOSchedModel &model, const VPTOSchedDAG &dag,
    const VPTOScheduleResult &result, VPTOSchedulingBudget &budget,
    VPTOScheduleFailure &failure) {
  VPTOSchedBoundary boundary(dag, model, VPTOSchedDirection::Top);
  for (const VPTOScheduleEntry &entry : result.entries) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    if (failed(
            replayPressureDrivenIdle(entry, boundary, budget, failure))) {
      return mlir::failure();
    }
    if (boundary.getAvailable().empty()) {
      FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
      if (failed(advanced)) {
        setWorkBudgetFailure(failure, budget);
        return mlir::failure();
      }
      if (!*advanced) {
        setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
                   "replay has no candidate or pending event");
        return mlir::failure();
      }
    }
    bool isCandidate = boundary.isAvailable(entry.unit);
    bool hasRecordedDirection = entry.direction == boundary.getDirection();
    bool hasRecordedCycle = entry.issueCycle == boundary.getCurrentCycle();
    if (!isCandidate || !hasRecordedDirection || !hasRecordedCycle) {
      setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
                 "recorded node is not dependency-ready in its direction and "
                 "issue cycle");
      return mlir::failure();
    }
    std::string detail;
    if (failed(
            boundary.commit(*entry.unit, entry.issueCycle, budget, detail))) {
      if (budget.hasExceeded()) {
        setWorkBudgetFailure(failure, budget);
        return mlir::failure();
      }
      setFailure(failure, VPTOScheduleFailureKind::ModelReplay, detail);
      return mlir::failure();
    }
  }
  bool scheduleComplete = boundary.isComplete();
  bool peakMatches =
      llvm::equal(boundary.getPressureTracker().getPeak(), result.peakPressure);
  if (!scheduleComplete || !peakMatches) {
    setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
               "fresh replay does not reproduce the schedule summary");
    return mlir::failure();
  }
  return success();
}

LogicalResult mlir::pto::applyVPTOScheduleResult(
    const VPTOSchedDAG &dag, const VPTOScheduleResult &result,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  if (!budget.consume(result.entries.size())) {
    setWorkBudgetFailure(failure, budget);
    return mlir::failure();
  }
  const VPTOSchedRegion &region = dag.getRegion();
  for (const VPTOScheduleEntry &entry : result.entries) {
    if (region.followingBoundary) {
      entry.unit->getOperation()->moveBefore(region.followingBoundary);
    } else {
      entry.unit->getOperation()->moveBefore(region.block, region.block->end());
    }
  }
  return success();
}

StringRef
mlir::pto::stringifyVPTOScheduleFailureKind(VPTOScheduleFailureKind kind) {
  switch (kind) {
  case VPTOScheduleFailureKind::None:
    return "none";
  case VPTOScheduleFailureKind::Budget:
    return "budget";
  case VPTOScheduleFailureKind::InvalidModel:
    return "invalid-model";
  case VPTOScheduleFailureKind::Scheduling:
    return "scheduling";
  case VPTOScheduleFailureKind::SemanticVerification:
    return "semantic-verification";
  case VPTOScheduleFailureKind::ModelReplay:
    return "model-replay";
  case VPTOScheduleFailureKind::Apply:
    return "apply";
  }
  llvm_unreachable("unknown VPTO schedule failure kind");
}
