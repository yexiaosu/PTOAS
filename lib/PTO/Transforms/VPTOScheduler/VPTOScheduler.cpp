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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

static void saturatingMultiplyAdd(int64_t lhs, int64_t rhs, int64_t &total) {
  int64_t product = 0;
  int64_t updated = 0;
  bool scoreWouldOverflow = llvm::MulOverflow(lhs, rhs, product) ||
                            llvm::AddOverflow(total, product, updated);
  if (scoreWouldOverflow) {
    total = std::numeric_limits<int64_t>::max();
    return;
  }
  total = updated;
}

static bool isCriticalPressure(const VPTOSchedBoundary &boundary,
                               const VPTOSchedModel &model) {
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    if (pressureSet.limit &&
        boundary.getPressureTracker().getCurrent()[index] * 2 >=
            static_cast<int64_t>(*pressureSet.limit)) {
      return true;
    }
  }
  return false;
}

static int64_t getEvaluationExcess(
    const VPTORegPressureEvaluation &evaluation,
    ArrayRef<VPTORegPressureSet> pressureSets) {
  int64_t cost = 0;
  for (auto [index, pressureSet] : llvm::enumerate(pressureSets)) {
    saturatingMultiplyAdd(evaluation.projectedExcess[index],
                          pressureSet.spillCost, cost);
  }
  return cost;
}

static int64_t getEvaluationPressure(
    const VPTORegPressureEvaluation &evaluation,
    ArrayRef<VPTORegPressureSet> pressureSets) {
  int64_t cost = 0;
  for (auto [index, pressureSet] : llvm::enumerate(pressureSets)) {
    saturatingMultiplyAdd(evaluation.projected[index], pressureSet.weight,
                          cost);
  }
  return cost;
}

static FailureOr<VPTOSchedCandidate>
buildCandidate(VPTOSUnit &unit, const VPTOSchedBoundary &boundary,
               const VPTOSchedModel &model, VPTOSchedulingBudget &budget) {
  VPTOSchedDirection direction = boundary.getDirection();
  unsigned criticalPath =
      direction == VPTOSchedDirection::Top ? unit.getHeight() : unit.getDepth();
  VPTORegPressureEvaluation pressure =
      boundary.evaluatePressure(unit);
  VPTOSchedCandidate candidate{&unit,
                               direction,
                               boundary.getCurrentCycle(),
                               criticalPath,
                               unit.getOriginalIndex(),
                               pressure,
                               pressure.projected,
                               pressure.projected,
                               0,
                               false};
  candidate.opensPressureFrontier =
      llvm::any_of(pressure.introduced,
                   [](int64_t value) { return value > 0; }) &&
      llvm::all_of(pressure.released,
                   [](int64_t value) { return value == 0; });
  if (!isCriticalPressure(boundary, model)) {
    return candidate;
  }

  // Follow only dependency nodes made ready by this candidate. This bounded
  // simulation estimates whether the newly opened chain soon reaches a
  // live-range-closing consumer without copying the full ready queue.
  constexpr unsigned kLookaheadDepth = 8;
  VPTORegPressureTracker simulatedTracker = boundary.getPressureTracker();
  SmallVector<VPTOSUnit *, kLookaheadDepth> ready{&unit};
  DenseMap<VPTOSUnit *, unsigned> remaining;
  DenseSet<VPTOSUnit *> queued;
  queued.insert(&unit);
  ArrayRef<VPTORegPressureSet> pressureSets = model.getPressureSets();
  for (unsigned step = 0; step < kLookaheadDepth && !ready.empty(); ++step) {
    if (!budget.consume(ready.size())) {
      return failure();
    }
    auto best = llvm::min_element(ready, [&](VPTOSUnit *lhs, VPTOSUnit *rhs) {
      VPTORegPressureEvaluation lhsPressure = simulatedTracker.evaluate(*lhs);
      VPTORegPressureEvaluation rhsPressure = simulatedTracker.evaluate(*rhs);
      int64_t lhsExcess = getEvaluationExcess(lhsPressure, pressureSets);
      int64_t rhsExcess = getEvaluationExcess(rhsPressure, pressureSets);
      if (lhsExcess != rhsExcess) {
        return lhsExcess < rhsExcess;
      }
      int64_t lhsPressureCost =
          getEvaluationPressure(lhsPressure, pressureSets);
      int64_t rhsPressureCost =
          getEvaluationPressure(rhsPressure, pressureSets);
      if (lhsPressureCost != rhsPressureCost) {
        return lhsPressureCost < rhsPressureCost;
      }
      return lhs->getOriginalIndex() < rhs->getOriginalIndex();
    });
    VPTOSUnit *selected = *best;
    ready.erase(best);
    if (failed(simulatedTracker.commit(*selected))) {
      return failure();
    }
    ++candidate.lookaheadSteps;
    candidate.lookaheadEnd.assign(simulatedTracker.getCurrent().begin(),
                                  simulatedTracker.getCurrent().end());
    for (auto [index, value] : llvm::enumerate(candidate.lookaheadEnd)) {
      candidate.lookaheadPeak[index] =
          std::max(candidate.lookaheadPeak[index], value);
    }

    ArrayRef<VPTOSchedEdge *> edges =
        direction == VPTOSchedDirection::Top ? selected->getSuccessors()
                                             : selected->getPredecessors();
    for (VPTOSchedEdge *edge : edges) {
      if (!budget.consume()) {
        return failure();
      }
      if (!edge->isMust()) {
        continue;
      }
      VPTOSUnit *neighbor = direction == VPTOSchedDirection::Top
                               ? edge->getSuccessor()
                               : edge->getPredecessor();
      bool alreadyHandled =
          boundary.isScheduled(neighbor) || queued.contains(neighbor);
      if (alreadyHandled) {
        continue;
      }
      auto [position, inserted] = remaining.try_emplace(
          neighbor, boundary.getRemainingDependencyCount(*neighbor));
      (void)inserted;
      if (position->second == 0) {
        continue;
      }
      --position->second;
      if (position->second == 0) {
        ready.push_back(neighbor);
        queued.insert(neighbor);
      }
    }
  }
  return candidate;
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
buildCandidates(const VPTOSchedBoundary &boundary, const VPTOSchedModel &model,
                VPTOSchedulingBudget &budget,
                VPTOScheduleFailure &failure) {
  SmallVector<VPTOSchedCandidate> candidates;
  candidates.reserve(boundary.getAvailable().size());
  for (VPTOSUnit *unit : boundary.getAvailable()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    FailureOr<VPTOSchedCandidate> candidate =
        buildCandidate(*unit, boundary, model, budget);
    if (failed(candidate)) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    candidates.push_back(std::move(*candidate));
  }
  return candidates;
}

static bool increasesPressureRisk(
    const VPTOSchedModel &model, ArrayRef<int64_t> currentPressure,
    const VPTOSchedCandidate &candidate) {
  for (auto [index, pressureSet] : llvm::enumerate(model.getPressureSets())) {
    if (!pressureSet.limit) {
      continue;
    }
    int64_t currentExcess = std::max<int64_t>(
        0, currentPressure[index] - static_cast<int64_t>(*pressureSet.limit));
    if (candidate.pressure.projectedExcess[index] > currentExcess) {
      return true;
    }
    int64_t limit = static_cast<int64_t>(*pressureSet.limit);
    bool critical = currentPressure[index] * 2 >= limit;
    if (critical &&
        candidate.pressure.projected[index] > currentPressure[index])
      return true;
  }
  return false;
}

static bool allCandidatesIncreasePressureRisk(
    const VPTOSchedModel &model, ArrayRef<int64_t> currentPressure,
    ArrayRef<VPTOSchedCandidate> candidates) {
  return !candidates.empty() &&
         llvm::all_of(candidates, [&](const VPTOSchedCandidate &candidate) {
           return increasesPressureRisk(model, currentPressure, candidate);
         });
}

static FailureOr<bool>
advanceForPressure(VPTOSchedBoundary &boundary,
                   ArrayRef<VPTOSchedCandidate> candidates,
                   const VPTOSchedModel &model,
                   VPTOSchedulingBudget &budget,
                   VPTOScheduleFailure &failure) {
  if (!allCandidatesIncreasePressureRisk(
          model, boundary.getPressureTracker().getCurrent(), candidates)) {
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
    const VPTOSchedModel &model,
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
        buildCandidates(boundary, model, budget, failure);
    if (failed(candidates)) {
      return mlir::failure();
    }
    if (!allCandidatesIncreasePressureRisk(
            model, boundary.getPressureTracker().getCurrent(), *candidates)) {
      setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
                 "pressure idle occurred while a non-growing-risk candidate was "
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
          buildCandidates(boundary, model, budget, failure);
      if (failed(builtCandidates)) {
        return mlir::failure();
      }
      candidates = std::move(*builtCandidates);
      FailureOr<bool> advanced =
          advanceForPressure(boundary, candidates, model, budget, failure);
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
            replayPressureDrivenIdle(entry, boundary, model, budget,
                                     failure))) {
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
