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

constexpr unsigned kMaxClosureGroupNodes = 96;

struct PressureClosureGroup {
  VPTOSUnit *target = nullptr;
  std::optional<unsigned> pressureSet;
  DenseSet<VPTOSUnit *> units;
  SmallVector<int64_t, 2> peak;
  SmallVector<int64_t, 2> end;
  unsigned steps = 0;

  void clear() {
    target = nullptr;
    pressureSet.reset();
    units.clear();
    peak.clear();
    end.clear();
    steps = 0;
  }
};

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
               const VPTOSchedModel &model, VPTOSchedulingBudget &budget,
               const PressureClosureGroup *closureGroup = nullptr) {
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
  candidate.advancesPressureClosure =
      closureGroup && closureGroup->units.contains(&unit);
  if (closureGroup && closureGroup->target) {
    return candidate;
  }
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

static bool isNearPressureSet(const VPTOSchedBoundary &boundary,
                              ArrayRef<VPTORegPressureSet> pressureSets,
                              unsigned index) {
  bool invalidIndex = index >= pressureSets.size();
  bool hasNoLimit = !invalidIndex && !pressureSets[index].limit;
  if (invalidIndex || hasNoLimit) {
    return false;
  }
  int64_t current = boundary.getPressureTracker().getCurrent()[index];
  int64_t limit = static_cast<int64_t>(*pressureSets[index].limit);
  return current * 2 >= limit;
}

static std::optional<unsigned>
selectHighestPressureSet(const VPTOSchedBoundary &boundary,
                         ArrayRef<VPTORegPressureSet> pressureSets) {
  std::optional<unsigned> selected;
  ArrayRef<int64_t> current = boundary.getPressureTracker().getCurrent();
  for (unsigned index = 0; index < pressureSets.size(); ++index) {
    if (!isNearPressureSet(boundary, pressureSets, index)) {
      continue;
    }
    if (!selected) {
      selected = index;
      continue;
    }
    int64_t selectedLimit =
        static_cast<int64_t>(*pressureSets[*selected].limit);
    int64_t candidateLimit =
        static_cast<int64_t>(*pressureSets[index].limit);
    int64_t selectedExcess = std::max<int64_t>(
        0, current[*selected] - selectedLimit);
    int64_t candidateExcess =
        std::max<int64_t>(0, current[index] - candidateLimit);
    bool hasHigherExcess = candidateExcess > selectedExcess;
    bool hasHigherRatio =
        candidateExcess == selectedExcess &&
        current[index] * selectedLimit >
            current[*selected] * candidateLimit;
    if (hasHigherExcess || hasHigherRatio) {
      selected = index;
    }
  }
  return selected;
}

static bool valueContributesToPressureSet(const VPTOSchedModel &model,
                                          Value value,
                                          VPTOPressureSetID pressureSet) {
  return llvm::any_of(
      model.getPressure(value), [&](const VPTORegPressureContribution &item) {
        return item.pressureSet == pressureSet && item.units > 0;
      });
}

static LogicalResult collectPressureClosureBundles(
    const VPTOSchedBoundary &boundary, const VPTOSchedModel &model,
    const VPTOSchedDAG &dag, unsigned pressureSet,
    VPTOSchedulingBudget &budget,
    SmallVectorImpl<SmallVector<Value, 2>> &bundles) {
  constexpr unsigned kMaxClosureTargetBundles = 1;
  constexpr unsigned kMaxClosureDirectUsers = 8;
  VPTOPressureSetID pressureSetID = model.getPressureSets()[pressureSet].id;
  const VPTORegPressureTracker &tracker = boundary.getPressureTracker();
  SmallVector<Value> liveValues(tracker.getLiveValues().begin(),
                                tracker.getLiveValues().end());
  llvm::sort(liveValues, [&](Value lhs, Value rhs) {
    VPTOSUnit *lhsUnit =
        lhs.getDefiningOp() ? dag.lookup(lhs.getDefiningOp()) : nullptr;
    VPTOSUnit *rhsUnit =
        rhs.getDefiningOp() ? dag.lookup(rhs.getDefiningOp()) : nullptr;
    unsigned lhsIndex = lhsUnit ? lhsUnit->getOriginalIndex()
                                : std::numeric_limits<unsigned>::max();
    unsigned rhsIndex = rhsUnit ? rhsUnit->getOriginalIndex()
                                : std::numeric_limits<unsigned>::max();
    if (lhsIndex != rhsIndex) {
      return lhsIndex < rhsIndex;
    }
    auto lhsResult = dyn_cast<OpResult>(lhs);
    auto rhsResult = dyn_cast<OpResult>(rhs);
    if (lhsResult && rhsResult) {
      return lhsResult.getResultNumber() < rhsResult.getResultNumber();
    }
    auto lhsLiveIn = llvm::find(dag.getLiveIns(), lhs);
    auto rhsLiveIn = llvm::find(dag.getLiveIns(), rhs);
    return lhsLiveIn < rhsLiveIn;
  });

  DenseSet<Operation *> checkedDefinitions;
  for (Value value : liveValues) {
    if (!budget.consume()) {
      return failure();
    }
    if (!valueContributesToPressureSet(model, value, pressureSetID)) {
      continue;
    }
    SmallVector<Value, 2> bundle;
    Operation *definingOp = value.getDefiningOp();
    VPTOSUnit *definingUnit = definingOp ? dag.lookup(definingOp) : nullptr;
    if (definingUnit) {
      if (!checkedDefinitions.insert(definingOp).second) {
        continue;
      }
      for (Value result : definingOp->getResults()) {
        bool isLiveInPressureSet =
            tracker.isLive(result) &&
            valueContributesToPressureSet(model, result, pressureSetID);
        if (isLiveInPressureSet) {
          bundle.push_back(result);
        }
      }
    } else {
      bundle.push_back(value);
    }

    DenseSet<VPTOSUnit *> users;
    bool hasUnscheduledUser = false;
    for (Value targetValue : bundle) {
      for (Operation *user : targetValue.getUsers()) {
        if (!budget.consume()) {
          return failure();
        }
        VPTOSUnit *unit = dag.lookup(user);
        if (unit) {
          users.insert(unit);
          hasUnscheduledUser |= !boundary.isScheduled(unit);
        }
      }
    }
    bool hasTooManyUsers = users.size() > kMaxClosureDirectUsers;
    if (!hasUnscheduledUser || hasTooManyUsers) {
      continue;
    }
    bundles.push_back(std::move(bundle));
    bool reachedBundleLimit = bundles.size() == kMaxClosureTargetBundles;
    if (reachedBundleLimit) {
      break;
    }
  }
  return success();
}

static bool isBetterClosureGroup(const PressureClosureGroup &candidate,
                                 const PressureClosureGroup &selected,
                                 unsigned pressureSet) {
  if (candidate.peak[pressureSet] != selected.peak[pressureSet]) {
    return candidate.peak[pressureSet] < selected.peak[pressureSet];
  }
  if (candidate.end[pressureSet] != selected.end[pressureSet]) {
    return candidate.end[pressureSet] < selected.end[pressureSet];
  }
  if (candidate.steps != selected.steps) {
    return candidate.steps < selected.steps;
  }
  return candidate.target->getOriginalIndex() <
         selected.target->getOriginalIndex();
}

struct PressureClosureSimulation {
  SmallVector<VPTOSUnit *, 8> ready;
  DenseMap<VPTOSUnit *, unsigned> remaining;
  DenseSet<VPTOSUnit *> discovered;
  DenseSet<VPTOSUnit *> expanded;
  DenseSet<VPTOSUnit *> enqueued;
  DenseSet<VPTOSUnit *> core;
  DenseSet<VPTOSUnit *> scheduled;
};

static FailureOr<bool> discoverClosureUnit(
    VPTOSUnit &root, bool isCore, const VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, PressureClosureSimulation &simulation) {
  SmallVector<std::pair<VPTOSUnit *, bool>, 8> worklist{{&root, isCore}};
  while (!worklist.empty()) {
    if (!budget.consume()) {
      return failure();
    }
    auto [unit, coreUnit] = worklist.pop_back_val();
    if (boundary.isScheduled(unit)) {
      continue;
    }
    if (coreUnit) {
      simulation.core.insert(unit);
    }
    bool newlyDiscovered = simulation.discovered.insert(unit).second;
    if (newlyDiscovered &&
        simulation.discovered.size() > kMaxClosureGroupNodes) {
      return false;
    }
    auto found = simulation.remaining.find(unit);
    if (found == simulation.remaining.end()) {
      unsigned dependencies = boundary.getRemainingDependencyCount(*unit);
      for (VPTOSchedEdge *edge : unit->getPredecessors()) {
        if (!budget.consume()) {
          return failure();
        }
        bool dependencyAlreadySelected =
            edge->isMust() &&
            simulation.scheduled.contains(edge->getPredecessor());
        if (dependencyAlreadySelected) {
          if (dependencies == 0) {
            return failure();
          }
          --dependencies;
        }
      }
      found = simulation.remaining.try_emplace(unit, dependencies).first;
    }
    auto position = found;
    if (position->second == 0) {
      if (simulation.enqueued.insert(unit).second) {
        simulation.ready.push_back(unit);
      }
      continue;
    }
    if (!simulation.expanded.insert(unit).second) {
      continue;
    }
    for (VPTOSchedEdge *edge : unit->getPredecessors()) {
      if (!budget.consume()) {
        return failure();
      }
      if (!edge->isMust()) {
        continue;
      }
      VPTOSUnit *predecessor = edge->getPredecessor();
      bool needsPredecessor = !boundary.isScheduled(predecessor) &&
                              !simulation.discovered.contains(predecessor);
      if (needsPredecessor) {
        worklist.push_back({predecessor, false});
      }
    }
  }
  return true;
}

static FailureOr<int64_t> getLiveSupportPressure(
    const PressureClosureSimulation &simulation,
    const VPTORegPressureTracker &tracker, const VPTOSchedModel &model,
    unsigned pressureSet, VPTOSchedulingBudget &budget) {
  VPTOPressureSetID pressureSetID = model.getPressureSets()[pressureSet].id;
  int64_t pressure = 0;
  DenseSet<Value> countedValues;
  for (VPTOSUnit *unit : simulation.scheduled) {
    if (simulation.core.contains(unit)) {
      continue;
    }
    for (Value result : unit->getOperation()->getResults()) {
      if (!budget.consume()) {
        return failure();
      }
      Value representative = tracker.getPressureRepresentative(result);
      bool isLive = tracker.isLive(representative);
      bool isFirstContribution = countedValues.insert(representative).second;
      if (!isLive || !isFirstContribution) {
        continue;
      }
      for (const VPTORegPressureContribution &contribution :
           model.getPressure(representative)) {
        if (contribution.pressureSet != pressureSetID) {
          continue;
        }
        int64_t updated = 0;
        if (llvm::AddOverflow(
                pressure, static_cast<int64_t>(contribution.units), updated)) {
          return failure();
        }
        pressure = updated;
      }
    }
  }
  return pressure;
}

static FailureOr<bool> populatePressureClosureGroup(
    ArrayRef<Value> targetValues, const VPTOSchedBoundary &boundary,
    const VPTOSchedModel &model, const VPTOSchedDAG &dag,
    unsigned pressureSet, VPTOSchedulingBudget &budget,
    PressureClosureGroup &group) {
  VPTORegPressureTracker simulatedTracker = boundary.getPressureTracker();
  ArrayRef<int64_t> initialPressure = simulatedTracker.getCurrent();
  ArrayRef<VPTORegPressureSet> pressureSets = model.getPressureSets();
  int64_t start = initialPressure[pressureSet];
  int64_t limit = static_cast<int64_t>(*pressureSets[pressureSet].limit);

  PressureClosureSimulation simulation;
  for (Value targetValue : targetValues) {
    for (Operation *user : targetValue.getUsers()) {
      if (!budget.consume()) {
        return failure();
      }
      VPTOSUnit *unit = dag.lookup(user);
      if (!unit || boundary.isScheduled(unit)) {
        continue;
      }
      FailureOr<bool> discovered = discoverClosureUnit(
          *unit, true, boundary, budget, simulation);
      if (failed(discovered)) {
        return failure();
      }
      if (!*discovered) {
        return false;
      }
    }
  }

  group.clear();
  group.pressureSet = pressureSet;
  group.peak.assign(initialPressure.begin(), initialPressure.end());
  group.end.assign(initialPressure.begin(), initialPressure.end());
  for (unsigned step = 0;
       step < kMaxClosureGroupNodes && !simulation.ready.empty(); ++step) {
    if (!budget.consume(simulation.ready.size())) {
      return failure();
    }
    auto best = llvm::min_element(
        simulation.ready, [&](VPTOSUnit *lhs, VPTOSUnit *rhs) {
          VPTORegPressureEvaluation lhsEvaluation =
              simulatedTracker.evaluate(*lhs);
          VPTORegPressureEvaluation rhsEvaluation =
              simulatedTracker.evaluate(*rhs);
          int64_t lhsExcess =
              getEvaluationExcess(lhsEvaluation, pressureSets);
          int64_t rhsExcess =
              getEvaluationExcess(rhsEvaluation, pressureSets);
          if (lhsExcess != rhsExcess) {
            return lhsExcess < rhsExcess;
          }
          int64_t lhsPressure =
              getEvaluationPressure(lhsEvaluation, pressureSets);
          int64_t rhsPressure =
              getEvaluationPressure(rhsEvaluation, pressureSets);
          if (lhsPressure != rhsPressure) {
            return lhsPressure < rhsPressure;
          }
          return lhs->getOriginalIndex() < rhs->getOriginalIndex();
        });
    VPTOSUnit *selected = *best;
    simulation.ready.erase(best);
    if (failed(simulatedTracker.commit(*selected))) {
      return failure();
    }
    simulation.scheduled.insert(selected);
    group.units.insert(selected);
    group.target = selected;
    ++group.steps;
    group.end.assign(simulatedTracker.getCurrent().begin(),
                     simulatedTracker.getCurrent().end());
    for (auto [index, pressure] : llvm::enumerate(group.end)) {
      group.peak[index] = std::max(group.peak[index], pressure);
    }

    bool targetsAreDead = llvm::all_of(targetValues, [&](Value value) {
      return !simulatedTracker.isLive(value);
    });
    bool avoidsNewExcess = group.peak[pressureSet] <= std::max(start, limit);
    if (targetsAreDead && avoidsNewExcess) {
      FailureOr<int64_t> supportPressure = getLiveSupportPressure(
          simulation, simulatedTracker, model, pressureSet, budget);
      if (failed(supportPressure)) {
        return failure();
      }
      if (*supportPressure > group.end[pressureSet]) {
        return failure();
      }
      bool closesWithoutNetGrowth =
          group.end[pressureSet] - *supportPressure <= start;
      if (closesWithoutNetGrowth) {
        return true;
      }
    }

    for (VPTOSchedEdge *edge : selected->getSuccessors()) {
      if (!budget.consume()) {
        return failure();
      }
      if (!edge->isMust()) {
        continue;
      }
      VPTOSUnit *successor = edge->getSuccessor();
      if (boundary.isScheduled(successor)) {
        continue;
      }
      bool followsClosure =
          simulation.core.contains(selected) &&
          edge->getKind() == VPTOSchedEdgeKind::Data;
      bool alreadyDiscovered = simulation.discovered.contains(successor);
      if (!followsClosure && !alreadyDiscovered) {
        continue;
      }
      if (followsClosure) {
        simulation.core.insert(successor);
      }
      auto position = simulation.remaining.find(successor);
      if (position != simulation.remaining.end()) {
        if (position->second == 0) {
          continue;
        }
        --position->second;
      }
      FailureOr<bool> discovered = discoverClosureUnit(
          *successor, followsClosure, boundary, budget, simulation);
      if (failed(discovered)) {
        return failure();
      }
      if (!*discovered) {
        return false;
      }
    }
  }
  return false;
}

static LogicalResult
refreshPressureClosureGroup(PressureClosureGroup &closureGroup,
                            const VPTOSchedBoundary &boundary,
                            const VPTOSchedModel &model,
                            const VPTOSchedDAG &dag,
                            VPTOSchedulingBudget &budget) {
  ArrayRef<VPTORegPressureSet> pressureSets = model.getPressureSets();
  std::optional<unsigned> pressureSet =
      selectHighestPressureSet(boundary, pressureSets);
  bool keepsCurrentClosure =
      closureGroup.target && closureGroup.pressureSet &&
      !boundary.isScheduled(closureGroup.target) &&
      pressureSet == closureGroup.pressureSet;
  if (keepsCurrentClosure) {
    return success();
  }
  closureGroup.clear();

  if (!pressureSet) {
    return success();
  }

  SmallVector<SmallVector<Value, 2>, 2> bundles;
  LogicalResult collected = collectPressureClosureBundles(
      boundary, model, dag, *pressureSet, budget, bundles);
  if (failed(collected)) {
    return failure();
  }
  PressureClosureGroup selected;
  for (ArrayRef<Value> bundle : bundles) {
    PressureClosureGroup candidate;
    FailureOr<bool> built = populatePressureClosureGroup(
        bundle, boundary, model, dag, *pressureSet, budget, candidate);
    if (failed(built)) {
      return failure();
    }
    if (*built &&
        (!selected.target ||
         isBetterClosureGroup(candidate, selected, *pressureSet))) {
      selected = std::move(candidate);
    }
  }
  closureGroup = std::move(selected);
  return success();
}

static FailureOr<SmallVector<VPTOSchedCandidate>>
buildCandidates(const VPTOSchedBoundary &boundary, const VPTOSchedModel &model,
                VPTOSchedulingBudget &budget,
                VPTOScheduleFailure &failure,
                const PressureClosureGroup *closureGroup = nullptr) {
  SmallVector<VPTOSchedCandidate> candidates;
  candidates.reserve(boundary.getAvailable().size());
  for (VPTOSUnit *unit : boundary.getAvailable()) {
    if (!budget.consume()) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    FailureOr<VPTOSchedCandidate> candidate =
        buildCandidate(*unit, boundary, model, budget, closureGroup);
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

static LogicalResult replayMandatoryDependencyIdle(
    const VPTOScheduleEntry &entry, VPTOSchedBoundary &boundary,
    VPTOSchedulingBudget &budget, VPTOScheduleFailure &failure) {
  while (boundary.getAvailable().empty()) {
    unsigned currentCycle = boundary.getCurrentCycle();
    if (currentCycle >= entry.issueCycle) {
      break;
    }
    FailureOr<bool> advanced = boundary.advanceToNextPendingCycle(budget);
    if (failed(advanced)) {
      setWorkBudgetFailure(failure, budget);
      return mlir::failure();
    }
    if (!*advanced) {
      setFailure(failure, VPTOScheduleFailureKind::ModelReplay,
                 "replay has no pending dependency event");
      return mlir::failure();
    }
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
  PressureClosureGroup closureGroup;
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

      LogicalResult refreshed =
          refreshPressureClosureGroup(closureGroup, boundary, model, dag,
                                      budget);
      if (failed(refreshed)) {
        setWorkBudgetFailure(failure, budget);
        return mlir::failure();
      }
      FailureOr<SmallVector<VPTOSchedCandidate>> builtCandidates =
          buildCandidates(boundary, model, budget, failure, &closureGroup);
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
                                boundary.getPressureTracker().getCurrent(),
                                closureGroup.pressureSet};
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
    LogicalResult mandatoryIdle =
        replayMandatoryDependencyIdle(entry, boundary, budget, failure);
    if (failed(mandatoryIdle)) {
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
