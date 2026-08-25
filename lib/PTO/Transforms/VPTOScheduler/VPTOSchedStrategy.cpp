// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedStrategy.cpp - VPTO scheduling strategy -----------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedStrategy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct RankedCandidate {
  const VPTOSchedCandidate *candidate = nullptr;
  bool exceedsLimit = false;
  int64_t excessGrowthCost = 0;
  int64_t projectedExcessCost = 0;
  int64_t lookaheadExcessCost = 0;
  int64_t lookaheadRiskCost = 0;
  int64_t lookaheadEndCost = 0;
  int64_t highPressureProjectedCost = 0;
  int64_t highPressureReleaseCredit = 0;
  int64_t nearLimitProjectedCost = 0;
  int64_t nearLimitReleaseCredit = 0;
  int64_t pressureDeltaCost = 0;
  int64_t closureProjected = 0;
  int64_t closureReleaseCredit = 0;
  bool urgentCriticalPath = false;
  bool opensPressureFrontier = false;
  bool advancesPressureClosure = false;
};

struct RankingContext {
  SmallVector<bool> nearLimitPressureSets;
  SmallVector<bool> highPressureSets;
  unsigned longestCriticalPath = 0;
  unsigned urgentSlack = 0;
  bool hasNearLimitPressure = false;
  bool hasHighPressure = false;
};

static bool checkedMultiplyAdd(int64_t lhs, int64_t rhs, int64_t &total) {
  int64_t product = 0;
  int64_t updated = 0;
  if (llvm::MulOverflow(lhs, rhs, product)) {
    return false;
  }
  if (llvm::AddOverflow(total, product, updated)) {
    return false;
  }
  total = updated;
  return true;
}

static bool hasCompletePressureResult(const VPTOScheduleContext &context,
                                      const VPTOSchedCandidate &candidate) {
  size_t pressureSetCount = context.model.getPressureSets().size();
  return context.currentPressure.size() == pressureSetCount &&
         candidate.pressure.delta.size() == pressureSetCount &&
         candidate.pressure.released.size() == pressureSetCount &&
         candidate.pressure.introduced.size() == pressureSetCount &&
         candidate.pressure.projected.size() == pressureSetCount &&
         candidate.pressure.projectedExcess.size() == pressureSetCount &&
         candidate.lookaheadPeak.size() == pressureSetCount &&
         candidate.lookaheadEnd.size() == pressureSetCount;
}

static FailureOr<RankingContext>
buildRankingContext(const VPTOScheduleContext &context,
                    ArrayRef<VPTOSchedCandidate> candidates,
                    std::string &detail) {
  ArrayRef<VPTORegPressureSet> pressureSets = context.model.getPressureSets();
  bool pressureCountMatches =
      context.currentPressure.size() == pressureSets.size();
  if (!pressureCountMatches) {
    detail = "current pressure does not match target pressure sets";
    return failure();
  }
  if (context.closurePressureSet &&
      *context.closurePressureSet >= pressureSets.size()) {
    detail = "closure pressure set does not match the target model";
    return failure();
  }

  RankingContext rankingContext;
  rankingContext.nearLimitPressureSets.assign(pressureSets.size(), false);
  rankingContext.highPressureSets.assign(pressureSets.size(), false);
  for (const VPTOSchedCandidate &candidate : candidates) {
    if (!candidate.unit || candidate.direction != context.direction ||
        candidate.issueCycle != context.issueCycle ||
        !hasCompletePressureResult(context, candidate)) {
      detail = "candidate does not match the current scheduling context";
      return failure();
    }
    Operation *op = candidate.unit->getOperation();
    const VPTOSchedClass &schedClass = context.model.getSchedClass(op);
    VPTOSchedParameters parameters = context.model.getSchedParameters(op);
    if (!schedClass.known) {
      detail = "candidate has an unknown scheduling class";
      return failure();
    }
    if (candidate.criticalPath > rankingContext.longestCriticalPath) {
      rankingContext.longestCriticalPath = candidate.criticalPath;
      rankingContext.urgentSlack = parameters.writeLatency;
    } else if (candidate.criticalPath == rankingContext.longestCriticalPath) {
      rankingContext.urgentSlack =
          std::max(rankingContext.urgentSlack, parameters.writeLatency);
    }
  }

  for (auto [index, pressureSet] : llvm::enumerate(pressureSets)) {
    if (context.currentPressure[index] < 0) {
      detail = "current pressure contains a negative value";
      return failure();
    }
    if (!pressureSet.limit) {
      continue;
    }
    int64_t limit = static_cast<int64_t>(*pressureSet.limit);
    // Enter the pressure-critical state at half capacity. This is early
    // enough to redirect a producer-heavy frontier before the next single
    // instruction would spill, while risk bands still leave room for an
    // urgent critical-path candidate when alternatives are equally safe.
    bool nearLimit = context.currentPressure[index] * 2 >= limit;
    bool highPressure = context.currentPressure[index] * 3 >= limit * 2;
    rankingContext.nearLimitPressureSets[index] = nearLimit;
    rankingContext.highPressureSets[index] = highPressure;
    rankingContext.hasNearLimitPressure |= nearLimit;
    rankingContext.hasHighPressure |= highPressure;
  }
  return rankingContext;
}

static FailureOr<RankedCandidate>
rankCandidate(const VPTOScheduleContext &context,
              const RankingContext &rankingContext,
              const VPTOSchedCandidate &candidate, std::string &detail) {
  if (!candidate.unit || candidate.direction != context.direction ||
      candidate.issueCycle != context.issueCycle) {
    detail = "candidate does not match the current scheduling context";
    return failure();
  }

  ArrayRef<VPTORegPressureSet> pressureSets = context.model.getPressureSets();
  bool hasCompletePressure = hasCompletePressureResult(context, candidate);
  if (!hasCompletePressure) {
    detail = "candidate pressure does not match target pressure sets";
    return failure();
  }

  RankedCandidate rank;
  rank.candidate = &candidate;
  rank.opensPressureFrontier = candidate.opensPressureFrontier;
  rank.advancesPressureClosure = candidate.advancesPressureClosure;
  for (auto [index, pressureSet] : llvm::enumerate(pressureSets)) {
    if (pressureSet.weight < 0 || pressureSet.spillCost < 0 ||
        context.currentPressure[index] < 0 ||
        candidate.pressure.released[index] < 0 ||
        candidate.pressure.introduced[index] < 0 ||
        candidate.pressure.projected[index] < 0 ||
        candidate.pressure.projectedExcess[index] < 0) {
      detail =
          "pressure set or candidate contains an invalid scoring parameter";
      return failure();
    }
    int64_t expectedDelta = 0;
    int64_t expectedProjected = 0;
    if (llvm::SubOverflow(candidate.pressure.introduced[index],
                          candidate.pressure.released[index], expectedDelta) ||
        llvm::AddOverflow(context.currentPressure[index],
                          candidate.pressure.delta[index],
                          expectedProjected) ||
        expectedDelta != candidate.pressure.delta[index] ||
        expectedProjected != candidate.pressure.projected[index]) {
      detail = "candidate pressure snapshot is inconsistent or overflows";
      return failure();
    }
    int64_t currentExcess = 0;
    int64_t projectedExcess = 0;
    if (pressureSet.limit) {
      currentExcess = std::max<int64_t>(0, context.currentPressure[index] -
                                               *pressureSet.limit);
      projectedExcess = candidate.pressure.projectedExcess[index];
      int64_t expectedExcess = std::max<int64_t>(
          0, expectedProjected - static_cast<int64_t>(*pressureSet.limit));
      if (projectedExcess != expectedExcess) {
        detail = "candidate projected pressure excess is inconsistent";
        return failure();
      }
      rank.exceedsLimit |= projectedExcess > 0;
    } else if (candidate.pressure.projectedExcess[index] != 0) {
      detail = "unbounded pressure set has projected excess";
      return failure();
    }
    int64_t excessGrowth =
        std::max<int64_t>(0, projectedExcess - currentExcess);
    if (!checkedMultiplyAdd(pressureSet.spillCost, excessGrowth,
                            rank.excessGrowthCost) ||
        !checkedMultiplyAdd(pressureSet.spillCost, projectedExcess,
                            rank.projectedExcessCost) ||
        !checkedMultiplyAdd(pressureSet.weight, candidate.pressure.delta[index],
                            rank.pressureDeltaCost)) {
      detail = "candidate pressure score overflow";
      return failure();
    }
    if (rankingContext.nearLimitPressureSets[index] &&
        (!checkedMultiplyAdd(pressureSet.spillCost,
                             candidate.pressure.projected[index],
                             rank.nearLimitProjectedCost) ||
         !checkedMultiplyAdd(pressureSet.spillCost,
                             candidate.pressure.released[index],
                             rank.nearLimitReleaseCredit))) {
      detail = "candidate near-limit pressure score overflow";
      return failure();
    }
    if (rankingContext.highPressureSets[index] &&
        (!checkedMultiplyAdd(pressureSet.spillCost,
                             candidate.pressure.projected[index],
                             rank.highPressureProjectedCost) ||
         !checkedMultiplyAdd(pressureSet.spillCost,
                             candidate.pressure.released[index],
                             rank.highPressureReleaseCredit))) {
      detail = "candidate high-pressure score overflow";
      return failure();
    }
    if (rankingContext.nearLimitPressureSets[index]) {
      int64_t limit = static_cast<int64_t>(*pressureSet.limit);
      int64_t criticalThreshold = (limit + 1) / 2;
      int64_t bandWidth =
          std::max<int64_t>(1, (limit - criticalThreshold + 3) / 4);
      int64_t lookaheadPeak = candidate.lookaheadPeak[index];
      int64_t lookaheadEnd = candidate.lookaheadEnd[index];
      if (lookaheadPeak < 0 || lookaheadEnd < 0) {
        detail = "candidate contains negative lookahead pressure";
        return failure();
      }
      int64_t lookaheadExcess =
          std::max<int64_t>(0, lookaheadPeak - limit);
      int64_t lookaheadRisk =
          std::max<int64_t>(0, candidate.pressure.projected[index] -
                                   criticalThreshold) /
          bandWidth;
      if (!checkedMultiplyAdd(pressureSet.spillCost, lookaheadExcess,
                              rank.lookaheadExcessCost) ||
          !checkedMultiplyAdd(pressureSet.weight, lookaheadRisk,
                              rank.lookaheadRiskCost) ||
          !checkedMultiplyAdd(pressureSet.weight, lookaheadEnd,
                              rank.lookaheadEndCost)) {
        detail = "candidate lookahead pressure score overflow";
        return failure();
      }
    }
  }
  unsigned criticalPathSlack =
      rankingContext.longestCriticalPath - candidate.criticalPath;
  rank.urgentCriticalPath = criticalPathSlack <= rankingContext.urgentSlack;
  if (context.closurePressureSet) {
    unsigned index = *context.closurePressureSet;
    rank.closureProjected = candidate.pressure.projected[index];
    rank.closureReleaseCredit = candidate.pressure.released[index];
  }
  return rank;
}

static bool isBetterCandidate(const RankedCandidate &lhs,
                              const RankedCandidate &rhs,
                              bool hasNearLimitPressure,
                              bool hasHighPressure,
                              bool hasPressureClosure) {
  if (lhs.exceedsLimit != rhs.exceedsLimit) {
    return !lhs.exceedsLimit;
  }
  if (lhs.excessGrowthCost != rhs.excessGrowthCost) {
    return lhs.excessGrowthCost < rhs.excessGrowthCost;
  }
  if (lhs.projectedExcessCost != rhs.projectedExcessCost) {
    return lhs.projectedExcessCost < rhs.projectedExcessCost;
  }
  if (hasPressureClosure) {
    if (lhs.advancesPressureClosure != rhs.advancesPressureClosure) {
      return lhs.advancesPressureClosure;
    }
    if (lhs.closureProjected != rhs.closureProjected) {
      return lhs.closureProjected < rhs.closureProjected;
    }
    if (lhs.closureReleaseCredit != rhs.closureReleaseCredit) {
      return lhs.closureReleaseCredit > rhs.closureReleaseCredit;
    }
  }
  if (hasHighPressure) {
    if (lhs.highPressureProjectedCost != rhs.highPressureProjectedCost) {
      return lhs.highPressureProjectedCost < rhs.highPressureProjectedCost;
    }
    if (lhs.highPressureReleaseCredit != rhs.highPressureReleaseCredit) {
      return lhs.highPressureReleaseCredit > rhs.highPressureReleaseCredit;
    }
  }
  if (hasNearLimitPressure) {
    if (lhs.lookaheadExcessCost != rhs.lookaheadExcessCost) {
      return lhs.lookaheadExcessCost < rhs.lookaheadExcessCost;
    }
    if (lhs.lookaheadRiskCost != rhs.lookaheadRiskCost) {
      return lhs.lookaheadRiskCost < rhs.lookaheadRiskCost;
    }
    if (lhs.urgentCriticalPath != rhs.urgentCriticalPath) {
      return lhs.urgentCriticalPath;
    }
    if (lhs.opensPressureFrontier != rhs.opensPressureFrontier) {
      return !lhs.opensPressureFrontier;
    }
    if (lhs.lookaheadEndCost != rhs.lookaheadEndCost) {
      return lhs.lookaheadEndCost < rhs.lookaheadEndCost;
    }
    if (lhs.nearLimitProjectedCost != rhs.nearLimitProjectedCost) {
      return lhs.nearLimitProjectedCost < rhs.nearLimitProjectedCost;
    }
    if (lhs.nearLimitReleaseCredit != rhs.nearLimitReleaseCredit) {
      return lhs.nearLimitReleaseCredit > rhs.nearLimitReleaseCredit;
    }
  }
  if (lhs.candidate->criticalPath != rhs.candidate->criticalPath) {
    return lhs.candidate->criticalPath > rhs.candidate->criticalPath;
  }
  if (lhs.pressureDeltaCost != rhs.pressureDeltaCost) {
    return lhs.pressureDeltaCost < rhs.pressureDeltaCost;
  }
  return lhs.candidate->originalIndex < rhs.candidate->originalIndex;
}

static StringRef getDecisionReason(const RankedCandidate &selected,
                                   const RankedCandidate &runnerUp,
                                   bool hasNearLimitPressure,
                                   bool hasHighPressure,
                                   bool hasPressureClosure) {
  if (selected.exceedsLimit != runnerUp.exceedsLimit) {
    return "pressure-safe-candidate";
  }
  if (selected.excessGrowthCost != runnerUp.excessGrowthCost) {
    return "lower-excess-growth";
  }
  if (selected.projectedExcessCost != runnerUp.projectedExcessCost) {
    return "lower-projected-excess";
  }
  if (hasPressureClosure) {
    if (selected.advancesPressureClosure !=
        runnerUp.advancesPressureClosure) {
      return "advance-pressure-closure";
    }
    if (selected.closureProjected != runnerUp.closureProjected) {
      return "closure-pressure-preserving";
    }
    if (selected.closureReleaseCredit != runnerUp.closureReleaseCredit) {
      return "closure-live-range-closing";
    }
  }
  if (hasHighPressure) {
    if (selected.highPressureProjectedCost !=
        runnerUp.highPressureProjectedCost) {
      return "high-pressure-preserving";
    }
    if (selected.highPressureReleaseCredit !=
        runnerUp.highPressureReleaseCredit) {
      return "high-pressure-live-range-closing";
    }
  }
  if (hasNearLimitPressure) {
    if (selected.lookaheadExcessCost != runnerUp.lookaheadExcessCost) {
      return "bounded-lookahead-avoids-excess";
    }
    if (selected.lookaheadRiskCost != runnerUp.lookaheadRiskCost) {
      return "bounded-lookahead-lower-risk";
    }
    if (selected.urgentCriticalPath != runnerUp.urgentCriticalPath) {
      return "urgent-critical-path";
    }
    if (selected.opensPressureFrontier != runnerUp.opensPressureFrontier) {
      return "continue-open-pressure-frontier";
    }
    if (selected.lookaheadEndCost != runnerUp.lookaheadEndCost) {
      return "bounded-lookahead-lower-ending-pressure";
    }
    if (selected.nearLimitProjectedCost != runnerUp.nearLimitProjectedCost) {
      if (selected.nearLimitReleaseCredit >
          runnerUp.nearLimitReleaseCredit) {
        return "near-limit-live-range-closing";
      }
      return "near-limit-pressure-preserving";
    }
    if (selected.nearLimitReleaseCredit !=
        runnerUp.nearLimitReleaseCredit) {
      return "near-limit-live-range-closing";
    }
  }
  if (selected.candidate->criticalPath != runnerUp.candidate->criticalPath) {
    return "longer-critical-path";
  }
  if (selected.pressureDeltaCost != runnerUp.pressureDeltaCost) {
    return "lower-pressure-delta";
  }
  if (selected.candidate->originalIndex != runnerUp.candidate->originalIndex) {
    return "deterministic-tie-break";
  }
  return "stable-candidate-order";
}

} // namespace

FailureOr<VPTOSchedDecision>
VPTODefaultSchedStrategy::pickCandidate(const VPTOScheduleContext &context,
                                        ArrayRef<VPTOSchedCandidate> candidates,
                                        std::string &detail) const {
  if (candidates.empty()) {
    detail = "strategy received no candidates";
    return failure();
  }

  SmallVector<RankedCandidate> ranks;
  ranks.reserve(candidates.size());
  FailureOr<RankingContext> rankingContext =
      buildRankingContext(context, candidates, detail);
  if (failed(rankingContext)) {
    return failure();
  }
  for (const VPTOSchedCandidate &candidate : candidates) {
    FailureOr<RankedCandidate> rank =
        rankCandidate(context, *rankingContext, candidate, detail);
    if (failed(rank)) {
      return failure();
    }
    ranks.push_back(*rank);
  }

  const RankedCandidate *selected = &ranks.front();
  for (const RankedCandidate &rank : llvm::drop_begin(ranks)) {
    if (isBetterCandidate(rank, *selected,
                          rankingContext->hasNearLimitPressure,
                          rankingContext->hasHighPressure,
                          context.closurePressureSet.has_value())) {
      selected = &rank;
    }
  }

  const RankedCandidate *runnerUp = nullptr;
  for (const RankedCandidate &rank : ranks) {
    if (&rank == selected) {
      continue;
    }
    if (!runnerUp || isBetterCandidate(rank, *runnerUp,
                                       rankingContext->hasNearLimitPressure,
                                       rankingContext->hasHighPressure,
                                       context.closurePressureSet.has_value())) {
      runnerUp = &rank;
    }
  }

  StringRef reason = runnerUp ? getDecisionReason(
                                    *selected, *runnerUp,
                                    rankingContext->hasNearLimitPressure,
                                    rankingContext->hasHighPressure,
                                    context.closurePressureSet.has_value())
                              : StringRef("only-candidate");
  const VPTOSchedCandidate &candidate = *selected->candidate;
  return VPTOSchedDecision{candidate.unit, candidate.direction,
                           candidate.issueCycle, reason.str()};
}

const VPTOSchedStrategy &mlir::pto::getDefaultVPTOSchedStrategy() {
  static const VPTODefaultSchedStrategy strategy;
  return strategy;
}
