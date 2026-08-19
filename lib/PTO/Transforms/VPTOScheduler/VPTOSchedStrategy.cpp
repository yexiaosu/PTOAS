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
  int64_t nearLimitProjectedCost = 0;
  int64_t nearLimitReleaseCredit = 0;
  int64_t pressureDeltaCost = 0;
  bool urgentCriticalPath = false;
};

struct RankingContext {
  SmallVector<bool> nearLimitPressureSets;
  unsigned longestCriticalPath = 0;
  unsigned urgentSlack = 0;
  bool hasNearLimitPressure = false;
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
         candidate.pressure.projectedExcess.size() == pressureSetCount;
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

  RankingContext rankingContext;
  rankingContext.nearLimitPressureSets.assign(pressureSets.size(), false);
  for (const VPTOSchedCandidate &candidate : candidates) {
    if (!candidate.unit || candidate.direction != context.direction ||
        candidate.issueCycle != context.issueCycle ||
        !hasCompletePressureResult(context, candidate)) {
      detail = "candidate does not match the current scheduling context";
      return failure();
    }
    const VPTOSchedClass &schedClass =
        context.model.getSchedClass(candidate.unit->getOperation());
    if (!schedClass.known) {
      detail = "candidate has an unknown scheduling class";
      return failure();
    }
    if (candidate.criticalPath > rankingContext.longestCriticalPath) {
      rankingContext.longestCriticalPath = candidate.criticalPath;
      rankingContext.urgentSlack = schedClass.writeLatency;
    } else if (candidate.criticalPath == rankingContext.longestCriticalPath) {
      rankingContext.urgentSlack =
          std::max(rankingContext.urgentSlack, schedClass.writeLatency);
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
    int64_t maxIntroduced = 0;
    for (const VPTOSchedCandidate &candidate : candidates) {
      int64_t introduced = candidate.pressure.introduced[index];
      if (introduced < 0) {
        detail = "candidate contains negative introduced pressure";
        return failure();
      }
      maxIntroduced = std::max(maxIntroduced, introduced);
    }
    int64_t headroom =
        std::max<int64_t>(0, limit - context.currentPressure[index]);
    bool nearLimit = maxIntroduced > 0 && maxIntroduced >= headroom;
    rankingContext.nearLimitPressureSets[index] = nearLimit;
    rankingContext.hasNearLimitPressure |= nearLimit;
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
  }
  unsigned criticalPathSlack =
      rankingContext.longestCriticalPath - candidate.criticalPath;
  rank.urgentCriticalPath = criticalPathSlack <= rankingContext.urgentSlack;
  return rank;
}

static bool isBetterCandidate(const RankedCandidate &lhs,
                              const RankedCandidate &rhs,
                              bool hasNearLimitPressure) {
  if (lhs.exceedsLimit != rhs.exceedsLimit) {
    return !lhs.exceedsLimit;
  }
  if (lhs.excessGrowthCost != rhs.excessGrowthCost) {
    return lhs.excessGrowthCost < rhs.excessGrowthCost;
  }
  if (lhs.projectedExcessCost != rhs.projectedExcessCost) {
    return lhs.projectedExcessCost < rhs.projectedExcessCost;
  }
  if (hasNearLimitPressure) {
    if (lhs.urgentCriticalPath != rhs.urgentCriticalPath) {
      return lhs.urgentCriticalPath;
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
                                   bool hasNearLimitPressure) {
  if (selected.exceedsLimit != runnerUp.exceedsLimit) {
    return "pressure-safe-candidate";
  }
  if (selected.excessGrowthCost != runnerUp.excessGrowthCost) {
    return "lower-excess-growth";
  }
  if (selected.projectedExcessCost != runnerUp.projectedExcessCost) {
    return "lower-projected-excess";
  }
  if (hasNearLimitPressure) {
    if (selected.urgentCriticalPath != runnerUp.urgentCriticalPath) {
      return "urgent-critical-path";
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
                          rankingContext->hasNearLimitPressure)) {
      selected = &rank;
    }
  }

  const RankedCandidate *runnerUp = nullptr;
  for (const RankedCandidate &rank : ranks) {
    if (&rank == selected) {
      continue;
    }
    if (!runnerUp || isBetterCandidate(rank, *runnerUp,
                                       rankingContext->hasNearLimitPressure)) {
      runnerUp = &rank;
    }
  }

  StringRef reason = runnerUp ? getDecisionReason(
                                    *selected, *runnerUp,
                                    rankingContext->hasNearLimitPressure)
                              : StringRef("only-candidate");
  const VPTOSchedCandidate &candidate = *selected->candidate;
  return VPTOSchedDecision{candidate.unit, candidate.direction,
                           candidate.issueCycle, reason.str()};
}

const VPTOSchedStrategy &mlir::pto::getDefaultVPTOSchedStrategy() {
  static const VPTODefaultSchedStrategy strategy;
  return strategy;
}
