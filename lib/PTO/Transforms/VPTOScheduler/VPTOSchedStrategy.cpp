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
  int64_t excessGrowthCost = 0;
  int64_t projectedExcessCost = 0;
  int64_t pressureDeltaCost = 0;
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

static FailureOr<RankedCandidate>
rankCandidate(const VPTOScheduleContext &context,
              const VPTOSchedCandidate &candidate, std::string &detail) {
  if (!candidate.unit || candidate.direction != context.direction ||
      candidate.issueCycle != context.issueCycle) {
    detail = "candidate does not match the current scheduling context";
    return failure();
  }

  ArrayRef<VPTORegPressureSet> pressureSets = context.model.getPressureSets();
  bool hasCompletePressureResult =
      context.currentPressure.size() == pressureSets.size() &&
      candidate.pressure.delta.size() == pressureSets.size() &&
      candidate.pressure.projected.size() == pressureSets.size() &&
      candidate.pressure.projectedExcess.size() == pressureSets.size();
  if (!hasCompletePressureResult) {
    detail = "candidate pressure does not match target pressure sets";
    return failure();
  }

  RankedCandidate rank;
  rank.candidate = &candidate;
  for (auto [index, pressureSet] : llvm::enumerate(pressureSets)) {
    if (pressureSet.weight < 0 || pressureSet.spillCost < 0 ||
        context.currentPressure[index] < 0 ||
        candidate.pressure.projectedExcess[index] < 0) {
      detail =
          "pressure set or candidate contains an invalid scoring parameter";
      return failure();
    }
    int64_t currentExcess = 0;
    int64_t projectedExcess = 0;
    if (pressureSet.limit) {
      currentExcess = std::max<int64_t>(0, context.currentPressure[index] -
                                               *pressureSet.limit);
      projectedExcess = candidate.pressure.projectedExcess[index];
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
  }
  return rank;
}

static bool isBetterCandidate(const RankedCandidate &lhs,
                              const RankedCandidate &rhs) {
  if (lhs.excessGrowthCost != rhs.excessGrowthCost) {
    return lhs.excessGrowthCost < rhs.excessGrowthCost;
  }
  if (lhs.projectedExcessCost != rhs.projectedExcessCost) {
    return lhs.projectedExcessCost < rhs.projectedExcessCost;
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
                                   const RankedCandidate &runnerUp) {
  if (selected.excessGrowthCost != runnerUp.excessGrowthCost) {
    return "lower-excess-growth";
  }
  if (selected.projectedExcessCost != runnerUp.projectedExcessCost) {
    return "lower-projected-excess";
  }
  if (selected.candidate->criticalPath != runnerUp.candidate->criticalPath) {
    return "longer-critical-path";
  }
  if (selected.pressureDeltaCost != runnerUp.pressureDeltaCost) {
    return "lower-pressure-delta";
  }
  if (selected.candidate->originalIndex != runnerUp.candidate->originalIndex) {
    return "earlier-original-order";
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
  for (const VPTOSchedCandidate &candidate : candidates) {
    FailureOr<RankedCandidate> rank = rankCandidate(context, candidate, detail);
    if (failed(rank)) {
      return failure();
    }
    ranks.push_back(*rank);
  }

  const RankedCandidate *selected = &ranks.front();
  for (const RankedCandidate &rank : llvm::drop_begin(ranks)) {
    if (isBetterCandidate(rank, *selected)) {
      selected = &rank;
    }
  }

  const RankedCandidate *runnerUp = nullptr;
  for (const RankedCandidate &rank : ranks) {
    if (&rank == selected) {
      continue;
    }
    if (!runnerUp || isBetterCandidate(rank, *runnerUp)) {
      runnerUp = &rank;
    }
  }

  StringRef reason = runnerUp ? getDecisionReason(*selected, *runnerUp)
                              : StringRef("only-candidate");
  const VPTOSchedCandidate &candidate = *selected->candidate;
  return VPTOSchedDecision{candidate.unit, candidate.direction,
                           candidate.issueCycle, reason.str()};
}

const VPTOSchedStrategy &mlir::pto::getDefaultVPTOSchedStrategy() {
  static const VPTODefaultSchedStrategy strategy;
  return strategy;
}
