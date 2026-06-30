// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOBmu.h - Shared constants/helpers for BMU integration -*- C++ -*-===//
//
// Shared names and helpers for the BMU (Buffer Management Unit) memory
// planning path. See docs/designs/BMU-integration-design.md.
//
//   * `kPtoPlanClassAttrName` ("S"/"D"/"H") is written on each memref.alloc by
//     `pto-classify-buffers` to select the planning mode (static / dynamic /
//     hybrid). PlanMemory only plans the "S" allocs; "H"/"D" allocs are
//     materialized by `pto-plan-bmu-layout` into BMU ops.
//   * `kPtoHGroupIdAttrName` groups the memref.allocs that share a BMU
//     allocation (an H group).
//   * `kPtoMultiBufPlacementAttrName` ("auto"/"static"/"bmu") is propagated by
//     PTOViewToMemref from the `multi_tile_buf` placement field down onto the
//     lowered memref.alloc, so the classifier (which runs on memref) can see
//     the user's explicit placement choice.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_IR_PTOBMU_H
#define PTO_IR_PTOBMU_H

// Pulls in the generated `pto::AddressSpace` enum (via its include guard).
#include "PTO/IR/PTO.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace pto {

/// Per-alloc planning class written by `pto-classify-buffers`.
inline constexpr llvm::StringLiteral kPtoPlanClassAttrName = "pto.plan_class";
inline constexpr llvm::StringLiteral kPtoPlanClassStatic = "S";
inline constexpr llvm::StringLiteral kPtoPlanClassDynamic = "D";
inline constexpr llvm::StringLiteral kPtoPlanClassHybrid = "H";

/// Group id (i32) shared by the memref.allocs of one H group.
inline constexpr llvm::StringLiteral kPtoHGroupIdAttrName = "pto.h_group_id";

/// Placement choice propagated from the `multi_tile_buf` type onto the
/// lowered memref.alloc ("auto" / "static" / "bmu").
inline constexpr llvm::StringLiteral kPtoMultiBufPlacementAttrName =
    "pto.multi_buf_placement";
inline constexpr llvm::StringLiteral kPtoMultiBufPlacementAuto = "auto";
inline constexpr llvm::StringLiteral kPtoMultiBufPlacementStatic = "static";
inline constexpr llvm::StringLiteral kPtoMultiBufPlacementBmu = "bmu";

/// BMU slice size (bytes) per buffer kind (BMU design §2.1):
///   UB(VEC) 8KB, L1(MAT) 8KB, L0A(LEFT)/L0B(RIGHT) 4KB, L0C(ACC) 16KB,
///   BIAS(BT) 256B, SCALING(FB) one 32-entry table treated as 256B granule.
inline uint64_t bmuSliceBytes(AddressSpace scope) {
  switch (scope) {
  case AddressSpace::VEC:
    return 8 * 1024;
  case AddressSpace::MAT:
    return 8 * 1024;
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
    return 4 * 1024;
  case AddressSpace::ACC:
    return 16 * 1024;
  case AddressSpace::BIAS:
    return 256;
  case AddressSpace::SCALING:
    return 256;
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return 0;
  }
  return 0;
}

/// Approximate A5 per-scope local-memory byte capacity, used by the
/// classifier's `placement=auto` threshold. Mirrors the kA5 MemSpec in
/// PTOPlanMemory (values there are in bits; converted to bytes here). This is
/// a heuristic threshold only — exact carve-up happens in pto-plan-bmu-layout.
inline uint64_t a5ScopeStaticCapacityBytes(AddressSpace scope) {
  switch (scope) {
  case AddressSpace::VEC:
    return 2031616 / 8;
  case AddressSpace::MAT:
    return 4194304 / 8;
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
    return 524288 / 8;
  case AddressSpace::ACC:
    return 2097152 / 8;
  case AddressSpace::BIAS:
    return 524288 / 8;
  case AddressSpace::SCALING:
    return 2031616 / 8;
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return 0;
  }
  return 0;
}

} // namespace pto
} // namespace mlir

#endif // PTO_IR_PTOBMU_H
