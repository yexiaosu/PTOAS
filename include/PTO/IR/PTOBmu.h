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

#include "llvm/ADT/SmallVector.h"
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

/// Total number of BMU slices in each buffer kind (BMU.md CoreV920 slice
/// table). This is the hardware upper bound for a BMU_SEGM_* tail: the config
/// constraint is 0 <= tail_seg0 <= tail_seg1 <= tail_seg2 <= tail_seg3 <=
/// total_slices. Non-BMU scopes (GM/Zero) return 0.
inline uint64_t bmuTotalSlices(AddressSpace scope) {
  switch (scope) {
  case AddressSpace::VEC:
    return 48; // UB  384KB / 8KB
  case AddressSpace::MAT:
    return 64; // L1  512KB / 8KB
  case AddressSpace::LEFT:
    return 16; // L0A 64KB / 4KB
  case AddressSpace::RIGHT:
    return 16; // L0B 64KB / 4KB
  case AddressSpace::ACC:
    return 16; // L0C 256KB / 16KB
  case AddressSpace::BIAS:
    return 16; // BT  4KB / 256B
  case AddressSpace::SCALING:
    return 16; // FB  16 entries
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return 0;
  }
  return 0;
}

/// Total physical byte capacity of a buffer kind, derived from the BMU.md
/// CoreV920 slice table (total_slices * slice_size) — the single source of
/// truth for scope capacity. Non-BMU scopes (GM/Zero) return 0.
inline uint64_t bmuScopeTotalBytes(AddressSpace scope) {
  return bmuTotalSlices(scope) * bmuSliceBytes(scope);
}

/// Pipes that may carry a BMU alloc (producer) / free (consumer) for a buffer
/// kind. This is the single source of truth shared by the op verifier
/// (`verifyBmuPipeForBuffer`) and pto-plan-bmu-layout's pipe selection, so the
/// two never drift. BMU.md §4.4 (alloc / producer pipes) and §5.4 (free /
/// consumer pipes). `PIPE_S` (scalar) is a universal fallback the verifier
/// accepts in addition to these. Non-BMU scopes (GM/Zero) return {}.
inline llvm::SmallVector<PIPE, 4> bmuValidPipesFor(AddressSpace scope,
                                                   bool isAlloc) {
  if (isAlloc) {
    switch (scope) {
    case AddressSpace::VEC: // UB <- MTE2 (GM->UB) / V (compute) / F (fixpipe)
      return {PIPE::PIPE_MTE2, PIPE::PIPE_V, PIPE::PIPE_FIX};
    case AddressSpace::MAT: // L1 <- MTE2 (GM->L1)
      return {PIPE::PIPE_MTE2};
    case AddressSpace::LEFT:  // L0A <- MTE1 (L1->L0A)
    case AddressSpace::RIGHT: // L0B <- MTE1 (L1->L0B)
      return {PIPE::PIPE_MTE1};
    case AddressSpace::ACC: // L0C <- M (MAD)
      return {PIPE::PIPE_M};
    case AddressSpace::BIAS: // BT <- MTE1 (from L1) / MTE2 (from GM)
      return {PIPE::PIPE_MTE1, PIPE::PIPE_MTE2};
    case AddressSpace::SCALING: // FB <- MTE1 (L1->FB)
      return {PIPE::PIPE_MTE1};
    default:
      return {};
    }
  }
  switch (scope) {
  case AddressSpace::VEC: // UB read by V (in-place) / MTE3 (UB->GM)
    return {PIPE::PIPE_V, PIPE::PIPE_MTE3};
  case AddressSpace::MAT: // L1 read by MTE1 (L1->L0) / F (L1->FB)
    return {PIPE::PIPE_MTE1, PIPE::PIPE_FIX};
  case AddressSpace::LEFT:  // L0A read by M (MAD)
  case AddressSpace::RIGHT: // L0B read by M (MAD)
    return {PIPE::PIPE_M};
  case AddressSpace::ACC:     // L0C read by F (fixpipe)
  case AddressSpace::BIAS:    // BT  read by F (fixpipe)
  case AddressSpace::SCALING: // FB  read by F (fixpipe)
    return {PIPE::PIPE_FIX};
  default:
    return {};
  }
}

/// Canonical carrier pipe pto-plan-bmu-layout emits for a BMU alloc of `scope`
/// (the dominant producer pipe). Always a member of
/// bmuValidPipesFor(scope, /*isAlloc=*/true).
inline PIPE bmuAllocPipeFor(AddressSpace scope) {
  switch (scope) {
  case AddressSpace::VEC:     // GM->UB load
  case AddressSpace::MAT:     // GM->L1 load
  case AddressSpace::BIAS:    // bias from GM
    return PIPE::PIPE_MTE2;
  case AddressSpace::LEFT:    // L1->L0A
  case AddressSpace::RIGHT:   // L1->L0B
  case AddressSpace::SCALING: // L1->FB
    return PIPE::PIPE_MTE1;
  case AddressSpace::ACC: // MAD writes L0C
    return PIPE::PIPE_M;
  default:
    return PIPE::PIPE_S;
  }
}

/// Canonical carrier pipe pto-plan-bmu-layout emits for a BMU free of `scope`
/// (the dominant last-consumer pipe). Always a member of
/// bmuValidPipesFor(scope, /*isAlloc=*/false). Because this differs from the
/// alloc pipe, the producer->consumer data dependency is protected by
/// InsertSync's normal cross-pipe set/wait_flag, not by the free itself (BMU
/// does no cross-pipe data sync — BMU.md §2.1 / design §4.8(a)).
inline PIPE bmuFreePipeFor(AddressSpace scope) {
  switch (scope) {
  case AddressSpace::VEC: // UB->GM store drains UB
    return PIPE::PIPE_MTE3;
  case AddressSpace::MAT: // L1->L0 drains L1
    return PIPE::PIPE_MTE1;
  case AddressSpace::LEFT:  // MAD reads L0A
  case AddressSpace::RIGHT: // MAD reads L0B
    return PIPE::PIPE_M;
  case AddressSpace::ACC:     // fixpipe consumes L0C
  case AddressSpace::BIAS:    // fixpipe consumes BT
  case AddressSpace::SCALING: // fixpipe consumes FB
    return PIPE::PIPE_FIX;
  default:
    return PIPE::PIPE_S;
  }
}

} // namespace pto
} // namespace mlir

#endif // PTO_IR_PTOBMU_H
