// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOClassifyBuffers.cpp ---------------------------------------------===//
//
// Step 1 of the BMU memory-planning pipeline (BMU design §4.5).
//
// Annotates each local-memory `memref.alloc` with a planning class
// (`pto.plan_class` = "S" / "D" / "H") and, for H allocs, an
// `pto.h_group_id`. PlanMemory plans only the "S" allocs; "H"/"D" allocs are
// later materialized into BMU ops by `pto-plan-bmu-layout`.
//
// Rules (BMU design §4.5 table), by priority:
//   * rule 0: A2/A3, or A5 without `pto.uses_bmu`, => no-op (all stays S).
//   * rule 1: explicit `multi_tile_buf` placement (kStatic→S, kBmu→H), seen
//     here as the `pto.multi_buf_placement` attribute propagated by
//     PTOViewToMemref.
//   * rule 2/3: multi-buffer `placement=auto` capacity decision — H when
//     N×slot_size overflows the scope's static quota, else S.
//   * rule 5: a local buffer whose whole lifetime stays inside one scf.if
//     branch => D (see isBranchLocalToIf).
//   * rule 8: default => S (left unannotated).
//
// Deferred rules and why:
//   * rule 4 (concurrent-bytes dry-run => D): needs a scope-wide liveness
//     dry-run; not yet wired.
//   * rule 6 (pypto multibuffer-unroll pair): the design keeps these S for now
//     (phase 2-3), which is exactly the default here — no reclassification.
//   * rule 7 (scf.for iter_args implicit H group): the general-H materialization
//     it targets already exists in pto-plan-bmu-layout (driven by
//     `pto.h_group_id`), but the loop-carried-memref IR shape is rejected
//     downstream ("region block arguments and loop-carried memref values are
//     unsupported here" in tile-handle materialization), so an auto-detector is
//     blocked on unrelated infrastructure outside BMU scope.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOBmu.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCLASSIFYBUFFERS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#define DEBUG_TYPE "pto-classify-buffers"

using namespace mlir;

namespace {

// Byte size of one physical slot of a multi-buffer memref.alloc.
static uint64_t slotByteSize(memref::AllocOp alloc) {
  auto ty = cast<MemRefType>(alloc.getType());
  if (!ty.hasStaticShape())
    return 0;
  uint64_t elems = 1;
  for (int64_t d : ty.getShape())
    elems *= static_cast<uint64_t>(d);
  uint64_t elemBytes = pto::getPTOStorageElemByteSize(ty.getElementType());
  return elems * elemBytes;
}

static pto::AddressSpace allocScope(memref::AllocOp alloc) {
  auto ty = cast<MemRefType>(alloc.getType());
  if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(ty.getMemorySpace()))
    return as.getAddressSpace();
  return pto::AddressSpace::GM;
}

// True if `region` is `op`'s region or a transitive parent of it.
static bool regionContains(Region *region, Operation *op) {
  for (Region *r = op->getParentRegion(); r;
       r = r->getParentOp() ? r->getParentOp()->getParentRegion() : nullptr) {
    if (r == region)
      return true;
  }
  return false;
}

// Rule 5: the alloc sits directly in one branch region of an scf.if and every
// use stays inside that same region, i.e. its lifetime never crosses the
// branch. Such a buffer is a dynamic (D) candidate — its liveness cannot
// overlap the sibling branch, so BMU can bump-allocate it on demand.
static bool isBranchLocalToIf(memref::AllocOp alloc) {
  Region *region = alloc->getParentRegion();
  Operation *parent = region ? region->getParentOp() : nullptr;
  if (!isa_and_nonnull<scf::IfOp>(parent))
    return false;
  for (Operation *user : alloc->getUsers()) {
    if (!regionContains(region, user))
      return false;
    // If the buffer feeds this branch's scf.yield it escapes through the
    // scf.if result and outlives the branch — not branch-local. Materializing
    // it as D would free it before the yield and use-after-free the result.
    if (isa<scf::YieldOp>(user) && user->getParentRegion() == region)
      return false;
  }
  return true;
}

struct PTOClassifyBuffersPass
    : public mlir::pto::impl::PTOClassifyBuffersBase<PTOClassifyBuffersPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOClassifyBuffersPass)

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Rule 0: only A5 + uses_bmu participates. Otherwise leave every alloc
    // unannotated so the static path is byte-for-byte unchanged.
    if (!pto::usesBmu(mod))
      return;

    MLIRContext *ctx = &getContext();
    auto strAttr = [&](llvm::StringRef s) { return StringAttr::get(ctx, s); };
    int32_t nextHGroupId = 0;

    mod.walk([&](memref::AllocOp alloc) {
      // Phase 2 only classifies multi-buffer allocs; everything else defaults
      // to S (left unannotated). Rules 4–8 (capacity dry-run, scf.if/for
      // groups) arrive in a later phase.
      auto nAttr =
          alloc->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName);
      if (!nAttr) {
        // Non-multi-buffer alloc. Rule 5 (scf.if branch-local -> D) is the only
        // Phase-4 rule that fires here; everything else defaults to S.
        if (pto::bmuSliceBytes(allocScope(alloc)) != 0 &&
            isBranchLocalToIf(alloc))
          alloc->setAttr(pto::kPtoPlanClassAttrName,
                         strAttr(pto::kPtoPlanClassDynamic));
        return;
      }
      uint64_t n = nAttr.getValue().getZExtValue();

      // Read the placement propagated by PTOViewToMemref (default auto).
      llvm::StringRef placement = pto::kPtoMultiBufPlacementAuto;
      if (auto p = alloc->getAttrOfType<StringAttr>(
              pto::kPtoMultiBufPlacementAttrName))
        placement = p.getValue();

      bool hybrid = false;
      if (placement == pto::kPtoMultiBufPlacementStatic) {
        hybrid = false; // rule 1: explicit static
      } else if (placement == pto::kPtoMultiBufPlacementBmu) {
        hybrid = true; // rule 1: explicit bmu
      } else {
        // rule 2/3: auto — promote to H only when the N slots would overflow
        // the scope's static quota; otherwise keep S to retain the const-addr
        // disjoint-sync optimizations.
        pto::AddressSpace scope = allocScope(alloc);
        uint64_t demand = n * slotByteSize(alloc);
        uint64_t quota = pto::a5ScopeStaticCapacityBytes(scope);
        hybrid = (quota != 0) && (demand > quota);
      }

      alloc->setAttr(pto::kPtoPlanClassAttrName,
                     strAttr(hybrid ? pto::kPtoPlanClassHybrid
                                    : pto::kPtoPlanClassStatic));
      if (hybrid) {
        // Each multi-buffer alloc is its own H group (the group is the IR
        // node — see BMU design §1.4).
        alloc->setAttr(pto::kPtoHGroupIdAttrName,
                       IntegerAttr::get(IntegerType::get(ctx, 32),
                                        nextHGroupId++));
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOClassifyBuffersPass() {
  return std::make_unique<PTOClassifyBuffersPass>();
}
