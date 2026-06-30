// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOPlanBmuLayout.cpp -----------------------------------------------===//
//
// Step 2 of the BMU memory-planning pipeline (BMU design §4.6). Runs after
// pto-classify-buffers, before pto-plan-memory.
//
// For each H-class multi-buffer `memref.alloc` it emits a runtime BMU
// allocation and rewrites the alloc into a multi-address `pto.pointer_cast`
// whose operands are `arith.addi %base, constK` (mirroring AllocToPointerCast,
// but with the SSA base from `pto.bmu_alloc` instead of constant addresses):
//
//   %cnt  = arith.constant <slice_count> : index
//   %base = pto.bmu_alloc <scope>, <pipe>, segm 0, count %cnt : index -> i64
//   %o0 = arith.constant 0 : i64        %a0 = arith.addi %base, %o0
//   %o1 = arith.constant <slotBytes>    %a1 = arith.addi %base, %o1   ...
//   %cast = pto.pointer_cast (%a0, ..., %aN-1) : memref<...>
//   ... (slot_marker ops untouched) ...
//   pto.bmu_free <scope>, <pipe>, base %base, count %cnt   (before each return)
//
// A single `pto.bmu_config` per buffer kind is emitted at the function entry,
// sizing segment 0 to the concurrent H slice demand. Subpath (b) (general H
// groups) and the D class are left for a later phase. No-op unless the module
// is A5 with `pto.uses_bmu = true`.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOBmu.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOPLANBMULAYOUT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#define DEBUG_TYPE "pto-plan-bmu-layout"

using namespace mlir;

namespace {

static pto::AddressSpace allocScope(memref::AllocOp alloc) {
  auto ty = cast<MemRefType>(alloc.getType());
  if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(ty.getMemorySpace()))
    return as.getAddressSpace();
  return pto::AddressSpace::GM;
}

// Byte size of one physical slot of a multi-buffer memref.alloc.
static uint64_t slotByteSize(memref::AllocOp alloc) {
  auto ty = cast<MemRefType>(alloc.getType());
  if (!ty.hasStaticShape())
    return 0;
  uint64_t elems = 1;
  for (int64_t d : ty.getShape())
    elems *= static_cast<uint64_t>(d);
  return elems * pto::getPTOStorageElemByteSize(ty.getElementType());
}

// Pull the config attr from the alloc's bind_tile user (same contract as
// AllocToPointerCast::inferBindTileConfig).
static pto::TileBufConfigAttr inferBindTileConfig(memref::AllocOp op) {
  pto::TileBufConfigAttr configAttr;
  for (Operation *user : op.getResult().getUsers()) {
    auto bind = dyn_cast<pto::BindTileOp>(user);
    if (!bind || bind.getSource() != op.getResult())
      continue;
    if (!configAttr)
      configAttr = bind.getConfigAttr();
  }
  return configAttr;
}

static std::pair<Value, Value> getDynamicValidShapeValues(memref::AllocOp op) {
  Value vRow, vCol;
  auto dynSizes = op.getDynamicSizes();
  if (dynSizes.size() >= 2) {
    vRow = dynSizes[0];
    vCol = dynSizes[1];
  } else if (dynSizes.size() == 1) {
    vCol = dynSizes[0];
  }
  return {vRow, vCol};
}

static uint64_t ceilDiv(uint64_t a, uint64_t b) {
  return b == 0 ? 0 : (a + b - 1) / b;
}

struct PTOPlanBmuLayoutPass
    : public mlir::pto::impl::PTOPlanBmuLayoutBase<PTOPlanBmuLayoutPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOPlanBmuLayoutPass)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!pto::usesBmu(mod))
      return;

    mod.walk([&](func::FuncOp func) { processFunc(func); });
  }

  void processFunc(func::FuncOp func) {
    // Collect the H-class multi-buffer allocs in this function.
    SmallVector<memref::AllocOp> hAllocs;
    func.walk([&](memref::AllocOp alloc) {
      auto cls = alloc->getAttrOfType<StringAttr>(pto::kPtoPlanClassAttrName);
      if (cls && cls.getValue() == pto::kPtoPlanClassHybrid &&
          alloc->hasAttr(pto::kPtoMultiBufferAttrName))
        hAllocs.push_back(alloc);
    });
    if (hAllocs.empty())
      return;

    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // Stage 1: per-scope concurrent slice demand. Phase 2 conservatively sums
    // all H groups in the function (ignores liveness), so segment 0 never
    // stalls. Liveness-aware sizing is a Phase 5 optimization.
    llvm::MapVector<pto::AddressSpace, uint64_t> scopeSlices;
    for (memref::AllocOp alloc : hAllocs) {
      pto::AddressSpace scope = allocScope(alloc);
      uint64_t n =
          alloc->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName)
              .getValue()
              .getZExtValue();
      uint64_t sliceBytes = pto::bmuSliceBytes(scope);
      scopeSlices[scope] += ceilDiv(n * slotByteSize(alloc), sliceBytes);
    }

    // Emit one bmu_config per buffer kind at the function entry, before any
    // bmu_alloc. Segment 0 holds the whole demand; segments 1-3 are empty.
    Block &entry = func.getBody().front();
    rewriter.setInsertionPointToStart(&entry);
    for (auto &kv : scopeSlices) {
      auto bufferAttr = pto::AddressSpaceAttr::get(ctx, kv.first);
      uint32_t tail = static_cast<uint32_t>(kv.second);
      rewriter.create<pto::BmuConfigOp>(func.getLoc(), bufferAttr, tail, tail,
                                        tail, tail);
    }

    // Stage 3a: materialize each H alloc.
    auto pipeAttr = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE2);
    auto i64Ty = rewriter.getI64Type();
    for (memref::AllocOp alloc : hAllocs) {
      pto::AddressSpace scope = allocScope(alloc);
      auto bufferAttr = pto::AddressSpaceAttr::get(ctx, scope);
      uint64_t n =
          alloc->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName)
              .getValue()
              .getZExtValue();
      uint64_t slotBytes = slotByteSize(alloc);
      uint64_t sliceCount = ceilDiv(n * slotBytes, pto::bmuSliceBytes(scope));
      Location loc = alloc.getLoc();

      rewriter.setInsertionPoint(alloc);
      Value cnt = rewriter.create<arith::ConstantIndexOp>(
          loc, static_cast<int64_t>(sliceCount));
      Value base = rewriter.create<pto::BmuAllocOp>(
          loc, i64Ty, pipeAttr, bufferAttr, /*segm=*/0u, cnt);

      SmallVector<Value> addrs;
      addrs.reserve(n);
      for (uint64_t k = 0; k < n; ++k) {
        Value off = rewriter.create<arith::ConstantIntOp>(
            loc, static_cast<int64_t>(k * slotBytes), 64);
        addrs.push_back(rewriter.create<arith::AddIOp>(loc, base, off));
      }

      auto [vRow, vCol] = getDynamicValidShapeValues(alloc);
      auto cast = rewriter.create<pto::PointerCastOp>(
          loc, alloc.getType(), ValueRange(addrs), vRow ? vRow : Value(),
          vCol ? vCol : Value(), inferBindTileConfig(alloc));
      rewriter.replaceOp(alloc, cast.getResult());

      // Free before every function return. base / cnt dominate the returns
      // because alloc_multi_tile allocs live at function scope.
      func.walk([&](func::ReturnOp ret) {
        rewriter.setInsertionPoint(ret);
        rewriter.create<pto::BmuFreeOp>(ret.getLoc(), pipeAttr, bufferAttr,
                                        base, cnt);
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOPlanBmuLayoutPass() {
  return std::make_unique<PTOPlanBmuLayoutPass>();
}
