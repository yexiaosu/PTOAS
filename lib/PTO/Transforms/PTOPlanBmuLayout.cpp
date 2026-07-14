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
// A single `pto.bmu_config` per buffer kind is emitted at the function entry.
// Its four segment tails come from the segmentation strategy
// (docs/designs/BMU-segmentation-strategy-design.md): each `(class, sliceCount)`
// partition gets its own bump-pointer segment (up to 4; the overflow bin-merges
// into seg3), so same-size allocs stay uniform-grained and the short-lived D
// class is isolated from the long-lived H slots. No-op unless the module is A5
// with `pto.uses_bmu = true`.
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <tuple>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOPLANBMULAYOUT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#define DEBUG_TYPE "pto-plan-bmu-layout"

using namespace mlir;

namespace {

static pto::AddressSpace getAllocScope(memref::AllocOp alloc) {
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

static uint64_t alignUp(uint64_t v, uint64_t a) {
  return a == 0 ? v : ((v + a - 1) / a) * a;
}

// Conservative intra-group member alignment (bytes). 512B covers the largest
// fractal/tile alignment seen on the BMU scopes (L0C fractal, 16x16xf16 s_frac).
static constexpr uint64_t kHGroupMemberAlignBytes = 512;

// §4.7 intra-group layout subroutine: given the members of one H group (all
// co-living by construction — the group is exactly the set of buffers that live
// and die together), assign each a byte offset inside a single BMU allocation
// and return the group's total byte size via `groupBytes`.
//
// Because the members are co-living they can never overlap, so a bump-pointer
// walk in allocation order is a correct layout; for equal-size members this
// naturally yields the ping/pong offsets the design calls for. Inplace reuse of
// dead holes (delegating to the full MemPlan allocator) is a later refinement;
// it only tightens packing and never changes correctness.
static SmallVector<uint64_t> assignGroupOffsets(ArrayRef<memref::AllocOp> members,
                                                uint64_t &groupBytes) {
  SmallVector<uint64_t> offsets;
  offsets.reserve(members.size());
  uint64_t cursor = 0;
  for (memref::AllocOp m : members) {
    cursor = alignUp(cursor, kHGroupMemberAlignBytes);
    offsets.push_back(cursor);
    cursor += slotByteSize(m);
  }
  groupBytes = cursor;
  return offsets;
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

// The nearest ancestor region that re-executes per loop iteration: an scf.for
// body or an scf.while `after` region. Null if `op` is not inside any loop.
static Region *enclosingLoopBody(Operation *op) {
  for (Region *r = op->getParentRegion(); r;) {
    Operation *p = r->getParentOp();
    if (!p)
      return nullptr;
    if (isa<scf::ForOp, scf::WhileOp>(p))
      return r;
    r = p->getParentRegion();
  }
  return nullptr;
}

// §4.6 (a-1) shape match: the clean explicit-multi-buffer ping-pong, where the
// alloc's single `bind_tile` user is consumed only by `slot_marker`s:
//   %alloc = memref.alloc {multi_buffer=N, H}
//   %bt    = bind_tile %alloc ...          (alloc's only user)
//   ... slot_marker %bt[%k] ...            (bt's only users)
// Fills `bindOut` / `smsOut` and returns true on a match, else false.
static bool collectPingPongShape(memref::AllocOp alloc,
                                 pto::BindTileOp &bindOut,
                                 SmallVectorImpl<pto::SlotMarkerOp> &smsOut) {
  if (!alloc->hasOneUse())
    return false;
  auto bind = dyn_cast<pto::BindTileOp>(*alloc->getUsers().begin());
  if (!bind || bind.getSource() != alloc.getResult())
    return false;
  SmallVector<pto::SlotMarkerOp> sms;
  for (Operation *u : bind->getUsers()) {
    auto sm = dyn_cast<pto::SlotMarkerOp>(u);
    if (!sm || sm.getSource() != bind.getResult())
      return false;
    sms.push_back(sm);
  }
  if (sms.empty())
    return false;
  bindOut = bind;
  smsOut.assign(sms.begin(), sms.end());
  return true;
}

// §4.6 (a-1) precondition: the slot's lifetime is closed inside one loop
// iteration — every transitive user of `sm` stays inside `loopBody` and none is
// the loop terminator (which would carry the slot value across iterations).
static bool slotStaysInIteration(pto::SlotMarkerOp sm, Region *loopBody) {
  Operation *term = loopBody->back().getTerminator();
  SmallVector<Operation *> work(sm->getUsers().begin(), sm->getUsers().end());
  llvm::DenseSet<Operation *> seen;
  while (!work.empty()) {
    Operation *op = work.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    if (!regionContains(loopBody, op) || op == term)
      return false;
    for (Operation *u : op->getUsers())
      work.push_back(u);
  }
  return true;
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
    // Collect the three materializable categories:
    //   * multi-buffer H allocs           -> subpath (a), multi-address cast
    //   * general H groups (h_group_id)   -> subpath (b), one base per group
    //   * D-class allocs                  -> stage 2, pure dynamic alloc/free
    SmallVector<memref::AllocOp> mbHAllocs;
    llvm::MapVector<int32_t, SmallVector<memref::AllocOp>> hGroups;
    SmallVector<memref::AllocOp> dAllocs;
    func.walk([&](memref::AllocOp alloc) {
      auto cls = alloc->getAttrOfType<StringAttr>(pto::kPtoPlanClassAttrName);
      if (!cls)
        return;
      StringRef c = cls.getValue();
      if (c == pto::kPtoPlanClassHybrid) {
        if (alloc->hasAttr(pto::kPtoMultiBufferAttrName)) {
          mbHAllocs.push_back(alloc);
        } else if (auto gid = alloc->getAttrOfType<IntegerAttr>(
                       pto::kPtoHGroupIdAttrName)) {
          hGroups[static_cast<int32_t>(gid.getInt())].push_back(alloc);
        }
      } else if (c == pto::kPtoPlanClassDynamic) {
        dAllocs.push_back(alloc);
      }
    });
    if (mbHAllocs.empty() && hGroups.empty() && dAllocs.empty())
      return;

    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // §4.6 (a-1): decide which multi-buffer H allocs are materialized
    // per-advance (loop-internal alloc/free, gets the bump-pointer implicit
    // reuse sync) vs whole-ring (a-2, one function-scope alloc). A multi-buffer
    // qualifies when its slots are used through the clean ping-pong shape inside
    // one loop and every slot's lifetime closes within the iteration.
    struct PerAdvanceInfo {
      pto::BindTileOp bind;
      SmallVector<pto::SlotMarkerOp> sms;
      Region *loopBody;
    };
    llvm::DenseMap<Operation *, PerAdvanceInfo> perAdvance;
    for (memref::AllocOp alloc : mbHAllocs) {
      pto::BindTileOp bind;
      SmallVector<pto::SlotMarkerOp> sms;
      if (!collectPingPongShape(alloc, bind, sms))
        continue;
      // Per-advance materializes one INDEPENDENT bmu_alloc per slot_marker (see
      // stage 3a). That is sound only when the multi-buffer is touched through a
      // single rotating slot per iteration. Two or more slot_markers means either
      //   * a cross-iteration prefetch (distinct indices: write slot[(i+1)%N] now,
      //     read it next iteration), or
      //   * aliasing views of one slot (same index used by separate get's),
      // both of which rely on the physical slot persisting/aliasing -- exactly
      // what per-slot_marker allocation severs (the consume would read a fresh,
      // never-written buffer). Fall back to whole-ring (a-2), which keeps the N
      // slots persistent for the whole loop. See test bmu_multi_tile_h_inloop_
      // prefetch.pto and design doc ptoas-multi-buffer-explicit-design.md §7.2.
      if (sms.size() != 1)
        continue;
      pto::SlotMarkerOp sm = sms.front();
      Region *loopBody = enclosingLoopBody(sm);
      if (!loopBody || !loopBody->hasOneBlock())
        continue;
      if (!slotStaysInIteration(sm, loopBody))
        continue;
      perAdvance[alloc] = {bind, std::move(sms), loopBody};
    }

    // Stage 1 (segmentation strategy §4 carve-up). Instead of summing all
    // dynamic demand into segment 0, split each BMU scope into up to 4
    // independent bump-pointer segments — one per `(class, sliceCount)`
    // partition. Uniform per-segment granularity removes fragment-stall / rule-6
    // deadlock risk, and putting D (short-lived scratch) in different segments
    // from H (long-lived slots) stops D churn from pushing H's bump pointer.
    //
    // Capacity per partition is the conservative sum of its members' concurrent
    // slice demand (no liveness yet — a later optimization). Per-advance
    // multi-buffers need N * perSlotSlices (N = max concurrent slot count:
    // up to N pipeline iterations overlap, each holding one slot); whole-ring
    // packs the N slots into one contiguous allocation.
    struct PartEntry {
      char planClass;        // 'H' (long-lived slot) or 'D' (scratch)
      uint64_t slicePerSlot; // slices one slot occupies (partition key)
      uint64_t capSlices;    // total concurrent slices held = slicePerSlot * slotCnt
      uint64_t slotCnt;      // concurrent slot count
      uint64_t memberCnt;    // #members (sort tiebreak)
    };
    llvm::MapVector<pto::AddressSpace, SmallVector<PartEntry>> scopeParts;
    // (scope, cls, slot) -> index into scopeParts[scope], to accumulate members.
    std::map<std::tuple<int, char, uint64_t>, size_t> partIndex;
    auto addMember = [&](pto::AddressSpace scope, char cls,
                         uint64_t slicePerSlot, uint64_t slotCnt) {
      if (slicePerSlot == 0)
        return;
      uint64_t cap = slicePerSlot * slotCnt;
      auto key = std::make_tuple(static_cast<int>(scope), cls, slicePerSlot);
      SmallVector<PartEntry> &vec = scopeParts[scope];
      auto it = partIndex.find(key);
      if (it == partIndex.end()) {
        partIndex[key] = vec.size();
        vec.push_back({cls, slicePerSlot, cap, slotCnt, 1});
      } else {
        PartEntry &e = vec[it->second];
        e.capSlices += cap;
        e.slotCnt += slotCnt;
        e.memberCnt += 1;
      }
    };

    for (memref::AllocOp alloc : mbHAllocs) {
      pto::AddressSpace scope = getAllocScope(alloc);
      uint64_t n =
          alloc->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName)
              .getValue()
              .getZExtValue();
      uint64_t slotBytes = slotByteSize(alloc);
      uint64_t sliceBytes = pto::bmuSliceBytes(scope);
      if (perAdvance.count(alloc)) {
        uint64_t perSlot = ceilDiv(slotBytes, sliceBytes);
        addMember(scope, 'H', perSlot, n);
      } else {
        uint64_t ringSlices = ceilDiv(n * slotBytes, sliceBytes);
        addMember(scope, 'H', ringSlices, 1);
      }
    }
    for (auto &kv : hGroups) {
      pto::AddressSpace scope = getAllocScope(kv.second.front());
      uint64_t groupBytes = 0;
      (void)assignGroupOffsets(kv.second, groupBytes);
      uint64_t groupSlices = ceilDiv(groupBytes, pto::bmuSliceBytes(scope));
      addMember(scope, 'H', groupSlices, 1);
    }
    for (memref::AllocOp alloc : dAllocs) {
      pto::AddressSpace scope = getAllocScope(alloc);
      uint64_t slotSlices =
          ceilDiv(slotByteSize(alloc), pto::bmuSliceBytes(scope));
      addMember(scope, 'D', slotSlices, 1);
    }

    // Per-scope: sort partitions (§4 Step C), assign <=4 segments, compute
    // tails (Step E) and record `(scope, cls, slot) -> segment`. `segmFor`
    // below reproduces the same key at materialization time.
    struct SegInfo {
      uint32_t segm;
      bool isBin;
      uint64_t binSlice;
    };
    std::map<std::tuple<int, char, uint64_t>, SegInfo> segmentOf;

    // S-first: plan-memory already ran and planned S allocs with liveness-based
    // reuse. It exported the precise tail_seg3 per scope as func attrs
    // (pto.bmu_tail_<scope>). BMU dynamic segments carve within [0, tail_seg3).
    Block &entry = func.getBody().front();
    rewriter.setInsertionPointToStart(&entry);
    for (auto &kv : scopeParts) {
      pto::AddressSpace scope = kv.first;
      SmallVector<PartEntry> parts = kv.second;
      uint64_t totalSlices = pto::bmuTotalSlices(scope);
      // Read the tail_seg3 that plan-memory computed from exact S peak usage.
      std::string tailAttrName =
          "pto.bmu_tail_" + std::string(pto::stringifyAddressSpace(scope));
      uint64_t bmuAvailable = totalSlices; // default: no S in this scope
      if (auto tailAttr = func->getAttrOfType<IntegerAttr>(tailAttrName))
        bmuAvailable = static_cast<uint64_t>(tailAttr.getInt());

      // Step C priority: H before D, then larger cap, then more churn, then
      // slot size (last key only for deterministic ordering).
      llvm::stable_sort(parts, [](const PartEntry &a, const PartEntry &b) {
        if (a.planClass != b.planClass)
          return a.planClass == 'H';
        if (a.capSlices != b.capSlices)
          return a.capSlices > b.capSlices;
        if (a.memberCnt != b.memberCnt)
          return a.memberCnt > b.memberCnt;
        return a.slicePerSlot > b.slicePerSlot;
      });

      // Step D: <=4 partitions map 1:1 to segments; the overflow bin-merges
      // into seg3 with a common `binSlice` granularity (its members round their
      // alloc up to binSlice, so seg3 stays uniform at the cost of internal
      // fragmentation). A merged slot occupies binSlice slices, so seg3 needs
      // `sum(slotCnt) * binSlice` — not `sum(capSlices)` (which under-provisions
      // when the merged slot sizes differ).
      bool bin = parts.size() > 4;
      uint64_t binSlice = 0;
      if (bin)
        for (size_t i = 3; i < parts.size(); ++i)
          binSlice = std::max(binSlice, parts[i].slicePerSlot);

      uint64_t segCap[4] = {0, 0, 0, 0};
      SmallVector<std::tuple<char, uint64_t, SegInfo>> records;
      records.reserve(parts.size());
      for (size_t i = 0; i < parts.size(); ++i) {
        const PartEntry &p = parts[i];
        if (!bin || i < 3) {
          segCap[i] += p.capSlices;
          records.push_back(
              {p.planClass, p.slicePerSlot, SegInfo{static_cast<uint32_t>(i), false, 0}});
        } else {
          segCap[3] += p.slotCnt * binSlice;
          records.push_back({p.planClass, p.slicePerSlot, SegInfo{3u, true, binSlice}});
        }
      }

      uint64_t t0 = segCap[0];
      uint64_t t1 = t0 + segCap[1];
      uint64_t t2 = t1 + segCap[2];
      uint64_t t3 = t2 + segCap[3];

      // S-first feasibility: BMU segments must fit in the remainder after
      // reserving space for S. bmuAvailable = totalSlices - sReservedSlices.
      if (totalSlices != 0 && t3 > bmuAvailable) {
        func.emitError()
            << "pto-plan-bmu-layout: BMU dynamic segments for "
            << pto::stringifyAddressSpace(scope) << " need " << t3
            << " slices but only " << bmuAvailable
            << " available (tail_seg3 from plan-memory S planning)";
        signalPassFailure();
        return;
      }

      for (auto &rec : records)
        segmentOf[std::make_tuple(static_cast<int>(scope), std::get<0>(rec),
                                  std::get<1>(rec))] = std::get<2>(rec);

      auto bufferAttr = pto::AddressSpaceAttr::get(ctx, scope);
      rewriter.create<pto::BmuConfigOp>(
          func.getLoc(), bufferAttr, static_cast<uint32_t>(t0),
          static_cast<uint32_t>(t1), static_cast<uint32_t>(t2),
          static_cast<uint32_t>(t3));
    }

    // Look up the segment (and effective slice count, bin-rounded) for an alloc
    // by reproducing its partition key. Falls back to seg0 if unseen.
    auto segmFor = [&](pto::AddressSpace scope, char cls,
                       uint64_t naturalSlot) -> std::pair<uint32_t, uint64_t> {
      auto it = segmentOf.find(
          std::make_tuple(static_cast<int>(scope), cls, naturalSlot));
      if (it == segmentOf.end())
        return {0u, naturalSlot};
      const SegInfo &si = it->second;
      return {si.segm, si.isBin ? si.binSlice : naturalSlot};
    };

    auto i64Ty = rewriter.getI64Type();

    // Alloc rides the producer pipe, free the last-consumer pipe (§5.4). They
    // differ, so the producer->consumer data dependency is still carried by
    // InsertSync's cross-pipe set/wait_flag (BMU does no cross-pipe data sync —
    // §4.8(a)); the free only drains its own consumer pipe.
    auto allocPipeAttr = [&](pto::AddressSpace s) {
      return pto::PipeAttr::get(ctx, pto::bmuAllocPipeFor(s));
    };
    auto freePipeAttr = [&](pto::AddressSpace s) {
      return pto::PipeAttr::get(ctx, pto::bmuFreePipeFor(s));
    };

    // Emit a bmu_alloc at `alloc`'s site and a matching bmu_free. Function-scope
    // allocs free before every return; nested allocs free before the enclosing
    // region's terminator (where base is still in scope).
    auto emitAllocAndFree = [&](memref::AllocOp alloc, uint64_t sliceCount,
                                pto::AddressSpaceAttr bufferAttr,
                                uint32_t segm) -> Value {
      pto::AddressSpace scope = bufferAttr.getAddressSpace();
      Location loc = alloc.getLoc();
      rewriter.setInsertionPoint(alloc);
      Value cnt = rewriter.create<arith::ConstantIndexOp>(
          loc, static_cast<int64_t>(sliceCount));
      Value base = rewriter.create<pto::BmuAllocOp>(
          loc, i64Ty, allocPipeAttr(scope), bufferAttr, segm, cnt);
      // Place the free so `base` dominates it. A function-scope alloc frees
      // before every return; an alloc nested in an scf region (e.g. a
      // branch-local D buffer, whose lifetime does not cross the branch) frees
      // before that region's terminator, where the base is still in scope.
      if (alloc->getParentRegion() == &func.getBody()) {
        func.walk([&](func::ReturnOp ret) {
          rewriter.setInsertionPoint(ret);
          rewriter.create<pto::BmuFreeOp>(ret.getLoc(), freePipeAttr(scope),
                                          bufferAttr, base, cnt);
        });
      } else {
        Operation *term = alloc->getBlock()->getTerminator();
        rewriter.setInsertionPoint(term);
        rewriter.create<pto::BmuFreeOp>(term->getLoc(), freePipeAttr(scope),
                                        bufferAttr, base, cnt);
      }
      return base;
    };

    // Stage 3a: multi-buffer H allocs.
    for (memref::AllocOp alloc : mbHAllocs) {
      pto::AddressSpace scope = getAllocScope(alloc);
      auto bufferAttr = pto::AddressSpaceAttr::get(ctx, scope);
      uint64_t n =
          alloc->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName)
              .getValue()
              .getZExtValue();
      uint64_t slotBytes = slotByteSize(alloc);

      // §4.6 (a-1) per-advance: replace each in-loop slot_marker with its own
      // bmu_alloc/free so consecutive iterations acquire distinct slots from the
      // hardware bump pointer and the reuse WAR is absorbed by the alloc stall.
      auto paIt = perAdvance.find(alloc);
      if (paIt != perAdvance.end()) {
        PerAdvanceInfo &info = paIt->second;
        uint64_t perSlot = ceilDiv(slotBytes, pto::bmuSliceBytes(scope));
        auto [segm, effSlot] = segmFor(scope, 'H', perSlot);
        Attribute cfg = inferBindTileConfig(alloc);
        Operation *term = info.loopBody->back().getTerminator();
        for (pto::SlotMarkerOp sm : info.sms) {
          Location loc = sm.getLoc();
          rewriter.setInsertionPoint(sm);
          Value cnt = rewriter.create<arith::ConstantIndexOp>(
              loc, static_cast<int64_t>(effSlot));
          auto bmuAllocOp = rewriter.create<pto::BmuAllocOp>(
              loc, i64Ty, allocPipeAttr(scope), bufferAttr, segm, cnt);
          // Carry the multi-buffer depth so InsertSync's GetEventIdNum can
          // recover N from a single-address per-advance pointer_cast.
          bmuAllocOp->setAttr(pto::kPtoMultiBufferAttrName,
                              rewriter.getI32IntegerAttr(static_cast<int32_t>(n)));
          Value base = bmuAllocOp.getResult();
          auto pc = rewriter.create<pto::PointerCastOp>(
              loc, alloc.getType(), ValueRange(base), Value(), Value(), cfg);
          // Reproduce the slot's result type by cloning the func-scope bind_tile
          // with its source rewired to this iteration's cast.
          IRMapping map;
          map.map(info.bind.getSource(), pc.getResult());
          Operation *newBind = rewriter.clone(*info.bind.getOperation(), map);
          // Keep slot_marker alive for sync analysis: it carries the slot index
          // %k that InsertSync needs for dyn event-id derivation. Rewire its
          // source to the cloned bind; ResolveBufferSelect will naturally remove
          // it later (single-address pointer_cast → identity pass-through).
          sm.getSourceMutable().assign(newBind->getResult(0));
          rewriter.setInsertionPoint(term);
          rewriter.create<pto::BmuFreeOp>(term->getLoc(), freePipeAttr(scope),
                                          bufferAttr, base, cnt);
        }
        rewriter.eraseOp(info.bind);
        rewriter.eraseOp(alloc);
        continue;
      }

      // §4.6 (a-2) whole-ring fallback: one base + N slot addresses.
      uint64_t sliceCount = ceilDiv(n * slotBytes, pto::bmuSliceBytes(scope));
      auto [segm, effCount] = segmFor(scope, 'H', sliceCount);
      Value base = emitAllocAndFree(alloc, effCount, bufferAttr, segm);

      Location loc = alloc.getLoc();
      rewriter.setInsertionPointAfter(base.getDefiningOp());
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
    }

    // Stage 3b: general H groups -> one group base + per-member static offset.
    mlir::DominanceInfo dom(func);
    for (auto &kv : hGroups) {
      SmallVector<memref::AllocOp> &members = kv.second;
      pto::AddressSpace scope = getAllocScope(members.front());
      auto bufferAttr = pto::AddressSpaceAttr::get(ctx, scope);
      uint64_t groupBytes = 0;
      SmallVector<uint64_t> offsets = assignGroupOffsets(members, groupBytes);
      uint64_t sliceCount = ceilDiv(groupBytes, pto::bmuSliceBytes(scope));
      auto [segm, effCount] = segmFor(scope, 'H', sliceCount);
      // The shared base is emitted once and every member reads `base + offset`, so
      // the anchor op must DOMINATE all members. Blindly anchoring at walk-order
      // front is ill-formed when members live in incomparable regions (e.g. one in
      // an scf.if branch and another after it): base would not dominate them, and
      // emitAllocAndFree's region-terminator free could even precede a member.
      // Pick a member that dominates all the others; if none exists, bail loudly
      // instead of emitting invalid IR.
      memref::AllocOp anchor;
      for (memref::AllocOp cand : members) {
        if (llvm::all_of(members, [&](memref::AllocOp m) {
              return m == cand ||
                     dom.dominates(cand.getOperation(), m.getOperation());
            })) {
          anchor = cand;
          break;
        }
      }
      if (!anchor) {
        members.front()->emitError(
            "pto-plan-bmu-layout: H group members have no common dominator; "
            "shared-base BMU materialization is unsupported");
        signalPassFailure();
        return;
      }
      Value base = emitAllocAndFree(anchor, effCount, bufferAttr, segm);
      for (auto [member, offset] : llvm::zip(members, offsets)) {
        Location loc = member.getLoc();
        rewriter.setInsertionPoint(member);
        Value off = rewriter.create<arith::ConstantIntOp>(
            loc, static_cast<int64_t>(offset), 64);
        Value addr = rewriter.create<arith::AddIOp>(loc, base, off);
        auto [vRow, vCol] = getDynamicValidShapeValues(member);
        auto cast = rewriter.create<pto::PointerCastOp>(
            loc, member.getType(), ValueRange(addr), vRow ? vRow : Value(),
            vCol ? vCol : Value(), inferBindTileConfig(member));
        rewriter.replaceOp(member, cast.getResult());
      }
    }

    // Stage 2: D-class allocs -> pure dynamic single-address alloc/free.
    for (memref::AllocOp alloc : dAllocs) {
      pto::AddressSpace scope = getAllocScope(alloc);
      auto bufferAttr = pto::AddressSpaceAttr::get(ctx, scope);
      uint64_t sliceCount =
          ceilDiv(slotByteSize(alloc), pto::bmuSliceBytes(scope));
      auto [segm, effCount] = segmFor(scope, 'D', sliceCount);
      Value base = emitAllocAndFree(alloc, effCount, bufferAttr, segm);

      Location loc = alloc.getLoc();
      rewriter.setInsertionPointAfter(base.getDefiningOp());
      auto [vRow, vCol] = getDynamicValidShapeValues(alloc);
      auto cast = rewriter.create<pto::PointerCastOp>(
          loc, alloc.getType(), ValueRange(base), vRow ? vRow : Value(),
          vCol ? vCol : Value(), inferBindTileConfig(alloc));
      rewriter.replaceOp(alloc, cast.getResult());
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOPlanBmuLayoutPass() {
  return std::make_unique<PTOPlanBmuLayoutPass>();
}
