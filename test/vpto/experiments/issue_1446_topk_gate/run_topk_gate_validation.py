#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the issue #1446 TopK fixture through the repository SIM runner."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np


def _require_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    path = Path(value).resolve()
    if not path.exists():
        raise RuntimeError(f"{name} does not exist: {path}")
    return path


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _normalize_generated_ptodsl(source: str) -> str:
    """Remove redundant guards emitted inside an already equivalent guard."""
    output: list[str] = []
    unindent_next_mte = False
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("__cond_") and " = pto.const(0," in stripped:
            continue
        if stripped.startswith("__cond_") and " = _tl_wrap_surface_value(" in stripped:
            continue
        if stripped.startswith("if bool(__cond_"):
            unindent_next_mte = True
            continue
        if unindent_next_mte and stripped.startswith(
            ("pto.mte_gm_ub(", "pto.mte_ub_gm(")
        ):
            line = line[2:]
            unindent_next_mte = False
        output.append(line)
    return "".join(output).replace(
        'pto.ptr(pto.si32, "ub")', 'pto.ptr(pto.i32, "ub")'
    )


def _mark_generated_pto_vector_kernel(pto_path: Path) -> None:
    source = pto_path.read_text(encoding="utf-8")
    marker = 'module attributes {pto.backend = "vpto", pto.target_arch = "a5"}'
    replacement = (
        'module attributes {pto.backend = "vpto", '
        'pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"}'
    )
    if marker not in source:
        raise RuntimeError("generated PTO is missing the expected VPTO child module")
    pto_path.write_text(source.replace(marker, replacement, 1), encoding="utf-8")


def _install_fixture_frontend_compat(dump_pto: Path) -> None:
    from pto_compile_patch import install_pto_compile_patch
    from tilelang.jit.adapter import libgen

    install_pto_compile_patch()
    compile_ptodsl = libgen.LibraryGenerator._compile_ptodsl_source_to_pto

    def compile_normalized(ptodsl_source, kernel_name, src_path, out_path):
        result = compile_ptodsl(
            _normalize_generated_ptodsl(ptodsl_source),
            kernel_name,
            src_path,
            out_path,
        )
        _mark_generated_pto_vector_kernel(Path(out_path))
        dump_pto.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, dump_pto)
        return result

    libgen.LibraryGenerator._compile_ptodsl_source_to_pto = staticmethod(
        compile_normalized
    )


def _scores(pattern: str, seed: int, tokens: int, experts: int) -> np.ndarray:
    if pattern == "random":
        return _ORIGINAL_RANDOM_STATE(seed).randn(tokens, experts).astype(np.float32)
    if pattern == "equal":
        return np.ones((tokens, experts), dtype=np.float32)

    indices = np.arange(experts, dtype=np.float32)
    if pattern == "increasing":
        row = indices
    elif pattern == "decreasing":
        row = -indices
    elif pattern == "cross_chunk_ties":
        row = -indices
        tie_indices = sorted(
            {
                min(experts - 1, 70),
                min(experts - 1, experts // 2 + 17),
                experts - 1,
            }
        )
        row[tie_indices] = np.float32(experts + 1)
    else:
        raise ValueError(f"unknown TOPK_GATE_PATTERN: {pattern}")
    return np.broadcast_to(row, (tokens, experts)).copy()


_ORIGINAL_RANDOM_STATE = np.random.RandomState


def main() -> int:
    fixture_root = _require_path("TOPK_GATE_FIXTURE_ROOT")
    fixture_dir = fixture_root / "examples" / "ascend" / "vmi" / "topk_gate"
    if not fixture_dir.is_dir():
        raise RuntimeError(f"TopK fixture directory does not exist: {fixture_dir}")
    sys.path.insert(0, str(fixture_dir))

    tokens = _positive_int("TOPK_GATE_TOKENS", 4)
    experts = _positive_int("TOPK_GATE_EXPERTS", 768)
    topk = _positive_int("TOPK_GATE_K", 9)
    sms = _positive_int("TOPK_GATE_SMS", 1)
    token_tile = _positive_int("TOPK_GATE_TOKEN_TILE", 2)
    seed = int(os.environ.get("TOPK_GATE_SEED", "0"))
    pattern = os.environ.get("TOPK_GATE_PATTERN", "random")
    so_path = _require_path("TOPK_GATE_SO_PARENT") / "kernel.so"
    dump_pto = Path(os.environ.get("TOPK_GATE_DUMP_PTO", "topk_gate.pto")).resolve()

    from run_topk_gate_msprof import run_case

    class PatternRandomState:
        def __init__(self, requested_seed: int):
            self.seed = requested_seed

        def randn(self, requested_tokens: int, requested_experts: int) -> np.ndarray:
            return _scores(
                pattern,
                self.seed,
                requested_tokens,
                requested_experts,
            )

    np.random.RandomState = PatternRandomState  # type: ignore[misc]
    if not so_path.is_file():
        _install_fixture_frontend_compat(dump_pto)

    print(
        f"issue1446-input pattern={pattern} seed={seed} tokens={tokens} "
        f"experts={experts} topk={topk} tile={token_tile} sms={sms}"
    )
    run_case(
        kernel="vmi_w128",
        tokens=tokens,
        experts=experts,
        topk=topk,
        sms=sms,
        so_path=so_path,
        token_tile=token_tile,
        schedule="normal",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
