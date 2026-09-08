# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Specialize the saved, normalized E768 fixture to its eight-chunk E512 form.

The five explicitly unrolled chunk sections follow the original TileLang fixture:
padding, index initialization, score loads, tournament updates, and winner masking.
Keep the original text and orchestration; reject unexpected section counts.
"""

import argparse
import ast
from collections import Counter
import importlib.util
import io
from pathlib import Path
import tokenize


def large_chunk(node, array):
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == array and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int) and 8 <= node.slice.value < 12)


def specialize(source):
    removed = set()
    counts = Counter()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if large_chunk(target, "index_vec"):
                removed.add(node.lineno)
                counts["index"] += 1
            elif large_chunk(target, "score_vec"):
                if isinstance(node.value, ast.Name):
                    removed.update(range(node.lineno - 2, node.lineno + 1))
                    counts["load"] += 1
                else:
                    removed.add(node.lineno)
                    counts["mask"] += 1
            elif large_chunk(node.value, "score_vec"):
                removed.update(range(node.lineno, node.lineno + 4))
                counts["tournament"] += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (not isinstance(call.func, ast.Attribute) or call.func.attr != "vstore"
                    or call.keywords or len(call.args) != 3):
                continue
            pointer = call.args[1]
            if not isinstance(pointer, ast.Call) or len(pointer.args) != 2:
                continue
            offset = pointer.args[1]
            if (isinstance(offset, ast.BinOp) and isinstance(offset.op, ast.Add)
                    and isinstance(offset.right, ast.Constant)
                    and offset.right.value in (512, 576, 640, 704)):
                removed.add(node.lineno)
                counts["padding"] += 1
    expected = {name: 4 for name in ("index", "load", "mask", "tournament", "padding")}
    if counts != expected:
        raise ValueError(f"unexpected fixture section counts: {counts}")
    reduced = "".join(line for line_number, line in enumerate(source.splitlines(keepends=True), 1)
                      if line_number not in removed)
    replacements = {"12": "8", "768": "512", "1536": "1024", "3072": "2048", "9216": "6144"}
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(reduced).readline):
        if token.type == tokenize.NUMBER and token.string in replacements:
            token = token._replace(string=replacements[token.string])
        tokens.append(token)
    result = tokenize.untokenize(tokens)
    for node in ast.walk(ast.parse(result)):
        if large_chunk(node, "score_vec") or large_chunk(node, "index_vec"):
            raise ValueError("discarded chunk remains referenced")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "e512.ptodsl.py"
    path.write_text(specialize(args.source.read_text(encoding="utf-8")), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("e512_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generated fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.topk_gate_kernel_vmi_w128_kernel.compile()
    (args.output / "e512.pto").write_text(kernel.mlir_text(), encoding="utf-8")
    print(args.output / "e512.pto")


if __name__ == "__main__":
    main()
