#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate a control and a single-mask scheduling ablation from emitted IR."""

import argparse
from pathlib import Path
import re
import shutil


def sink_boundary_mask(source):
    """Move the unique b32 boundary predicate immediately before its loop use."""
    lines = source.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines)
               if "pto.vcmps" in line and "%c196_i32" in line and '"lt"' in line]
    if len(matches) != 1:
        raise ValueError("Expected exactly one b32 boundary predicate")
    position = matches[0]
    definition = lines[position]
    indent = definition[:len(definition) - len(definition.lstrip())]
    name = definition.strip().split(" = ", 1)[0]
    destinations = [index for index in range(position + 1, len(lines))
                    if lines[index].startswith(indent + "scf.for ")]
    if not destinations:
        raise ValueError("Boundary predicate has no following loop")
    destination = destinations[0]
    uses = re.compile(re.escape(name) + r"(?![\w])")
    if any(uses.search(line) for line in lines[position + 1:destination]):
        raise ValueError("Cannot sink predicate across a use")
    lines.insert(destination, definition)
    del lines[position]
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve().read_text(encoding="utf-8")
    fixture = Path(__file__).resolve().parents[1] / "cases/kernels/issue-1506-vec-misched-minimal"
    variants = {"control": source, "sink-mask": sink_boundary_mask(source)}
    for name, text in variants.items():
        case = args.output.resolve() / name
        case.mkdir(parents=True, exist_ok=False)
        (case / "kernel.pto").write_text(text, encoding="utf-8")
        for filename in ("main.cpp", "launch.cpp", "golden.py", "compare.py"):
            shutil.copyfile(fixture / filename, case / filename)


if __name__ == "__main__":
    main()
