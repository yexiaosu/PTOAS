#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Attach a LICM hint to one explicit LLVM loop backedge for diagnosis.

This helper deliberately rejects ambiguous or already annotated backedges;
it preserves all instructions and existing metadata in the input LLVM IR.
"""

import argparse
from pathlib import Path
import re


def add_hint(source, latch, header, disabled):
    """Add a fresh self-referential loop ID to the selected unconditional edge."""
    label = re.compile(r"^([a-zA-Z0-9_.$-]+):")
    block = None
    matches = []
    lines = source.splitlines(keepends=True)
    edge = "br label %" + header
    for index, line in enumerate(lines):
        match = label.match(line)
        if match:
            block = match.group(1)
        if block == latch and line.strip() == edge:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError("Expected one unannotated backedge in the selected latch")
    numbers = [int(value) for value in re.findall(r"^!(\d+) =", source, re.MULTILINE)]
    loop_id = max(numbers, default=-1) + 1
    hint_id = loop_id + 1
    index = matches[0]
    lines[index] = lines[index].rstrip("\r\n") + f", !llvm.loop !{loop_id}\n"
    hint = '!"llvm.licm.disable"' if disabled else '!"llvm.licm.disable", i1 false'
    metadata = f"\n!{loop_id} = distinct !{{!{loop_id}, !{hint_id}}}\n!{hint_id} = !{{{hint}}}\n"
    return "".join(lines) + metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--latch", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--value", choices=("true", "false"), default="true")
    args = parser.parse_args()
    source = args.input.resolve().read_text(encoding="utf-8")
    result = add_hint(source, args.latch, args.header, args.value == "true")
    # Refuse to overwrite the control IR or any existing diagnostic artifact.
    with args.output.resolve().open("x", encoding="utf-8") as output:
        output.write(result)


if __name__ == "__main__":
    main()
