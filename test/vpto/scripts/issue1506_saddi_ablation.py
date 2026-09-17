#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Test-only PTOAS wrapper moving the issue-1506 inner-loop pointer increment.

Compile normally, then move only SADDI within the twelve-instruction loop.
Instruction count, loop boundaries, registers and all other object bytes remain
unchanged. Require a unique exact match against an existing validated SIM trace.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def loop_words(reference, direction):
    """Recover and validate the exact loop words from a prior instruction log."""
    size, offset = (0x54, 0x11C) if direction == "early" else (0x5A, 0x134)
    text = reference.read_text(encoding="utf-8")
    addresses = set(re.findall(rf"VF  addr: (0x[0-9a-f]+),instr_num: 0x{size:x},", text))
    if len(addresses) != 1:
        raise ValueError("Expected one reference VF address with the selected code size")
    start = int(addresses.pop(), 16) + offset
    pattern = re.compile(r"\(PC: (0x[0-9a-f]+)\).*?Binary: (0x[0-9a-f]+).*?\(ID: \d+\) (RV_\w+)")
    instructions = {}
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        pc = int(match[1], 16)
        if start <= pc < start + 48:
            value = (int(match[2], 16).to_bytes(4, "little"), match[3])
            if pc in instructions and instructions[pc] != value:
                raise ValueError("Reference instruction encoding changed at one PC")
            instructions[pc] = value
    ordered = [instructions[pc] for pc in range(start, start + 48, 4)]
    expected = ["RV_SADD", "RV_VGATHER2", "RV_VLD", "RV_VGATHER2", "RV_VCVT_F2F",
                "RV_VMULS", "RV_VCVT_F2F", "RV_VADD", "RV_VADD", "RV_VSEL", "RV_VST", "RV_SADDI"]
    if direction == "late":
        expected.insert(4, expected.pop())
    if [name for _, name in ordered] != expected:
        raise ValueError("Reference does not match the intended twelve-instruction loop")
    return [word for word, _ in ordered]


def patch_object(output, words, direction):
    """Keep an exact original and replace one fixed-size instruction sequence."""
    original = output.read_bytes()
    before = b"".join(words)
    if not original.startswith(b"\x7fELF") or original.count(before) != 1:
        raise ValueError("Expected one exact loop encoding in the ELF fat object")
    reordered = words.copy()
    source, destination = (11, 4) if direction == "early" else (4, 11)
    reordered.insert(destination, reordered.pop(source))
    after = b"".join(reordered)
    offset = original.index(before)
    changed = original[:offset] + after + original[offset + len(before):]
    with output.with_suffix(".unpatched.o").open("xb") as backup:
        backup.write(original)
    output.write_bytes(changed)
    record = {"direction": direction, "offset": offset, "before": before.hex(), "after": after.hex(),
              "sha256_before": hashlib.sha256(original).hexdigest(),
              "sha256_after": hashlib.sha256(changed).hexdigest()}
    with output.with_suffix(".ablation.json").open("x", encoding="utf-8") as report:
        report.write(json.dumps(record, indent=2) + "\n")
    print("Issue-1506 SADDI ablation: " + json.dumps(record), flush=True)


def main():
    """Run the real worktree compiler before applying the scoped ablation."""
    direction = os.environ["PTOAS_ABLATION_DIRECTION"]
    if direction not in ("early", "late"):
        raise ValueError("Direction must be early or late")
    compiler = Path(os.environ["PTOAS_REAL_BIN"]).resolve(strict=True)
    if compiler == Path(__file__).resolve():
        raise ValueError("Real compiler cannot be the diagnostic wrapper")
    reference = Path(os.environ["PTOAS_ABLATION_REFERENCE"]).resolve(strict=True)
    words = loop_words(reference, direction)
    args = sys.argv[1:]
    if args.count("-o") != 1 or args[-2] != "-o":
        raise ValueError("Expected the repository runner's final -o output argument")
    output = Path(args[-1]).resolve()
    if not output.name.endswith(".fatobj.o"):
        raise ValueError("Expected the repository runner's fat object output")
    subprocess.run([str(compiler), *args], check=True, timeout=600)
    patch_object(output, words, direction)


if __name__ == "__main__":
    main()
