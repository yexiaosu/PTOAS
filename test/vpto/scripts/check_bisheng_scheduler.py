#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise real Bisheng scheduling through the strict CA-model validation runner.

The callback unit tests do not establish toolchain compatibility. This test
requires a real CANN installation and simulator, checks the actual auto decision,
and fixes PTOAS scheduling while comparing Bisheng off/on/auto. It also measures
full PTOAS fatobj compilation separately from host compilation and simulation.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
MODES = ("off", "on", "auto")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="micro-op/dsa-sfu/vmula")
    parser.add_argument("--work-space", required=True, type=Path)
    parser.add_argument("--ptoas", default=os.environ.get("PTOAS_BIN", "ptoas"))
    parser.add_argument("--ptoas-scheduler", choices=("on", "off"), default="on")
    parser.add_argument("--expect", choices=("zero", "compared", "unavailable", "recovered", "any"), default="any")
    parser.add_argument("--repetitions", type=int, choices=range(1, 11), default=1)
    return parser.parse_args()


def auto_decision(log):
    if "on compilation failed; selected off." in log:
        if "recovered Bisheng on compilation failure:" not in log:
            raise RuntimeError("Recovered compilation did not retain the on failure reason")
        return "recovered"
    match = re.search(r"SIMD VF stack bytes on=(\d+), off=(\d+); selected (on|off)\.", log)
    if match:
        on_bytes, off_bytes = int(match[1]), int(match[2])
        expected = "off" if off_bytes < on_bytes else "on"
        if on_bytes == 0 or match[3] != expected:
            raise RuntimeError("Auto decision disagrees with the reported stack sizes")
        return "compared"
    if "SIMD VF stack bytes on=0; selected on without retry." in log:
        return "zero"
    if "Warning: Bisheng scheduler auto: SIMD VF stack report unavailable; keeping on" in log:
        return "unavailable"
    raise RuntimeError("No supported auto decision found in real compiler output")


def run_logged(command, log_path, env):
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    return time.perf_counter() - start


def check_auto(log, expected):
    decision = auto_decision(log)
    if expected != "any" and decision != expected:
        raise RuntimeError(f"Expected {expected}, got {decision}")
    return decision


def compile_sample(options, mode, run_dir, kernel, env):
    flags = ["--pto-arch=a5", "--pto-backend=vpto", "--vpto-scheduler-remat=false",
             f"--vpto-scheduler={options.ptoas_scheduler}", f"--bisheng-vec-misched={mode}"]
    output = run_dir / "timed.fatobj.o"
    log_path = run_dir / "compile.log"
    elapsed = run_logged([options.ptoas, *flags, str(kernel), "-o", str(output)], log_path, env)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Compiler did not produce a nonempty fatobj")
    if mode == "auto":
        check_auto(log_path.read_text(encoding="utf-8"), options.expect)
    return flags, elapsed


def validate_sample(options, mode, run_dir, kernel, env):
    flags, compile_seconds = compile_sample(options, mode, run_dir, kernel, env)
    runner_env = dict(env, PTOAS_BIN=options.ptoas, PTOAS_FLAGS=" ".join(flags),
                      CASE_NAME=options.case, WORK_SPACE=str(run_dir / "sim"),
                      DEVICE="SIM", COMPILE_ONLY="0", HOST_RUNNER="")
    log_path = run_dir / "runner.log"
    runner = ROOT / "test/vpto/scripts/run_host_vpto_validation.sh"
    run_logged(["bash", str(runner)], log_path, runner_env)
    log = log_path.read_text(encoding="utf-8")
    ticks = re.findall(r"Total tick:\s*(\d+)", log)
    if not ticks or "Model stopped successfully" not in log or f"[{options.case}] compare passed" not in log:
        raise RuntimeError("Missing real simulator execution or strict comparison evidence")
    result = {"mode": mode, "compile_seconds": compile_seconds, "ticks": [int(value) for value in ticks]}
    if mode == "auto":
        result["decision"] = check_auto(log, options.expect)
        result["decision_lines"] = [line for line in log.splitlines() if "Bisheng scheduler auto:" in line]
    return result


def validate_inputs(options):
    cases = Path(os.environ.get("CASES_ROOT", ROOT / "test/vpto/cases")).resolve()
    case = (cases / options.case).resolve()
    if not case.is_relative_to(cases) or not (case / "kernel.pto").is_file():
        raise ValueError("--case must name an existing kernel.pto case under CASES_ROOT")
    if (case / "ptoas.flags").exists():
        raise ValueError("Case-specific ptoas.flags would override the controlled scheduler flags")
    compiler = shutil.which(options.ptoas)
    if compiler is None:
        raise ValueError("PTOAS executable not found")
    options.ptoas = str(Path(compiler).absolute())
    options.work_space = options.work_space.resolve()
    options.work_space.mkdir(parents=True, exist_ok=False)
    return case / "kernel.pto"


def main():
    options = parse_args()
    kernel = validate_inputs(options)
    results = []
    for repetition in range(options.repetitions):
        for mode in MODES:
            run_dir = options.work_space / f"{mode}-{repetition + 1}"
            run_dir.mkdir()
            result = validate_sample(options, mode, run_dir, kernel, os.environ.copy())
            results.append(result)
            print(json.dumps(result), flush=True)
            (options.work_space / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for mode in MODES:
        samples = [item["compile_seconds"] for item in results if item["mode"] == mode]
        print(f"{mode}: median PTOAS fatobj compilation {statistics.median(samples):.6f} s")


if __name__ == "__main__":
    main()
