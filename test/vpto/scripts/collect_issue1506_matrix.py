# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect portable evidence from an issue 1506 simulator matrix run."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def collect_run(run, evidence):
    log_path = run / "runner.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    case = run / "kernels_issue-1506-vec-misched"
    compare_path = case / "compare.json"
    comparison = json.loads(compare_path.read_text()) if compare_path.is_file() else None
    record = {
        "group": run.parent.name, "repeat": run.name, "directory": str(run),
        "total_ticks": [int(x) for x in re.findall(r"Total tick:\s*(\d+)", text)],
        "model_stopped": "Model stopped successfully" in text,
        "compare_passed": "compare passed" in text, "comparison": comparison,
        "output_sha256": digest(case / "output.bin"),
        "fatobj_sha256": digest(case / "kernel.fatobj.o"),
        "input_sha256": {p.name: digest(p) for p in sorted(case.glob("input_*.bin"))},
    }
    target = evidence / run.parent.name / run.name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_path, target / "runner.log")
    if compare_path.is_file():
        shutil.copy2(compare_path, target / "compare.json")
    return record


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    result = {}
    for name, runs in groups.items():
        ticks = [x["total_ticks"][0] for x in runs if len(x["total_ticks"]) == 1]
        result[name] = {
            "ticks": ticks,
            "median_ticks": statistics.median(ticks) if ticks else None,
            "all_passed": len(runs) == 3 and all(x["compare_passed"] and x["model_stopped"] for x in runs),
            "output_hashes": sorted({x["output_sha256"] for x in runs if x["output_sha256"]}),
        }
    baseline = result.get("ptoas-off_bisheng-false", {}).get("median_ticks")
    for group in result.values():
        ticks = group["median_ticks"]
        group["tick_reduction_percent"] = 100 * (baseline - ticks) / baseline if baseline and ticks else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    root, evidence = args.result_root.resolve(), args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    records = [collect_run(p.parent, evidence) for p in sorted(root.glob("ptoas-*/run*/runner.log"))]
    summary = {"runs": records, "groups": summarize(records)}
    for name in ("smoke.log", "provenance.log", "matrix.log"):
        if (root / name).is_file():
            shutil.copy2(root / name, evidence / name)
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
