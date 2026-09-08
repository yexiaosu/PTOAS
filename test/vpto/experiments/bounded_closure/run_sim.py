# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage locally generated candidates and run the repository's strict SIM runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--experts", type=int, choices=(512, 768), default=768)
    args = parser.parse_args()
    output = args.output.resolve()
    inputs = output / "candidates"
    cases = output / "cases"
    host = Path(__file__).resolve().parent / "host"
    sources = {path.stem: path for path in sorted(inputs.glob("*.pto"))}
    sources["exact-on"] = args.baseline / "on.vpto.mlir"
    sources["exact-off"] = args.baseline / "off.vpto.mlir"
    names = args.names or list(sources)
    for name in names:
        if name not in sources:
            raise ValueError(f"unknown generated candidate {name}")
    results_path = output / "sim-results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in names:
        case_dir = cases / name
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[name], case_dir / "kernel.pto")
        for filename in ("main.cpp", "launch.cpp", "golden.py", "compare.py"):
            source = (host / filename).read_text(encoding="utf-8")
            if args.experts == 512:
                source = source.replace("4U * 768U", "4U * 512U").replace("E = 768", "E = 512")
            (case_dir / filename).write_text(source, encoding="utf-8")
        env = os.environ.copy()
        env.update({"DEVICE": "SIM", "COMPILE_ONLY": "0", "COMPARE_STRICT": "1",
                    "CASES_ROOT": str(cases), "CASE_NAME": name,
                    "WORK_SPACE": str(output / "runs")})
        log_path = output / f"{name}.runner.log"
        print(f"Running {name}: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            run = subprocess.run(["bash", str(args.runner.resolve())], env=env,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=900, check=False)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        result = {"returncode": run.returncode,
                  "input_sha256": hashlib.sha256(sources[name].read_bytes()).hexdigest(),
                  "total_tick": re.findall(r"Total tick: (\d+)", log_text),
                  "compare": "exact int32 compare passed" in log_text,
                  "log": str(log_path),
                  "errors": [line for line in log_text.splitlines()
                             if any(word in line.lower() for word in ("error:", "[error]", "spills"))][:12]}
        results[name] = result
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
