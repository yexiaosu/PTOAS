# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate bounded closure permutations and globally span-checked candidates."""

import argparse
import json
from pathlib import Path

from dag_model import LoopDAG


def closure(dag, state, target, extra, steps, budget):
    required = set(dag.users[target]) - state.done
    work = list(required)
    while work:
        node = work.pop()
        for pred in dag.preds[node] - state.done - required:
            required.add(pred)
            work.append(pred)
        if len(required) > steps:
            return None
    sim = state.copy()
    start_v, start_p = sim.pressure()
    witness, area = [], 0
    while required:
        ready = [n for n in required if dag.preds[n] <= sim.done]
        choices = []
        for n in ready:
            trial = sim.copy()
            trial.commit(n)
            vector, predicate = trial.pressure()
            if vector <= start_v + extra and predicate <= max(7, start_p):
                choices.append(((vector, predicate, n), trial))
        if not choices:
            return None
        (_, _, selected), sim = min(choices, key=lambda x: x[0])
        area += max(0, sim.pressure()[0] - start_v)
        if area > budget:
            return None
        witness.append(selected)
        required.remove(selected)
    relief = start_v - sim.pressure()[0]
    if target in sim.live or relief < 1:
        return None
    return witness, relief, area


def rank(metrics):
    return (metrics["positive_span"], metrics["max_span_regression"],
            metrics["peak_v"], metrics["excess_v"])


def optimize(dag, extra, steps, budget, span_guard):
    order = dag.current.copy()
    changes = []
    for iteration in range(8):
        base_metrics = dag.metrics(order)
        candidates = []
        for position, state in enumerate(dag.states(order)[:-1]):
            if state.pressure()[0] <= 32:
                continue
            for target in sorted(state.live & set(dag.defs)):
                if dag.kinds.get(target) != "V" or target in dag.liveout:
                    continue
                found = closure(dag, state, target, extra, steps, budget)
                if found is None:
                    continue
                witness, relief, area = found
                taken = set(witness)
                proposed = order[:position] + witness + [n for n in order[position:] if n not in taken]
                if proposed == order:
                    continue
                metrics = dag.metrics(proposed)
                if metrics["peak_p"] > max(7, base_metrics["peak_p"]):
                    continue
                if span_guard and rank(metrics) >= rank(base_metrics):
                    continue
                if span_guard:
                    base_spans = dag.spans(order)
                    spans = dag.spans(proposed)
                    if any(span > max(base_spans[value], dag.reference_spans[value])
                           for value, span in spans.items()):
                        continue
                if not span_guard and metrics["excess_v"] >= base_metrics["excess_v"]:
                    continue
                score = rank(metrics) if span_guard else (metrics["excess_v"], metrics["area_v"])
                candidates.append((score, proposed, {"position": position, "target": target,
                    "witness": witness, "relief": relief, "extra_area": area, "metrics": metrics}))
        if not candidates:
            break
        _, order, detail = min(candidates, key=lambda x: (x[0], x[2]["position"], x[2]["witness"]))
        changes.append(detail)
    return order, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=3)
    args = parser.parse_args()
    dag = LoopDAG(args.ir.read_text(), args.trace.read_text(), args.block)
    args.output.mkdir(parents=True, exist_ok=True)
    variants = {"off": (dag.original, []), "current": (dag.current, [])}
    for extra, steps, budget in ((1, 6, 4), (1, 12, 8), (2, 12, 12)):
        for guarded in (False, True):
            name = f"bounded-e{extra}-s{steps}-a{budget}" + ("-span" if guarded else "")
            variants[name] = optimize(dag, extra, steps, budget, guarded)
    best = min(variants, key=lambda name: rank(dag.metrics(variants[name][0])))
    variants["selected-with-original-fallback"] = variants[best]
    report = {"selected": best, "variants": {}}
    for name, (order, changes) in variants.items():
        (args.output / f"{name}.pto").write_text(dag.render(order))
        report["variants"][name] = {"metrics": dag.metrics(order), "changes": changes, "order": order}
        print(name, json.dumps(report["variants"][name]["metrics"]), "changes", len(changes), flush=True)
    (args.output / "candidates.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
