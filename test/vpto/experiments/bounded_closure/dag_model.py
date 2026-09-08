# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Offline scheduling experiment on a traced, flat loop DAG; no graph rewrites."""

import re
from collections import Counter, defaultdict


VALUE = re.compile(r"%[\w]+")


def parse_op(line):
    assignment = re.match(r"^(%\w+(?::\d+)?) = (.*)", line.strip())
    result = assignment.group(1) if assignment else None
    rhs = assignment.group(2) if assignment else line.strip()
    head = rhs.split(" : ", 1)[0]
    operands = VALUE.findall(head)
    result_type = rhs.rsplit(" -> ", 1)[-1].rsplit(" : ", 1)[-1]
    kind = "V" if result_type.startswith("!pto.vreg") else "P"
    if not result_type.startswith(("!pto.vreg", "!pto.mask")):
        kind = None
    return result, operands, kind


class LoopDAG:
    def __init__(self, ir, trace, block):
        self.lines = ir.splitlines()
        starts = [i for i, line in enumerate(self.lines)
                  if "scf.for" in line and "iter_args(" in line]
        # Select the flat loop matching the supplied traced region size.
        part = trace.split(f"vpto-scheduler: block={block} regions=1\n", 1)[1]
        part = part.split("vpto-scheduler: block=", 1)[0]
        nodes = re.findall(r"node=(\d+) original-index=\d+ op=", part)
        size = len(nodes)
        matches = []
        for start in starts:
            end = next(i for i in range(start + 1, len(self.lines))
                       if "scf.yield" in self.lines[i])
            if end - start - 1 == size:
                matches.append((start + 1, end))
        if len(matches) != 1:
            raise ValueError("expected one flat loop matching the traced DAG")
        self.start, self.end = matches[0]
        self.ops = [parse_op(line) for line in self.lines[self.start:self.end]]
        self.defs = {result: n for n, (result, _, _) in enumerate(self.ops) if result}
        self.kinds = {}
        for line in self.lines[:self.start - 1]:
            result, _, kind = parse_op(line)
            if result:
                self.kinds[result] = kind
        args = re.findall(r"(%\w+) = (%\w+)", self.lines[self.start - 1])
        for arg, value in args:
            self.kinds[arg] = self.kinds.get(value)
        for result, _, kind in self.ops:
            if result:
                self.kinds[result] = kind
        self.users = defaultdict(set)
        for n, (_, operands, _) in enumerate(self.ops):
            for operand in operands:
                self.users[operand].add(n)
        self.external = set(self.users) - set(self.defs)
        # Loop invariants remain live across the backedge; iter_args do not.
        loop_args = {arg for arg, _ in args}
        self.liveout = set(VALUE.findall(self.lines[self.end])) | (self.external - loop_args)
        self.preds = [set() for _ in self.ops]
        self.succs = [set() for _ in self.ops]
        for pred, succ in re.findall(r"edge=(\d+)->(\d+) .*strength=must", part):
            a, b = int(pred), int(succ)
            self.preds[b].add(a)
            self.succs[a].add(b)
        self.current = [int(n) for n in re.findall(r"result-position=\d+ node=(\d+)", part)]
        self.original = list(range(size))
        self.validate(self.current)
        details = re.findall(r"decision-detail position=\d+ .*?current=\{vector:(\d+),predicate:(\d+)\}", part)
        states = self.states(self.current)
        for position, (vector, predicate) in enumerate(details):
            if states[position].pressure() != (int(vector), int(predicate)):
                raise ValueError(f"pressure replay mismatch at {position}: {states[position].pressure()} vs {(vector, predicate)}")
        self.reference_spans = self.spans(self.original)
        self.template = None

    def set_template(self, ir):
        """Keep ON's surrounding regions, translating SSA names by the traced permutation."""
        lines = ir.splitlines()
        starts = [i for i, line in enumerate(lines) if "scf.for" in line and "iter_args(" in line]
        matches = []
        for start in starts:
            end = next(i for i in range(start + 1, len(lines)) if "scf.yield" in lines[i])
            if end - start - 1 == len(self.ops):
                matches.append((start + 1, end))
        if len(matches) != 1:
            raise ValueError("expected exactly one corresponding template loop")
        start, end = matches[0]
        mapping = {}
        for node, line in zip(self.current, lines[start:end]):
            result, operands, _ = self.ops[node]
            new_result, new_operands, _ = parse_op(line)
            before = ([result] if result else []) + operands
            after = ([new_result] if new_result else []) + new_operands
            if len(before) != len(after):
                raise ValueError("template operand count mismatch")
            for old, new in zip(before, after):
                if old in mapping and mapping[old] != new:
                    raise ValueError("template SSA mapping is inconsistent")
                mapping[old] = new
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("template SSA mapping is not bijective")
        translated = [VALUE.sub(lambda match: mapping[match.group()], line)
                      for line in self.lines[self.start:self.end]]
        if [translated[n] for n in self.current] != lines[start:end]:
            raise ValueError("current schedule does not exactly match template body")
        self.template = (lines, start, end, translated)

    def validate(self, order):
        if sorted(order) != self.original:
            raise ValueError("candidate is not a permutation")
        positions = {n: p for p, n in enumerate(order)}
        for n, preds in enumerate(self.preds):
            if any(positions[pred] >= positions[n] for pred in preds):
                raise ValueError(f"candidate violates a Must edge into node {n}")

    def states(self, order):
        state = State(self)
        output = [state.copy()]
        for node in order:
            state.commit(node)
            output.append(state.copy())
        return output

    def spans(self, order):
        positions = {n: p for p, n in enumerate(order)}
        spans = {}
        for value, users in self.users.items():
            if self.kinds.get(value) != "V":
                continue
            first = positions[self.defs[value]] if value in self.defs else min(positions[n] for n in users)
            last = max(positions[n] for n in users)
            if value in self.defs and value in self.liveout:
                last = len(order)
            spans[value] = last - first
        return spans

    def metrics(self, order):
        self.validate(order)
        pressure = [state.pressure() for state in self.states(order)]
        spans = self.spans(order)
        regressions = [max(0, span - self.reference_spans[value]) for value, span in spans.items()]
        return {"peak_v": max(v for v, _ in pressure), "peak_p": max(p for _, p in pressure),
                "area_v": sum(v for v, _ in pressure[1:]),
                "excess_v": sum(max(0, v - 32) for v, _ in pressure[1:]),
                "positive_span": sum(regressions), "max_span_regression": max(regressions, default=0)}

    def render(self, order):
        self.validate(order)
        if self.template is not None:
            lines, start, end, translated = self.template
            return "\n".join(lines[:start] + [translated[n] for n in order] + lines[end:]) + "\n"
        body = [self.lines[self.start + n] for n in order]
        return "\n".join(self.lines[:self.start] + body + self.lines[self.end:]) + "\n"


class State:
    def __init__(self, dag):
        self.dag = dag
        self.live = set(dag.external)
        self.remaining = Counter(x for _, operands, _ in dag.ops for x in operands)
        self.done = set()

    def copy(self):
        other = object.__new__(State)
        other.dag = self.dag
        other.live = self.live.copy()
        other.remaining = self.remaining.copy()
        other.done = self.done.copy()
        return other

    def pressure(self):
        counts = Counter(self.dag.kinds.get(x) for x in self.live)
        return counts["V"], counts["P"]

    def commit(self, node):
        if node in self.done or not self.dag.preds[node] <= self.done:
            raise ValueError("attempted to schedule an unready node")
        result, operands, _ = self.dag.ops[node]
        for value in operands:
            self.remaining[value] -= 1
            if self.remaining[value] == 0 and value not in self.dag.liveout:
                self.live.discard(value)
        if result and (self.remaining[result] or result in self.dag.liveout):
            self.live.add(result)
        self.done.add(node)
