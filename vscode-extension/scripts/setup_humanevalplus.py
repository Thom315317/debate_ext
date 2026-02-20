#!/usr/bin/env python3
"""
Download HumanEval+ via EvalPlus and export to data/HumanEvalPlus.jsonl.

Usage:
    pip install evalplus
    python scripts/setup_humanevalplus.py

Output:
    data/HumanEvalPlus.jsonl — one JSON object per line
    Same schema as MbppPlus.jsonl: task_id, prompt, code, test_list, test_setup_code, entry_point, test_list_plus
"""

import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_PATH = os.path.join(DATA_DIR, "HumanEvalPlus.jsonl")


class SafeEncoder(json.JSONEncoder):
    """Handle complex numbers and other non-serializable types."""
    def default(self, obj):
        if isinstance(obj, complex):
            return repr(obj)
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def extract_asserts_from_check(test_str: str, entry_point: str) -> list:
    """Extract assert lines from a HumanEval test string containing def check(candidate):."""
    # Replace 'candidate' with the actual function name
    test_str = test_str.replace("candidate", entry_point)
    # Extract all assert lines
    asserts = []
    for line in test_str.split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert"):
            asserts.append(stripped)
    return asserts


def _build_asserts(entry_point: str, inputs: list, canonical_solution: str, contract: str = "", atol: float = 0) -> list:
    """Build assert strings by running canonical_solution on each input set."""
    if not inputs or not canonical_solution:
        return []
    namespace = {}
    try:
        exec(contract + "\n" + canonical_solution if contract else canonical_solution, namespace)
    except Exception:
        return []
    func = namespace.get(entry_point)
    if func is None:
        return []
    asserts = []
    for inp in inputs:
        if not isinstance(inp, (list, tuple)):
            inp = [inp]
        try:
            expected = func(*inp)
            args_str = ', '.join(repr(x) for x in inp)
            exp_str = repr(expected)
            if len(args_str) + len(exp_str) > 10_000:
                continue
            call = f"{entry_point}({args_str})"
            if atol and isinstance(expected, float):
                asserts.append(f"assert abs({call} - {exp_str}) <= {atol}")
            elif isinstance(expected, (set, frozenset)):
                asserts.append(f"assert set({call}) == {exp_str}")
            elif isinstance(expected, (tuple, list)):
                try:
                    sorted_exp = sorted(expected)
                    asserts.append(f"assert sorted({call}) == {repr(sorted_exp)}")
                except TypeError:
                    asserts.append(f"assert {call} == {exp_str}")
            else:
                asserts.append(f"assert {call} == {exp_str}")
        except Exception:
            pass
    return asserts


def load_via_evalplus():
    """Load HumanEval+ via the evalplus package."""
    from evalplus.data import get_human_eval_plus

    raw = get_human_eval_plus()
    tasks = []
    for task_id, item in raw.items():
        entry_point = item.get("entry_point", "unknown")
        prompt = item.get("prompt", "")
        canonical = item.get("canonical_solution", "")

        # The full function = prompt (signature+docstring) + canonical_solution (body)
        full_code = prompt + canonical

        # Base tests: extract from 'assertion' field or 'test' field
        assertion_str = item.get("assertion", "")
        if assertion_str:
            test_list = [line.strip() for line in assertion_str.strip().split("\n")
                         if line.strip().startswith("assert")]
        else:
            # Fallback: extract from test string (def check(candidate):)
            test_str = item.get("test", "")
            test_list = extract_asserts_from_check(test_str, entry_point) if test_str else []

        # Plus tests: build from plus_input + canonical_solution
        test_list_plus = _build_asserts(
            entry_point, item.get("plus_input", []),
            full_code, item.get("contract", ""), item.get("atol", 0)
        )

        # test_setup_code: imports that the prompt may assume
        test_setup = ""
        if "from typing import" in prompt or "import " in prompt:
            # Extract import lines from the prompt
            imports = [line for line in prompt.split("\n") if line.startswith("from ") or line.startswith("import ")]
            test_setup = "\n".join(imports)

        tasks.append({
            "task_id": task_id,
            "prompt": prompt,
            "code": full_code,
            "test_list": test_list,
            "test_list_plus": test_list_plus,
            "test_setup_code": test_setup,
            "entry_point": entry_point,
        })
    return tasks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        tasks = load_via_evalplus()
        print(f"Loaded {len(tasks)} tasks via evalplus")
    except ImportError:
        print("ERROR: evalplus not installed.", file=sys.stderr)
        print("  pip install evalplus", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False, cls=SafeEncoder) + "\n")

    # Stats
    with_plus = sum(1 for t in tasks if t["test_list_plus"])
    print(f"Wrote {len(tasks)} tasks to {OUT_PATH}")
    print(f"  {with_plus}/{len(tasks)} have plus tests")


if __name__ == "__main__":
    main()
