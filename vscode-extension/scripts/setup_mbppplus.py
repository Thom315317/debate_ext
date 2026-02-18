#!/usr/bin/env python3
"""
Download MBPP+ via EvalPlus and export to data/MbppPlus.jsonl.

Usage:
    pip install evalplus   # or: pip install datasets
    python scripts/setup_mbppplus.py

Output:
    data/MbppPlus.jsonl  — one JSON object per line
    Fields: task_id, prompt, code, test_list, test_setup_code, entry_point
"""

import json


class SafeEncoder(json.JSONEncoder):
    """Handle complex numbers and other non-serializable types."""
    def default(self, obj):
        if isinstance(obj, complex):
            return repr(obj)
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_PATH = os.path.join(DATA_DIR, "MbppPlus.jsonl")


def extract_entry_point(code: str) -> str:
    """Extract the first function name from a code snippet."""
    m = re.search(r"def\s+(\w+)\s*\(", code)
    return m.group(1) if m else "unknown"


def _build_asserts(entry_point: str, inputs: list, canonical_solution: str, atol: float = 0) -> list:
    """Build assert strings by running canonical_solution on each input set."""
    if not inputs or not canonical_solution:
        return []
    # Compile canonical solution once
    namespace = {}
    try:
        exec(canonical_solution, namespace)
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
            if atol and isinstance(expected, float):
                asserts.append(f"assert abs({entry_point}({args_str}) - {repr(expected)}) <= {atol}")
            else:
                asserts.append(f"assert {entry_point}({args_str}) == {repr(expected)}")
        except Exception:
            pass  # skip inputs that fail on canonical solution
    return asserts


def load_via_evalplus():
    """Try loading via the evalplus package (preferred)."""
    from evalplus.data import get_mbpp_plus  # type: ignore

    raw = get_mbpp_plus()
    tasks = []
    for task_id, item in raw.items():
        entry_point = item.get("entry_point", extract_entry_point(
            item.get("canonical_solution", item.get("code", ""))
        ))

        # ── Base tests: use 'assertion' field (proper assert strings) ──
        assertion_str = item.get("assertion", "")
        if assertion_str:
            test_list = [line.strip() for line in assertion_str.strip().split("\n") if line.strip().startswith("assert")]
        else:
            # Fallback: build asserts from base_input + canonical_solution
            test_list = _build_asserts(entry_point, item.get("base_input", []),
                                       item.get("canonical_solution", ""), item.get("atol", 0))

        # ── Plus tests: build asserts from plus_input + canonical_solution ──
        test_list_plus = _build_asserts(entry_point, item.get("plus_input", []),
                                        item.get("canonical_solution", ""), item.get("atol", 0))

        tasks.append({
            "task_id": task_id,
            "prompt": item.get("prompt", ""),
            "code": item.get("canonical_solution", item.get("code", "")),
            "test_list": test_list,
            "test_list_plus": test_list_plus,
            "test_setup_code": item.get("test_setup_code", ""),
            "entry_point": entry_point,
        })
    return tasks


def load_via_datasets():
    """Fallback: load via HuggingFace datasets."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("evalplus/mbppplus", split="test")
    tasks = []
    for item in ds:
        code = item.get("canonical_solution", item.get("code", ""))
        test_list_plus = item.get("test_list_plus", item.get("plus", []))
        if not isinstance(test_list_plus, list):
            test_list_plus = []
        tasks.append({
            "task_id": item.get("task_id", f"Mbpp/{len(tasks)}"),
            "prompt": item.get("prompt", ""),
            "code": code,
            "test_list": item.get("test_list", []),
            "test_list_plus": test_list_plus,
            "test_setup_code": item.get("test_setup_code", ""),
            "entry_point": item.get("entry_point", extract_entry_point(code)),
        })
    return tasks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    tasks = None

    # Strategy 1: evalplus
    try:
        tasks = load_via_evalplus()
        print(f"Loaded {len(tasks)} tasks via evalplus")
    except ImportError:
        print("evalplus not installed, trying datasets...")
    except Exception as e:
        print(f"evalplus failed: {e}, trying datasets...")

    # Strategy 2: HuggingFace datasets
    if tasks is None:
        try:
            tasks = load_via_datasets()
            print(f"Loaded {len(tasks)} tasks via datasets")
        except ImportError:
            print("ERROR: Neither evalplus nor datasets is installed.", file=sys.stderr)
            print("  pip install evalplus   # or: pip install datasets", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: datasets failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Write JSONL
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False, cls=SafeEncoder) + "\n")

    print(f"Wrote {len(tasks)} tasks to {OUT_PATH}")


if __name__ == "__main__":
    main()
