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


def load_via_evalplus():
    """Try loading via the evalplus package (preferred)."""
    from evalplus.data import get_mbpp_plus  # type: ignore

    raw = get_mbpp_plus()
    tasks = []
    for task_id, item in raw.items():
        entry_point = item.get("entry_point", extract_entry_point(
            item.get("canonical_solution", item.get("code", ""))
        ))
        # EvalPlus+ augmented tests
        test_list_plus = []
        plus_inputs = item.get("plus_input", item.get("plus", []))
        if isinstance(plus_inputs, list) and len(plus_inputs) > 0:
            expected = item.get("plus", item.get("expected_output", []))
            if isinstance(plus_inputs[0], (list, tuple)):
                # Format: pairs of (input, expected_output)
                for inp, exp in zip(plus_inputs, expected if isinstance(expected, list) else []):
                    try:
                        test_list_plus.append(f"assert {entry_point}({', '.join(repr(x) for x in inp)}) == {repr(exp)}")
                    except Exception:
                        pass
            elif isinstance(plus_inputs[0], str):
                # Already assertion strings
                test_list_plus = list(plus_inputs)
        tasks.append({
            "task_id": task_id,
            "prompt": item.get("prompt", ""),
            "code": item.get("canonical_solution", item.get("code", "")),
            "test_list": item.get("test_list", item.get("base_input", [])),
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
