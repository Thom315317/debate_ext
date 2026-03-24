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
    # Extract assert statements, joining multi-line continuations
    asserts = []
    current_assert = ""
    for line in test_str.split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert"):
            if current_assert:
                asserts.append(current_assert)
            current_assert = stripped
        elif current_assert and stripped:
            current_assert += " " + stripped
    if current_assert:
        asserts.append(current_assert)
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


def extract_helper_functions(prompt: str, entry_point: str) -> str:
    """Extract all function definitions from prompt that are NOT the entry_point."""
    lines = prompt.split('\n')
    helpers = []
    current_func = []
    in_func = False
    is_entry = False

    for line in lines:
        m = re.match(r'^def\s+(\w+)', line)
        if m:
            if current_func and not is_entry:
                helpers.append('\n'.join(current_func))
            func_name = m.group(1)
            is_entry = (func_name == entry_point)
            current_func = [line]
            in_func = True
        elif in_func:
            current_func.append(line)

    if current_func and not is_entry:
        helpers.append('\n'.join(current_func))

    return '\n\n'.join(helpers)


def fix_broken_tests(tasks: list) -> int:
    """Fix tasks with broken test_list and inject helper functions into test_setup_code.
    Returns the number of tasks fixed."""
    # Bug 2: replace entire test_list for tasks with undefined variables
    fixes = {
        "HumanEval/32": [
            "assert math.fabs(poly([1, 2], find_zero([1, 2]))) < 1e-4",
            "assert math.fabs(poly([1, 0, -1], find_zero([1, 0, -1]))) < 1e-4",
            "assert math.fabs(poly([-6, 11, -6, 1], find_zero([-6, 11, -6, 1]))) < 1e-4",
            "assert math.fabs(poly([1, 1], find_zero([1, 1]))) < 1e-4",
            "assert math.fabs(poly([2, -3, 1], find_zero([2, -3, 1]))) < 1e-4",
        ],
        "HumanEval/38": [
            "assert decode_cyclic(encode_cyclic('')) == ''",
            "assert decode_cyclic(encode_cyclic('hello')) == 'hello'",
            "assert decode_cyclic(encode_cyclic('abcdefghij')) == 'abcdefghij'",
            "assert decode_cyclic(encode_cyclic('a')) == 'a'",
            "assert decode_cyclic(encode_cyclic('ab')) == 'ab'",
            "assert decode_cyclic(encode_cyclic('abcabcabcabc')) == 'abcabcabcabc'",
        ],
        "HumanEval/50": [
            "assert decode_shift(encode_shift('')) == ''",
            "assert decode_shift(encode_shift('hello')) == 'hello'",
            "assert decode_shift(encode_shift('abcdefghijklmnopqrstuvwxyz')) == 'abcdefghijklmnopqrstuvwxyz'",
            "assert decode_shift(encode_shift('a')) == 'a'",
            "assert decode_shift(encode_shift('test string here')) == 'test string here'",
        ],
    }
    # Bug 2: trim broken trailing tests (undefined vars + for-loop garbage from multi-line joiner)
    trim_last = {
        "HumanEval/44": 5,   # test[5] has 'for x in range' joined from multi-line
        "HumanEval/53": 4,   # test[4] has 'for i in range' joined from multi-line
        "HumanEval/151": 5,  # test[5] has trailing garbage from multi-line
    }
    count = 0
    for task in tasks:
        tid = task["task_id"]
        if tid in fixes:
            task["test_list"] = fixes[tid]
            count += 1
            print(f"  Fixed {tid}: replaced test_list ({len(fixes[tid])} tests)")
        elif tid in trim_last:
            n = trim_last[tid]
            old_len = len(task["test_list"])
            task["test_list"] = task["test_list"][:n]
            count += 1
            print(f"  Fixed {tid}: trimmed test_list from {old_len} to {n}")

    # Bug 3: inject helper functions from prompt into test_setup_code
    helpers_injected = 0
    for task in tasks:
        helpers = extract_helper_functions(task["prompt"], task["entry_point"])
        if helpers:
            existing_setup = task.get("test_setup_code", "")
            task["test_setup_code"] = (existing_setup + "\n\n" + helpers).strip()
            helpers_injected += 1
    if helpers_injected:
        print(f"  Injected helper functions into test_setup_code for {helpers_injected} tasks")

    return count


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


def validate_tests(tasks: list) -> int:
    """Validate all test_list entries for syntax errors.
    Returns the number of broken tests found."""
    broken = 0
    for task in tasks:
        tid = task["task_id"]
        for i, test in enumerate(task["test_list"]):
            try:
                compile(test, "<test>", "exec")
            except SyntaxError as e:
                print(f"  BROKEN {tid} test[{i}]: SyntaxError — {e.msg}")
                broken += 1
    return broken


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

    # Fix known broken tests
    print("Fixing broken tests...")
    fixed = fix_broken_tests(tasks)
    print(f"  {fixed} tasks fixed")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False, cls=SafeEncoder) + "\n")

    # Stats
    with_plus = sum(1 for t in tasks if t["test_list_plus"])
    total_tests = sum(len(t["test_list"]) for t in tasks)
    print(f"Wrote {len(tasks)} tasks to {OUT_PATH}")
    print(f"  {with_plus}/{len(tasks)} have plus tests")

    # Validation
    print("Validating tests...")
    broken = validate_tests(tasks)
    if broken == 0:
        print(f"  Validation: {len(tasks)}/{len(tasks)} tasks OK, {total_tests} tests checked, 0 broken")
    else:
        print(f"  WARNING: {broken} broken tests found!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
