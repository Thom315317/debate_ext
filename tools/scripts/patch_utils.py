#!/usr/bin/env python3
"""
patch_utils.py — Auxiliary diff/patch validation and repair utility.

Usage:
    python3 patch_utils.py validate <diff_file>
    python3 patch_utils.py repair <diff_file>

Exit codes:
    0 = valid / repaired successfully (prints cleaned diff to stdout)
    1 = invalid / could not repair
"""
import sys
import re
from pathlib import Path


def parse_hunks(diff_text: str) -> list[dict]:
    """Parse a unified diff into a list of hunk dicts."""
    hunks = []
    current_file = ""
    current_hunk = None

    for line in diff_text.splitlines():
        # File header
        m = re.match(r'^\+\+\+ (?:b/)?(.+)', line)
        if m:
            current_file = m.group(1).strip()
            continue

        m = re.match(r'^--- (?:a/)?(.+)', line)
        if m and not current_file:
            current_file = m.group(1).strip()
            continue

        # Hunk header
        m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if m:
            if current_hunk is not None:
                hunks.append(current_hunk)
            current_hunk = {
                "file": current_file,
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2) or "1"),
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4) or "1"),
                "lines": [],
            }
            continue

        if current_hunk is not None and line and line[0] in ('+', '-', ' '):
            current_hunk["lines"].append(line)

    if current_hunk is not None:
        hunks.append(current_hunk)

    return hunks


def validate_hunk(hunk: dict) -> tuple[bool, str]:
    """Validate a single hunk's line counts."""
    old_lines = sum(1 for l in hunk["lines"] if l[0] in ('-', ' '))
    new_lines = sum(1 for l in hunk["lines"] if l[0] in ('+', ' '))

    errors = []
    if old_lines != hunk["old_count"]:
        errors.append(f"old count mismatch: header says {hunk['old_count']}, actual {old_lines}")
    if new_lines != hunk["new_count"]:
        errors.append(f"new count mismatch: header says {hunk['new_count']}, actual {new_lines}")

    return (len(errors) == 0, "; ".join(errors))


def repair_hunk_header(hunk: dict) -> dict:
    """Fix hunk header counts to match actual content."""
    old_lines = sum(1 for l in hunk["lines"] if l[0] in ('-', ' '))
    new_lines = sum(1 for l in hunk["lines"] if l[0] in ('+', ' '))
    hunk["old_count"] = old_lines
    hunk["new_count"] = new_lines
    return hunk


def hunks_to_diff(hunks: list[dict]) -> str:
    """Reconstruct a unified diff string from parsed hunks."""
    output = []
    current_file = ""

    for hunk in hunks:
        if hunk["file"] != current_file:
            current_file = hunk["file"]
            output.append(f"--- a/{current_file}")
            output.append(f"+++ b/{current_file}")

        header = f"@@ -{hunk['old_start']},{hunk['old_count']} +{hunk['new_start']},{hunk['new_count']} @@"
        output.append(header)
        output.extend(hunk["lines"])

    return "\n".join(output) + "\n"


def cmd_validate(diff_path: str) -> int:
    """Validate a diff file. Prints cleaned diff to stdout on success."""
    text = Path(diff_path).read_text(encoding="utf-8")
    hunks = parse_hunks(text)

    if not hunks:
        print("ERROR: No valid hunks found", file=sys.stderr)
        return 1

    all_valid = True
    for i, hunk in enumerate(hunks):
        valid, msg = validate_hunk(hunk)
        if not valid:
            print(f"WARN: Hunk {i+1} ({hunk['file']}): {msg}", file=sys.stderr)
            all_valid = False

    if all_valid:
        print(text)
        return 0

    # Try repair
    print("Attempting repair...", file=sys.stderr)
    repaired = [repair_hunk_header(h) for h in hunks]
    result = hunks_to_diff(repaired)
    print(result)
    return 0


def cmd_repair(diff_path: str) -> int:
    """Repair a diff file. Prints repaired diff to stdout."""
    text = Path(diff_path).read_text(encoding="utf-8")
    hunks = parse_hunks(text)

    if not hunks:
        print("ERROR: No valid hunks found", file=sys.stderr)
        return 1

    repaired = [repair_hunk_header(h) for h in hunks]
    print(hunks_to_diff(repaired))
    return 0


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <validate|repair> <diff_file>", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    diff_path = sys.argv[2]

    if not Path(diff_path).exists():
        print(f"ERROR: File not found: {diff_path}", file=sys.stderr)
        sys.exit(1)

    if cmd == "validate":
        sys.exit(cmd_validate(diff_path))
    elif cmd == "repair":
        sys.exit(cmd_repair(diff_path))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
