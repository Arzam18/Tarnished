#!/usr/bin/env python3
"""Summarize Clang/GCC .su files from an Android AArch64 build."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <stack-usage-directory>")
    root = Path(sys.argv[1])
    reports = sorted(root.glob("*.su"))
    if not reports:
        raise SystemExit(f"ERROR: no .su stack-usage reports found in {root}")

    entries: list[tuple[int, str, str]] = []
    for report in reports:
        for line in report.read_text(errors="replace").splitlines():
            fields = line.rsplit("\t", 2)
            if len(fields) < 2:
                continue
            try:
                size = int(fields[-2])
            except ValueError:
                continue
            kind = fields[-1] if len(fields) == 3 else "unknown"
            entries.append((size, kind, f"{report.name}: {fields[0]}"))

    if not entries:
        raise SystemExit("ERROR: .su files contained no parseable stack entries")

    entries.sort(reverse=True)
    print(f"Parsed {len(entries)} stack frames from {len(reports)} reports")
    print("Largest 25 static/dynamic frames:")
    for size, kind, name in entries[:25]:
        print(f"{size:>8} bytes  {kind:<18} {name}")
    print("\nSearch recursion frames:")
    pattern = re.compile(r"(Search6searchI|Search7qsearchI|search<|qsearch<|iterativeDeepening|startSearching)")
    matches = [(size, kind, name) for size, kind, name in entries if pattern.search(name)]
    for size, kind, name in matches:
        print(f"{size:>8} bytes  {kind:<18} {name}")
    if not matches:
        print("No matching mangled/search function names; inspect raw .su files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
