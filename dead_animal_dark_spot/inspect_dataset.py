"""
dataset_prep/kaggle_converters/inspect_dataset.py
Run this FIRST on anything you download from Kaggle, before running any
of the converter scripts. Kaggle dataset internals can drift between
the version documented online and what you actually download, so this
prints the real structure (folders, file counts/extensions, and the
header row of any CSVs it finds) instead of assuming.

Usage:
    python dataset_prep/kaggle_converters/inspect_dataset.py /path/to/downloaded/dataset
"""

import argparse
import os
import csv
from collections import Counter


def inspect(root, max_depth=4, max_files_listed=5):
    print(f"\n=== Inspecting: {root} ===\n")
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        return

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []  # don't descend further
            continue

        indent = "  " * depth
        rel = os.path.relpath(dirpath, root)
        print(f"{indent}{rel}/  ({len(filenames)} files)")

        if filenames:
            ext_counts = Counter(os.path.splitext(f)[1].lower() for f in filenames)
            ext_summary = ", ".join(f"{ext or '(no ext)'}:{n}" for ext, n in ext_counts.most_common())
            print(f"{indent}  extensions -> {ext_summary}")
            for f in sorted(filenames)[:max_files_listed]:
                print(f"{indent}  e.g. {f}")

            # Peek at CSV headers — this is the part that matters most for
            # writing a correct converter, since column names vary.
            for f in filenames:
                if f.lower().endswith(".csv"):
                    csv_path = os.path.join(dirpath, f)
                    try:
                        with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
                            reader = csv.reader(fh)
                            header = next(reader, None)
                            first_row = next(reader, None)
                        print(f"{indent}  CSV '{f}' header -> {header}")
                        if first_row:
                            print(f"{indent}  CSV '{f}' row1   -> {first_row}")
                    except Exception as e:
                        print(f"{indent}  (couldn't read {f}: {e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Path to the downloaded/extracted Kaggle dataset folder")
    parser.add_argument("--max_depth", type=int, default=4)
    args = parser.parse_args()
    inspect(args.root, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
