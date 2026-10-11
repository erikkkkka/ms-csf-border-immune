#!/usr/bin/env python3
"""Check tracked-file integrity and, optionally, extracted processed checkpoints."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, manifest: Path, required: bool) -> tuple[int, int]:
    checked, absent = 0, 0
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = root / row["path"]
            if not path.is_file():
                if required:
                    raise FileNotFoundError(f"Missing file: {row['path']}")
                absent += 1
                continue
            if path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
                raise ValueError(f"Checksum mismatch: {row['path']}")
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=row["path"])
            checked += 1
    return checked, absent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checked, _ = verify(root, root / "reproducibility/REPOSITORY_FILE_MANIFEST.csv", True)
    print(f"PASS: {checked} tracked files match their checksums; Python syntax parsed.")
    checked, absent = verify(root, root / "reproducibility/PROCESSED_DATA_MANIFEST.csv", args.require_data)
    print(f"Processed data: {checked} files verified, {absent} files not extracted.")
    if absent:
        print("Extract the four Processed_GSE*.zip release assets to verify all checkpoints.")


if __name__ == "__main__":
    main()
