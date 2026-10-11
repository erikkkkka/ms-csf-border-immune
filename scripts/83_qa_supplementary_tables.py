#!/usr/bin/env python3
"""Audit supplementary workbooks, source hashes and spreadsheet usability."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Supplementary_Table_1_Cohorts_and_QC.xlsx": 9,
    "Supplementary_Table_2_Annotation_and_State.xlsx": 11,
    "Supplementary_Table_3_Trajectory_TwoAxis_LAM.xlsx": 16,
    "Supplementary_Table_4_Receiver_Sender_Regulators.xlsx": 11,
    "Supplementary_Table_5_InVivo_CRISPR_Bridge.xlsx": 13,
    "Supplementary_Table_6_Virtual_Perturbation.xlsx": 14,
    "Supplementary_Table_7_Drug_Target_Structure.xlsx": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_workbooks(root: Path) -> pd.DataFrame:
    output = root / "supplementary"
    rows: list[dict[str, object]] = []
    for filename, expected_sheets in EXPECTED.items():
        path = output / filename
        row: dict[str, object] = {
            "workbook": filename,
            "exists": path.is_file(),
            "expected_sheets": expected_sheets,
        }
        if not path.is_file():
            row["passed"] = False
            rows.append(row)
            continue
        workbook = load_workbook(path, read_only=False, data_only=False)
        formula_cells = 0
        error_cells = 0
        data_sheets_ok = True
        table_count = 0
        arial_headers = True
        for sheet_name in workbook.sheetnames:
            ws = workbook[sheet_name]
            for row_cells in ws.iter_rows():
                for cell in row_cells:
                    formula_cells += int(cell.data_type == "f")
                    error_cells += int(cell.data_type == "e")
            if sheet_name == "Overview":
                data_sheets_ok &= ws.freeze_panes == "A11" and bool(ws.auto_filter.ref)
                arial_headers &= ws["A1"].font.name == "Arial"
            else:
                data_sheets_ok &= (
                    ws.max_row >= 5
                    and ws.max_column >= 1
                    and ws.freeze_panes == "A5"
                    and bool(ws.auto_filter.ref)
                    and len(ws.tables) == 1
                )
                table_count += len(ws.tables)
                arial_headers &= all(ws.cell(4, col).font.name == "Arial" for col in range(1, ws.max_column + 1))
        row.update({
            "actual_sheets": len(workbook.sheetnames),
            "data_sheets": len(workbook.sheetnames) - 1,
            "table_count": table_count,
            "formula_cells": formula_cells,
            "error_cells": error_cells,
            "freeze_filter_table_checks": bool(data_sheets_ok),
            "arial_header_checks": bool(arial_headers),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        row["passed"] = bool(
            len(workbook.sheetnames) == expected_sheets
            and workbook.sheetnames[0] == "Overview"
            and data_sheets_ok
            and arial_headers
            and formula_cells == 0
            and error_cells == 0
            and path.stat().st_size > 10_000
        )
        rows.append(row)
    return pd.DataFrame(rows)


def audit_sources(root: Path) -> pd.DataFrame:
    manifest_path = root / "supplementary/SUPPLEMENTARY_SOURCE_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    for entry in manifest.itertuples(index=False):
        source_paths = [part.strip() for part in str(entry.source).split(";")]
        expected_hashes = [part.strip() for part in str(entry.source_sha256).split(";")]
        actual_hashes = [sha256(root / source_path) if (root / source_path).is_file() else "MISSING" for source_path in source_paths]
        rows.append({
            "workbook": entry.workbook,
            "sheet": entry.sheet,
            "source_files": len(source_paths),
            "reported_rows": int(entry.rows),
            "reported_columns": int(entry.columns),
            "sources_exist": all(value != "MISSING" for value in actual_hashes),
            "source_hashes_match": actual_hashes == expected_hashes,
            "passed": all(value != "MISSING" for value in actual_hashes) and actual_hashes == expected_hashes,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "supplementary"
    workbook_qa = audit_workbooks(root)
    source_qa = audit_sources(root)
    workbook_qa.to_csv(output / "SUPPLEMENTARY_WORKBOOK_QA.csv", index=False)
    source_qa.to_csv(output / "SUPPLEMENTARY_SOURCE_QA.csv", index=False)

    recorded = pd.read_csv(output / "SUPPLEMENTARY_WORKBOOK_MANIFEST.csv")
    recorded_hashes = dict(zip(recorded.workbook, recorded.sha256))
    manifest_match = all(recorded_hashes.get(row.workbook) == row.sha256 for row in workbook_qa.itertuples())
    index_exists = (output / "SUPPLEMENTARY_TABLES_INDEX.md").is_file()
    overall_pass = bool(
        workbook_qa.passed.all() and source_qa.passed.all() and manifest_match and index_exists
    )
    total_data_sheets = int(workbook_qa.data_sheets.sum())
    report = f"""# Supplementary table QA report

Audited at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}

## Status

**{'PASS' if overall_pass else 'FAIL'}**

- Workbooks: {int(workbook_qa.passed.sum())}/{len(workbook_qa)} passed.
- Data sheets: {total_data_sheets} across seven workbooks.
- Source links and hashes: {int(source_qa.passed.sum())}/{len(source_qa)} passed.
- Workbook manifest hashes: {'matched' if manifest_match else 'mismatch'}.
- Formula cells: {int(workbook_qa.formula_cells.sum())}; Excel error cells: {int(workbook_qa.error_cells.sum())}.

## Checks

- Every workbook opens successfully and begins with an Overview sheet.
- Every data sheet is non-empty, uses Arial headers, and has a frozen header,
  autofilter and formatted Excel table.
- Data are static result exports; no spreadsheet formulas are introduced.
- Every declared source path exists and matches its recorded SHA-256 digest.
- Workbook digests match `SUPPLEMENTARY_WORKBOOK_MANIFEST.csv`.
"""
    (output / "SUPPLEMENTARY_QA_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    if not overall_pass:
        raise SystemExit("Supplementary table QA failed")


if __name__ == "__main__":
    main()
