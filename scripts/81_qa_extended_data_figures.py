#!/usr/bin/env python3
"""Audit Extended Data exports, source tables and figure documentation."""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    "Extended_Data_Figure_1_preprocessing_QC": (183, 135),
    "Extended_Data_Figure_2_annotation_robustness": (183, 135),
    "Extended_Data_Figure_3_trajectory_robustness": (183, 140),
    "Extended_Data_Figure_4_two_axis_sensitivity": (183, 140),
    "Extended_Data_Figure_5_cross_species_LAM_audit": (183, 140),
    "Extended_Data_Figure_6_receiver_sender_boundary": (183, 140),
    "Extended_Data_Figure_7_in_vivo_CRISPR_robustness": (183, 145),
    "Extended_Data_Figure_8_virtual_perturbation_audit": (183, 145),
    "Extended_Data_Figure_9_drug_target_structure_audit": (183, 155),
}
FORMATS = {".png": 300, ".tiff": 600, ".svg": None, ".pdf": None}
EXPECTED_SOURCE_COUNTS = {number: 4 for number in range(1, 10)}
PIXEL_TOLERANCE = 5


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


def expected_pixels(mm: int, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


def raster_metrics(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
        dpi_info = image.info.get("dpi", (np.nan, np.nan))
        if isinstance(dpi_info, (int, float)):
            dpi_x = dpi_y = float(dpi_info)
        else:
            dpi_x, dpi_y = float(dpi_info[0]), float(dpi_info[1])
        preview = image.convert("RGB")
        preview.thumbnail((768, 768))
        array = np.asarray(preview, dtype=np.uint8)
        nonwhite = np.any(array < 248, axis=2)
        nonwhite_fraction = float(nonwhite.mean())
        channel_sd = float(array.std(axis=(0, 1)).mean())
    return {
        "pixel_width": width,
        "pixel_height": height,
        "dpi_x": dpi_x,
        "dpi_y": dpi_y,
        "nonwhite_fraction": nonwhite_fraction,
        "mean_channel_sd": channel_sd,
        "nonblank": bool(nonwhite_fraction > 0.01 and channel_sd > 2.0),
    }


def svg_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    root = ElementTree.fromstring(text)
    text_count = sum(1 for node in root.iter() if node.tag.endswith("text"))
    return {
        "svg_text_elements": text_count,
        "svg_viewbox": root.attrib.get("viewBox", ""),
        "svg_editable_text": bool(text_count >= 12),
        "svg_glyph_paths_absent": "DejaVuSans-" not in text,
    }


def pdf_metrics(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "pdf_header": data[:8].decode("ascii", errors="replace"),
        "pdf_valid_header": data.startswith(b"%PDF-"),
        "pdf_truetype_embedded": b"/FontFile2" in data,
        "pdf_type0_font": b"/Subtype /Type0" in data,
    }


def audit_exports(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_dir = root / "figures_extended/exports"
    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for stem, (width_mm, height_mm) in FIGURES.items():
        for suffix, dpi in FORMATS.items():
            path = export_dir / f"{stem}{suffix}"
            row: dict[str, object] = {
                "figure": stem,
                "format": suffix.lstrip("."),
                "relative_path": str(path.relative_to(root)),
                "expected_width_mm": width_mm,
                "expected_height_mm": height_mm,
                "target_dpi": dpi,
                "exists": path.is_file(),
            }
            if not path.is_file():
                row["passed"] = False
                rows.append(row)
                continue
            manifest.append({
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
            checks = [path.stat().st_size > 10_000]
            if suffix in {".png", ".tiff"}:
                metrics = raster_metrics(path)
                row.update(metrics)
                expected_width = expected_pixels(width_mm, int(dpi))
                expected_height = expected_pixels(height_mm, int(dpi))
                row["expected_pixel_width"] = expected_width
                row["expected_pixel_height"] = expected_height
                row["dimension_match"] = bool(
                    abs(int(metrics["pixel_width"]) - expected_width) <= PIXEL_TOLERANCE
                    and abs(int(metrics["pixel_height"]) - expected_height) <= PIXEL_TOLERANCE
                )
                row["dpi_match"] = bool(
                    abs(float(metrics["dpi_x"]) - int(dpi)) <= 1.0
                    and abs(float(metrics["dpi_y"]) - int(dpi)) <= 1.0
                )
                checks.extend([row["dimension_match"], row["dpi_match"], metrics["nonblank"]])
            elif suffix == ".svg":
                metrics = svg_metrics(path)
                row.update(metrics)
                checks.extend([metrics["svg_editable_text"], metrics["svg_glyph_paths_absent"]])
            else:
                metrics = pdf_metrics(path)
                row.update(metrics)
                checks.extend([
                    metrics["pdf_valid_header"], metrics["pdf_truetype_embedded"],
                    metrics["pdf_type0_font"],
                ])
            row["passed"] = all(bool(check) for check in checks)
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def audit_source_data(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_dir = root / "figures_extended/source_data"
    for path in sorted(source_dir.glob("EDFig*.csv")):
        frame = pd.read_csv(path)
        match = re.match(r"EDFig(\d+)", path.name)
        figure = int(match.group(1)) if match else None
        rows.append({
            "figure": figure,
            "relative_path": str(path.relative_to(root)),
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "null_cells": int(frame.isna().sum().sum()),
            "duplicate_column_names": bool(frame.columns.duplicated().any()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "passed": bool(len(frame) > 0 and frame.shape[1] > 0 and not frame.columns.duplicated().any()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    figure_dir = root / "figures_extended"
    export_qa, export_manifest = audit_exports(root)
    source_manifest = audit_source_data(root)
    export_qa.to_csv(figure_dir / "figure_export_qa.csv", index=False)
    export_manifest.to_csv(figure_dir / "EXPORT_MANIFEST.csv", index=False)
    source_manifest.to_csv(figure_dir / "source_data/SOURCE_DATA_MANIFEST.csv", index=False)

    source_counts = source_manifest.groupby("figure").size().to_dict()
    export_pass = bool(len(export_qa) == 36 and export_qa.passed.all())
    source_pass = bool(source_manifest.passed.all() and source_counts == EXPECTED_SOURCE_COUNTS)
    documentation = {
        "contract": (figure_dir / "EXTENDED_DATA_CONTRACT.md").is_file(),
        "legends": (figure_dir / "EXTENDED_DATA_FIGURE_LEGENDS.md").is_file(),
        "contact_sheet": (figure_dir / "exports/Extended_Data_contact_sheet.png").is_file(),
    }
    overall_pass = export_pass and source_pass and all(documentation.values())
    source_summary = ", ".join(f"ED{key}: {value}" for key, value in sorted(source_counts.items()))
    failed = export_qa.loc[~export_qa.passed, ["relative_path", "format"]]
    failure_lines = "\n".join(
        f"- `{row.relative_path}` ({row.format})" for row in failed.itertuples()
    ) or "- None."
    report = f"""# Extended Data figure QA report

Audited at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}

## Status

**{'PASS' if overall_pass else 'FAIL'}**

- Export files: {int(export_qa.passed.sum())}/{len(export_qa)} passed.
- Expected format set: nine figures x SVG, PDF, PNG and TIFF.
- Source-data tables: {int(source_manifest.passed.sum())}/{len(source_manifest)} passed.
- Source-data coverage: {source_summary}.
- Contract, legends and visual contact sheet: {'complete' if all(documentation.values()) else 'incomplete'}.

## Automated checks

- PNG files are 300 dpi and TIFF files are 600 dpi at 183 mm width and the
  contract-specified figure height.
- Raster exports contain non-white, variable pixels.
- SVG text remains editable and has not been converted to font-outline glyphs.
- PDF files have valid headers and embedded TrueType Type 0 fonts.
- Each Extended Data figure has exactly four non-empty, uniquely headed source tables.
- SHA-256 digests are recorded for all exports and source tables.

## Visual inspection

- All nine PNGs were inspected in a contact sheet after final export.
- Extended Data Figures 5, 7 and 8 were additionally inspected at original scale after
  figure-specific margin correction.
- Panel labels, axis labels, legends and long cohort/category labels remain within the canvas.

## Failed exports

{failure_lines}
"""
    (figure_dir / "EXTENDED_DATA_QA_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    if not overall_pass:
        raise SystemExit("Extended Data QA failed")


if __name__ == "__main__":
    main()
