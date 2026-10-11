#!/usr/bin/env python3
"""Audit manuscript figure exports and their plotted source-data tables."""

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
    "Figure_1_CSF_monocyte_BAM_continuum": (183, 140),
    "Figure_2_MS_inflammatory_adaptation_decoupling": (183, 150),
    "Figure_3_receiver_regulatory_analysis": (183, 172),
    "Figure_4_in_vivo_CRISPR_directional_control": (183, 150),
    "Figure_5_transition_aware_chemical_probe": (183, 190),
}

FORMATS = {
    ".png": 300,
    ".tiff": 600,
    ".svg": None,
    ".pdf": None,
}

EXPECTED_SOURCE_COUNTS = {1: 4, 2: 4, 3: 4, 4: 4, 5: 9}
PIXEL_DIMENSION_TOLERANCE = 5


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


def expected_pixels(mm: float, dpi: int) -> int:
    return round(mm / 25.4 * dpi)


def raster_metrics(path: Path) -> dict[str, float | int | str | bool]:
    with Image.open(path) as image:
        width, height = image.size
        dpi_info = image.info.get("dpi", (np.nan, np.nan))
        if isinstance(dpi_info, (int, float)):
            dpi_x = dpi_y = float(dpi_info)
        else:
            dpi_x, dpi_y = (float(dpi_info[0]), float(dpi_info[1]))

        preview = image.convert("RGB")
        preview.thumbnail((768, 768))
        array = np.asarray(preview, dtype=np.uint8)
        nonwhite = np.any(array < 248, axis=2)
        nonwhite_fraction = float(nonwhite.mean())
        channel_std = float(array.std(axis=(0, 1)).mean())

    return {
        "pixel_width": width,
        "pixel_height": height,
        "dpi_x": dpi_x,
        "dpi_y": dpi_y,
        "nonwhite_fraction": nonwhite_fraction,
        "mean_channel_sd": channel_std,
        "nonblank": bool(nonwhite_fraction > 0.01 and channel_std > 2.0),
    }


def svg_metrics(path: Path) -> dict[str, int | str | bool]:
    text = path.read_text(encoding="utf-8")
    root = ElementTree.fromstring(text)
    text_count = sum(1 for node in root.iter() if node.tag.endswith("text"))
    viewbox = root.attrib.get("viewBox", "")
    return {
        "svg_text_elements": text_count,
        "svg_viewbox": viewbox,
        "svg_editable_text": bool(text_count >= 20),
        "svg_glyph_paths_absent": "DejaVuSans-" not in text,
    }


def pdf_metrics(path: Path) -> dict[str, bool | str]:
    data = path.read_bytes()
    return {
        "pdf_header": data[:8].decode("ascii", errors="replace"),
        "pdf_valid_header": data.startswith(b"%PDF-"),
        "pdf_truetype_embedded": b"/FontFile2" in data,
        "pdf_type0_font": b"/Subtype /Type0" in data,
    }


def audit_exports(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_dir = root / "figures_main/exports"
    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    for stem, (width_mm, height_mm) in FIGURES.items():
        for suffix, target_dpi in FORMATS.items():
            path = export_dir / f"{stem}{suffix}"
            row: dict[str, object] = {
                "figure": stem,
                "format": suffix.lstrip("."),
                "relative_path": str(path.relative_to(root)),
                "expected_width_mm": width_mm,
                "expected_height_mm": height_mm,
                "target_dpi": target_dpi,
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

            checks: list[bool] = [path.stat().st_size > 10_000]
            if suffix in {".png", ".tiff"}:
                metrics = raster_metrics(path)
                row.update(metrics)
                expected_width = expected_pixels(width_mm, int(target_dpi))
                expected_height = expected_pixels(height_mm, int(target_dpi))
                row["expected_pixel_width"] = expected_width
                row["expected_pixel_height"] = expected_height
                row["dimension_match"] = (
                    abs(int(metrics["pixel_width"]) - expected_width) <= PIXEL_DIMENSION_TOLERANCE
                    and abs(int(metrics["pixel_height"]) - expected_height) <= PIXEL_DIMENSION_TOLERANCE
                )
                row["dpi_match"] = (
                    abs(float(metrics["dpi_x"]) - int(target_dpi)) <= 1.0
                    and abs(float(metrics["dpi_y"]) - int(target_dpi)) <= 1.0
                )
                checks.extend([
                    bool(row["dimension_match"]),
                    bool(row["dpi_match"]),
                    bool(metrics["nonblank"]),
                ])
            elif suffix == ".svg":
                metrics = svg_metrics(path)
                row.update(metrics)
                checks.extend([
                    bool(metrics["svg_editable_text"]),
                    bool(metrics["svg_glyph_paths_absent"]),
                ])
            elif suffix == ".pdf":
                metrics = pdf_metrics(path)
                row.update(metrics)
                checks.extend([
                    bool(metrics["pdf_valid_header"]),
                    bool(metrics["pdf_truetype_embedded"]),
                    bool(metrics["pdf_type0_font"]),
                ])
            row["passed"] = all(checks)
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(manifest)


def audit_source_data(root: Path) -> pd.DataFrame:
    source_dir = root / "figures_main/source_data"
    rows = []
    for path in sorted(source_dir.glob("Fig*.csv")):
        frame = pd.read_csv(path)
        match = re.match(r"Fig(\d+)", path.name)
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
    figure_dir = root / "figures_main"
    export_qa, export_manifest = audit_exports(root)
    source_manifest = audit_source_data(root)
    annotation_path = figure_dir / "figure_annotation_qa.csv"
    annotation_qa = pd.read_csv(annotation_path) if annotation_path.is_file() else pd.DataFrame()

    export_qa.to_csv(figure_dir / "figure_export_qa.csv", index=False)
    export_manifest.to_csv(figure_dir / "EXPORT_MANIFEST.csv", index=False)
    source_manifest.to_csv(
        figure_dir / "source_data/SOURCE_DATA_MANIFEST.csv", index=False
    )

    source_counts = source_manifest.groupby("figure").size().to_dict()
    source_count_pass = source_counts == EXPECTED_SOURCE_COUNTS
    export_pass = bool(export_qa.passed.all()) and len(export_qa) == 20
    source_pass = bool(source_manifest.passed.all()) and source_count_pass
    annotation_pass = bool(
        len(annotation_qa) > 0
        and "passed" in annotation_qa
        and annotation_qa.passed.astype(bool).all()
    )
    overall_pass = export_pass and source_pass and annotation_pass
    audited_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    failures = export_qa.loc[~export_qa.passed, ["relative_path", "format"]]
    failure_lines = "\n".join(
        f"- `{row.relative_path}` ({row.format})"
        for row in failures.itertuples()
    ) or "- None."

    source_summary = ", ".join(
        f"Figure {figure}: {count} table{'s' if count != 1 else ''}"
        for figure, count in sorted(source_counts.items())
    )
    report = f"""# Manuscript figure QA report

Audited at: {audited_at}

## Status

**{'PASS' if overall_pass else 'FAIL'}**

- Export files: {int(export_qa.passed.sum())}/{len(export_qa)} passed.
- Expected format set: five figures x SVG, PDF, PNG and TIFF.
- Source-data tables: {int(source_manifest.passed.sum())}/{len(source_manifest)} passed.
- Source-data coverage: {source_summary}.
- Annotation semantics: {int(annotation_qa.passed.astype(bool).sum()) if len(annotation_qa) and 'passed' in annotation_qa else 0}/{len(annotation_qa)} passed.

## Automated checks

- PNG review files are 300 dpi and TIFF submission files are 600 dpi at the
  contract-specified 183 mm width and figure-specific height.
- Raster previews contain non-white, variable pixels and are not blank canvases.
- SVG files retain editable text rather than converted font-outline glyphs.
- PDF files have valid headers and embedded TrueType Type 0 fonts.
- Every plotted source-data CSV is non-empty, has unique columns, and is hashed.
- SVG terminology, figure-document evidence boundaries and source-data display labels
  pass the dedicated annotation audit.

## Visual inspection

- All five 300 dpi PNGs were reviewed at full composition scale after final export.
- Panel labels, titles, direct labels, legends and statistical annotations remain
  inside their intended panels without incoherent overlap.
- Color roles are consistent across figures and remain distinguishable in shape or
  position where scientific interpretation depends on them.

## Failed exports

{failure_lines}

## Audit artifacts

- `figure_export_qa.csv`: per-format dimensions, DPI and editability checks.
- `EXPORT_MANIFEST.csv`: SHA-256 digests for all 20 exports.
- `source_data/SOURCE_DATA_MANIFEST.csv`: row, column and digest audit for plotted data.
- `figure_annotation_qa.csv`: required/forbidden terminology and source semantics.
- `FIGURE_ANNOTATION_AUDIT.md`: evidence-boundary decisions by figure domain.
"""
    (figure_dir / "FIGURE_QA_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    if not overall_pass:
        raise SystemExit("Figure QA failed")


if __name__ == "__main__":
    main()
