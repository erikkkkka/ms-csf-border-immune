#!/usr/bin/env python3
"""Audit figure annotation semantics across SVGs, legends and source-data tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FIGURE_RULES = {
    "Figure_1_CSF_monocyte_BAM_continuum": {
        "required": [
            "PBMC monocyte root",
            "CSF monocyte entry",
            "C1Q/APOE-high BAM-like pole",
            "Programs across BAM annotation confidence",
            "High-confidence BAM endpoint versus root AUC",
        ],
        "forbidden": ["BAM-like intermediate", "BAM-like terminal"],
    },
    "Figure_2_MS_inflammatory_adaptation_decoupling": {
        "required": [
            "Decoupled inflammatory state",
            "BAM-like adaptation pole",
            "GSE328265 new-only: frozen projection",
            "Cross-species LAM-program excess",
            "GM-CSF LAM precursor program",
        ],
        "forbidden": [
            "Decoupled inflammatory precursor",
            "Cross-species LAM-state identity",
            "blind projection",
            "Unseen cohort",
        ],
    },
    "Figure_3_receiver_regulatory_analysis": {
        "required": [
            "IFNGR1/2 receiver state",
            "TNFRSF1A/B receiver state",
            "CSF2RA/B",
            "Late - mid trajectory window",
            "CSF T-cell ligand scores",
            "Receiver circuit ranking",
        ],
        "forbidden": [
            "IFNG receptor state",
            "TNF receptor state",
            "Terminal - transition",
            "terminal adaptation",
            "Circuit priority",
        ],
    },
    "Figure_4_in_vivo_CRISPR_directional_control": {
        "required": [
            "In vivo receptor KO",
            "Frozen human CSF",
            "IFNGR1 response",
            "Direct KO-like",
            "KO-like inflammatory reversal",
            "STAT1 adaptation veto",
        ],
        "forbidden": [
            "clinical efficacy",
            "validated therapy",
        ],
    },
    "Figure_5_transition_aware_chemical_probe": {
        "required": [
            "76 NIBR U2OS profiles",
            "1 phenotypic profile",
            "Only one U2OS profile",
            "PDPK1-pathway comparators",
            "phenotypic probe",
            "unresolved",
        ],
        "forbidden": ["preclinical probe", "PDPK1 controls do not reproduce"],
    },
}

SOURCE_COLUMNS = {
    "Fig1d_module_progression.csv": ["stage_display", "annotation_role"],
    "Fig2b_donor_continuous_decoupling.csv": ["validation_role"],
    "Fig2d_external_two_axis_cells.csv": ["validation_role"],
    "Fig2e_cross_species_LAM_cohort_effects.csv": ["endpoint_display", "evidence_type"],
    "Fig3b_decoupled_receiver_regulator_meta.csv": ["display_label", "molecular_layer"],
    "Fig3c_regulator_trajectory_dynamics.csv": ["contrast_display", "contrast_definition"],
    "Fig3d_receiver_circuit_ranking.csv": ["circuit_display", "evidence_scope"],
    "Fig3e_sender_precursor_associations.csv": ["ligand_display", "sender_cell_type"],
    "Fig4b_in_vivo_response_bridge.csv": ["perturbation_display", "endpoint_display", "signature_scope"],
    "Fig4c_IFNGR1_cohort_directionality.csv": ["signature_display", "endpoint_display", "statistical_unit"],
    "Fig4d_direct_KO_triangulation.csv": ["perturbation_display", "interpretation_scope"],
    "Fig4e_virtual_downstream_control.csv": ["virtual_perturbation", "endpoint_display", "evidence_role"],
    "Fig5a_screen_cascade.csv": ["perturbation_cell_line", "screen_role"],
    "Fig5b_NIBR_profile_screen.csv": ["perturbation_cell_line", "evidence_role"],
    "Fig5g_developability_gate.csv": ["display_value", "assessment"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def svg_text(path: Path) -> str:
    root = ElementTree.parse(path).getroot()
    chunks = []
    for node in root.iter():
        if node.tag.endswith("text"):
            value = " ".join(part.strip() for part in node.itertext() if part.strip())
            if value:
                chunks.append(value)
    return "\n".join(chunks)


def add_check(rows: list[dict[str, object]], check_id: str, scope: str, passed: bool, detail: str) -> None:
    rows.append({
        "check_id": check_id,
        "scope": scope,
        "passed": bool(passed),
        "detail": detail,
    })


def audit_svg_text(root: Path, rows: list[dict[str, object]]) -> None:
    export_dir = root / "figures_main/exports"
    for stem, rules in FIGURE_RULES.items():
        path = export_dir / f"{stem}.svg"
        if not path.is_file():
            add_check(rows, f"svg_exists::{stem}", "SVG", False, str(path))
            continue
        text = svg_text(path)
        normalized = " ".join(text.split())
        for phrase in rules["required"]:
            add_check(
                rows,
                f"svg_required::{stem}::{phrase}",
                "SVG annotation",
                phrase in normalized,
                f"required phrase: {phrase}",
            )
        for phrase in rules["forbidden"]:
            add_check(
                rows,
                f"svg_forbidden::{stem}::{phrase}",
                "SVG annotation",
                phrase not in normalized,
                f"forbidden phrase: {phrase}",
            )


def audit_documents(root: Path, rows: list[dict[str, object]]) -> None:
    figure_dir = root / "figures_main"
    contract = (figure_dir / "FIGURE_CONTRACT.md").read_text(encoding="utf-8")
    legends = (figure_dir / "FIGURE_LEGENDS_DRAFT.md").read_text(encoding="utf-8")
    contract_normalized = " ".join(contract.split())
    legends_normalized = " ".join(legends.split())
    required = {
        "contract::BAM-like scope": (contract_normalized, "not presented as cell-identity assignments"),
        "contract::trajectory windows": (contract_normalized, "analysis strata, not discrete cell types"),
        "contract::experimental bridge": (contract_normalized, "external in vivo receptor-KO screen"),
        "contract::virtual perturbation": (contract_normalized, "Virtual IRF1 and STAT1 knockout"),
        "legend::receptor components": (legends_normalized, "standardized minimum expression across the required receptor components"),
        "legend::new-only projection": (legends_normalized, "new-only GSE328265 cohort"),
        "legend::program projection": (legends_normalized, "gene-program projections, not LAM cell-identity assignments"),
        "legend::U2OS scope": (legends_normalized, "U2OS-derived phenotypic chemical probe with unresolved target attribution"),
    }
    for check_id, (text, phrase) in required.items():
        add_check(rows, check_id, "Figure documents", phrase in text, phrase)
    forbidden = [
        "Cross-species LAM-state identity",
        "IFNG-receptor state",
        "TNF-receptor state",
        "preclinical probe",
    ]
    combined = contract_normalized + " " + legends_normalized
    for phrase in forbidden:
        add_check(
            rows,
            f"documents_forbidden::{phrase}",
            "Figure documents",
            phrase not in combined,
            phrase,
        )


def audit_source_data(root: Path, rows: list[dict[str, object]]) -> None:
    source_dir = root / "figures_main/source_data"
    frames: dict[str, pd.DataFrame] = {}
    for filename, columns in SOURCE_COLUMNS.items():
        path = source_dir / filename
        if not path.is_file():
            add_check(rows, f"source_exists::{filename}", "Source data", False, str(path))
            continue
        frame = pd.read_csv(path)
        frames[filename] = frame
        missing = [column for column in columns if column not in frame.columns]
        add_check(
            rows,
            f"source_columns::{filename}",
            "Source data",
            not missing,
            f"required columns: {', '.join(columns)}; missing: {', '.join(missing) or 'none'}",
        )

    semantic_checks = [
        (
            "source_semantics::Fig1d confidence",
            frames["Fig1d_module_progression.csv"].annotation_role.str.contains("not a discrete cell type").all(),
            "BAM non-core/core labels are confidence definitions",
        ),
        (
            "source_semantics::Fig2 external new-only",
            frames["Fig2b_donor_continuous_decoupling.csv"].loc[
                lambda x: x.dataset.str.contains("GSE328265", regex=False), "validation_role"
            ].str.contains("frozen projection without refitting").all(),
            "GSE328265 donor rows carry the new-only frozen-projection role",
        ),
        (
            "source_semantics::Fig2 program not identity",
            frames["Fig2e_cross_species_LAM_cohort_effects.csv"].evidence_type.str.contains("not cell-identity").all(),
            "LAM endpoints are program projections",
        ),
        (
            "source_semantics::Fig3 receptor labels",
            {"IFNGR1/2", "TNFRSF1A/B", "CSF2RA/B"}.issubset(
                set(frames["Fig3b_decoupled_receiver_regulator_meta.csv"].display_label)
            ),
            "ligand-family keys are displayed as receptor complexes",
        ),
        (
            "source_semantics::Fig3 trajectory windows",
            frames["Fig3c_regulator_trajectory_dynamics.csv"].contrast_definition.str.contains("pseudotime").all(),
            "every trajectory contrast carries an explicit pseudotime definition",
        ),
        (
            "source_semantics::Fig4 in vivo response",
            frames["Fig4b_in_vivo_response_bridge.csv"].signature_scope.eq(
                "author-defined in vivo cytokine-response signature"
            ).all(),
            "all primary bridge rows identify the in vivo response-signature scope",
        ),
        (
            "source_semantics::Fig4 direct KO",
            frames["Fig4d_direct_KO_triangulation.csv"].interpretation_scope.eq(
                "direct-KO sensitivity projection"
            ).all(),
            "direct KO panel is labelled as a sensitivity projection",
        ),
        (
            "source_semantics::Fig4 virtual",
            frames["Fig4e_virtual_downstream_control.csv"].virtual_perturbation.eq("KO").all(),
            "downstream IRF1/STAT1 controls remain explicitly virtual",
        ),
        (
            "source_semantics::Fig5 U2OS",
            frames["Fig5b_NIBR_profile_screen.csv"].perturbation_cell_line.eq("U2OS").all(),
            "all screening profiles identify the U2OS system",
        ),
        (
            "source_semantics::Fig5 unresolved target",
            (
                frames["Fig5g_developability_gate.csv"].loc[
                    lambda x: x.metric.eq("Target attribution"), "display_value"
                ].eq("unresolved").all()
                and frames["Fig5g_developability_gate.csv"].metric.eq("Target attribution").any()
            ),
            "BX-912 target attribution is unresolved",
        ),
    ]
    for check_id, passed, detail in semantic_checks:
        add_check(rows, check_id, "Source semantics", bool(passed), detail)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: list[dict[str, object]] = []
    audit_svg_text(root, rows)
    audit_documents(root, rows)
    audit_source_data(root, rows)
    audit = pd.DataFrame(rows)
    figure_dir = root / "figures_main"
    audit.to_csv(figure_dir / "figure_annotation_qa.csv", index=False)

    n_pass = int(audit.passed.sum())
    n_total = len(audit)
    failures = audit.loc[~audit.passed]
    failure_lines = "\n".join(
        f"- `{row.check_id}`: {row.detail}" for row in failures.itertuples()
    ) or "- None."
    audited_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = f"""# Figure annotation audit

Audited at: {audited_at}

## Status

**{'PASS' if n_pass == n_total else 'FAIL'}** ({n_pass}/{n_total} checks passed)

## Annotation decisions

| Domain | Display decision | Evidence boundary |
|---|---|---|
| BAM continuum | BAM-like pole; non-core/core are confidence definitions | No lineage-origin claim |
| Two-axis state | Decoupled inflammatory state and BAM-like adaptation pole | Continuous regions, not cell types |
| External validation | GSE328265 new-only frozen projection | No refitting; newly sequenced subjects only |
| Cross-species LAM | LAM gene-program excess | No LAM identity transfer |
| Receiver circuits | IFNGR1/2, TNFRSF1A/B and CSF2RA/B receiver states | No ligand-causality claim |
| Trajectory contrasts | Early, mid and late pseudotime windows | Analysis strata, not clusters |
| Perturbation | Virtual KO/OE in observed GRNs | Network sensitivity, not efficacy |
| BX-912 | U2OS phenotypic probe; target unresolved | Not PDPK1-confirmed or CNS-ready |

## Failed checks

{failure_lines}

## Audit scope

- Required and forbidden terminology in editable SVG exports.
- Evidence-boundary language in the figure contract and legends.
- Display-label, validation-role and evidence-type fields in plotted source data.
"""
    (figure_dir / "FIGURE_ANNOTATION_AUDIT.md").write_text(report, encoding="utf-8")
    print(report)
    if n_pass != n_total:
        raise SystemExit("Figure annotation audit failed")


if __name__ == "__main__":
    main()
