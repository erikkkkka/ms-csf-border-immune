#!/usr/bin/env python3
"""Prepare the BX-320/PDPK1 structure 1Z5M for same-series redocking."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--padding", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source = root / "data/reference/1Z5M.pdb"
    output = root / "results/docking/1Z5M_PDPK1_BX320_site"
    output.mkdir(parents=True, exist_ok=True)
    lines = source.read_text().splitlines()

    ligand_lines = []
    ligand_serials = set()
    coordinates = []
    receptor_lines = []
    for line in lines:
        record = line[:6].strip()
        is_atom = record in {"ATOM", "HETATM"}
        resname = line[17:20].strip() if is_atom else ""
        chain = line[21].strip() if is_atom else ""
        resnum = int(line[22:26]) if is_atom and line[22:26].strip() else None
        if record == "ATOM":
            receptor_lines.append(line)
        elif record == "HETATM" and resname == "SEP" and chain == "A" and resnum == 241:
            receptor_lines.append(line)
        elif record == "TER":
            receptor_lines.append(line)
        elif record == "HETATM" and resname == "LI8" and chain == "A" and resnum == 501:
            ligand_lines.append(line)
            ligand_serials.add(int(line[6:11]))
            coordinates.append([
                float(line[30:38]), float(line[38:46]), float(line[46:54])
            ])
    if len(ligand_lines) != 35:
        raise RuntimeError(f"Expected 35 LI8/BX-320 heavy atoms, found {len(ligand_lines)}")

    conect_lines = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        serials = [
            int(line[index:index + 5])
            for index in range(6, len(line), 5)
            if line[index:index + 5].strip()
        ]
        if serials and serials[0] in ligand_serials:
            retained = [serial for serial in serials if serial in ligand_serials]
            if len(retained) >= 2:
                conect_lines.append("CONECT" + "".join(f"{serial:5d}" for serial in retained))

    receptor_path = output / "1Z5M_PDPK1_phospho_receptor.pdb"
    ligand_path = output / "1Z5M_BX320_crystal.pdb"
    receptor_path.write_text("\n".join(receptor_lines + ["END"]) + "\n")
    ligand_path.write_text("\n".join(ligand_lines + conect_lines + ["END"]) + "\n")

    xyz = np.asarray(coordinates, dtype=float)
    center = xyz.mean(axis=0)
    size = xyz.max(axis=0) - xyz.min(axis=0) + 2 * args.padding
    pd.DataFrame([{
        "pdb_id": "1Z5M",
        "target": "human_PDPK1",
        "ligand": "LI8_BX320",
        "chain": "A",
        "residue_number": 501,
        "n_heavy_atoms": len(xyz),
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "padding_each_side_A": args.padding,
        "structure_resolution_A": 2.17,
        "phosphoserine_241_retained": True,
        "site_type": "ATP_competitive_kinase_site",
        "same_chemical_series_as_BX912": True,
    }]).to_csv(output / "1Z5M_BX320_grid_definition.csv", index=False)
    pd.DataFrame([{
        "structure_issue": "ARG_A75_missing_sidechain_atoms",
        "prespecified_action": "remove_incomplete_residue",
        "rationale": "unresolved_crystallographic_sidechain_outside_ATP_site",
        "minimum_distance_to_BX320_A": 19.070,
        "pocket_relevance": "remote",
    }]).to_csv(output / "1Z5M_receptor_preparation_decisions.csv", index=False)
    print(f"Prepared 1Z5M receptor and {len(xyz)}-atom BX-320 control", flush=True)


if __name__ == "__main__":
    main()
