#!/usr/bin/env python3
"""Dock BX-912 to the same-series-redocking-validated PDPK1 ATP site."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from vina import Vina


ROOT = Path(__file__).resolve().parents[1]
BX912_SMILES = "BrC1=CN=C(N=C1NCCC2=CN=CN2)NC3=CC(NC(N4CCCC4)=O)=CC=C3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--exhaustiveness", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def split_models(path: Path) -> list[list[str]]:
    models = []
    current = []
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith(("ATOM  ", "HETATM")):
            current.append(line)
        elif line.startswith("ENDMDL") and current:
            models.append(current)
            current = []
    return models


def xyz(line: str) -> np.ndarray:
    return np.asarray([
        float(line[30:38]), float(line[38:46]), float(line[46:54])
    ])


def atom_type(line: str) -> str:
    return line.split()[-1].upper()


def is_hydrogen(line: str) -> bool:
    return atom_type(line).startswith("H")


def scores(path: Path) -> list[float]:
    output = []
    for line in path.read_text().splitlines():
        match = re.match(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            output.append(float(match.group(1)))
    return output


def receptor_atoms(path: Path) -> tuple[np.ndarray, list[str]]:
    coordinates = []
    labels = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or is_hydrogen(line):
            continue
        coordinates.append(xyz(line))
        labels.append(
            f"{line[17:20].strip()}{line[22:26].strip()}:{line[21].strip()}:{line[12:16].strip()}"
        )
    return np.asarray(coordinates), labels


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    env_dir = root / ".conda_docking/bin"
    validation_dir = root / "results/docking/1Z5M_PDPK1_BX320_site"
    validation = pd.read_csv(validation_dir / "1Z5M_BX320_redocking_validation.csv")
    top_validation = validation.loc[validation["pose_rank"] == 1].iloc[0]
    if not bool(top_validation["top_pose_pass"]):
        raise RuntimeError("Same-series PDPK1 top-pose redocking gate has not passed")
    output = root / "results/docking/1Z5M_PDPK1_BX912"
    output.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(validation_dir / "1Z5M_BX320_grid_definition.csv").iloc[0]
    receptor = validation_dir / "1Z5M_PDPK1_phospho_receptor.pdbqt"
    ligand_sdf = output / "BX912_pH7.4.sdf"
    ligand_pdbqt = output / "BX912_pH7.4.pdbqt"
    docked = output / "BX912_docked.pdbqt"

    run([
        str(env_dir / "obabel"), f"-:{BX912_SMILES}", "--gen3d", "-p", "7.4",
        "-O", str(ligand_sdf),
    ])
    run([
        str(env_dir / "mk_prepare_ligand.py"), "-i", str(ligand_sdf),
        "-o", str(ligand_pdbqt), "--add_index_map",
    ])
    molecule = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
    if molecule is None:
        raise RuntimeError("RDKit could not read BX-912 SDF")
    formal_charge = int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))

    center = [float(grid[f"center_{axis}"]) for axis in "xyz"]
    size = [float(grid[f"size_{axis}"]) for axis in "xyz"]
    vina = Vina(sf_name="vina", seed=args.seed, verbosity=1)
    vina.set_receptor(str(receptor))
    vina.set_ligand_from_file(str(ligand_pdbqt))
    vina.compute_vina_maps(center=center, box_size=size)
    vina.dock(exhaustiveness=args.exhaustiveness, n_poses=20)
    vina.write_poses(str(docked), n_poses=20, overwrite=True)

    receptor_xyz, receptor_labels = receptor_atoms(receptor)
    poses = split_models(docked)
    vina_scores = scores(docked)
    rows = []
    contacts = []
    for rank, pose in enumerate(poses, start=1):
        heavy_xyz = np.asarray([xyz(line) for line in pose if not is_hydrogen(line)])
        distance_matrix = np.linalg.norm(
            receptor_xyz[:, None, :] - heavy_xyz[None, :, :], axis=2
        )
        receptor_distance = distance_matrix.min(axis=1)
        residues = sorted({
            label.rsplit(":", 1)[0]
            for label, value in zip(receptor_labels, receptor_distance)
            if value <= 4.0
        })
        ala162_indices = [
            index for index, label in enumerate(receptor_labels)
            if label.startswith("ALA162:A:")
        ]
        ala162_minimum = float(distance_matrix[ala162_indices].min())
        ala162_backbone_indices = [
            index for index, label in enumerate(receptor_labels)
            if label in {"ALA162:A:N", "ALA162:A:O"}
        ]
        ala162_backbone_minimum = float(distance_matrix[ala162_backbone_indices].min())
        rows.append({
            "pose_rank": rank,
            "vina_score_kcal_mol": vina_scores[rank - 1] if rank <= len(vina_scores) else np.nan,
            "n_residues_within_4A": len(residues),
            "residues_within_4A": ";".join(residues),
            "minimum_distance_to_ALA162_A": ala162_minimum,
            "minimum_distance_to_ALA162_backbone_N_or_O_A": ala162_backbone_minimum,
            "prepared_ligand_formal_charge": formal_charge,
            "prepared_at_pH": 7.4,
            "exhaustiveness": args.exhaustiveness,
            "seed": args.seed,
            "control_ligand": "BX320_LI8",
            "control_top_pose_RMSD_A": top_validation["symmetry_aware_heavy_atom_rmsd_A"],
            "interpretation": "validated_site_pose_hypothesis_not_affinity",
        })
        contacts.extend({"pose_rank": rank, "contact_residue": value} for value in residues)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "BX912_PDPK1_pose_summary.csv", index=False)
    pd.DataFrame(contacts).to_csv(output / "BX912_PDPK1_contacts.csv", index=False)
    (output / "BX912_input_smiles.txt").write_text(BX912_SMILES + "\n")
    print(summary.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
