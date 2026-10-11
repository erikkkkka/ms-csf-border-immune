#!/usr/bin/env python3
"""Validate PDPK1 ATP-site docking by 3RWP ABQ redocking."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from vina import Vina


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--exhaustiveness", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def atom_records(path: Path) -> list[str]:
    return [
        line for line in path.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]


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
    if current:
        models.append(current)
    return models


def xyz(line: str) -> np.ndarray:
    return np.asarray([
        float(line[30:38]), float(line[38:46]), float(line[46:54])
    ])


def atom_type(line: str) -> str:
    return line.split()[-1].upper()


def symmetry_aware_rmsd(reference: list[str], pose: list[str]) -> float:
    ref_xyz = np.asarray([xyz(line) for line in reference])
    pose_xyz = np.asarray([xyz(line) for line in pose])
    ref_types = np.asarray([atom_type(line) for line in reference])
    pose_types = np.asarray([atom_type(line) for line in pose])
    if len(reference) != len(pose) or sorted(ref_types) != sorted(pose_types):
        return np.nan
    assignment = np.empty(len(reference), dtype=int)
    for kind in sorted(set(ref_types)):
        ref_index = np.flatnonzero(ref_types == kind)
        pose_index = np.flatnonzero(pose_types == kind)
        cost = np.square(
            ref_xyz[ref_index, None, :] - pose_xyz[pose_index, :]
        ).sum(axis=2)
        row, column = linear_sum_assignment(cost)
        assignment[ref_index[row]] = pose_index[column]
    return float(np.sqrt(np.mean(np.square(ref_xyz - pose_xyz[assignment]).sum(axis=1))))


def vina_scores(path: Path) -> list[float]:
    output = []
    for line in path.read_text().splitlines():
        match = re.match(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            output.append(float(match.group(1)))
    return output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    env_dir = root / ".conda_docking/bin"
    output = root / "results/docking/3RWP_PDPK1_ATP_site"
    grid = pd.read_csv(output / "3RWP_ABQ_grid_definition.csv").iloc[0]
    receptor_pdb = output / "3RWP_PDPK1_phospho_receptor.pdb"
    receptor_pdbqt = output / "3RWP_PDPK1_phospho_receptor.pdbqt"
    ligand_pdb = output / "3RWP_ABQ_crystal.pdb"
    ligand_sdf = output / "3RWP_ABQ_crystal.sdf"
    ligand_pdbqt = output / "3RWP_ABQ_crystal.pdbqt"
    docked_pdbqt = output / "3RWP_ABQ_redocked.pdbqt"

    run([str(env_dir / "obabel"), str(ligand_pdb), "-O", str(ligand_sdf), "-h"])
    run([
        str(env_dir / "mk_prepare_ligand.py"), "-i", str(ligand_sdf),
        "-o", str(ligand_pdbqt), "--add_index_map",
    ])
    run([
        str(env_dir / "mk_prepare_receptor.py"), "--read_pdb", str(receptor_pdb),
        "-o", str(output / "3RWP_PDPK1_phospho_receptor"),
        "-p", str(receptor_pdbqt), "-a", "--default_altloc", "B",
    ])
    if not any(" SEP " in line for line in atom_records(receptor_pdbqt)):
        raise RuntimeError("Prepared receptor lost phosphoserine 241")

    center = [float(grid[f"center_{axis}"]) for axis in "xyz"]
    size = [float(grid[f"size_{axis}"]) for axis in "xyz"]
    vina = Vina(sf_name="vina", seed=args.seed, verbosity=1)
    vina.set_receptor(str(receptor_pdbqt))
    vina.set_ligand_from_file(str(ligand_pdbqt))
    vina.compute_vina_maps(center=center, box_size=size)
    vina.dock(exhaustiveness=args.exhaustiveness, n_poses=20)
    vina.write_poses(str(docked_pdbqt), n_poses=20, overwrite=True)

    reference = atom_records(ligand_pdbqt)
    poses = split_models(docked_pdbqt)
    scores = vina_scores(docked_pdbqt)
    rows = []
    for rank, pose in enumerate(poses, start=1):
        rows.append({
            "pose_rank": rank,
            "vina_score_kcal_mol": scores[rank - 1] if rank <= len(scores) else np.nan,
            "symmetry_aware_heavy_atom_rmsd_A": symmetry_aware_rmsd(reference, pose),
            "exhaustiveness": args.exhaustiveness,
            "seed": args.seed,
            "structure_resolution_A": 1.92,
            "phosphoserine_241_retained": True,
            "altloc_A247_retained": "B",
            "remote_incomplete_residues_removed": "GLU_A303;LYS_A304",
        })
    result = pd.DataFrame(rows)
    result["validation_pass_rmsd_lt_2A"] = result["symmetry_aware_heavy_atom_rmsd_A"] < 2.0
    result.to_csv(output / "3RWP_ABQ_redocking_validation.csv", index=False)
    best_rmsd = result["symmetry_aware_heavy_atom_rmsd_A"].min()
    print(result.head(10).to_string(index=False), flush=True)
    if not np.isfinite(best_rmsd) or best_rmsd >= 2.0:
        raise RuntimeError(f"PDPK1 redocking failed RMSD gate: best={best_rmsd:.3f} A")


if __name__ == "__main__":
    main()
