#!/usr/bin/env python3
"""Assess BX-912 PDPK1 pose robustness by blind and congeneric-seeded docking."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdMolAlign import AlignMol
from vina import Vina


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260813, 20260817, 20260823, 20260829, 20260831)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--exhaustiveness", type=int, default=64)
    return parser.parse_args()


def xyz(line: str) -> np.ndarray:
    return np.asarray([
        float(line[30:38]), float(line[38:46]), float(line[46:54])
    ])


def first_model(path: Path) -> list[str]:
    output = []
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "HETATM")):
            output.append(line)
    return output


def vina_score(path: Path) -> float:
    for line in path.read_text().splitlines():
        match = re.match(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            return float(match.group(1))
    return np.nan


def receptor_ala162(path: Path) -> dict[str, np.ndarray]:
    output = {}
    for line in path.read_text().splitlines():
        if (
            line.startswith(("ATOM  ", "HETATM"))
            and line[17:20].strip() == "ALA"
            and line[22:26].strip() == "162"
            and line[12:16].strip() in {"N", "O"}
        ):
            output[line[12:16].strip()] = xyz(line)
    return output


def pyrimidine_nitrogen_distance(pose: list[str], ala162: dict[str, np.ndarray]) -> float:
    nitrogen_xyz = [
        xyz(line) for line in pose
        if line[12:16].strip() == "N" and line.split()[-1].upper() in {"N", "NA"}
    ]
    if not nitrogen_xyz:
        return np.nan
    return float(min(
        np.linalg.norm(atom - backbone)
        for atom in nitrogen_xyz
        for backbone in ala162.values()
    ))


def congeneric_seeded_sdf(reference_sdf: Path, bx912_sdf: Path, output_sdf: Path) -> float:
    reference = Chem.RemoveHs(Chem.SDMolSupplier(str(reference_sdf), removeHs=False)[0])
    bx912 = Chem.RemoveHs(Chem.SDMolSupplier(str(bx912_sdf), removeHs=False)[0])
    mcs = rdFMCS.FindMCS(
        [reference, bx912], atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder, ringMatchesRingOnly=True,
        completeRingsOnly=True, timeout=60,
    )
    query = Chem.MolFromSmarts(mcs.smartsString)
    reference_match = reference.GetSubstructMatch(query)
    bx912_match = bx912.GetSubstructMatch(query)
    if len(reference_match) < 20 or len(reference_match) != len(bx912_match):
        raise RuntimeError("Congeneric BX-320/BX-912 scaffold match was insufficient")
    rmsd = float(AlignMol(
        bx912, reference, atomMap=list(zip(bx912_match, reference_match))
    ))
    bx912 = Chem.AddHs(bx912, addCoords=True)
    Chem.MolToMolFile(bx912, str(output_sdf))
    return rmsd


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    validation_dir = root / "results/docking/1Z5M_PDPK1_BX320_site"
    output = root / "results/docking/1Z5M_PDPK1_BX912"
    receptor = validation_dir / "1Z5M_PDPK1_phospho_receptor.pdbqt"
    ligand = output / "BX912_pH7.4.pdbqt"
    grid = pd.read_csv(validation_dir / "1Z5M_BX320_grid_definition.csv").iloc[0]
    center = [float(grid[f"center_{axis}"]) for axis in "xyz"]
    size = [float(grid[f"size_{axis}"]) for axis in "xyz"]
    ala162 = receptor_ala162(receptor)

    rows = []
    for seed in SEEDS:
        docked = output / f"BX912_blind_seed_{seed}.pdbqt"
        vina = Vina(sf_name="vina", seed=seed, verbosity=0)
        vina.set_receptor(str(receptor))
        vina.set_ligand_from_file(str(ligand))
        vina.compute_vina_maps(center=center, box_size=size)
        vina.dock(exhaustiveness=args.exhaustiveness, n_poses=20)
        vina.write_poses(str(docked), n_poses=20, overwrite=True)
        pose = first_model(docked)
        rows.append({
            "analysis": "blind_docking_top_pose",
            "seed": seed,
            "vina_score_kcal_mol": vina_score(docked),
            "minimum_ligand_nitrogen_to_ALA162_backbone_A":
                pyrimidine_nitrogen_distance(pose, ala162),
            "known_hinge_orientation_recovered_lt_3.5A":
                pyrimidine_nitrogen_distance(pose, ala162) < 3.5,
            "congeneric_scaffold_alignment_RMSD_A": np.nan,
        })

    seeded_sdf = output / "BX912_BX320_congeneric_seeded.sdf"
    alignment_rmsd = congeneric_seeded_sdf(
        validation_dir / "1Z5M_BX320_crystal.sdf",
        output / "BX912_pH7.4.sdf",
        seeded_sdf,
    )
    seeded_pdbqt = output / "BX912_BX320_congeneric_seeded.pdbqt"
    import subprocess
    subprocess.run([
        str(root / ".conda_docking/bin/mk_prepare_ligand.py"), "-i", str(seeded_sdf),
        "-o", str(seeded_pdbqt), "--add_index_map",
    ], check=True)
    vina = Vina(sf_name="vina", seed=SEEDS[0], verbosity=0)
    vina.set_receptor(str(receptor))
    vina.set_ligand_from_file(str(seeded_pdbqt))
    vina.compute_vina_maps(center=center, box_size=size)
    seeded_score = float(vina.score()[0])
    optimized_score = float(vina.optimize()[0])
    optimized = output / "BX912_BX320_congeneric_local_optimized.pdbqt"
    vina.write_pose(str(optimized), overwrite=True)
    optimized_pose = [
        line for line in optimized.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    rows.append({
        "analysis": "congeneric_seeded_local_optimization",
        "seed": SEEDS[0],
        "vina_score_kcal_mol": optimized_score,
        "preoptimization_score_kcal_mol": seeded_score,
        "minimum_ligand_nitrogen_to_ALA162_backbone_A":
            pyrimidine_nitrogen_distance(optimized_pose, ala162),
        "known_hinge_orientation_recovered_lt_3.5A":
            pyrimidine_nitrogen_distance(optimized_pose, ala162) < 3.5,
        "congeneric_scaffold_alignment_RMSD_A": alignment_rmsd,
    })
    result = pd.DataFrame(rows)
    result.to_csv(output / "BX912_pose_robustness.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
