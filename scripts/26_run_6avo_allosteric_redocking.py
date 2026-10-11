#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from vina import Vina


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def pdbqt_coordinates(path: Path) -> np.ndarray:
    coords = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coords)


def split_models(path: Path) -> list[np.ndarray]:
    models = []
    current = []
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith(("ATOM  ", "HETATM")):
            current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL") and current:
            models.append(np.asarray(current))
            current = []
    if current:
        models.append(np.asarray(current))
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    env_dir = root / ".conda_docking" / "bin"
    out_dir = root / "results" / "docking" / "6AVO_allosteric_redocking"
    grid = pd.read_csv(out_dir / "6AVO_BZ7_grid_definition.csv").iloc[0]
    receptor_pdb = out_dir / "6AVO_protein_only.pdb"
    ligand_pdb = out_dir / "6AVO_BZ7_chainC_crystal.pdb"
    ligand_sdf = out_dir / "6AVO_BZ7_chainC_crystal.sdf"
    ligand_pdbqt = out_dir / "6AVO_BZ7_chainC_crystal.pdbqt"
    receptor_pdbqt = out_dir / "6AVO_protein_only.pdbqt"
    docked_pdbqt = out_dir / "6AVO_BZ7_chainC_redocked.pdbqt"

    run([str(env_dir / "obabel"), str(ligand_pdb), "-O", str(ligand_sdf), "-h"])
    run([
        str(env_dir / "mk_prepare_ligand.py"), "-i", str(ligand_sdf),
        "-o", str(ligand_pdbqt), "--add_index_map",
    ])
    run([
        str(env_dir / "mk_prepare_receptor.py"), "--read_pdb", str(receptor_pdb),
        "-o", str(out_dir / "6AVO_protein_only"), "-p", str(receptor_pdbqt), "-a",
    ])

    center = [float(grid[f"center_{axis}"]) for axis in "xyz"]
    size = [float(grid[f"size_{axis}"]) for axis in "xyz"]
    vina = Vina(sf_name="vina", seed=args.seed, verbosity=1)
    vina.set_receptor(str(receptor_pdbqt))
    vina.set_ligand_from_file(str(ligand_pdbqt))
    vina.compute_vina_maps(center=center, box_size=size)
    vina.dock(exhaustiveness=args.exhaustiveness, n_poses=10)
    vina.write_poses(str(docked_pdbqt), n_poses=10, overwrite=True)

    reference = pdbqt_coordinates(ligand_pdbqt)
    poses = split_models(docked_pdbqt)
    rows = []
    for rank, pose in enumerate(poses, start=1):
        if pose.shape != reference.shape:
            rmsd = np.nan
        else:
            rmsd = float(np.sqrt(np.mean(np.sum((pose - reference) ** 2, axis=1))))
        rows.append({"pose_rank": rank, "heavy_atom_rmsd_A": rmsd})
    result = pd.DataFrame(rows)
    result["validation_pass_rmsd_lt_2A"] = result["heavy_atom_rmsd_A"] < 2.0
    result["exhaustiveness"] = args.exhaustiveness
    result["seed"] = args.seed
    result["structure_resolution_A"] = 3.8
    result.to_csv(out_dir / "6AVO_BZ7_redocking_validation.csv", index=False)


if __name__ == "__main__":
    main()
