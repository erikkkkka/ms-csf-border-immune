#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from vina import Vina


ACEBILUSTAT_SMILES = (
    "O=C(O)c1ccc(CN2C[C@@H]3C[C@H]2CN3Cc2ccc(Oc3ccc(-c4ncco4)cc3)cc2)cc1"
)


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
    if current:
        models.append(current)
    return models


def vina_scores(path: Path) -> list[float]:
    scores = []
    for line in path.read_text().splitlines():
        match = re.match(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            scores.append(float(match.group(1)))
    return scores


def xyz(line: str) -> np.ndarray:
    return np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def element(line: str) -> str:
    pdb_element = line[76:78].strip()
    if pdb_element:
        return pdb_element.upper()
    atom_type = line.split()[-1]
    return re.sub(r"[^A-Za-z]", "", atom_type).upper()


def receptor_atoms(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    coords = []
    labels = []
    zinc = None
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        coord = xyz(line)
        resname = line[17:20].strip()
        chain = line[21].strip()
        resnum = line[22:26].strip()
        atom_name = line[12:16].strip()
        if resname == "ZN":
            zinc = coord
        elif not element(line).startswith("H"):
            coords.append(coord)
            labels.append(f"{resname}{resnum}:{chain}:{atom_name}")
    if zinc is None:
        raise RuntimeError("Catalytic zinc was not found in receptor PDBQT")
    return np.asarray(coords), labels, zinc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--exhaustiveness", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    env_dir = root / ".conda_docking" / "bin"
    validation_dir = root / "results" / "docking" / "8QQ4_LTA4H_redocking"
    out_dir = root / "results" / "docking" / "8QQ4_LTA4H_acebilustat"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(validation_dir / "8QQ4_WEE_grid_definition.csv").iloc[0]
    receptor_pdbqt = validation_dir / "8QQ4_LTA4H_protein_Zn.pdbqt"
    ligand_sdf = out_dir / "acebilustat_pH7.4.sdf"
    ligand_pdbqt = out_dir / "acebilustat_pH7.4.pdbqt"
    docked_pdbqt = out_dir / "acebilustat_docked.pdbqt"

    run([
        str(env_dir / "obabel"), f"-:{ACEBILUSTAT_SMILES}", "--gen3d", "-p", "7.4",
        "-O", str(ligand_sdf),
    ])
    run([
        str(env_dir / "mk_prepare_ligand.py"), "-i", str(ligand_sdf),
        "-o", str(ligand_pdbqt), "--add_index_map",
    ])

    mol = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
    if mol is None:
        raise RuntimeError("RDKit could not read the prepared acebilustat SDF")
    formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))

    center = [float(grid[f"center_{axis}"]) for axis in "xyz"]
    size = [float(grid[f"size_{axis}"]) for axis in "xyz"]
    vina = Vina(sf_name="vina", seed=args.seed, verbosity=1)
    vina.set_receptor(str(receptor_pdbqt))
    vina.set_ligand_from_file(str(ligand_pdbqt))
    vina.compute_vina_maps(center=center, box_size=size)
    vina.dock(exhaustiveness=args.exhaustiveness, n_poses=20)
    vina.write_poses(str(docked_pdbqt), n_poses=20, overwrite=True)

    protein_xyz, protein_labels, zinc_xyz = receptor_atoms(receptor_pdbqt)
    poses = split_models(docked_pdbqt)
    scores = vina_scores(docked_pdbqt)
    rows = []
    contact_rows = []
    for rank, pose in enumerate(poses, start=1):
        heavy_lines = [line for line in pose if not element(line).startswith("H")]
        oxygen_lines = [line for line in heavy_lines if element(line).startswith("O")]
        heavy_xyz = np.asarray([xyz(line) for line in heavy_lines])
        oxygen_xyz = np.asarray([xyz(line) for line in oxygen_lines])
        min_zn_o = float(np.min(np.linalg.norm(oxygen_xyz - zinc_xyz, axis=1)))
        distances = np.linalg.norm(
            protein_xyz[:, None, :] - heavy_xyz[None, :, :], axis=2
        ).min(axis=1)
        contacts = sorted({
            label.rsplit(":", 1)[0]
            for label, distance in zip(protein_labels, distances)
            if distance <= 4.0
        })
        rows.append({
            "pose_rank": rank,
            "vina_score_kcal_mol": scores[rank - 1] if rank <= len(scores) else np.nan,
            "minimum_Zn_to_ligand_oxygen_A": min_zn_o,
            "n_protein_residues_within_4A": len(contacts),
            "protein_residues_within_4A": ";".join(contacts),
            "prepared_ligand_formal_charge": formal_charge,
            "prepared_at_pH": 7.4,
            "exhaustiveness": args.exhaustiveness,
            "seed": args.seed,
            "interpretation": "pose_hypothesis_only_not_affinity",
        })
        for contact in contacts:
            contact_rows.append({"pose_rank": rank, "contact_residue": contact})

    pd.DataFrame(rows).to_csv(out_dir / "acebilustat_docking_pose_summary.csv", index=False)
    pd.DataFrame(contact_rows).to_csv(out_dir / "acebilustat_docking_contacts.csv", index=False)
    (out_dir / "acebilustat_input_smiles.txt").write_text(ACEBILUSTAT_SMILES + "\n")


if __name__ == "__main__":
    main()
