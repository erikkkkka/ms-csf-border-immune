#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--padding", type=float, default=7.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source = root / "data" / "reference" / "8QQ4.pdb"
    out_dir = root / "results" / "docking" / "8QQ4_LTA4H_redocking"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = source.read_text().splitlines()

    ligand_lines = []
    ligand_serials = set()
    coordinates = []
    receptor_lines = []
    for line in lines:
        record = line[:6].strip()
        is_atom_record = record in {"ATOM", "HETATM"}
        resname = line[17:20].strip() if is_atom_record else ""
        chain = line[21].strip() if is_atom_record else ""
        resnum = int(line[22:26]) if is_atom_record and line[22:26].strip() else None
        if record == "ATOM":
            receptor_lines.append(line)
        elif record == "TER":
            receptor_lines.append(line)
        elif record == "HETATM" and resname == "ZN" and chain == "A" and resnum == 701:
            receptor_lines.append(line)
        elif record == "HETATM" and resname == "WEE" and chain == "A" and resnum == 708:
            ligand_lines.append(line)
            ligand_serials.add(int(line[6:11]))
            coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    if len(ligand_lines) != 25:
        raise RuntimeError(f"Expected 25 WEE atoms, found {len(ligand_lines)}")
    if not any(line[17:20].strip() == "ZN" for line in receptor_lines if line.startswith("HETATM")):
        raise RuntimeError("Catalytic zinc was not retained")

    conect_lines = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        serials = [int(line[i:i + 5]) for i in range(6, len(line), 5) if line[i:i + 5].strip()]
        if serials and serials[0] in ligand_serials:
            retained = [serial for serial in serials if serial in ligand_serials]
            if len(retained) >= 2:
                conect_lines.append("CONECT" + "".join(f"{serial:5d}" for serial in retained))

    receptor_path = out_dir / "8QQ4_LTA4H_protein_Zn.pdb"
    ligand_path = out_dir / "8QQ4_WEE_crystal.pdb"
    receptor_path.write_text("\n".join(receptor_lines + ["END"]) + "\n")
    ligand_path.write_text("\n".join(ligand_lines + conect_lines + ["END"]) + "\n")

    xyz = np.asarray(coordinates)
    center = xyz.mean(axis=0)
    size = xyz.max(axis=0) - xyz.min(axis=0) + 2 * args.padding
    pd.DataFrame([{
        "pdb_id": "8QQ4",
        "target": "human_LTA4H",
        "ligand": "WEE",
        "chain": "A",
        "residue_number": 708,
        "n_heavy_atoms": len(xyz),
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "padding_each_side_A": args.padding,
        "structure_resolution_A": 1.6,
        "catalytic_zinc_retained": True,
        "site_type": "zinc_metalloenzyme_active_site",
    }]).to_csv(out_dir / "8QQ4_WEE_grid_definition.csv", index=False)


if __name__ == "__main__":
    main()
