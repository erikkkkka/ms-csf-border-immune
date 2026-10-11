#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--chain", default="C")
    parser.add_argument("--resname", default="BZ7")
    parser.add_argument("--resnum", type=int, default=301)
    parser.add_argument("--padding", type=float, default=6.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source = root / "data" / "reference" / "6AVO.pdb"
    out_dir = root / "results" / "docking" / "6AVO_allosteric_redocking"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = source.read_text().splitlines()

    ligand_lines = []
    ligand_serials = set()
    coordinates = []
    receptor_lines = []
    for line in lines:
        record = line[:6].strip()
        if record == "ATOM":
            receptor_lines.append(line)
        elif record == "TER":
            receptor_lines.append(line)
        elif record == "HETATM":
            resname = line[17:20].strip()
            chain = line[21].strip()
            resnum = int(line[22:26])
            if resname == args.resname and chain == args.chain and resnum == args.resnum:
                ligand_lines.append(line)
                ligand_serials.add(int(line[6:11]))
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    if not ligand_lines:
        raise RuntimeError("Requested ligand instance was not found")
    conect_lines = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        serials = [int(line[i:i + 5]) for i in range(6, len(line), 5) if line[i:i + 5].strip()]
        if serials and serials[0] in ligand_serials:
            conect_lines.append(line)

    receptor_path = out_dir / "6AVO_protein_only.pdb"
    ligand_path = out_dir / f"6AVO_{args.resname}_chain{args.chain}_crystal.pdb"
    receptor_path.write_text("\n".join(receptor_lines + ["END"]) + "\n")
    ligand_path.write_text("\n".join(ligand_lines + conect_lines + ["END"]) + "\n")

    xyz = np.asarray(coordinates)
    center = xyz.mean(axis=0)
    size = xyz.max(axis=0) - xyz.min(axis=0) + 2 * args.padding
    pd.DataFrame([{
        "pdb_id": "6AVO",
        "ligand": args.resname,
        "chain": args.chain,
        "n_heavy_atoms": len(xyz),
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "padding_each_side_A": args.padding,
        "structure_resolution_A": 3.8,
        "site_type": "noncompetitive_allosteric_immunoproteasome_site",
    }]).to_csv(out_dir / "6AVO_BZ7_grid_definition.csv", index=False)


if __name__ == "__main__":
    main()
