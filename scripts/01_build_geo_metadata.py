#!/usr/bin/env python3
import argparse
import csv
import gzip
import re
from pathlib import Path


def read_series_matrix(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip().strip('"') for p in line.split("\t")]
            key = parts[0].lstrip("!")
            rows.setdefault(key, []).append(parts[1:])
    return rows


def get_value(rows, key, idx, default=""):
    values = rows.get(key, [])
    if not values:
        return default
    if idx >= len(values[0]):
        return default
    return values[0][idx]


def get_repeated(rows, key, idx):
    out = []
    for values in rows.get(key, []):
        if idx < len(values) and values[idx]:
            out.append(values[idx])
    return out


def parse_characteristics(items):
    parsed = {}
    for item in items:
        m = re.match(r"([^:]+):\s*(.*)$", item)
        if m:
            parsed[m.group(1).strip().lower()] = m.group(2).strip()
    return parsed


def infer_tissue(title, source, chars):
    tissue = chars.get("tissue", "")
    text = " ".join([title, source, tissue]).upper()
    if "PBMC" in text:
        return "PBMC"
    if "CSF" in text or "CEREBROSPINAL" in text:
        return "CSF"
    return tissue or "unknown"


def infer_subject_id(gse, title, chars):
    if "patient id" in chars:
        return chars["patient id"]
    if "sample id" in chars:
        return chars["sample id"]
    clean = title.replace("CSF cells, ", "").replace("CSF cells", "").strip()
    clean = re.sub(r"_(CSF|PBMCs?|GEX|uns).*$", "", clean, flags=re.I)
    clean = re.sub(r"\s+", "_", clean)
    return clean or f"{gse}_unknown_subject"


def infer_diagnosis(gse, title, chars):
    for key in ("disease condition", "disease", "disease state"):
        value = chars.get(key, "")
        if value:
            diagnosis = value
            break
    else:
        diagnosis = ""

    if gse == "GSE286068":
        upper = title.upper()
        if "RR-MS" in upper:
            diagnosis = "RRMS"
        elif "PP-MS" in upper:
            diagnosis = "PPMS"
        elif "SP-MS" in upper:
            diagnosis = "SPMS"
        elif "HD" in upper:
            diagnosis = "Healthy donor"
        elif "NIND" in upper:
            diagnosis = "NIND"
        elif not diagnosis:
            diagnosis = "unknown"

    if not diagnosis:
        diagnosis = "unknown"
    return diagnosis


def normalize_group(diagnosis):
    text = diagnosis.lower()
    if text in {"ms", "multiple sclerosis", "rrms", "ppms", "spms"}:
        return "MS"
    if "multiple sclerosis" in text:
        return "MS"
    if "idiopathic intracranial hypertension" in text:
        return "IIH_control"
    if text in {"normal", "healthy donor"}:
        return "non_inflammatory_control"
    if "nind" in text:
        return "non_inflammatory_control"
    if "ab-mediated idd" in text:
        return "ab_mediated_idd"
    return "unknown"


def parse_one(gse, matrix_path):
    rows = read_series_matrix(matrix_path)
    accessions = rows.get("Sample_geo_accession", [[]])[0]
    records = []
    for idx, gsm in enumerate(accessions):
        title = get_value(rows, "Sample_title", idx)
        source = get_value(rows, "Sample_source_name_ch1", idx)
        chars = parse_characteristics(get_repeated(rows, "Sample_characteristics_ch1", idx))
        diagnosis = infer_diagnosis(gse, title, chars)
        tissue = infer_tissue(title, source, chars)
        supplementary = get_repeated(rows, "Sample_supplementary_file", idx)
        records.append(
            {
                "gse": gse,
                "sample_accession": gsm,
                "sample_title": title,
                "subject_id": infer_subject_id(gse, title, chars),
                "tissue": tissue,
                "source_name": source,
                "diagnosis": diagnosis,
                "disease_group": normalize_group(diagnosis),
                "technology": "10x scRNA-seq/CITE-seq" if gse == "GSE286068" else "10x scRNA-seq",
                "organism": "Homo sapiens",
                "raw_matrix_available": "yes",
                "analysis_role": "discovery" if gse == "GSE138266" else "external_validation",
                "supplementary_files": ";".join(supplementary),
            }
        )
    return records


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys()) if records else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_counts(path, records):
    counts = {}
    for r in records:
        key = (r["gse"], r["tissue"], r["disease_group"], r["diagnosis"])
        counts[key] = counts.get(key, 0) + 1
    fields = ["gse", "tissue", "disease_group", "diagnosis", "n_samples"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, n in sorted(counts.items()):
            writer.writerow(
                {
                    "gse": key[0],
                    "tissue": key[1],
                    "disease_group": key[2],
                    "diagnosis": key[3],
                    "n_samples": n,
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()
    root = Path(args.root)
    inputs = {
        "GSE138266": root / "data/raw/GSE138266_series_matrix.txt.gz",
        "GSE194078": root / "data/raw/GSE194078_series_matrix.txt.gz",
        "GSE286068": root / "data/raw/GSE286068_series_matrix.txt.gz",
    }
    records = []
    for gse, path in inputs.items():
        if path.exists():
            records.extend(parse_one(gse, path))

    write_csv(root / "metadata/all_geo_sample_metadata.csv", records)
    write_counts(root / "metadata/sample_count_summary.csv", records)
    gse138 = [r for r in records if r["gse"] == "GSE138266"]
    if gse138:
        write_csv(root / "metadata/GSE138266_sample_metadata.csv", gse138)

    manifest = [
        {
            "gse": "GSE138266",
            "title": "Integrated single cell analysis of blood and cerebrospinal fluid leukocytes in multiple sclerosis",
            "role": "paired CSF-PBMC trajectory and disease discovery",
        },
        {
            "gse": "GSE194078",
            "title": "AXL+SIGLEC6+ dendritic cells in CSF and brain tissues of IDD",
            "role": "independent paired trajectory replication",
        },
        {
            "gse": "GSE286068",
            "title": "CITE-seq analysis of human CSF cells in multiple sclerosis",
            "role": "CSF trajectory and disease discovery",
        },
    ]
    for row in manifest:
        row["raw_archive_present"] = (root / "data/raw" / f"{row['gse']}_RAW.tar").is_file()
    write_csv(root / "metadata/raw_data_inventory.csv", manifest)
    print(f"Wrote {len(records)} sample records")


if __name__ == "__main__":
    main()
