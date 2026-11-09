#!/usr/bin/env python3
"""Prepare regional eQTL and MS summary statistics in GRCh37."""
import argparse
import csv
import gzip
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
GENES = {'ENSG00000027697': 'IFNGR1', 'ENSG00000125347': 'IRF1', 'ENSG00000115415': 'STAT1'}
CELLS = ('CD14_Mono', 'CD16_Mono')

def rc(allele):
    return allele.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def map_variant(row, mappings):
    pos = int(row['pos_GRCh38'])
    end = pos + len(row['other_allele']) - 1
    found = []
    for block in mappings:
        old, new = block['original'], block['mapped']
        if old['start'] <= pos <= end <= old['end']:
            assert old['end'] - old['start'] == new['end'] - new['start']
            if new['strand'] == 1:
                mapped = new['start'] + pos - old['start']
                ea, oa = row['effect_allele'], row['other_allele']
            else:
                mapped = new['end'] - end + old['start']
                ea, oa = rc(row['effect_allele']), rc(row['other_allele'])
            found.append((str(new['seq_region_name']), mapped, ea, oa, int(new['strand'])))
    return list(set(found))



def write_table(path, rows, delimiter=','):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def eqtl_rows(archive, member):
    if archive.is_dir():
        with gzip.open(archive / (member + '.gz'), 'rt') as stream:
            yield from csv.DictReader(stream, delimiter='\t')
    else:
        with ZipFile(archive) as source, source.open(member) as raw:
            yield from csv.DictReader(io.TextIOWrapper(raw), delimiter='\t')


def exposure(archive, window):
    mappings = {gene: json.loads((ROOT / 'reference' / f'{gene}_{window}_GRCh37.json').read_text())['mappings']
                for gene in GENES.values()}
    suffix = '_1000000bp' if window == '1Mb' else ''
    rows = []
    for cell in CELLS:
        for row in eqtl_rows(archive, f'{cell}_common_all_cis_raw_pvalues{suffix}.tsv'):
            if row['gene'] not in GENES:
                continue
            gene = GENES[row['gene']]
            chrom, pos, ref, alt = row['MarkerID'].split(':')
            beta, se, af, p = (float(row[k]) for k in ('BETA', 'SE', 'AF_Allele2', 'p.value'))
            n = int(row['N'])
            if not all(math.isfinite(x) for x in (beta, se, af, p)) or not (se > 0 and 0 < af < 1 and 0 <= p <= 1 and n > 0):
                continue
            record = dict(gene=gene, cell_type=cell, variant_GRCh38=row['MarkerID'], chr_GRCh38=chrom,
                          pos_GRCh38=int(pos), other_allele=ref, effect_allele=alt, beta=beta, se=se,
                          effect_allele_frequency=af, MAF=min(af, 1-af), p=p, N=n, F_wald_approx=(beta/se)**2,
                          palindromic_SNP={ref, alt} in ({'A', 'T'}, {'C', 'G'}))
            mapped = map_variant(record, mappings[gene])
            if len(mapped) != 1:
                continue
            c37, p37, ea37, oa37, strand = mapped[0]
            record.update(chr_GRCh37=c37, pos_GRCh37=p37, effect_allele_GRCh37=ea37,
                          other_allele_GRCh37=oa37, map_strand=strand)
            rows.append(record)
    rows.sort(key=lambda row: (row['gene'], row['cell_type'], row['pos_GRCh38'], row['variant_GRCh38']))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eqtl-100kb', type=Path, required=True)
    parser.add_argument('--eqtl-1mb', type=Path, required=True)
    parser.add_argument('--imsgc', type=Path, required=True)
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    with (ROOT / 'reference/lead_variants.csv').open() as f:
        leads = list(csv.DictReader(f))
    regions = defaultdict(list)
    disease = defaultdict(list)
    for window, archive, filename in [('100kb', args.eqtl_100kb, 'exposure_regions_mapped.tsv'),
                                      ('1Mb', args.eqtl_1mb, 'exposure_1Mb_mapped.tsv')]:
        rows = exposure(archive, window)
        write_table(WORK / filename, rows, '\t')
        for gene in GENES.values():
            subset = [r for r in rows if r['gene'] == gene]
            regions[subset[0]['chr_GRCh37']].append((window, gene, min(r['pos_GRCh37'] for r in subset),
                                                     max(r['pos_GRCh37'] for r in subset)))
        summary = []
        for lead in leads:
            subset = [r for r in rows if (r['gene'], r['cell_type']) == (lead['gene'], lead['cell_type'])]
            if window == '100kb':
                top = next((r for r in subset if r['variant_GRCh38'] == lead['lead_variant_GRCh38']), None)
                if subset and top is None:
                    raise ValueError(f"Missing eQTL for {lead['rsid']} in {lead['cell_type']}")
                summary.append({**lead, 'n_regional_variants': len(subset), 'lead_beta': top['beta'] if top else '',
                                'F_wald_approx': top['F_wald_approx'] if top else ''})
            else:
                summary.append({'gene': lead['gene'], 'cell_type': lead['cell_type'], 'n_source_rows': len(subset),
                                'n_eqtl_P_le_1e5_and_F_ge_10_before_LD': sum(r['p'] <= 1e-5 and r['F_wald_approx'] >= 10 for r in subset),
                                'status': 'available' if subset else 'missing_exposure_statistics'})
        write_table(ROOT / 'results' / ('lead_variants.csv' if window == '100kb' else 'exposure_summary_1Mb.csv'), summary)
    with gzip.open(args.imsgc, 'rt') as stream:
        header = next(stream).split()
        if header != ['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'P', 'OR']:
            raise ValueError('Unexpected IMSGC discovery file header.')
        for line in stream:
            row = line.split()
            chrom, pos = row[0], int(row[1])
            for window, gene, start, end in regions.get(chrom, ()):
                if start <= pos <= end:
                    disease[window].append({'region_gene': gene, **dict(zip(header, row))})
    for window, filename in [('100kb', 'IMSGC_three_regions_GRCh37.tsv.gz'), ('1Mb', 'IMSGC_1Mb_regions_GRCh37.tsv.gz')]:
        write_table(WORK / filename, disease[window], '\t')


if __name__ == '__main__':
    main()
