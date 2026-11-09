#!/usr/bin/env python3
"""Retrieve the 1000 Genomes European reference for IFNGR1."""
from pathlib import Path
import gzip
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
import pysam

BASE = 'https://1000genomes.s3.amazonaws.com/release/20130502/'
VCF = 'ALL.chr6.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'
PANEL = 'integrated_call_samples_v3.20130502.ALL.panel'
START, END = 136040520, 139040520


def cached_download(name):
    p = ROOT / 'work' / name
    if not p.exists():
        with urllib.request.urlopen(BASE + name, timeout=45) as h:
            p.write_bytes(h.read())
    return p


def main():
    panel = cached_download(PANEL)
    index = cached_download(VCF + '.tbi')
    eur = {line.split()[0] for line in panel.read_text().splitlines()[1:]
           if len(line.split()) >= 3 and line.split()[2] == 'EUR'}
    assert len(eur) == 503
    data = pysam.TabixFile(BASE + VCF, index=str(index))
    header = next(x for x in data.header if x.startswith('#CHROM'))
    columns = header.split('\t')
    indices = [i for i, name in enumerate(columns) if i >= 9 and name in eur]
    assert len(indices) == 503
    out = ROOT / 'work/IFNGR1_1KG_Phase3_EUR_GRCh37.vcf.gz'
    temp = out.with_suffix('.part')
    with gzip.open(temp, 'wt') as f:
        f.write('##fileformat=VCFv4.2\n')
        f.write('##source=1000Genomes_Phase3_20130502_EUR_subset\n')
        f.write('##reference=GRCh37\n')
        f.write('\t'.join(columns[:9] + [columns[i] for i in indices]) + '\n')
        for line in data.fetch('6', START - 1, END):
            a = line.split('\t')
            # Keep all biallelic SNPs, including rare ones; monomorphic variants
            # are handled only when preparing a correlation matrix.
            if len(a[3]) != 1 or len(a[4]) != 1 or ',' in a[4]:
                continue
            f.write('\t'.join(a[:9] + [a[i] for i in indices]) + '\n')
    temp.replace(out)


if __name__ == '__main__':
    main()
