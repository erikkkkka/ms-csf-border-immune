#!/usr/bin/env python3
"""Construct trait-specific signed LD matrices."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import gc
import gzip
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from mr_wald import orient, reconstruct

def read(p, compressed=False):
    with (gzip.open(p, 'rt') if compressed else p.open()) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def build(window, gene):
    if window == '100kb':
        e = read(ROOT / 'work/exposure_regions_mapped.tsv')
        g = read(ROOT / 'work/IMSGC_three_regions_GRCh37.tsv.gz', True)
    else:
        e = read(ROOT / 'work/exposure_1Mb_mapped.tsv')
        g = read(ROOT / 'work/IMSGC_1Mb_regions_GRCh37.tsv.gz', True)
    e = [r for r in e if r['gene'] == gene]
    g = [r for r in g if r['region_gene'] == gene]
    wanted = {int(r['pos_GRCh37']) for r in e} | {int(r['BP']) for r in g}
    reference = defaultdict(list)
    with gzip.open(ROOT / f'work/{gene}_1KG_Phase3_EUR_GRCh37.vcf.gz', 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                samples = line.rstrip().split('\t')[9:]
                continue
            if line.startswith('#') or int(line.split('\t', 2)[1]) not in wanted:
                continue
            a = line.rstrip().split('\t'); gi = a[8].split(':').index('GT')
            gt = []
            for call in a[9:]:
                x = call.split(':')[gi].replace('|','/').split('/')
                gt.append(np.nan if '.' in x else sum(int(v) for v in x))
            genotype = np.array(gt, dtype=float)
            missing_count = int(np.isnan(genotype).sum())
            af = float(np.nanmean(genotype)/2)
            genotype[np.isnan(genotype)] = np.nanmean(genotype)
            reference[int(a[1])].append({'snp': ':'.join([a[0],a[1],a[3],a[4]]), 'ref': a[3], 'alt': a[4],
                                         'dosage': genotype, 'af': af, 'missing_count': missing_count})
    assert len(samples) == 503
    source_groups = [('CD14_Mono_expression', [r for r in e if r['cell_type']=='CD14_Mono']),
                     ('CD16_Mono_expression', [r for r in e if r['cell_type']=='CD16_Mono']), ('MS',g)]
    for label, source in source_groups:
        kept, geno, excluded = [], [], []
        for row in source:
            disease = label == 'MS'
            pos = int(row['BP'] if disease else row['pos_GRCh37'])
            a1 = row['A1'] if disease else row['effect_allele_GRCh37']
            a2 = row['A2'] if disease else row['other_allele_GRCh37']
            if disease and {a1,a2} in [{'A','T'},{'C','G'}]:
                excluded.append({'position':pos,'reason':'GWAS_palindromic_direction_unresolved'})
                continue
            try:
                if disease:
                    if int(row['N']) < 2:
                        raise ValueError('fewer_than_two_studies')
                    pair, reason = reconstruct(float(row['OR']),float(row['P']))
                    if pair is None:
                        raise ValueError(reason)
                    b, se = pair
                else:
                    b, se = float(row['beta']),float(row['se'])
            except ValueError as ex:
                excluded.append({'position':pos,'reason':str(ex)});continue
            matches = []
            for ref in reference[pos]:
                sign, how = orient(ref['alt'],ref['ref'],a1,a2)
                if sign is not None:
                    matches.append((ref,sign,how))
            if len(matches)!=1:
                excluded.append({'position':pos,'reason':'no_unique_reference_allele_match'});continue
            ref,sign,how=matches[0]
            if not np.isfinite(ref['dosage']).all() or np.var(ref['dosage'])==0:
                excluded.append({'position':pos,'reason':'reference_monomorphic_or_all_missing'});continue
            kept.append({'snp_GRCh37':ref['snp'], 'variant_GRCh38':row.get('variant_GRCh38',''),
                         'pos_GRCh37':pos,'effect_allele':ref['alt'],'other_allele':ref['ref'],
                         'beta':sign*b,'se':se,'z':sign*b/se,'p':float(row['P'] if disease else row['p']),
                         'N_model':41505 if disease else int(row['N']),
                         'N_status':'paper_total_nominal_approximation' if disease else 'actual_exposure_donors',
                         'source_GWAS_N_studies':row['N'] if disease else '',
                         'reference_EAF':ref['af'],'orientation':how,'reference_missing_calls':ref['missing_count']})
            geno.append(ref['dosage'])
        counts = Counter(r['snp_GRCh37'] for r in kept)
        duplicate_ids = {k for k,v in counts.items() if v>1}
        if duplicate_ids:
            selected=[i for i,r in enumerate(kept) if r['snp_GRCh37'] not in duplicate_ids]
            kept=[kept[i] for i in selected];geno=[geno[i] for i in selected]
        order=sorted(range(len(kept)),key=lambda i:(kept[i]['pos_GRCh37'],kept[i]['snp_GRCh37']))
        kept=[kept[i] for i in order];G=np.column_stack([geno[i] for i in order])
        X=G-G.mean(axis=0);X/=np.sqrt(np.mean(X**2,axis=0))
        R=X.T@X/len(samples);np.fill_diagonal(R,1.)
        assert np.isfinite(R).all() and np.max(abs(R-R.T))<1e-12
        folder=ROOT/'work'/f'{window}_{gene}_{label}';folder.mkdir(exist_ok=True)
        R.astype('<f8').tofile(folder/'LD.bin');X.astype('<f8').tofile(folder/'reference_standardized_genotypes.bin')
        with (folder/'statistics.tsv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(kept[0]),delimiter='\t');w.writeheader();w.writerows(kept)
        if excluded:
            with (folder/'excluded_variants.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(excluded[0]));w.writeheader();w.writerows(excluded)
        summary={'window':window,'gene':gene,'trait':label,'n_source_variants':len(source),'n_reference_samples':len(samples),
                 'n_variants':len(kept),'excluded_by_reason':dict(Counter(r['reason'] for r in excluded)),
                 'duplicates_dropped':len(duplicate_ids),'N_model':sorted(set(r['N_model'] for r in kept)),
                 'genotype_format':'little_endian_float64_row_major_nref_by_p',
                 'expression_palindromes':'allowed_when_original_reference_genome_alleles_and_mapping_resolve_orientation',
                 'GWAS_palindromes':'excluded_without_frequency_or_confirmed_strand',
                 'SNPs': [r['snp_GRCh37'] for r in kept]}
        (folder/'metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
        del G,X,R;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--window',required=True,choices=['100kb','1Mb']);p.add_argument('--gene',default='IFNGR1')
    a=p.parse_args();build(a.window,a.gene)
