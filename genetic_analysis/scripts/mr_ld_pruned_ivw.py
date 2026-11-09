#!/usr/bin/env python3
"""Estimate LD-pruned IVW."""
from pathlib import Path
import csv
import math
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]


def read(p,delimiter=','):
    with p.open() as f:return list(csv.DictReader(f,delimiter=delimiter))


def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    summary=read(ROOT/'results/exposure_summary_1Mb.csv')
    inventory=[];instruments=[];estimates=[]
    for context in summary:
        gene,cell=context['gene'],context['cell_type']
        row={'gene':gene,'cell_type':cell,'association_P_threshold':1e-5,'F_threshold':10,'LD_r2_threshold':.01,
             'n_association_candidates':int(context['n_eqtl_P_le_1e5_and_F_ge_10_before_LD']),
             'n_reference_and_direction_eligible':0,'n_LD_independent_tools':0,'status':'',
             'selected_variants_GRCh38':'','selected_SNPs_GRCh37':''}
        if context['status']!='available':
            row['status']='missing_exposure_statistics'
        elif not row['n_association_candidates']:
            row['status']='no_eligible_instruments'
        else:
            folder=ROOT/'work'/f'1Mb_{gene}_{cell}_expression'
            if not (folder/'statistics.tsv').exists():
                raise RuntimeError('A qualifying context needs LD; do not silently skip it.')
            h=read(folder/'statistics.tsv','\t');p=len(h)
            R=np.memmap(folder/'LD.bin',dtype='<f8',mode='r',shape=(p,p))
            eligible=[i for i,r in enumerate(h) if float(r['p'])<=1e-5 and float(r['z'])**2>=10
                      and {r['effect_allele'],r['other_allele']} not in [{'A','T'},{'C','G'}]]
            eligible.sort(key=lambda i:(float(h[i]['p']),h[i]['snp_GRCh37']))
            selected=[]
            for i in eligible:
                if all(R[i,j]**2<.01 for j in selected):selected.append(i)
            row.update(n_reference_and_direction_eligible=len(eligible),n_LD_independent_tools=len(selected),
                       status='one_instrument' if len(selected)==1 else 'multiple_instruments',
                       selected_variants_GRCh38=';'.join(h[i]['variant_GRCh38'] for i in selected),
                       selected_SNPs_GRCh37=';'.join(h[i]['snp_GRCh37'] for i in selected))
            for i in selected:
                instruments.append({'gene':gene,'cell_type':cell,**h[i],
                                    'selection_basis':'expression_P_F_and_reference_LD_only'})
            write(ROOT/'work/ld_pruned_variants.csv',instruments)
            if len(selected)>=2:
                disease={r['snp_GRCh37']:r for r in read(ROOT/'work'/f'1Mb_{gene}_MS/statistics.tsv','\t')}
                usable=[i for i in selected if h[i]['snp_GRCh37'] in disease]
                if len(usable)>=2:
                    bx=np.array([float(h[i]['beta']) for i in usable])
                    by=np.array([float(disease[h[i]['snp_GRCh37']]['beta']) for i in usable])
                    sy=np.array([float(disease[h[i]['snp_GRCh37']]['se']) for i in usable])
                    w=1/sy**2;b=float(sum(w*bx*by)/sum(w*bx**2));se=float(math.sqrt(1/sum(w*bx**2)))
                    estimates.append({'gene':gene,'cell_type':cell,'method':'LD-pruned IVW',
                                      'n_exposure_selected':len(selected),'n_outcome_usable':len(usable),'beta':b,'SE_first_order':se,
                                      'odds_ratio':math.exp(b),'CI95_lower':math.exp(b-norm.ppf(.975)*se),
                                      'CI95_upper':math.exp(b+norm.ppf(.975)*se),'nominal_p':float(2*norm.sf(abs(b/se)))})
                else:row['status']='multiple_expression_tools_but_insufficient_matched_outcome_tools'
            del R
        inventory.append(row)
    write(ROOT/'results/ld_pruned_instruments.csv',inventory)
    if estimates:write(ROOT/'results/mr_ld_pruned_ivw.csv',estimates)


if __name__=='__main__':main()
