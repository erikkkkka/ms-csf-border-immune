#!/usr/bin/env python3
"""Evaluate instrument thresholds and colocalization priors."""
from pathlib import Path
import csv
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def read(p,delimiter=','):
    with p.open() as f:return list(csv.DictReader(f,delimiter=delimiter))

def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    source=read(ROOT/'work/exposure_1Mb_mapped.tsv','\t')
    contexts=read(ROOT/'results/exposure_summary_1Mb.csv')
    rows=[]
    for context in contexts:
        gene,cell=context['gene'],context['cell_type']
        raw=[r for r in source if r['gene']==gene and r['cell_type']==cell]
        h=[];R=None
        folder=ROOT/'work'/f'1Mb_{gene}_{cell}_expression'
        if folder.exists():
            h=read(folder/'statistics.tsv','\t')
            R=np.memmap(folder/'LD.bin',dtype='<f8',mode='r',shape=(len(h),len(h)))
        for cutoff in [1e-5,1e-4,1e-3]:
            assoc=[r for r in raw if float(r['p'])<=cutoff]
            strong=[r for r in assoc if float(r['F_wald_approx'])>=10]
            eligible=[i for i,r in enumerate(h) if float(r['p'])<=cutoff and float(r['z'])**2>=10
                      and {r['effect_allele'],r['other_allele']} not in [{'A','T'},{'C','G'}]]
            eligible.sort(key=lambda i:(float(h[i]['p']),h[i]['snp_GRCh37']))
            for r2 in [.01,.05,.1]:
                chosen=[]
                if R is not None:
                    for i in eligible:
                        if all(R[i,j]**2<r2 for j in chosen):chosen.append(i)
                rows.append({'gene':gene,'cell_type':cell,'P_threshold':cutoff,'F_threshold':10,'LD_r2_threshold':r2,
                             'n_source_rows':len(raw),'n_P_eligible':len(assoc),'n_P_and_F_eligible':len(strong),
                             'n_F_only_eligible':sum(float(r['F_wald_approx'])>=10 for r in raw),
                             'n_reference_and_direction_eligible':len(eligible) if R is not None else '',
                             'n_LD_clumped':len(chosen) if R is not None else '',
                             'selected_GRCh38':';'.join(h[i]['variant_GRCh38'] for i in chosen),
                             'LD_available':R is not None,'no_outcome_statistics_used':True})
        del R
    write(ROOT/'results/expression_threshold_sensitivity.csv',rows)
    priors=[]
    for label,path,suffix in [('single_signal',ROOT/'results/coloc_single_signal.csv',''),
                              ('100kb_multisignal',ROOT/'results/100kb_IFNGR1_coloc_susie.csv','.abf'),
                              ('1Mb_multisignal',ROOT/'results/1Mb_IFNGR1_coloc_susie.csv','.abf')]:
        for row in read(path):
            if not row.get('PP.H4'+suffix):continue
            old=np.array([float(row[f'PP.H{i}'+suffix]) for i in range(5)])
            for p12 in [1e-6,1e-5,1e-4]:
                weight=old.copy();weight[4]*=p12/1e-5;weight/=sum(weight)
                priors.append({'analysis':label,'gene':row['gene'],'cell_type':row['cell_type'],
                               'MS_component':row.get('idx1',''),'expression_component':row.get('idx2',''),
                               'p1':1e-4,'p2':1e-4,'p12':p12,
                               **{f'PP.H{i}':float(v) for i,v in enumerate(weight)},
                               'conditional_H4_over_H3_plus_H4':float(weight[4]/(weight[3]+weight[4])),
                               'calculation':'existing_component_BFs_unchanged_only_prior_reweighted'})
                if p12==1e-5:assert np.max(abs(weight-old))<1e-12
    write(ROOT/'results/coloc_prior_sensitivity.csv',priors)

if __name__=='__main__':main()
