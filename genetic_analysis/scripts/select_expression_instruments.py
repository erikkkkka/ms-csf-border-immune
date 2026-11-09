#!/usr/bin/env python3
"""Select expression instruments for PCA–IVW."""
from pathlib import Path
import csv
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def read(p,delimiter=','):
    with p.open() as f:return list(csv.DictReader(f,delimiter=delimiter))

def write(p,rows,delimiter=','):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=delimiter);w.writeheader();w.writerows(rows)

def main():
    contexts=read(ROOT/'results/exposure_summary_1Mb.csv')
    inventory=[];candidates=[]
    for context in contexts:
        gene,cell=context['gene'],context['cell_type']
        row={'gene':gene,'cell_type':cell,'source_rows':int(context['n_source_rows']),
             'association_candidates':int(context['n_eqtl_P_le_1e5_and_F_ge_10_before_LD']),
             'n_eligible_instruments':0,'status':''}
        if context['status']!='available':row['status']='missing_public_exposure_statistics'
        elif not row['association_candidates']:row['status']='no_eligible_instruments'
        else:
            folder=ROOT/'work'/f'1Mb_{gene}_{cell}_expression'
            h=read(folder/'statistics.tsv','\t');p=len(h)
            idx=[i for i,r in enumerate(h) if float(r['p'])<=1e-5 and float(r['z'])**2>=10
                 and {r['effect_allele'],r['other_allele']} not in [{'A','T'},{'C','G'}]]
            idx.sort(key=lambda i:(float(h[i]['p']),h[i]['snp_GRCh37']))
            fields=['snp_GRCh37','variant_GRCh38','effect_allele','other_allele','beta','se','z','p','N_model','reference_EAF']
            local=[{'gene':gene,'cell_type':cell,**{k:h[i][k] for k in fields},
                    'selection_basis':'expression_P_le_1e-5_F_ge_10_reference_and_direction_only_no_LD_clump'} for i in idx]
            write(ROOT/f'work/{gene}_{cell}_expression_instruments.csv',local)
            candidates.extend(local)
            R=np.memmap(folder/'LD.bin',dtype='<f8',mode='r',shape=(p,p))
            chosen=np.array(R[np.ix_(idx,idx)])
            assert np.isfinite(chosen).all() and np.max(abs(chosen-chosen.T))<1e-12
            chosen.astype('<f8').tofile(ROOT/f'work/{gene}_{cell}_candidate_LD.bin')
            row.update(n_eligible_instruments=len(local),status='instruments_available')
        inventory.append(row)
    write(ROOT/'work/expression_instruments.csv',candidates)
    write(ROOT/'results/expression_instrument_counts.csv',inventory)

if __name__=='__main__':main()
