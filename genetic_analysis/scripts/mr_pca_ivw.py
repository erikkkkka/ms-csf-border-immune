#!/usr/bin/env python3
"""Estimate PCA–IVW using signed LD."""
from pathlib import Path
import csv
import math
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]

def read(p,delimiter=','):
    with p.open() as f:return list(csv.DictReader(f,delimiter=delimiter))

def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pca_ivw(bx,by,sy,R,target=.999):
    signs=np.sign(bx);assert np.all(signs!=0)
    bx=np.abs(bx);by=by*signs;R=R*np.outer(signs,signs)
    Phi=np.outer(bx/sy,bx/sy)*R
    centered=Phi-Phi.mean(axis=0)
    _,s,vh=np.linalg.svd(centered,full_matrices=False)
    fractions=s*s/sum(s*s);cumulative=np.cumsum(fractions)
    k=int(np.searchsorted(cumulative,target)+1)
    loading=vh[:k,:].T
    x=loading.T@bx;y=loading.T@by
    Omega=np.outer(sy,sy)*R;C=loading.T@Omega@loading;C=(C+C.T)/2
    cf=cho_factor(C,lower=True);cx=cho_solve(cf,x);cy=cho_solve(cf,y)
    precision=float(x@cx);theta=float(x@cy/precision);se=float(precision**-.5)
    residual=y-theta*x;Q=float(residual@cho_solve(cf,residual))
    return {'theta':theta,'se':se,'k':k,'retained_variance':float(cumulative[k-1]),
            'previous_variance':float(cumulative[k-2]) if k>1 else 0.,'condition_number':float(np.linalg.cond(C)),
            'Q_descriptive':Q,'Q_df':k-1,'fractions':fractions,'loading':loading,
            'projected_x':x,'projected_y':y,'projected_covariance':C,'Phi':Phi}

def main():
    disease={r['snp_GRCh37']:r for r in read(ROOT/'work/1Mb_IFNGR1_MS/statistics.tsv','\t')}
    summary=[]
    for cell in ['CD14_Mono','CD16_Mono']:
        gene='IFNGR1';prefix=f'{gene}_{cell}'
        candidates=read(ROOT/f'work/{prefix}_expression_instruments.csv');p=len(candidates)
        R=np.fromfile(ROOT/f'work/{prefix}_candidate_LD.bin',dtype='<f8').reshape(p,p)
        usable=[];local=[]
        for i,e in enumerate(candidates):
            matched=disease.get(e['snp_GRCh37'])
            if matched:
                assert (e['effect_allele'],e['other_allele'])==(matched['effect_allele'],matched['other_allele'])
                usable.append(i)
                local.append({'snp_GRCh37':e['snp_GRCh37'],'variant_GRCh38':e['variant_GRCh38'],
                              'beta_exposure_ALT':float(e['beta']),'se_exposure':float(e['se']),
                              'P_exposure':float(e['p']),'F_marginal':float(e['z'])**2,
                              'beta_outcome_ALT':float(matched['beta']),'se_outcome_approx':float(matched['se']),
                              'source_GWAS_N_studies':matched['source_GWAS_N_studies']})
        rr=R[np.ix_(usable,usable)]
        bx=np.array([r['beta_exposure_ALT'] for r in local]);sx=np.array([r['se_exposure'] for r in local])
        by=np.array([r['beta_outcome_ALT'] for r in local]);sy=np.array([r['se_outcome_approx'] for r in local])
        fit=pca_ivw(bx,by,sy,rr)
        beta,se=fit['theta'],fit['se'];q=float(norm.ppf(.975))
        summary.append({'gene':gene,'cell_type':cell,'method':'PCA–IVW',
                        'n_expression_instruments':p,'n_outcome_usable':len(local),'n_PCs':fit['k'],
                        'PCA_target':.999,'PCA_retained_variance':fit['retained_variance'],
                        'PCA_previous_variance':fit['previous_variance'],
                        'beta':beta,'SE_first_order':se,'odds_ratio':math.exp(beta),
                        'CI95_lower':math.exp(beta-q*se),'CI95_upper':math.exp(beta+q*se),
                        'nominal_p':float(2*norm.sf(abs(beta/se))),
                        'min_marginal_F':float(min((bx/sx)**2)),'max_marginal_F':float(max((bx/sx)**2)),
                        'projected_covariance_condition_number':fit['condition_number'],
                        'Q_descriptive':fit['Q_descriptive'],'Q_df':fit['Q_df'],
                        'exposure_unit':'TenK10K_expression_model_unit',
                        'SE_outcome':'approximated_from_rounded_OR_and_P'})
    write(ROOT/'results/mr_pca_ivw.csv',summary)

if __name__=='__main__':main()
