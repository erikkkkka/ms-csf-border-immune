#!/usr/bin/env python3
"""Donor omission, rank scoring, and cell-type sensitivity analyses for Ban CSF."""
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from statsmodels.stats.multitest import multipletests

import analyze_ifngr1_signatures as signatures
import analyze_ban_csf as ban

OUT = signatures.BAN / 'sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
load = signatures.load
baseline=load(ban.BASELINE_CODE,'baseline')
projection=load(signatures.PROJECT/'scripts/54_project_gse328265_two_axis.py','projection')
cells=ad.read_h5ad(signatures.BAN_CACHE)
obs=cells.obs.copy();obs['subject_id']=obs.subject_id.astype(str);obs['pseudotime']=obs.projected_differentiation
panels=signatures.panels()
cohort,_=ban.clinical_cohort(cells,signatures.PROJECT,baseline,projection)
assert np.allclose(cohort.pt,obs.pseudotime)
eligible=obs.donor_eligible.to_numpy()
reference=(obs.disease.eq('NIND') & obs.donor_eligible).to_numpy()
donors=sorted(obs.loc[eligible,'subject_id'].unique())
assert len(donors)==23

# Full transcriptome rankings: log1p normalization preserves within-cell order.
ranks=stats.rankdata(-cells.X.toarray(),method='average',axis=1)
ranks=np.minimum(ranks,1500)
def ucell(genes):
    n=len(genes); rr=ranks[:,cells.var_names.get_indexer(genes)]
    return 1-(rr.sum(axis=1)-n*(n+1)/2)/(n*1500-n*(n+1)/2)

def scores(ref,method):
    out={}
    for endpoint in ['axis2','pseudotime']:
        panel=panels[endpoint]; w=panel.weight.to_numpy(); up=w>0;down=w<0
        fraction=w[up].sum()/np.abs(w).sum()
        if method=='weighted':
            x=cells[:,panel.human_gene].X.toarray().astype(float)
            mu,sd=signatures.bridge.weighted_mean_sd(x[ref],obs.loc[ref,'subject_id']);z=(x-mu)/sd
            arms={'up':z[:,up]@w[up]/w[up].sum(),'down':z[:,down]@(-w[down])/(-w[down]).sum()}
        else:
            arms={'up':ucell(panel.loc[up,'human_gene']), 'down':ucell(panel.loc[down,'human_gene'])}
        arms['combined']=fraction*arms['up']-(1-fraction)*arms['down']
        for arm,raw in arms.items():
            out[(endpoint,arm)]=signatures.bridge.residualize_score(raw,obs,'pseudotime',ref)[0] if endpoint=='axis2' else raw
    return out

def effects(sc,activation,keep,celltype='pooled',minimum=20):
    rows=[]
    for donor in donors:
        pos=np.flatnonzero(keep & obs.subject_id.eq(donor).to_numpy())
        if celltype!='pooled':pos=pos[obs.celltype.iloc[pos].eq(celltype).to_numpy()]
        for endpoint in ['axis2','pseudotime']:
            ii=pos[obs.incomplete.iloc[pos].to_numpy()] if endpoint=='axis2' else pos
            if len(ii)<minimum:continue
            y=activation[ii] if endpoint=='axis2' else obs.pseudotime.iloc[ii].to_numpy()
            if np.ptp(y)==0 or (endpoint=='pseudotime' and np.ptp(y)<.05):continue
            for arm in ['combined','up','down']:
                v=sc[(endpoint,arm)][ii]
                if np.ptp(v)==0:continue
                rows.append({'subject_id':donor,'disease':obs.disease.iloc[ii[0]],'endpoint':endpoint,'arm':arm,'n_cells':len(ii),'effect':stats.spearmanr(v,y).statistic})
    return pd.DataFrame(rows)

all_effects=[]
for method in ['weighted','ucell']:
    sc=scores(reference,method)
    configs=[('pooled',20)] if method=='ucell' else [('pooled',20),('monocyte',10),('Macrophage',10),('monocyte',20),('Macrophage',20)]
    for celltype,minimum in configs:
        e=effects(sc,obs.activation_residual.to_numpy(),eligible,celltype,minimum)
        all_effects.append(e.assign(method=method,celltype=celltype,min_cells=minimum))
all_effects=pd.concat(all_effects,ignore_index=True)
original=pd.read_csv(signatures.BAN/'signature_components/donor_effects.csv')
original=original[(~original.strict_joint_holdout)&original.endpoint.isin(['axis2_spearman','pseudotime_spearman'])].copy()
original['endpoint']=original.endpoint.str.replace('_spearman','')
original['subject_id']=original.subject_id.astype(str)
check=all_effects[(all_effects.method=='weighted')&(all_effects.celltype=='pooled')].merge(original,left_on=['subject_id','endpoint','arm'],right_on=['subject_id','endpoint','signature_type'],suffixes=('_new','_old'))
assert len(check)==129, len(check)
error=float(np.max(abs(check.effect_new-check.effect_old))); assert error<1e-10

rng=np.random.default_rng(20261114);summaries=[]
for keys,e in all_effects.groupby(['method','celltype','min_cells','endpoint','arm'],sort=True):
    v=e.effect.to_numpy();p,test=signatures.bridge.sign_flip_test(v,20261114)
    ci=np.quantile(rng.choice(v,(2000,len(v)),replace=True).mean(axis=1),[.025,.975])
    summaries.append(dict(zip(['method','celltype','min_cells','endpoint','arm'],keys),n_donors=len(v),mean_effect=float(v.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),p=float(p),n_positive=int((v>0).sum()),test=test))
summary=pd.DataFrame(summaries)
for _,inds in summary.groupby(['method','celltype','min_cells']).groups.items():
    if summary.loc[inds[0],'method']=='weighted' and summary.loc[inds[0],'celltype']=='pooled':
        prior=json.loads((signatures.BAN/'signature_components/results.json').read_text())['summaries']
        initial=json.loads((signatures.BAN/'signature_results.json').read_text())['results']
        for i in inds:
            row=summary.loc[i]
            s=next(x for x in prior if not x['strict'] and x['arm']==row.arm and x['endpoint']==row.endpoint+'_spearman')
            summary.loc[i,['p','ci_low','ci_high']]=[s['p_two_sided'],*s['ci95_fixed_scores']]
            q=s.get('q_bh_six_arm_tests')
            if row.arm=='combined':
                s=next(x for x in initial if not x['strict_joint_holdout'] and x['signature_type']=='direct_ko_consensus' and x['endpoint']==row.endpoint+'_spearman')
                q=s['q_bh_within_prespecified_family']
            summary.loc[i,'q_original_family']=q
    else:
        summary.loc[inds,'q_six']=multipletests(summary.loc[inds,'p'],method='fdr_bh')[1]
all_effects.to_csv(OUT/'robustness_donor_effects.csv',index=False)
summary.to_csv(OUT/'robustness_summary.csv',index=False)

loo=[];clinical=[]
for omitted in donors:
    keep=eligible & obs.subject_id.ne(omitted).to_numpy();ref=reference & keep
    sc=scores(ref,'weighted')
    mult=cohort.eligible.astype(int);mult[cohort.subjects==omitted]=0
    _,readings,thresholds,_=cohort.fit(cohort.case,mult)
    donor_scores=cohort.aggregate(readings,thresholds,np.flatnonzero(cohort.eligible))
    difference,d=cohort.effect(donor_scores,cohort.case,mult)
    clinical.append({'omitted':omitted,'difference':difference[0],'cohens_d':d[0]})
    e=effects(sc,readings[:,0],keep)
    for (endpoint,arm),g in e.groupby(['endpoint','arm']):
        loo.append({'omitted':omitted,'endpoint':endpoint,'arm':arm,'n_donors':len(g),'mean_effect':g.effect.mean()})
pd.DataFrame(loo).to_csv(OUT/'leave_one_donor_out.csv',index=False)
pd.DataFrame(clinical).to_csv(OUT/'clinical_leave_one_donor_out.csv',index=False)

# Independently evaluate selected rank scores directly from sorted values/tie blocks.
maxerr=0.
for i in [0,80,500,1000,1625]:
    x=cells.X[i].toarray().ravel()
    for panel_name in ['axis2','pseudotime']:
        pp=panels[panel_name]
        for sign in [1,-1]:
            genes=pp.loc[pp.weight*sign>0,'human_gene'];ix=cells.var_names.get_indexer(genes)
            r=np.array([min(1500,1+np.sum(x>x[j])+(np.sum(x==x[j])-1)/2) for j in ix])
            n=len(r);v=1-(r.sum()-n*(n+1)/2)/(1500*n-n*(n+1)/2)
            maxerr=max(maxerr,abs(v-ucell(genes)[i]))
assert maxerr<1e-12
print(summary.to_string(index=False))
