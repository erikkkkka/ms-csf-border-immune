"""Donor-weighted control baseline and inflammatory endpoints for Ban CSF."""

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.interpolate import BSpline

MODULES={
 'MHC_I':['B2M','TAP1','TAP2','TAPBP','PSMB8','PSMB9','PSMB10','NLRC5'],
 'IFN':['IFIT1','IFIT2','IFIT3','ISG15','MX1','OAS1','STAT1','IRF7']}
READOUTS=[f'{module}_{endpoint}' for module in ['combined','MHC_I','IFN'] for endpoint in ['continuous','fraction']]
SEED=20261101

def weighted_quantile(v,w,q):
    order=np.argsort(v)
    cumulative=np.cumsum(w[order]);cumulative/=cumulative[-1]
    return float(np.interp(q,cumulative,v[order]))

class Cohort:
 def fit(self,case,multiplicity,holdout=(),balanced=False,reference_check=False):
    active=multiplicity>0
    control=(~case)&active
    control[list(holdout)]=False
    idx=np.concatenate([np.tile(self.cell_rows[d],int(multiplicity[d])) for d in np.flatnonzero(control)]) if control.any() else np.array([],int)
    if len(idx)<8:raise ValueError('Insufficient training control cells')
    donor_balanced=self.external or balanced
    w=(1/self.cell_counts[self.donor[idx]]) if donor_balanced else np.ones(len(idx))
    if self.external:
        wn=w/w.sum();mu=np.sum(self.x[idx]*wn[:,None],axis=0)
        sd=np.sqrt(np.sum((self.x[idx]-mu)**2*wn[:,None],axis=0));sd[sd<1e-8]=1
        z=(self.x-mu)/sd
        y=np.column_stack([z[:,j].mean(axis=1) for j in self.module_indices])
    else:y=self.fixed_y
    # patsy bs(df=4, degree=3, include_intercept=False) has one interior median knot.
    knot=float(np.quantile(self.pt[idx],.5))
    knots=np.r_[np.zeros(4),knot,np.ones(4)]
    basis=BSpline.design_matrix(self.pt,knots,3,extrapolate=False).toarray()[:,1:]
    design=np.column_stack([np.ones(len(self.pt)),basis,self.tech])
    weighted_x=design[idx]*np.sqrt(w[:,None]);weighted_y=y[idx]*np.sqrt(w[:,None])
    beta,_,rank,_=np.linalg.lstsq(weighted_x,weighted_y,rcond=None)
    predicted=design@beta
    if reference_check:
        frame=pd.DataFrame({'pt':self.pt,'log_counts':self.tech_raw[:,0],'n_genes':self.tech_raw[:,1]})
        for j,c in enumerate(['log_counts','n_genes']):
            if self.external:
                wn=w/w.sum();mu=np.sum(frame[c].to_numpy()[idx]*wn)
                sd=np.sqrt(np.sum((frame[c].to_numpy()[idx]-mu)**2*wn))
            else:mu=frame[c].to_numpy()[idx].mean();sd=frame[c].to_numpy()[idx].std(ddof=0)
            frame[c+'_z']=(frame[c]-mu)/(sd if sd>0 else 1)
        formula='bs(pt, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound=1) + log_counts_z + n_genes_z'
        train=patsy.dmatrix(formula,frame.iloc[idx],return_type='dataframe')
        all_design=patsy.build_design_matrices([train.design_info],frame)[0]
        reference=np.column_stack([(sm.WLS(y[idx,j],train,weights=w).fit() if donor_balanced else sm.OLS(y[idx,j],train).fit()).predict(all_design) for j in range(2)])
        assert np.max(np.abs(predicted-reference))<1e-8,(self.name,np.max(np.abs(predicted-reference)))
    residual=y-predicted;scaled=np.empty_like(residual)
    for j in range(2):
        if donor_balanced:
            center=weighted_quantile(residual[idx,j],w,.5)
            mu=np.average(residual[idx,j]-center,weights=w)
            scale=np.sqrt(np.average((residual[idx,j]-center-mu)**2,weights=w))
        else:center=np.median(residual[idx,j]);scale=residual[idx,j].std(ddof=1)
        if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate control residual scale')
        scaled[:,j]=(residual[:,j]-center)/scale
    readings=np.column_stack([scaled.mean(axis=1),scaled])
    thresholds=np.array([weighted_quantile(readings[idx,j],w,.75) if donor_balanced else np.quantile(readings[idx,j],.75) for j in range(3)])
    return scaled,readings,thresholds,{'n_train_donors':int(control.sum()),'rank':int(rank),'knot':knot}

 def aggregate(self,readings,thresholds,donors):
    output=np.full((self.n,6),np.nan)
    for d in donors:
        idx=self.cell_rows[d]
        for j in range(3):
            output[d,2*j]=np.mean(readings[idx,j]*(1-self.pt[idx]))
            output[d,2*j+1]=np.mean((readings[idx,j]>thresholds[j])&(self.pt[idx]<.8))
    return output

 def score(self,case,multiplicity=None,crossfit=False,balanced=False,ledger=None):
    multiplicity=np.ones(self.n,dtype=int) if multiplicity is None else multiplicity
    evaluate=np.flatnonzero(self.eligible & (multiplicity>0))
    if not crossfit:
        scaled,readings,thresholds,diagnostics=self.fit(case,multiplicity,balanced=balanced)
        return self.aggregate(readings,thresholds,np.arange(self.n)),scaled
    controls=evaluate[~case[evaluate]];cases=evaluate[case[evaluate]]
    if len(controls)<2:raise ValueError('Fewer than two unique eligible bootstrap control donors')
    rng=np.random.default_rng(SEED)
    controls=controls[rng.permutation(len(controls))];cases=cases[rng.permutation(len(cases))]
    result=np.full((self.n,6),np.nan)
    covered=[]
    for fold,c in enumerate(controls):
        held=np.r_[c,cases[np.arange(len(cases))%len(controls)==fold]]
        _,readings,thresholds,diagnostic=self.fit(case,multiplicity,holdout=held,balanced=balanced)
        row=self.aggregate(readings,thresholds,held);result[held]=row[held];covered.extend(held.tolist())
        train=np.flatnonzero((~case)&(multiplicity>0)&~np.isin(np.arange(self.n),held))
        assert not set(train)&set(held)
        if ledger is not None:ledger.append({'dataset':self.name,'fold':fold,'training_controls':';'.join(self.subjects[train]),'held_out_donors':';'.join(self.subjects[held]),**diagnostic})
    assert sorted(covered)==sorted(evaluate.tolist())
    assert np.isfinite(result[evaluate]).all()
    return result,None

 def effect(self,scores,case,multiplicity=None):
    multiplicity=np.ones(self.n,dtype=int) if multiplicity is None else multiplicity
    active=self.eligible&(multiplicity>0)
    ci=np.repeat(np.flatnonzero(active&case),multiplicity[active&case])
    co=np.repeat(np.flatnonzero(active&~case),multiplicity[active&~case])
    c=scores[ci];r=scores[co]
    difference=c.mean(axis=0)-r.mean(axis=0)
    pooled=np.sqrt(((len(ci)-1)*c.var(axis=0,ddof=1)+(len(co)-1)*r.var(axis=0,ddof=1))/(len(ci)+len(co)-2))
    d=np.divide(difference,pooled,out=np.full(6,np.nan),where=pooled>0)
    return difference,d
