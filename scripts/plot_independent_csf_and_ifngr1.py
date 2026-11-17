"""Plot the independent CSF comparison and IFNGR1 perturbation evidence."""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAME = "Figure_3_Independent_CSF_and_IFNGR1"


STYLE = {'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False}


def build_figure(root: Path = ROOT) -> plt.Figure:
    with mpl.rc_context(STYLE):
        BAN = root / 'results/external_validation/Ban'
        THP = root / 'results/external_validation/GSE153056'
        def j(p):return json.loads(p.read_text())
        ban=j(BAN/'primary_results.json')
        mhc=j(THP/'mhc_i_pseudobulk.json')
        thp=pd.read_csv(THP/'knockout_signature/guide_and_replicate_contrasts.csv')
        sig=j(BAN/'signature_results.json')['results']
        s=pd.read_csv(BAN/'sensitivity/robustness_summary.csv')
        omission = pd.read_csv(BAN / 'sensitivity/donor_omission_ranges.csv')
        omission = omission.set_index(['endpoint', 'arm'])
        local_omissions = BAN / 'sensitivity/leave_one_donor_out.csv'
        if local_omissions.is_file():
            local = pd.read_csv(local_omissions)
            omission = (local.groupby(['endpoint', 'arm'])['mean_effect']
                        .agg(min_effect='min', max_effect='max'))
        index_summary = pd.read_csv(BAN / 'inflammatory_index_summary.csv', index_col='disease')
        donor_scores_path = BAN / 'primary_donor_scores.csv'
        donors = pd.read_csv(donor_scores_path) if donor_scores_path.is_file() else None
        C={'up':'#3B7C9D','down':'#BA8152','combined':'#765888','MS':'#A95A64','NIND':'#657E8A'}
        fig,axs=plt.subplots(4,2,figsize=(7.2,8.2));fig.subplots_adjust(left=.13,right=.98,top=.965,bottom=.065,hspace=.88,wspace=.74)
        def init(ax,letter,title,zero=True):
            ax.text(-.29,1.14,letter,transform=ax.transAxes,fontsize=11,fontweight='bold')
            ax.set_title(title,loc='left',fontweight='bold',pad=9)
            if zero:ax.axvline(0,c='#999999',lw=.6,zorder=0)
        def err(ax,x,lo,hi,y,col,marker='o'):
            ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,ms=4,color=col,lw=1,capsize=2)
        ax=axs[0,0];init(ax,'a','Independent CSF disease comparison',False)
        rng=np.random.default_rng(12)
        for k,g in enumerate(['NIND','MS']):
            q = index_summary.loc[g, ['q25', 'median', 'q75']].to_numpy()
            if donors is not None:
                v = donors.loc[donors.eligible & donors.disease.eq(g), 'combined_continuous'].to_numpy()
                ax.scatter(k + rng.uniform(-.13, .13, len(v)), v, s=20, c=C[g], edgecolor='white', lw=.35)
                q = np.quantile(v, [.25, .5, .75])
            ax.plot([k,k],[q[0],q[2]],c='k',lw=1.3);ax.plot([k-.18,k+.18],[q[1],q[1]],c='k',lw=1.3)
        ax.set_xticks([0,1],['NIND\nn=10','Untreated MS\nn=13']);ax.set_ylabel('Continuous inflammatory index');ax.set_xlim(-.5,1.5);ax.text(.03,.97,'Δ=0.184; P=0.0093',transform=ax.transAxes,va='top',fontsize=7)
        ax=axs[0,1];init(ax,'b','Disease-effect sensitivity')
        clinical=[]
        for name,x in [('Primary',ban['contrasts']['original_refit']['readouts']['continuous']),('Held-out donors',ban['contrasts']['held_out_donor']['readouts']['continuous']),('Raw UMI',j(BAN/'raw_umi/primary_results.json')['contrasts']['original_refit']['readouts']['continuous']),('Including treated',j(BAN/'including_treated_sensitivity.json')['contrasts']['original_refit']['readouts']['continuous'])]:
            clinical.append({'analysis':name,**x})
        for i,x in enumerate(clinical):err(ax,x['mean_MS_minus_NIND'],*x['mean_difference_ci95'],i,C['MS'])
        ax.set_yticks(range(4),[x['analysis'] for x in clinical]);ax.invert_yaxis();ax.set_xlabel('MS − NIND difference (95% CI)')
        ax=axs[1,0];init(ax,'c','Human IFNGR1 perturbation: MHC-I',False)
        v=np.array(mhc['primary']['paired_differences']);ax.axhline(0,c='#999999',lw=.6)
        for i,x in enumerate(v):ax.scatter(i,x,s=28,c=C['MS'])
        ax.set_xticks(range(3),['Replicate 1','Replicate 2','Replicate 3']);ax.set_ylabel('KO − NT, mean log2(CPM+1)');ax.set_ylim(-1.45,.12)
        ax=axs[1,1];init(ax,'d','Frozen knockout signature in THP-1',False);ax.axhline(0,c='#999999',lw=.6)
        for i,arm in enumerate(['combined','up','down']):
            v=thp[(thp.panel=='axis2')&(thp.arm==arm)&(thp.group=='IFNGR1')].difference.to_numpy()
            for k,x in enumerate(v):ax.scatter(i+(k-1)*.12,x,s=22,c=C[arm],marker=['o','s','^'][k])
            ax.plot([i-.23,i+.23],[v.mean(),v.mean()],c=C[arm],lw=1.5)
        ax.set_xticks(range(3),['Combined','Native up','Native down']);ax.set_ylabel('KO − NT signature score')
        ax=axs[2,0];init(ax,'e','Frozen signatures in Ban')
        sel=[('published_response','axis2_spearman','Response / Axis 2'),('direct_ko_consensus','axis2_spearman','KO / Axis 2'),('direct_ko_consensus','pseudotime_spearman','KO / trajectory')]
        for i,(kind,end,label) in enumerate(sel):
            r=next(x for x in sig if x['signature_type']==kind and x['endpoint']==end and not x['strict_joint_holdout']);err(ax,r['mean_effect'],*r['ci95_donor_bootstrap_fixed_scores'],i,C['up'] if i==0 else C['combined'])
        ax.set_yticks(range(3),[x[2] for x in sel]);ax.invert_yaxis();ax.set_xlabel('Mean donor Spearman ρ (95% CI)')
        ax=axs[2,1];init(ax,'f','Differentiation association by score')
        for i,arm in enumerate(['up','down','combined']):
            for method,off,marker in [('weighted',-.13,'o'),('ucell',.13,'s')]:
                r=s[(s.method==method)&(s.celltype=='pooled')&(s.endpoint=='pseudotime')&(s.arm==arm)].iloc[0];err(ax,r.mean_effect,r.ci_low,r.ci_high,i+off,C[arm],marker)
        ax.set_yticks(range(3),['Native up','Native down','Combined']);ax.invert_yaxis();ax.set_xlabel('Mean donor Spearman ρ (95% CI)')
        ax.text(.01,1.02,'● Weighted     ■ UCell arms',transform=ax.transAxes,fontsize=6.5)
        ax=axs[3,0];init(ax,'g','Up-arm association within cell types')
        config=[('pooled',20,'Pooled ≥20'),('Macrophage',10,'Mac ≥10'),('monocyte',10,'Mono ≥10'),('Macrophage',20,'Mac ≥20'),('monocyte',20,'Mono ≥20')]
        labels=[]
        for i,(ct,n,label) in enumerate(config):
            r=s[(s.method=='weighted')&(s.celltype==ct)&(s.min_cells==n)&(s.endpoint=='pseudotime')&(s.arm=='up')].iloc[0];err(ax,r.mean_effect,r.ci_low,r.ci_high,i,C['up']);labels.append(f'{label} (n={r.n_donors})')
        ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlabel('Mean donor Spearman ρ (95% CI)')
        ax=axs[3,1];init(ax,'h','Donor omission with recalibration')
        for i,arm in enumerate(['up','down','combined']):
            lo, hi = omission.loc[('pseudotime', arm), ['min_effect', 'max_effect']]
            ax.plot([lo,hi],[i,i],c=C[arm],lw=3)
            r=s[(s.method=='weighted')&(s.celltype=='pooled')&(s.endpoint=='pseudotime')&(s.arm==arm)].iloc[0];ax.scatter(r.mean_effect,i,c=C[arm],s=30,zorder=3)
        ax.set_yticks(range(3),['Native up','Native down','Combined']);ax.invert_yaxis();ax.set_xlabel('Mean ρ; full estimate and omission range')
        return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / "reproduced_figures"
    output.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args.root)
    with mpl.rc_context(STYLE):
        fig.savefig(output / f"{FIGURE_NAME}.pdf")
        fig.savefig(output / f"{FIGURE_NAME}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
