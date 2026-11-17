#!/usr/bin/env python3
"""Frozen IFNGR1 signature positive control, followed by Ban arm decomposition."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import patsy
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[1]
BAN = PROJECT / 'results/external_validation/Ban'
THP = PROJECT / 'results/external_validation/GSE153056'
SCRATCH = PROJECT / 'data/external_validation/GSE153056/work'
BAN_CACHE = PROJECT / 'data/external_validation/Ban/work/Ban_primary_scored.h5ad'
SEED = 20261113


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bridge = load(PROJECT / 'scripts/79_bridge_in_vivo_crispr_signatures.py', 'bridge')
from analyze_gse153056 import load_annotations, two_sided_sign_p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(data, path):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def panels():
    table = pd.read_csv(BAN / 'frozen_panels.csv', keep_default_na=False)
    return {name: group.drop(columns='panel') for name, group in table.groupby('panel', sort=False)}


def extract(meta, rna):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(meta)
    features, matrices = [], []
    with gzip.open(rna, 'rb') as f:
        barcodes = [v.strip(b'"').decode() for v in f.readline().rstrip().split(b'\t')[1:]]
        assert len(set(barcodes)) == len(barcodes) and annotations.keys() <= set(barcodes)
        take = np.array([i for i, bc in enumerate(barcodes) if bc in annotations])
        selected = [barcodes[i] for i in take]
        total = np.zeros(len(take), dtype=np.int64)
        for i, line in enumerate(f):
            name, values = line.split(b'\t', 1)
            feature = name.strip(b'"').decode()
            row = np.fromstring(values.decode(), sep='\t', dtype=np.int64)
            assert len(row) == len(barcodes) and (row >= 0).all()
            counts = row[take]
            total += counts
            features.append(feature)
            matrices.append(sparse.csr_matrix(counts.astype(np.int32)[None, :]))
            if (i + 1) % 5000 == 0:
                print('Extracted feature rows:', i + 1, flush=True)
    assert len(set(features)) == len(features)
    obs = pd.DataFrame([{'barcode': bc, 'replicate': annotations[bc][0],
                         'guide': annotations[bc][1], 'total_umi': annotations[bc][2]}
                        for bc in selected]).set_index('barcode')
    assert np.array_equal(total, obs.total_umi.to_numpy())
    cells = ad.AnnData(sparse.vstack(matrices).T.tocsr(), obs=obs,
                       var=pd.DataFrame(index=pd.Index(features)))
    cells.write_h5ad(SCRATCH / 'THP1_selected_counts.h5ad', compression='gzip')
    audit = {'n_cells': cells.n_obs, 'n_features': cells.n_vars,
             'barcode_unique': True, 'all_umi_totals_match': True,
             'metadata_sha256': sha(meta), 'rna_sha256': sha(rna)}
    write(audit, SCRATCH / 'input_checks.json')
    missing = sorted(set(panels()['full'].human_gene) - set(features))
    write({'missing_exact_symbols': missing,
           'features': features}, SCRATCH / 'feature_audit.json')
    print(json.dumps({'audit': audit, 'missing_exact_symbols': missing}, indent=2), flush=True)


def score_thp(output):
    output.mkdir(exist_ok=True, parents=True)
    cells = ad.read_h5ad(SCRATCH / 'THP1_selected_counts.h5ad')
    mapping = json.loads((THP / 'feature_mapping.json').read_text())['gene_to_feature']
    reference = cells.obs.guide.eq('NT').to_numpy()
    groups, contrasts, coverages, gene_stats = [], [], [], []
    cell_scores = cells.obs.copy()
    for name, panel in panels().items():
        panel = panel.copy()
        panel['feature'] = [mapping.get(g, g) for g in panel.human_gene]
        present = panel.feature.isin(cells.var_names)
        for sign, label in [(1, 'up'), (-1, 'down')]:
            arm = panel.weight * sign > 0
            coverages.append({'panel': name, 'arm': label, 'n_fixed': int(arm.sum()),
                             'n_available': int((arm & present).sum()),
                             'gene_coverage': float((arm & present).sum() / arm.sum()),
                             'weight_coverage': float(panel.loc[arm & present, 'weight'].abs().sum() / panel.loc[arm, 'weight'].abs().sum()),
                             'missing': ';'.join(panel.loc[arm & ~present, 'human_gene'])})
        usable = panel[present]
        assert usable.feature.is_unique
        x = cells[:, usable.feature].X.toarray().astype(float)
        x = np.log1p(x * 1e4 / cells.obs.total_umi.to_numpy()[:, None])
        mu, sd = bridge.weighted_mean_sd(x[reference], cells.obs.loc[reference, 'replicate'])
        z = (x - mu) / sd
        w = usable.weight.to_numpy(float)
        up, down = w > 0, w < 0
        scores = {'combined': z @ w / np.abs(w).sum(),
                  'up': z[:, up] @ w[up] / w[up].sum(),
                  'down': z[:, down] @ (-w[down]) / (-w[down]).sum()}
        fraction = w[up].sum() / np.abs(w).sum()
        assert np.allclose(scores['combined'], fraction*scores['up'] - (1-fraction)*scores['down'])
        for j, (_, row) in enumerate(usable.iterrows()):
            gene_stats.append({'panel': name, 'gene': row.human_gene, 'feature': row.feature,
                               'weight': row.weight, 'nt_mean': mu[j], 'nt_sd_used': sd[j],
                               'nt_detection': float((x[reference, j] > 0).mean())})
        for arm, values in scores.items():
            cell_scores[f'{name}__{arm}'] = values
            for rep in ('rep1', 'rep2', 'rep3'):
                is_rep = cells.obs.replicate.eq(rep).to_numpy()
                control = is_rep & reference
                for group in ('NT', 'IFNGR1', 'IFNGR1g1', 'IFNGR1g2', 'IFNGR1g3', 'IFNGR1g4'):
                    mask = is_rep & ((~reference) if group == 'IFNGR1' else cells.obs.guide.eq(group).to_numpy())
                    assert mask.sum() > 0
                    entry = {'panel': name, 'arm': arm, 'replicate': rep, 'group': group,
                             'n_cells': int(mask.sum()), 'mean_score': float(values[mask].mean())}
                    groups.append(entry)
                    if group != 'NT':
                        contrasts.append({**entry, 'n_nt': int(control.sum()),
                                          'difference': float(values[mask].mean()-values[control].mean())})
    coverage = pd.DataFrame(coverages)
    contrasts = pd.DataFrame(contrasts)
    summary = []
    for (panel, arm), rows in contrasts.groupby(['panel', 'arm'], sort=True):
        pooled = rows[rows.group.eq('IFNGR1')].sort_values('replicate')
        guides = rows[~rows.group.eq('IFNGR1')]
        effects = pooled.difference.to_numpy()
        direction = -1 if arm == 'down' else 1
        n_expected = {rep: int((part.difference * direction > 0).sum()) for rep, part in guides.groupby('replicate')}
        summary.append({'panel': panel, 'arm': arm, 'paired_differences': effects.tolist(),
                        'mean_difference': float(effects.mean()), 'range': [float(effects.min()), float(effects.max())],
                        'exact_two_sided_sign_p': two_sided_sign_p(effects),
                        'guide_expected_directions': n_expected,
                        'equal_guide_differences': guides.groupby('replicate').difference.mean().to_dict()})
    coverage.to_csv(output/'gene_coverage.csv', index=False)
    pd.DataFrame(groups).to_csv(output/'group_scores.csv', index=False)
    contrasts.to_csv(output/'guide_and_replicate_contrasts.csv', index=False)
    pd.DataFrame(gene_stats).to_csv(output/'gene_reference_diagnostics.csv', index=False)
    cell_scores.to_csv(SCRATCH/'THP1_signature_cell_scores.csv.gz')
    result = {'primary_panel': 'axis2', 'primary_arm': 'combined',
              'frozen_panel_sha256': sha(BAN/'frozen_panels.csv'),
              'feature_mapping_sha256': sha(THP/'feature_mapping.json'),
              'input_cache_sha256': sha(SCRATCH/'THP1_selected_counts.h5ad'), 'summaries': summary}
    write(result, output/'results.json')
    print(json.dumps(result, indent=2), flush=True)


def reference_residuals(x, obs, reference):
    frame = pd.DataFrame({'pt': obs.pseudotime.to_numpy(float),
                          'log_counts': np.log1p(obs.total_counts.to_numpy(float)),
                          'n_genes': obs.n_genes_by_counts.to_numpy(float)}, index=obs.index)
    subjects = obs.loc[reference, 'subject_id']
    for col in ('log_counts', 'n_genes'):
        mu, sd = bridge.weighted_mean_sd(frame.loc[reference, col].to_numpy()[:,None], subjects)
        frame[f'{col}_z'] = (frame[col] - mu[0]) / sd[0]
    d = patsy.dmatrix('bs(pt, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound=1) + log_counts_z + n_genes_z', frame.loc[reference], return_type='dataframe')
    all_d = np.asarray(patsy.build_design_matrices([d.design_info], frame)[0])
    counts = subjects.value_counts()
    weights = subjects.map(lambda s: 1/counts[s]).to_numpy()
    coef = np.linalg.lstsq(np.asarray(d)*np.sqrt(weights[:,None]), x[reference]*np.sqrt(weights[:,None]), rcond=None)[0]
    return x-all_d@coef


def correlations(matrix, y):
    ranked = stats.rankdata(matrix, axis=0)
    ranked -= ranked.mean(axis=0)
    target = stats.rankdata(y)-np.mean(stats.rankdata(y))
    den = np.sqrt(np.sum(ranked**2, axis=0) * np.sum(target**2))
    return np.divide(target@ranked, den, out=np.full(matrix.shape[1], np.nan), where=den>0)


def score_ban(output):
    output.mkdir(parents=True, exist_ok=True)
    cache = BAN_CACHE
    cells = ad.read_h5ad(cache)
    obs = cells.obs.copy()
    obs['dataset'] = 'Ban'
    obs['subject_id'] = obs.subject_id.astype(str)
    obs['diagnosis'] = obs.disease
    obs['disease_group'] = obs.disease
    obs['disease_eligible'] = obs.disease.eq('MS') & obs.donor_eligible
    obs['pseudotime'] = obs.projected_differentiation
    reference = (obs.disease.eq('NIND') & obs.donor_eligible).to_numpy()
    assert obs.loc[reference,'subject_id'].nunique() == 10 and reference.sum() == 672
    effect_tables, diagnostics, influence, contributions, arm_structure = [], [], [], [], []
    planned_panels = panels()
    for strict in (False, True):
        scored = obs.copy()
        for endpoint in ('axis2', 'pseudotime'):
            table = planned_panels['strict' if strict else endpoint]
            assert table.human_gene.isin(cells.var_names).all()
            genes = table.human_gene.to_list()
            x = bridge.dense(cells[:, genes].X).astype(float)
            mean, sd = bridge.weighted_mean_sd(x[reference], obs.loc[reference,'subject_id'])
            z = (x-mean)/sd
            w = table.weight.to_numpy(float)
            up, down = w>0, w<0
            values = {'combined': z@w/np.abs(w).sum(),
                      'up': z[:,up]@w[up]/w[up].sum(),
                      'down': z[:,down]@(-w[down])/(-w[down]).sum()}
            arm_structure.append({'strict': strict, 'endpoint': endpoint, 'n_up': int(up.sum()), 'n_down': int(down.sum()),
                                  'up_weight_fraction': float(w[up].sum()/np.abs(w).sum()),
                                  'down_weight_fraction': float(-w[down].sum()/np.abs(w).sum())})
            for arm, raw in values.items():
                scored[f'score__{arm}__Ifngr1__{endpoint}'] = raw
                if endpoint == 'axis2':
                    res, _ = bridge.residualize_score(raw, obs, 'pseudotime', reference)
                    scored[f'residual__{arm}__Ifngr1__{endpoint}'] = res
            gene_contributions = z * w[None,:] / np.abs(w).sum()
            if endpoint == 'axis2':
                gene_contributions = reference_residuals(gene_contributions, obs, reference)
                total_res = gene_contributions.sum(axis=1)
                exact_res, _ = bridge.residualize_score(values['combined'],obs,'pseudotime',reference)
                center = bridge.weighted_median(total_res[reference],obs.loc[reference,'subject_id'])
                _, scale = bridge.weighted_mean_sd((total_res[reference]-center)[:,None],obs.loc[reference,'subject_id'])
                assert np.allclose((total_res-center)/scale[0], exact_res, atol=1e-9)
                gene_contributions /= scale[0]
            covariance_by_donor, loo_by_donor, native_correlations = [], [], []
            for subject, positions in obs.groupby('subject_id', observed=True).indices.items():
                positions=np.asarray(positions)
                if not obs.donor_eligible.iloc[positions[0]]:
                    continue
                if endpoint == 'axis2':
                    positions=positions[obs.incomplete.iloc[positions].to_numpy()]
                    if len(positions)<20:
                        continue
                    y=obs.activation_residual.iloc[positions].to_numpy(float)
                else:
                    y=obs.pseudotime.iloc[positions].to_numpy(float)
                    if np.ptp(y)<.05:
                        continue
                gc=gene_contributions[positions]
                centered=gc-gc.mean(axis=0)
                cov=((y-y.mean())[:,None]*centered).mean(axis=0)
                total=gc.sum(axis=1)
                assert np.isclose(cov.sum(),np.mean((y-y.mean())*(total-total.mean())),atol=1e-9)
                covariance_by_donor.append(cov)
                loo_by_donor.append(correlations(total[:,None]-gc,y))
                native_correlations.append(float(stats.spearmanr(total,y).statistic))
            mean_cov=np.mean(covariance_by_donor,axis=0)
            loo=np.nanmean(loo_by_donor,axis=0)
            abs_sum=np.abs(mean_cov).sum()
            for j,gene in enumerate(genes):
                contributions.append({'strict':strict,'endpoint':endpoint,'gene':gene,'arm':'up' if w[j]>0 else 'down',
                                      'weight':float(w[j]),'mean_signed_covariance':float(mean_cov[j]),
                                      'fraction_absolute_mean_covariance':float(abs(mean_cov[j])/abs_sum) if abs_sum else 0})
                influence.append({'strict':strict,'endpoint':endpoint,'omitted_gene':gene,
                                  'original_mean_rho':float(np.mean(native_correlations)),
                                  'omission_mean_rho':float(loo[j]),
                                  'rho_change':float(loo[j]-np.mean(native_correlations))})
            if not strict and endpoint=='axis2':
                for (disease,celltype), inds in obs.loc[obs.donor_eligible].groupby(['disease','celltype'],observed=True).groups.items():
                    positions=obs.index.get_indexer(inds)
                    subjects=obs.subject_id.iloc[positions]
                    detection=(x[positions]>0).astype(float)
                    donor_detection=pd.DataFrame(detection,index=subjects).groupby(level=0).mean()
                    donor_variance=pd.DataFrame(x[positions],index=subjects).groupby(level=0).var(ddof=0)
                    for j,gene in enumerate(genes):
                        diagnostics.append({'disease':disease,'celltype':celltype,'gene':gene,'arm':'up' if w[j]>0 else 'down',
                                            'weight':float(w[j]),'n_cells':len(positions),'n_donors':len(donor_detection),
                                            'donor_weighted_detection':float(donor_detection.iloc[:,j].mean()),
                                            'donors_detection_ge10pct':int((donor_detection.iloc[:,j]>=.1).sum()),
                                            'donors_variable':int((donor_variance.iloc[:,j]>0).sum()),
                                            'nt_reference_sd_used':float(sd[j])})
        signatures={(arm,'Ifngr1'):None for arm in ('combined','up','down')}
        effects=bridge.donor_effects(scored.loc[scored.donor_eligible],signatures)
        effect_tables.append(effects.assign(strict_joint_holdout=strict))
    effects=pd.concat(effect_tables,ignore_index=True)
    original=pd.read_csv(BAN/'signature_donor_effects.csv')
    original=original[original.signature_type.eq('direct_ko_consensus')]
    merged=effects[effects.signature_type.eq('combined')].merge(original,on=['subject_id','endpoint','strict_joint_holdout'],suffixes=('_new','_old'))
    assert len(merged)==len(original)==len(effects[effects.signature_type.eq('combined')])
    difference=float(np.max(np.abs(merged.effect_new-merged.effect_old)))
    assert difference<1e-10, difference
    rng=np.random.default_rng(SEED+3)
    summaries=[]
    for (strict,arm,endpoint), frame in effects.groupby(['strict_joint_holdout','signature_type','endpoint'],sort=True):
        val=frame.effect.to_numpy(float)
        val=val[np.isfinite(val)]
        p,method=bridge.sign_flip_test(val,SEED)
        ci=np.quantile(rng.choice(val,(2000,len(val)),replace=True).mean(axis=1),[.025,.975])
        summaries.append({'strict':bool(strict),'arm':arm,'endpoint':endpoint,'n_donors':len(val),
                          'mean_effect':float(val.mean()),'ci95_fixed_scores':ci.tolist(),
                          'p_two_sided':p,'test_method':method,'n_positive':int((val>0).sum()),'n_negative':int((val<0).sum())})
    for strict in (False,True):
        family=[r for r in summaries if r['strict']==strict and r['arm']!='combined']
        assert len(family)==6
        for r,q in zip(family,multipletests([r['p_two_sided'] for r in family],method='fdr_bh')[1]):
            r['q_bh_six_arm_tests']=float(q)
    for name,data in [('donor_effects',effects),('gene_detection_diagnostics',pd.DataFrame(diagnostics)),
                      ('gene_covariance_contributions',pd.DataFrame(contributions)),('leave_one_gene_out',pd.DataFrame(influence)),
                      ('arm_structure',pd.DataFrame(arm_structure))]:
        data.to_csv(output/f'{name}.csv',index=False)
    result={'ban_cache_sha256':sha(cache),
            'combined_effects_max_error':difference,'combined_donor_effects_recovered':len(merged),
            'summaries':summaries}
    write(result,output/'results.json')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['extract','thp1','ban'])
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--counts', type=Path)
    args=parser.parse_args()
    if args.stage == 'extract':
        if args.metadata is None or args.counts is None:
            parser.error('extract requires --metadata and --counts')
        extract(args.metadata, args.counts)
    elif args.stage == 'thp1':
        score_thp(THP / 'knockout_signature')
    else:
        score_ban(BAN / 'signature_components')
