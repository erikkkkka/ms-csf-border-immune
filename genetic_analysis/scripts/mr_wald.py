#!/usr/bin/env python3
"""Align alleles and estimate lead-variant Mendelian randomization."""
from collections import defaultdict
from pathlib import Path
import csv
import gzip
import math
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]


def read(path, delimiter=',', compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, 'rt', newline='') as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write(path, rows, delimiter=','):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def complement(x):
    return x.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def orient(ea, oa, a1, a2):
    if (ea, oa) == (a1, a2):
        return 1, 'same_effect_allele'
    if (ea, oa) == (a2, a1):
        return -1, 'swapped_effect_allele'
    if (ea, oa) == (complement(a1), complement(a2)):
        return 1, 'strand_complement'
    if (ea, oa) == (complement(a2), complement(a1)):
        return -1, 'strand_complement_and_swap'
    return None, 'allele_mismatch'


def reconstruct(odds_ratio, p):
    if not math.isfinite(odds_ratio) or odds_ratio <= 0:
        return None, 'invalid_OR'
    if not math.isfinite(p) or not 0 < p < 1:
        return None, 'invalid_or_boundary_P'
    beta = math.log(odds_ratio)
    if beta == 0:
        return None, 'rounded_OR_equal_to_one'
    se = abs(beta) / norm.isf(p / 2)
    if not math.isfinite(se) or se <= 0:
        return None, 'invalid_reconstructed_SE'
    return (beta, se), 'SE_approximated_from_OR_and_P'


def main():
    eqtl = read(ROOT / 'work/exposure_regions_mapped.tsv', delimiter='\t')
    gwas = read(ROOT / 'work/IMSGC_three_regions_GRCh37.tsv.gz', delimiter='\t', compressed=True)
    by_position = defaultdict(list)
    for r in gwas:
        by_position[(r['CHR'], int(r['BP']))].append(r)
    harmonized, audit_rows = [], []
    for e in eqtl:
        audit = {'gene': e['gene'], 'cell_type': e['cell_type'],
                 'variant_GRCh38': e['variant_GRCh38'], 'pos_GRCh37': e['pos_GRCh37'],
                 'status': '', 'harmonization': '', 'gwas_study_count': ''}
        candidates = by_position[(e['chr_GRCh37'], int(e['pos_GRCh37']))]
        if not candidates:
            audit['status'] = 'no_outcome_at_position'
        else:
            matches = [(g, *orient(e['effect_allele_GRCh37'], e['other_allele_GRCh37'], g['A1'], g['A2'])) for g in candidates]
            matches = [x for x in matches if x[1] is not None]
            # A variant may appear in two region selections; deduplicate only
            # identical source records, never choose by the outcome p-value.
            unique = {tuple(g[k] for k in ['CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'P', 'OR']): (g, sign, how)
                      for g, sign, how in matches}
            if len(unique) != 1:
                audit['status'] = 'allele_mismatch' if not unique else 'ambiguous_multiple_source_matches'
            else:
                g, sign, how = next(iter(unique.values()))
                audit.update(harmonization=how, gwas_study_count=g['N'])
                try:
                    p, odds_ratio = float(g['P']), float(g['OR'])
                except ValueError:
                    audit['status'] = 'outcome_statistics_missing'
                else:
                    if int(g['N']) < 2:
                        audit['status'] = 'fewer_than_two_studies'
                    else:
                        estimates, status = reconstruct(odds_ratio, p)
                        audit['status'] = 'usable' if estimates else status
                        if estimates:
                            beta, se = estimates
                            harmonized.append({**e, 'gwas_SNP': g['SNP'], 'gwas_effect_allele_original': g['A1'],
                                               'gwas_other_allele_original': g['A2'], 'gwas_OR_original': odds_ratio,
                                               'gwas_P': p, 'gwas_study_count': int(g['N']),
                                               'gwas_beta_aligned': sign * beta, 'gwas_se_approx': se,
                                               'harmonization': how, 'SE_source': status,
                                               'palindromic_direction_unresolved': e['palindromic_SNP'] == 'True'})
        audit_rows.append(audit)
    assert len({(r['gene'], r['cell_type'], r['variant_GRCh38']) for r in harmonized}) == len(harmonized)
    write(ROOT / 'work/harmonized_regions.tsv', harmonized, delimiter='\t')
    write(ROOT / 'work/allele_alignment.csv', audit_rows)

    contexts = read(ROOT / 'results/lead_variants.csv')
    summaries, matched_leads = [], []
    for c in contexts:
        gene, cell = c['gene'], c['cell_type']
        local = [r for r in harmonized if r['gene'] == gene and r['cell_type'] == cell]
        lead = [r for r in local if r['variant_GRCh38'] == c['lead_variant_GRCh38']]
        row = {'gene': gene, 'cell_type': cell, 'planned_role': c['planned_role'],
               'instrument_rsid': c['rsid'], 'instrument_GRCh38': c['lead_variant_GRCh38'],
               'status': '', 'n_available_eqtl_variants': c['n_regional_variants'],
               'n_usable_common_variants': len(local), 'F_exposure_approx': c['F_wald_approx'],
               'beta_MR': '', 'SE_MR_first_order': '', 'odds_ratio': '', 'CI95_lower': '',
               'CI95_upper': '', 'nominal_p': '', 'P_delta_method': '',
               'exposure_unit': 'one_unit_on_author_model_expression_scale_not_per_SD',
               'outcome_SE': 'approximated_from_rounded_OR_and_P'}
        if not c['lead_beta']:
            row['status'] = 'not_estimated_missing_exposure_beta_SE'
        elif not lead:
            row['status'] = 'not_estimated_unavailable_or_unusable_outcome_lead'
        else:
            assert len(lead) == 1
            lead = lead[0]
            assert not lead['palindromic_direction_unresolved'], 'Do not guess MR direction.'
            bx, sx = float(lead['beta']), float(lead['se'])
            by, sy = lead['gwas_beta_aligned'], lead['gwas_se_approx']
            b, se = by / bx, sy / abs(bx)
            p = float(2 * norm.sf(abs(b / se)))
            assert math.isclose(p, lead['gwas_P'], rel_tol=1e-10)
            row.update(status='estimated_primary' if c['planned_role'] == 'primary' else 'estimated_exploratory',
                       beta_MR=b, SE_MR_first_order=se, odds_ratio=math.exp(b),
                       CI95_lower=math.exp(b - norm.ppf(.975) * se),
                       CI95_upper=math.exp(b + norm.ppf(.975) * se), nominal_p=p,
                       P_delta_method=float(2 * norm.sf(abs(b) / math.hypot(se, by * sx / bx**2))))
            matched_leads.append(lead)
        summaries.append(row)
    write(ROOT / 'results/mr_wald.csv', summaries)
    write(ROOT / 'work/MR_harmonized_leads.csv', matched_leads)


if __name__ == '__main__':
    main()
