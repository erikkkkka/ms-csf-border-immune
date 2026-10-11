#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


ROWS = [
    {
        "candidate": "Teriflunomide", "route": "phenotype_reversal",
        "screen_evidence": "NIBR Drug-seq rank 8",
        "clinical_context": "Approved MS disease-modifying therapy",
        "decision": "positive_control", "tier": "benchmark",
        "rationale": "Blind recovery supports pharmacologic validity of the BAM signature.",
        "source": "local:NIBR_DRUGseq_2025_directional_reversal_compounds.csv",
    },
    {
        "candidate": "Mitoxantrone", "route": "phenotype_reversal",
        "screen_evidence": "L1000CDS2 rank 33",
        "clinical_context": "Known MS therapy with major safety limitations",
        "decision": "positive_control", "tier": "benchmark",
        "rationale": "Blind recovery is useful validation but does not imply a preferred therapy.",
        "source": "local:L1000CDS2_reversers_top50.csv",
    },
    {
        "candidate": "Simvastatin", "route": "phenotype_reversal",
        "screen_evidence": "L1000CDS2 rank 13",
        "clinical_context": "MS-STAT2 phase 3 negative for SPMS disability progression",
        "decision": "clinical_veto", "tier": "negative_benchmark",
        "rationale": "A strong reversal rank cannot override a negative disease-relevant trial.",
        "source": "PMID:41045938",
    },
    {
        "candidate": "TNF blockade (etanercept/lenercept class)",
        "route": "phenotype_reversal", "screen_evidence": "GEO perturbation rank 7",
        "clinical_context": "Lenercept increased MS exacerbations in a phase 2 trial",
        "decision": "clinical_veto", "tier": "negative_benchmark",
        "rationale": "Transcriptomic reversal can nominate a clinically harmful immune intervention.",
        "source": "PMID:10449104",
    },
    {
        "candidate": "PI3K/mTOR inhibitors", "route": "phenotype_reversal",
        "screen_evidence": "Repeated high-ranking L1000 class",
        "clinical_context": "Broad cytostatic and immunosuppressive mechanism",
        "decision": "supplement_only", "tier": "nonspecific",
        "rationale": "Likely captures proliferation/stress reversal rather than BAM mechanism specificity.",
        "source": "local:L1000CDS2_reversers_top50.csv",
    },
    {
        "candidate": "HSP90 inhibitors (monorden class)", "route": "phenotype_reversal",
        "screen_evidence": "Old CMap rank 1; compound FDR 0.0083",
        "clinical_context": "Stress-response and toxicity-prone class",
        "decision": "exclude_from_lead_set", "tier": "nonspecific",
        "rationale": "Weak signature overlap and generic stress biology reduce mechanistic value.",
        "source": "local:Old_CMAP_directional_reversal_compounds.csv",
    },
    {
        "candidate": "Zetomipzomib", "route": "PSMB9_target_based",
        "screen_evidence": "Human PSMB9 IC50 0.623 nM in ChEMBL",
        "clinical_context": "Immunoproteasome strategy already established in EAE/inflammation",
        "decision": "pathway_benchmark", "tier": "crowded_target",
        "rationale": "Potent benchmark, but covalent chemistry invalidates ordinary Vina and novelty is limited.",
        "source": "local:ChEMBL_PSMB9_named_compounds.csv",
    },
    {
        "candidate": "LYS006", "route": "LTA4H_target_based",
        "screen_evidence": "Sub-nanomolar/picomolar recombinant activity and co-crystal PDB 7AUZ; no public transcriptomic perturbation signature found",
        "clinical_context": "Human phase 1/2 exposure outside MS",
        "decision": "secondary_target_probe", "tier": "Tier_3",
        "rationale": "Strong biochemical and structural probe, but public LTA4H KO consensus did not reverse BAM/MHC-I and no disease-relevant myeloid perturbation data exist.",
        "source": "PMID:33592148;PMID:38407540;PDB:7AUZ;local:lta4h_perturbation_audit/LTA4H_GENETIC_PERTURBATION_REPORT.md",
    },
    {
        "candidate": "Acebilustat", "route": "LTA4H_target_based",
        "screen_evidence": "Recombinant IC50 3-4.9 nM; 0/4 U2OS DRUG-seq doses showed corrected BAM reversal",
        "clinical_context": "Phase 2 exposure; no primary efficacy benefit in CF or mild/moderate COVID-19",
        "decision": "negative_directional_benchmark", "tier": "Tier_3",
        "rationale": "Clinically exposed and selective comparator, but direct transcriptomic evidence does not support BAM or MHC-I reversal.",
        "source": "PMID:34538755;PMID:36996150;local:lta4h_perturbation_audit/LTA4H_INHIBITOR_DRUGSEQ_REPORT.md",
    },
    {
        "candidate": "SC-57461A", "route": "LTA4H_target_based",
        "screen_evidence": "Primary mouse ILC2 RNA-seq: 53 FDR-significant genes; BAM reversal P=0.0718 and MHC-I suppression P=0.455",
        "clinical_context": "Preclinical inhibitor in a mucosal lymphoid, not CNS myeloid, context",
        "decision": "immune_context_benchmark", "tier": "Tier_3",
        "rationale": "Confirms an immune-cell metabolic response to LTA4H inhibition, but cross-species leave-one-out analyses do not robustly support BAM or MHC-I reversal.",
        "source": "GEO:GSE304475;PMID:41838192;local:lta4h_perturbation_audit/GSE304475/GSE304475_SC57461A_REPORT.md",
    },
    {
        "candidate": "Ubenimex", "route": "LTA4H_polypharmacology",
        "screen_evidence": "Nominal BAM reversal only at 10 uM (P=0.0261; FDR=0.2088); 0/4 corrected-positive doses",
        "clinical_context": "Launched drug targeting ANPEP, LTA4H, and RNPEP",
        "decision": "exclude_as_lta4h_specific_lead", "tier": "Tier_4",
        "rationale": "High-dose uncorrected signal and aminopeptidase polypharmacology prevent target-specific inference.",
        "source": "local:lta4h_perturbation_audit/LTA4H_INHIBITOR_DRUGSEQ_REPORT.md",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "results" / "drug_repurposing"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(out_dir / "drug_candidate_evidence_tiers.csv", index=False)


if __name__ == "__main__":
    main()
