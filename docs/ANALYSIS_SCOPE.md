# Analysis Scope

## Cohort roles

`metadata/cohort_analysis_roles.csv` defines eligibility for each analysis.
GSE138266 and GSE286068 are disease-discovery cohorts. GSE138266 and GSE194078
provide paired blood-CSF root references. Trajectories are independently
reconstructed in GSE138266, GSE194078 and GSE286068. GSE194078 is excluded from
disease contrasts because its non-MS participants have other inflammatory
demyelinating diseases. GSE328265 supplies the external fixed-score projection,
using newly sequenced participants after exclusion of previously published
participants.

The sample-level `analysis_role` field agrees with these cohort roles.
Supplementary Table 1 separates doublet-union singlet retention from expression
filtering after ambient-RNA correction.

## Trajectory endpoint comparisons

Figure 1e compares full-cohort primary multiroot DPT with method-matched
leave-one-donor reconstructions. Both comparisons use `terminal_candidate`
versus `root_candidate` cells from `trajectory_role`.
`results/trajectory/full_multiroot_endpoint_auc.csv` contains the primary and
marker-excluded full-cohort AUCs for all three trajectory cohorts. Script
`84_export_multiroot_endpoint_auc.py` reads saved cell scores; it does not refit
the graphs. Supplementary Table 3 retains the single-root topology diagnostics
separately from the multiroot endpoint comparisons.

The primary three-cohort disease meta-analysis and its two-endpoint multiple-test
correction are exported separately from the discovery-only two-cohort,
11-endpoint analysis. External projection topology is also exported separately.

## Regulatory networks

Virtual perturbations use networks fitted at the whole-cohort level. Effects are
aggregated by donor for statistical testing. These are not separately fitted
donor-specific networks. Supplementary Figure S8 retains separate statistics
for the expected-direction tests and the shuffled-network specificity tests.

## Versioned data

The four processed-cohort archives are unchanged from `v0.1.0`.
`reproducibility/RELEASE_ASSET_MANIFEST.csv` records their original download URLs
and SHA-256 checksums. `v0.1.1` provides updated code, result exports, figure
sources and PDFs, and download manifests without duplicating those large
archives.

## Figure exports

The archived PDFs are the final formatted figures, including author headers.
`reproducibility/FIGURE_EXPORT_MANIFEST.csv` records their dimensions and hashes.
The figure-generation scripts reproduce the underlying plotted values;
`scripts/render_bx912_properties.py` provides the vector structure and property
typesetting used in Figure 5g. This optional PDF-rendering step requires RDKit,
PyMuPDF, pypdfium2, ReportLab, svglib and Pillow in addition to NumPy.
