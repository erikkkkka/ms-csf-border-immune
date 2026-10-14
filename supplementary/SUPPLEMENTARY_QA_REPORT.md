# Supplementary table QA report

Audited at: 2026-10-14T07:31:59+00:00

## Status

**PASS**

- Workbooks: 7/7 passed.
- Data sheets: 92 across seven workbooks.
- Source links and hashes: 92/92 passed.
- Result scope, figure baselines and response-gene denominators: 16/16 passed.
- Workbook manifest hashes: matched.
- Formula cells: 0; Excel error cells: 0.

## Checks

- Every workbook opens successfully and begins with an Overview sheet.
- Every data sheet is non-empty, uses Arial headers, and has a frozen header,
  autofilter and formatted Excel table.
- Data are static result exports; no spreadsheet formulas are introduced.
- Every declared source path exists and matches its recorded SHA-256 digest.
- Workbook digests match `SUPPLEMENTARY_WORKBOOK_MANIFEST.csv`.
- Primary three-cohort disease results and discovery-only results are exported in separate sheets.
- Response-gene percentages use all originally published mouse genes, before ortholog filtering.
- Figure 1e compares primary multiroot DPT with method-matched donor omissions; complete AUCs are checked against saved cells.
- External projection topology, final cohort roles and post-decontamination expression-filter counts are exported separately.
- Cohort-GRN directional tests remain separate from shuffled-network specificity tests.
