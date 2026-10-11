# Supplementary table QA report

Audited at: 2026-10-11T04:46:51+00:00

## Status

**PASS**

- Workbooks: 7/7 passed.
- Data sheets: 87 across seven workbooks.
- Source links and hashes: 87/87 passed.
- Workbook manifest hashes: matched.
- Formula cells: 0; Excel error cells: 0.

## Checks

- Every workbook opens successfully and begins with an Overview sheet.
- Every data sheet is non-empty, uses Arial headers, and has a frozen header,
  autofilter and formatted Excel table.
- Data are static result exports; no spreadsheet formulas are introduced.
- Every declared source path exists and matches its recorded SHA-256 digest.
- Workbook digests match `SUPPLEMENTARY_WORKBOOK_MANIFEST.csv`.
