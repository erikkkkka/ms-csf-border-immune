# Third-party sources and licenses

This repository distinguishes original analysis code and generated outputs from
reused third-party inputs. The original data providers' licenses and access
conditions continue to apply to their materials.

## ChEMBL snapshots

`data/reference/chembl_bx912_molecule.json` and `chembl_bx912_activities.json` are
unmodified, small ChEMBL API snapshots used by the figure and compound-annotation
code. ChEMBL data are provided under **Creative Commons Attribution-ShareAlike 3.0
Unported**, not the license for the authors' generated data.

- Provider: EMBL-EBI ChEMBL, https://www.ebi.ac.uk/chembl/
- License: https://creativecommons.org/licenses/by-sa/3.0/
- Provider documentation: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions
- Molecule endpoint: https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=BX-912
- Activity endpoint: https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL3916849

## Reused data not mirrored

Original GEO archives, external papers/PDFs, published supplementary workbooks,
DoRothEA/OmniPath networks, Enrichr/PerturbSeqR libraries, NIBR input databases,
and original PDB records are referenced rather than rehosted. Obtain those resources
from the source records listed in `reproducibility/DATA_SOURCE_MANIFEST.csv` and
`docs/EXTERNAL_INPUTS.md`, subject to the original providers' terms.

Prepared receptors, ligand files, ortholog mappings, signature selections, and
analysis scores are derived research outputs. Their source accessions and processing
scripts remain identifiable. Applicable third-party attribution and other source
conditions remain in force.
