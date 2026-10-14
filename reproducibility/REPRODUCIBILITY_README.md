# Reproducibility

## Starting data

The downstream workflow starts from the cleaned, decontaminated, doublet-filtered
AnnData checkpoints in `results/cleaned/` and
`results/external_validation/GSE328265/`. Upstream acquisition, cell calling,
DecontX, Scrublet/scDblFinder, and annotation scripts are provided separately.

## Commands

Verify files after extracting the processed-data assets into the repository root:

```bash
python release_tools/verify_release.py --root . --require-data
```

Regenerate figures from released results:

```bash
python release_tools/reproduce_public.py --mode figures
```

Rebuild downstream analyses after obtaining the inputs in `docs/EXTERNAL_INPUTS.md`:

```bash
python release_tools/reproduce_public.py --mode analysis --dry-run
python release_tools/reproduce_public.py --mode analysis \
  --main-python .conda_trajectory/bin/python \
  --celloracle-python .conda_celloracle/bin/python \
  --docking-python .conda_docking/bin/python \
  --rscript Rscript
```

Rebuild supplementary workbooks:

```bash
python release_tools/reproduce_public.py --mode tables
```

Use `--list`, `--from-step NAME`, and `--to-step NAME` to inspect or resume the
analysis order. Non-zero command exit statuses stop execution.

## Environments

`docs/ENVIRONMENT.md` describes the separate main, CellOracle, docking, and R
runtimes. Explicit Conda exports, Python package lists, R session information,
and external tool versions are recorded in this directory.

## Provenance and determinism

`DATA_SOURCE_MANIFEST.csv` records source accessions, URLs, analysis roles, sizes,
and SHA-256 values. The external human cohort contains new-data subseries
GSE327243 and GSE328264. The in vivo CRISPR source files are identified in their
external source manifest.

`ANALYSIS_WORKFLOW.json` defines the 34 downstream analysis commands, including
`trajectory_endpoint_auc`, which exports endpoint AUCs directly from saved
multiroot DPT values without rebuilding the trajectory graphs.
`RUN_ORDER.csv` records the runtimes, arguments, and random seeds for analysis,
figure, table, and validation steps. Exact enumeration is used where feasible;
Monte Carlo tests and docking use fixed recorded seeds. Plot jitter affects
display only.
