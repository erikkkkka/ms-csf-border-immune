# Computational environments

The recorded platform is macOS ARM64. Three Python environments are separated because
CellOracle and the modern Scanpy/docking stack have different dependency constraints.
The original environment snapshot is retained in `reproducibility/`; version-only
package lists generated for this release are in `environments/`.

For the original platform, use the explicit Conda records:

```bash
conda create --prefix "$PWD/.conda_trajectory" --file reproducibility/python_main_conda_explicit.txt
conda create --prefix "$PWD/.conda_celloracle" --file reproducibility/python_celloracle_conda_explicit.txt
conda create --prefix "$PWD/.conda_docking" --file reproducibility/python_docking_conda_explicit.txt
```

The requirements files record all observed Python package versions, including
Conda-managed packages. They are provenance records, not a guarantee that pip alone
can reconstruct compiled packages on every platform. Use Conda-resolved dependencies
and install any remaining pip-only packages in the corresponding environment.
CellOracle/velocyto may require a compiler, OpenMP, and compatible NumPy.

The R layer requires Slingshot and the packages in `reproducibility/r_installed_packages.csv`.
The installed R session is recorded in `r_session_info.txt`. Pass `--rscript` to the
public entrypoint rather than relying on the original machine's R path.

Other tools include AutoDock Vina, Open Babel, and the platform-specific build tools
documented in `system_tools_versions.txt`. Linux and Windows installations have not
been independently validated for this release. No runtime directories or caches
are uploaded.

Use `release_tools/reproduce_public.py` to configure executable paths for each runtime.
