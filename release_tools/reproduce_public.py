#!/usr/bin/env python3
"""Run published analyses or regenerate figures without assembling a manuscript."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def pipeline(root: Path):
    records = json.loads((root / "reproducibility/ANALYSIS_WORKFLOW.json").read_text())
    return [SimpleNamespace(**record) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("figures", "analysis", "tables"), default="figures")
    parser.add_argument("--main-python", default=sys.executable)
    parser.add_argument("--celloracle-python", default=os.environ.get("MS_CSF_CELLORACLE_PYTHON", str(ROOT / ".conda_celloracle/bin/python")))
    parser.add_argument("--docking-python", default=os.environ.get("MS_CSF_DOCKING_PYTHON", str(ROOT / ".conda_docking/bin/python")))
    parser.add_argument("--rscript", default=os.environ.get("MS_CSF_RSCRIPT", "Rscript"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_steps")
    parser.add_argument("--from-step")
    parser.add_argument("--to-step")
    args = parser.parse_args()
    root = args.root.resolve()
    runtimes = {"main": args.main_python, "celloracle": args.celloracle_python,
                "docking": args.docking_python, "r": args.rscript}
    commands = []
    if args.mode == "analysis":
        for step in pipeline(root):
            command = [runtimes[step.runtime], str(root / step.script)]
            command += [str(root)] if step.runtime == "r" else ["--root", str(root)]
            command += list(step.args)
            commands.append((step.name, command))
    elif args.mode == "tables":
        commands.append(("supplementary_tables", [args.main_python,
                         str(root / "scripts/82_build_supplementary_tables.py"), "--root", str(root)]))
    else:
        for label, script in (("main", "75_generate_manuscript_figures.py"),
                              ("supplementary", "80_generate_extended_data_figures.py")):
            commands.append((f"{label}_figures", [args.main_python, str(root / "scripts" / script),
                             "--root", str(root), "--formats", "pdf", "--output-dir",
                             str(root / "reproduced_figures" / label), "--source-dir",
                             str(root / "qa_generated/source_data" / label), "--qa-dir",
                             str(root / "qa_generated/layout" / label)]))
    names = [name for name, _ in commands]
    if args.list_steps:
        for name, command in commands:
            print(name + "\t" + shlex.join(command))
        return
    start = names.index(args.from_step) if args.from_step else 0
    stop = names.index(args.to_step) + 1 if args.to_step else len(commands)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    env.setdefault("NUMBA_CACHE_DIR", str(root / ".numba_cache"))
    env.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    for name, command in commands[start:stop]:
        print(name + ": " + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        executable = shutil.which(command[0])
        if executable is None:
            raise FileNotFoundError(f"Runtime not found: {command[0]}")
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
