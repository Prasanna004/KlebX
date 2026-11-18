"""Stdlib-only launcher. The first KlebX command creates conda env hvkp-tools.

The console script is installed into whichever Python ran ``pip install``.
Scoring runs in hvkp-tools, which holds Python 3.11, the model libraries,
BLAST, Kleborate, mlst, ABRicate, and their databases. This module must stay
importable without those libraries.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ENV_NAME = "hvkp-tools"
# Bioinformatics binaries only. A conda solve that also includes XGBoost and SHAP
# downloads the full bioconda index and can sit for a long time. The model
# libraries are installed with pip wheels below.
CONDA_PACKAGES = [
    "python=3.11",
    "blast",
    "kleborate",
    "mlst",
    "abricate",
    "mash",
    "pip",
]
PIP_SPEC = {
    "numpy": "numpy>=1.26",
    "pandas": "pandas>=2.2",
    "sklearn": "scikit-learn>=1.4",
    "xgboost": "xgboost==3.1.2",
    "shap": "shap>=0.49",
    "joblib": "joblib>=1.2",
    "matplotlib": "matplotlib>=3.7",
    "openpyxl": "openpyxl>=3.1",
}
TOOL_BINARIES = ("blastn", "kleborate", "mlst", "abricate", "python")
ABRICATE_DBS = ("ncbi", "card", "vfdb", "plasmidfinder")
PY_PACKAGES = ("numpy", "pandas", "sklearn", "xgboost", "shap", "joblib", "matplotlib", "openpyxl", "kleborate")


def _conda_exe() -> str | None:
    for name in ("mamba", "micromamba", "conda"):
        hit = shutil.which(name)
        if hit:
            return hit
    home = Path.home()
    for path in (
        home / "anaconda3/bin/conda",
        home / "miniconda3/bin/conda",
        home / "mambaforge/bin/mamba",
        home / "miniforge3/bin/conda",
        Path("/opt/conda/bin/conda"),
    ):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _conda_roots(conda: str | None) -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    if conda:
        roots.append(Path(conda).resolve().parent.parent)
    if os.environ.get("CONDA_EXE"):
        roots.append(Path(os.environ["CONDA_EXE"]).resolve().parent.parent)
    roots.extend(
        [
            home / "anaconda3",
            home / "miniconda3",
            home / "mambaforge",
            home / "miniforge3",
            Path("/opt/conda"),
        ]
    )
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _env_prefix(conda: str | None) -> Path | None:
    for root in _conda_roots(conda):
        prefix = root / "envs" / ENV_NAME
        if (prefix / "bin").is_dir():
            return prefix
    if not conda:
        return None
    proc = subprocess.run([conda, "env", "list"], text=True, capture_output=True)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if ENV_NAME not in parts:
            continue
        for part in reversed(parts):
            path = Path(part)
            if path.name == ENV_NAME and (path / "bin").is_dir():
                return path
    return None


def _inside_managed_environment() -> bool:
    return Path(sys.prefix).name == ENV_NAME


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise SystemExit(
            "KlebX must be installed from its source checkout so the first run can "
            "install it into the hvkp-tools conda environment."
        )
    return root


def _run_checked(command: list[str], description: str) -> None:
    proc = subprocess.run(command)
    if proc.returncode:
        raise SystemExit(f"{description} failed (exit {proc.returncode})")


def _site_packages(prefix: Path) -> Path | None:
    python = prefix / "bin" / "python"
    if not python.is_file():
        return None
    proc = subprocess.run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    path = Path(proc.stdout.strip())
    return path if path.is_dir() else None


def _problems(prefix: Path | None) -> list[str]:
    if prefix is None:
        return ["environment"]
    missing: list[str] = []
    for name in TOOL_BINARIES:
        path = prefix / "bin" / name
        if not path.is_file() or not os.access(path, os.X_OK):
            missing.append(name)
    for db in ABRICATE_DBS:
        seq = prefix / "db" / db / "sequences"
        if not seq.is_file() or seq.stat().st_size == 0:
            missing.append(db)
    scheme = prefix / "db" / "pubmlst" / "klebsiella" / "klebsiella.txt"
    if not scheme.is_file():
        missing.append("mlst-klebsiella")
    site = _site_packages(prefix)
    if site is None:
        missing.append("python-site")
    else:
        for package in PY_PACKAGES:
            # The kleborate conda package provides the command. Its Python
            # folder is not a reason to solve bioconda again.
            if package == "kleborate":
                continue
            if not (site / package).exists():
                missing.append(package)
    # hv_xgb.joblib was saved with XGBoost 3.1.2. The 1.7 pin cannot load it.
    python = prefix / "bin" / "python"
    if python.is_file() and "xgboost" not in missing:
        proc = subprocess.run(
            [str(python), "-c", "import xgboost; print(xgboost.__version__)"],
            text=True,
            capture_output=True,
        )
        version = tuple(int(p) for p in proc.stdout.strip().split(".") if p.isdigit())
        if proc.returncode != 0 or version < (3, 1):
            missing.append("xgboost")
    return missing


def _checkout_installed(prefix: Path, root: Path) -> bool:
    site = _site_packages(prefix)
    if site is None:
        return False
    files = sorted(site.glob("klebx-*.dist-info/direct_url.json"))
    if not files:
        return False
    try:
        url = json.loads(files[-1].read_text()).get("url", "")
    except (OSError, json.JSONDecodeError):
        return False
    return url.rstrip("/") == root.resolve().as_uri().rstrip("/")


# Only a missing program may be passed to conda. Names from PIP_SPEC must never
# appear here: a conda solve for them downloads the bioconda index and stalls.
_CONDA_FOR_PROBLEM = {
    "blastn": "blast",
    "kleborate": "kleborate",
    "mlst": "mlst",
    "mlst-klebsiella": "mlst",
    "abricate": "abricate",
    "python": "python=3.11",
    "python-site": "python=3.11",
}


def _install_plan(prefix_exists: bool, problems: list[str]) -> tuple[list[str], list[str]]:
    """Split gaps into conda packages and pip specs.

    Python libraries always go to pip. ABRicate database names are omitted;
    ``abricate --setupdb`` fills those without a channel solve.
    """
    pip_specs: list[str] = []
    seen_pip: set[str] = set()
    for item in problems:
        spec = PIP_SPEC.get(item)
        if spec and spec not in seen_pip:
            seen_pip.add(spec)
            pip_specs.append(spec)
    if not prefix_exists or "environment" in problems:
        return list(CONDA_PACKAGES), pip_specs
    conda_packages: list[str] = []
    seen: set[str] = set()
    for problem, package in _CONDA_FOR_PROBLEM.items():
        if problem in problems and package not in seen:
            seen.add(package)
            conda_packages.append(package)
    return conda_packages, pip_specs


def _install_cmd(conda: str, prefix_exists: bool, packages: list[str]) -> list[str]:
    blocked = [
        pkg
        for pkg in packages
        if pkg in PIP_SPEC or pkg.split("=", 1)[0].split(">", 1)[0].split("<", 1)[0] in PIP_SPEC
    ]
    if blocked:
        raise SystemExit(
            "refusing to conda-install Python libraries ("
            + ", ".join(blocked)
            + "). They are installed with pip."
        )
    return [
        conda,
        "install" if prefix_exists else "create",
        "-y",
        "-n",
        ENV_NAME,
        "--override-channels",
        "--strict-channel-priority",
        "-c",
        "conda-forge",
        "-c",
        "bioconda",
        *packages,
    ]


def _repair_databases(prefix: Path) -> None:
    missing = [db for db in ABRICATE_DBS if not (prefix / "db" / db / "sequences").is_file()]
    abricate = prefix / "bin" / "abricate"
    if not missing or not abricate.is_file():
        return
    print("[KlebX] downloading ABRicate databases: " + ", ".join(missing), flush=True)
    _run_checked([str(abricate), "--setupdb"], "ABRicate database download")


def prepare_environment(allow_install: bool = True) -> Path:
    """Create hvkp-tools on first use and install every tool. Later calls only repair gaps."""
    conda = _conda_exe()
    if not conda:
        raise SystemExit(
            "conda, mamba, or micromamba is required to create the hvkp-tools environment\n"
            "and install BLAST, Kleborate, mlst, ABRicate, and the model libraries.\n"
            "Install Miniforge or Miniconda, then rerun KlebX."
        )
    root = _project_root()
    prefix = _env_prefix(conda)
    problems = _problems(prefix)
    if problems and not allow_install:
        raise SystemExit(
            "Conda environment hvkp-tools is not ready ("
            + ", ".join(problems)
            + "). Rerun without --skip-setup."
        )
    conda_packages, _pip_specs = _install_plan(prefix is not None, problems)
    if conda_packages:
        print(
            "[KlebX] first run: creating conda env hvkp-tools and installing "
            "BLAST, Kleborate, mlst, and ABRicate"
            if prefix is None
            else "[KlebX] installing missing tools into conda env hvkp-tools: "
            + ", ".join(conda_packages),
            flush=True,
        )
        _run_checked(
            _install_cmd(conda, prefix is not None, conda_packages),
            "Conda environment setup",
        )
        prefix = _env_prefix(conda)
        if prefix is None:
            raise SystemExit("conda reported success but hvkp-tools is still missing")
    if prefix is not None and any(item in ABRICATE_DBS for item in _problems(prefix)):
        _repair_databases(prefix)
    if prefix is not None:
        specs = [PIP_SPEC[item] for item in _problems(prefix) if item in PIP_SPEC]
        if specs:
            print("[KlebX] installing Python libraries with pip: " + ", ".join(specs), flush=True)
            _run_checked(
                [
                    str(prefix / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    *specs,
                ],
                "Python library installation",
            )
    if prefix is not None:
        still = _problems(prefix)
        if still:
            raise SystemExit("still missing after install: " + ", ".join(still))
    if prefix is None:
        raise SystemExit("hvkp-tools is missing")
    if not _checkout_installed(prefix, root):
        if not allow_install:
            raise SystemExit("KlebX is not installed in hvkp-tools. Rerun without --skip-setup.")
        print(f"[KlebX] installing KlebX into {prefix}", flush=True)
        _run_checked(
            [str(prefix / "bin" / "python"), "-m", "pip", "install", "--no-deps", "-e", str(root)],
            "KlebX installation into hvkp-tools",
        )
    return prefix


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    allow_install = "--skip-setup" not in args
    if _inside_managed_environment():
        prepare_environment(allow_install=allow_install)
        from KlebX.predict import main as predict_main

        return predict_main(args)

    prefix = prepare_environment(allow_install=allow_install)
    python = prefix / "bin" / "python"
    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = ENV_NAME
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    os.execve(str(python), [str(python), "-m", "KlebX", *args], env)
