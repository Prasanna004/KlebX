#!/usr/bin/env python3
"""Predict hypervirulent K. pneumoniae from a genome FASTA.

Core path (always):
  BLASTN the assembly against the six-gene marker database shipped with this
  package, score an XGBoost model, write a SHAP waterfall, and write a clear
  human-readable report (report.txt / report.md).

Optional path (conda env hvkp-tools, created on first launch):
  kleborate, mlst (Seemann), abricate, and their databases.
  Kleborate already reports ST (MLST) and a virulence score; run it whenever
  it is installed. Binaries in hvkp-tools are used ahead of any copy on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"

MIN_PIDENT = 80.0
# iuc 1 (classic ST23) and iuc 5 (the reference CDS) share ~79% nucleotide identity.
MIN_PIDENT_IUCD = 75.0
MIN_QCOV = 60.0


def _run(cmd: list[str], cwd: Path | None = None, extra_path: str | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
        # Seemann mlst/abricate need their conda lib/perl
        lib = str(Path(extra_path).parent / "lib")
        env["PERL5LIB"] = lib + os.pathsep + env.get("PERL5LIB", "")
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)


def _bin_dir(exe: str) -> str:
    return str(Path(exe).resolve().parent)


ENV_NAME = "hvkp-tools"
REQUIRED_TOOLS = {
    "blastn": "blast",
    "kleborate": "kleborate",
    "mlst": "mlst",
    "abricate": "abricate",
}
ABRICATE_DBS = ("ncbi", "card", "vfdb", "plasmidfinder")
_PREFIX: Path | None = None
_PREFIX_READY = False


def _conda_roots() -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    conda = _conda_exe()
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


def _conda_exe() -> str | None:
    for name in ("mamba", "micromamba", "conda"):
        hit = shutil.which(name)
        if hit:
            return hit
    home = Path.home()
    for p in (
        home / "anaconda3" / "bin" / "conda",
        home / "miniconda3" / "bin" / "conda",
        home / "mambaforge" / "bin" / "mamba",
        home / "miniforge3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
    ):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def _forget_prefix() -> None:
    global _PREFIX, _PREFIX_READY
    _PREFIX = None
    _PREFIX_READY = False


def tools_prefix() -> Path | None:
    """Directory of the hvkp-tools conda env, or None if it has not been created."""
    global _PREFIX, _PREFIX_READY
    if _PREFIX_READY:
        return _PREFIX
    found: Path | None = None
    for root in _conda_roots():
        prefix = root / "envs" / ENV_NAME
        if (prefix / "bin").is_dir():
            found = prefix
            break
    if found is None:
        conda = _conda_exe()
        if conda:
            proc = subprocess.run([conda, "env", "list"], text=True, capture_output=True)
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if ENV_NAME not in parts:
                        continue
                    for part in reversed(parts):
                        path = Path(part)
                        if path.name == ENV_NAME and (path / "bin").is_dir():
                            found = path
                            break
                    if found:
                        break
    _PREFIX = found
    _PREFIX_READY = True
    return _PREFIX


def _bin(prefix: Path | None, name: str) -> str | None:
    if prefix is None:
        return None
    path = prefix / "bin" / name
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    return None


def _missing_data(prefix: Path | None) -> list[str]:
    if prefix is None:
        return ["hvkp-tools environment"]
    missing: list[str] = []
    for db in ABRICATE_DBS:
        seq = prefix / "db" / db / "sequences"
        if not seq.is_file() or seq.stat().st_size == 0:
            missing.append(db)
    scheme = prefix / "db" / "pubmlst" / "klebsiella" / "klebsiella.txt"
    if not scheme.is_file():
        missing.append("mlst-klebsiella")
    if not list((prefix / "lib").glob("python*/site-packages/kleborate/__main__.py")):
        missing.append("kleborate-data")
    return missing


def _install_cmd(conda: str, prefix_exists: bool, packages: list[str]) -> list[str]:
    action = "install" if prefix_exists else "create"
    return [
        conda,
        action,
        "-y",
        "-n",
        ENV_NAME,
        "--override-channels",
        "-c",
        "conda-forge",
        "-c",
        "bioconda",
        *packages,
    ]


def _print_tools(found: dict[str, str | None], heading: str) -> None:
    print(heading, flush=True)
    for name, path in found.items():
        print(f"       {name:10s} {path or 'NOT FOUND'}", flush=True)


_VERSION_ARGS = {
    "blastn": ["-version"],
    "kleborate": ["--version"],
    "mlst": ["--version"],
    "abricate": ["--version"],
}


def _tool_runs(path: str, name: str) -> bool:
    proc = _run([path, *_VERSION_ARGS[name]], extra_path=_bin_dir(path))
    return proc.returncode == 0


def ensure_tools(install: bool = True) -> dict[str, str | None]:
    """Check blastn, kleborate, mlst, and abricate on every run, then let the caller score genomes.

    The first run installs the whole hvkp-tools environment. A copy of a tool
    already on PATH does not count, and a binary that does not execute is reinstalled.
    """
    from KlebX.bootstrap import prepare_environment

    prefix = tools_prefix()
    found = {name: _bin(prefix, name) for name in REQUIRED_TOOLS}
    for name, path in list(found.items()):
        if path and not _tool_runs(path, name):
            found[name] = None
    missing = [name for name, path in found.items() if not path]
    data_missing = [] if missing else _missing_data(prefix)
    if not missing and not data_missing:
        _print_tools(found, "[KlebX] tools checked:")
        return found

    if not install:
        resolved = {name: found[name] or _which_fallback(name) for name in REQUIRED_TOOLS}
        detail = ", ".join(missing or data_missing)
        _print_tools(resolved, f"[KlebX] setup skipped; still missing {detail}:")
        return resolved

    prepare_environment(allow_install=True)
    _forget_prefix()
    prefix = tools_prefix()
    found = {name: _bin(prefix, name) for name in REQUIRED_TOOLS}
    for name, path in list(found.items()):
        if path and not _tool_runs(path, name):
            found[name] = None
    missing = [name for name, path in found.items() if not path]
    data_missing = [] if missing else _missing_data(prefix)
    if missing or data_missing:
        _print_tools(found, "[KlebX] tools after install:")
        raise SystemExit("still missing after install: " + ", ".join(missing or data_missing))
    _print_tools(found, "[KlebX] tools checked:")
    return found


def _which_fallback(name: str) -> str | None:
    """Find a binary on PATH or in other conda envs when hvkp-tools is not used."""
    hit = shutil.which(name)
    if hit:
        return hit
    home = Path.home()
    roots = [
        Path(os.environ["CONDA_PREFIX"]) / "bin" if os.environ.get("CONDA_PREFIX") else None,
        home / "anaconda3" / "bin",
        home / "anaconda3" / "envs" / "hvkp-predictor" / "bin",
        home / "miniconda3" / "bin",
        home / "mambaforge" / "bin",
        home / "miniforge3" / "bin",
        Path("/opt/conda/bin"),
    ]
    for root in _conda_roots():
        envs = root / "envs"
        if envs.is_dir():
            for env in envs.glob("*/bin"):
                roots.append(env)
    seen: set[str] = set()
    for folder in roots:
        if folder is None:
            continue
        path = folder / name
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def which(name: str) -> str | None:
    """Find a binary in conda env hvkp-tools, then on PATH or other conda envs."""
    hit = _bin(tools_prefix(), name)
    if hit:
        return hit
    return _which_fallback(name)


def validate_fasta(fasta: Path) -> list[str]:
    """Return warnings; raise ValueError on empty / unusable FASTA."""
    warnings: list[str] = []
    if not fasta.exists():
        raise ValueError(f"FASTA not found: {fasta}")
    if fasta.stat().st_size == 0:
        raise ValueError(f"FASTA is empty: {fasta}")
    headers = 0
    empty_headers = 0
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                headers += 1
                if not line[1:].strip():
                    empty_headers += 1
    if headers == 0:
        raise ValueError(f"No FASTA headers (lines starting with '>') in {fasta}")
    if empty_headers:
        warnings.append(
            f"{empty_headers} empty FASTA header(s) (e.g. lone '>'). "
            "Seemann mlst / ABRicate may fail on those records; BLAST still runs."
        )
    return warnings


def blast_markers(fasta: Path, outdir: Path, threads: int = 4) -> pd.DataFrame:
    blastn = which("blastn")
    if not blastn:
        raise SystemExit("blastn is required. Run KlebX --tools to install it.")
    db = DATA / "markers"
    if not (DATA / "markers.nsq").exists() and not (DATA / "markers.nin").exists():
        raise SystemExit(f"BLAST database missing under {DATA} (expected markers.n*)")
    tsv = outdir / "marker_blast.tsv"
    cmd = [
        blastn,
        "-query", str(fasta),
        "-db", str(db),
        "-out", str(tsv),
        "-outfmt", "6 qseqid sseqid pident length qlen slen qstart qend sstart send evalue bitscore",
        "-perc_identity", str(MIN_PIDENT_IUCD),
        "-max_target_seqs", "20",
        "-culling_limit", "1",
        "-num_threads", str(max(1, int(threads))),
    ]
    p = _run(cmd, extra_path=_bin_dir(blastn))
    if p.returncode != 0:
        raise RuntimeError(f"blastn failed:\n{p.stderr}")
    cols = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
    if not tsv.exists() or tsv.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    hits = pd.read_csv(tsv, sep="\t", names=cols)
    hits["qcov"] = 100.0 * hits["length"] / hits["slen"]
    hits = hits[hits["qcov"] >= MIN_QCOV]
    floor = hits["sseqid"].map(lambda g: MIN_PIDENT_IUCD if g == "iucD" else MIN_PIDENT)
    hits = hits[hits["pident"] >= floor]
    return hits.sort_values("bitscore", ascending=False)


def presence_vector(hits: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    vec = np.zeros(len(features), dtype=np.float32)
    best = hits.groupby("sseqid", as_index=False).first() if len(hits) else hits
    by = {r.sseqid: r for r in best.itertuples()} if len(best) else {}
    for i, f in enumerate(features):
        if f in by:
            r = by[f]
            vec[i] = 1.0
            rows.append(
                {
                    "feature": f,
                    "present": 1,
                    "pident": float(r.pident),
                    "qcov": float(r.qcov),
                    "evalue": float(r.evalue),
                    "bitscore": float(r.bitscore),
                }
            )
        else:
            rows.append(
                {
                    "feature": f,
                    "present": 0,
                    "pident": np.nan,
                    "qcov": np.nan,
                    "evalue": np.nan,
                    "bitscore": np.nan,
                }
            )
    return vec, pd.DataFrame(rows)


def shap_waterfall(model, x: np.ndarray, names: list[str], out_png: Path, title: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(x.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.ravel(sv)
    base = expl.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.ravel(base)[-1])
    else:
        base = float(base)

    order = np.argsort(np.abs(sv))[::-1]
    keep = order[: min(12, len(order))]
    contrib = list(sv[keep])
    labs = [names[j] for j in keep]
    present = ["+" if x[j] > 0 else "−" for j in keep]
    starts, level = [], float(base)
    for c in contrib:
        starts.append(level)
        level += c
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ys = np.arange(len(contrib))[::-1]
    pos, neg = "#c0392b", "#1f6aa5"
    for y, start, c in zip(ys, starts, contrib):
        ax.barh(y, c, left=start, color=pos if c >= 0 else neg, height=0.62)
    ax.axvline(base, color="#868e96", ls="--", lw=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{p}  {lab}" for p, lab in zip(present, labs)])
    ax.set_xlabel("Model output (log-odds of hvKp)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "base_value": base,
        "shap": {names[i]: float(sv[i]) for i in range(len(names))},
        "prediction_margin": float(base + sv.sum()),
        "ranked": [{"feature": names[j], "shap": float(sv[j]), "present": bool(x[j] > 0)} for j in order],
    }


def run_kleborate(fasta: Path, outdir: Path) -> dict:
    exe = which("kleborate")
    if not exe:
        return {"available": False, "error": "kleborate not found (conda install -c bioconda kleborate)"}
    kb_dir = outdir / "kleborate_out"
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
    # Kleborate 3 writes a directory. -p kpsc is the K. pneumoniae species-complex preset.
    # -r means --resume in this version, so it is not used.
    cmd = [exe, "-a", str(fasta), "-o", str(kb_dir), "-p", "kpsc"]
    p = _run(cmd, extra_path=_bin_dir(exe))
    (outdir / "kleborate.log").write_text((p.stdout or "") + "\n" + (p.stderr or ""))
    out = outdir / "kleborate.tsv"
    produced = sorted(kb_dir.glob("*output.txt")) if kb_dir.is_dir() else []
    preferred = kb_dir / "klebsiella_pneumo_complex_output.txt"
    src = preferred if preferred.is_file() else (produced[0] if produced else None)
    if src is not None and src.is_file() and src.stat().st_size > 0:
        shutil.copyfile(src, out)
    if not out.exists() or out.stat().st_size == 0:
        return {"available": True, "error": (p.stderr or p.stdout or "no output").strip()[:1000], "cmd": " ".join(cmd)}
    df = pd.read_csv(out, sep="\t")
    raw = df.iloc[0].to_dict() if len(df) else {}

    def _clean(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        if s.lower() in {"nan", "na", "-", ""}:
            return None
        return v

    rec = {k: _clean(v) for k, v in raw.items()}
    rec["available"] = True
    rec["path"] = str(out)
    rec["binary"] = exe
    # Kleborate 3 prefixes each column with its module. Keep the short names the report uses.
    _alias_kleborate(rec)
    return rec


def _alias_kleborate(rec: dict) -> None:
    names = (
        "species", "ST", "virulence_score", "resistance_score",
        "Aerobactin", "Yersiniabactin", "Salmochelin", "Colibactin",
        "RmpADC", "rmpA2", "K_locus", "O_locus", "wzi",
        "Bla_acquired", "Bla_inhR_acquired", "Bla_ESBL_acquired", "Bla_ESBL_inhR_acquired",
        "Bla_Carb_acquired", "Col_acquired", "AGly_acquired", "Fcyn_acquired", "Flq_acquired",
        "Gly_acquired", "MLS_acquired", "Phe_acquired", "Rif_acquired", "Sul_acquired",
        "Tet_acquired", "Tgc_acquired", "Tmt_acquired",
        "QC_warnings", "N50", "total_size", "contig_count",
    )
    for name in names:
        if rec.get(name) not in (None, ""):
            continue
        suffix = f"__{name}"
        for key, value in rec.items():
            if str(key).endswith(suffix):
                rec[name] = value
                break


def run_mlst(fasta: Path, outdir: Path) -> dict:
    """Torsten Seemann mlst (https://github.com/tseemann/mlst), bioconda package mlst."""
    exe = which("mlst")
    if not exe:
        return {
            "available": False,
            "note": "Seemann mlst not found. Install: conda install -c bioconda mlst. Kleborate ST is used as fallback.",
        }
    cmd = [exe, "--scheme", "klebsiella", str(fasta)]
    p = _run(cmd, extra_path=_bin_dir(exe))
    text = (p.stdout or "") + (("\n" + p.stderr) if p.returncode else "")
    (outdir / "mlst.txt").write_text(text)
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip() and not ln.startswith("#")]
    line = lines[-1] if lines else ""
    parts = line.split("\t")
    rec = {"available": True, "binary": exe, "version": "seemann/mlst", "raw": line, "cmd": " ".join(cmd)}
    if p.returncode != 0:
        rec["error"] = (p.stderr or "").strip()[:1000]
    if len(parts) >= 3:
        st = parts[2]
        if st in {"-", "NA", "n/a", ""}:
            st = None
        rec.update({"file": parts[0], "scheme": parts[1], "ST": st, "alleles": parts[3:]})
    return rec


def run_abricate(fasta: Path, outdir: Path) -> dict:
    exe = which("abricate")
    if not exe:
        return {
            "available": False,
            "note": "abricate not found. Install: conda install -c bioconda abricate. Kleborate -r covers acquired AMR.",
        }
    extra = _bin_dir(exe)
    summary = {"available": True, "binary": exe, "databases": {}}
    for db in ("ncbi", "card", "vfdb", "plasmidfinder"):
        tsv = outdir / f"abricate_{db}.tsv"
        p = _run([exe, "--db", db, str(fasta)], extra_path=extra)
        # ABRicate writes the table to stdout; stderr is progress — do not treat stderr alone as failure
        tsv.write_text(p.stdout or "")
        lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
        if p.returncode != 0 and not lines:
            summary["databases"][db] = {"n": 0, "error": (p.stderr or "")[:500], "path": str(tsv)}
            continue
        hdr = lines[0].lstrip("#").split("\t") if lines else []
        start = 1 if lines and (lines[0].startswith("#") or "FILE" in hdr or "GENE" in hdr) else 0
        gene_col = hdr.index("GENE") if "GENE" in hdr else 5
        genes = []
        for ln in lines[start:]:
            if ln.startswith("#"):
                continue
            f = ln.split("\t")
            if len(f) > gene_col:
                genes.append(f[gene_col])
        summary["databases"][db] = {"n": len(genes), "genes": genes, "path": str(tsv)}
    return summary


def classify(prob: float) -> str:
    if prob >= 0.8:
        return "hvKp-like (high)"
    if prob >= 0.5:
        return "hvKp-like (moderate)"
    if prob >= 0.2:
        return "equivocal"
    return "non-hvKp-like"


def _fmt(v, default="-"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    s = str(v).strip()
    return default if s.lower() in {"", "nan", "na", "none", "-"} else s


def _st_display(result: dict) -> str:
    mlst = result.get("mlst") or {}
    kb = result.get("kleborate") or {}
    if mlst.get("ST"):
        return str(mlst["ST"])
    st = kb.get("ST")
    if st:
        s = str(st)
        return s[2:] if s.upper().startswith("ST") and s[2:].isdigit() else s
    return "NA"


def write_report(result: dict, outdir: Path) -> tuple[Path, Path]:
    """Write a clear human-readable report (txt + md). Returns (report.txt, report.md)."""
    markers = result.get("markers") or []
    shap = result.get("shap") or {}
    ranked = shap.get("ranked") or []
    kb = result.get("kleborate") or {}
    mlst = result.get("mlst") or {}
    ab = result.get("abricate") or {}
    warnings = result.get("warnings") or []

    present = [m for m in markers if m.get("present")]
    absent = [m for m in markers if not m.get("present")]
    st = _st_display(result)

    # ---- plain text ----
    L: list[str] = []
    bar = "=" * 78
    thin = "-" * 78
    L += [
        bar,
        "KlebX report",
        bar,
        f"Sample:          {result['sample']}",
        f"FASTA:           {result['fasta']}",
        f"Call:            {result['call']}",
        f"P(hvKp):         {result['p_hvkp']:.4f}",
        f"Markers present: {result['n_markers_present']} / {len(result['markers'])}",
        f"ST (best):       {st}",
        "",
        thin,
        "1. Marker panel  (BLASTN: identity ≥ 80%, iucD ≥ 75%, subject coverage ≥ 60%)",
        thin,
        f"{'gene':<12} {'hit':<6} {'identity%':>10} {'coverage%':>10} {'bitscore':>10}",
    ]
    for m in markers:
        if m.get("present"):
            L.append(
                f"{m['feature']:<12} {'YES':<6} {m['pident']:>10.1f} {m['qcov']:>10.1f} {m['bitscore']:>10.0f}"
            )
        else:
            L.append(f"{m['feature']:<12} {'no':<6} {'-':>10} {'-':>10} {'-':>10}")
    L += [
        "",
        f"Present ({len(present)}): {', '.join(m['feature'] for m in present) or 'none'}",
        f"Absent  ({len(absent)}): {', '.join(m['feature'] for m in absent) or 'none'}",
        "",
        thin,
        "2. Sequence type (MLST)",
        thin,
    ]
    if mlst.get("available") and mlst.get("ST"):
        alleles = " ".join(mlst.get("alleles") or [])
        L += [
            f"Seemann mlst:    ST {mlst['ST']}  (scheme {mlst.get('scheme', 'klebsiella')})",
            f"Alleles:         {alleles or '-'}",
        ]
    elif mlst.get("available") and mlst.get("error"):
        L.append(f"Seemann mlst:    error — {mlst['error'][:200]}")
    elif mlst.get("available"):
        L.append("Seemann mlst:    no ST called for this assembly")
    else:
        L.append(f"Seemann mlst:    not installed — {_fmt(mlst.get('note'))}")
    if kb.get("available") and kb.get("ST"):
        L.append(f"Kleborate ST:    {_fmt(kb.get('ST'))}")
    L += ["", thin, "3. Kleborate (species, virulence, acquired AMR)", thin]
    if not kb.get("available"):
        L.append(f"Kleborate:       not available — {_fmt(kb.get('error'))}")
    elif kb.get("error"):
        L.append(f"Kleborate:       failed — {kb['error'][:300]}")
    else:
        L += [
            f"Species:         {_fmt(kb.get('species'))}",
            f"ST:              {_fmt(kb.get('ST'))}",
            f"Virulence score: {_fmt(kb.get('virulence_score'))}",
            f"Resistance score: {_fmt(kb.get('resistance_score'))}",
            f"Aerobactin:      {_fmt(kb.get('Aerobactin'))}",
            f"Yersiniabactin:  {_fmt(kb.get('Yersiniabactin'))}",
            f"Salmochelin:     {_fmt(kb.get('Salmochelin'))}",
            f"RmpADC:          {_fmt(kb.get('RmpADC'))}",
            f"rmpA2:           {_fmt(kb.get('rmpA2'))}",
            f"K locus:         {_fmt(kb.get('K_locus'))}",
            f"wzi:             {_fmt(kb.get('wzi'))}",
            f"Carbapenemase:   {_fmt(kb.get('Bla_Carb_acquired'))}",
            f"ESBL:            {_fmt(kb.get('Bla_ESBL_acquired'))}",
            f"Full table:      {outdir / 'kleborate.tsv'}",
        ]
    L += ["", thin, "4. ABRicate (NCBI / CARD / VFDB / PlasmidFinder)", thin]
    if not ab.get("available"):
        L.append(f"ABRicate:        not available — {_fmt(ab.get('note'))}")
    else:
        for db in ("ncbi", "card", "vfdb", "plasmidfinder"):
            info = (ab.get("databases") or {}).get(db) or {}
            if "error" in info and not info.get("n"):
                L.append(f"{db:<16} error — {str(info['error'])[:120]}")
                continue
            genes = info.get("genes") or []
            preview = ", ".join(genes[:12])
            more = f" … (+{len(genes) - 12} more)" if len(genes) > 12 else ""
            L.append(f"{db:<16} {info.get('n', 0):>3} gene(s)  {preview}{more}")
        L.append(f"Tables:          {outdir}/abricate_*.tsv")
    L += ["", thin, "5. SHAP explanation of P(hvKp)", thin]
    L.append(f"Waterfall plot:  {outdir / 'shap_waterfall.png'}")
    if ranked:
        L.append("Top contributors (signed SHAP; + raises hvKp log-odds):")
        for row in ranked[:8]:
            sign = "+" if row["shap"] >= 0 else ""
            state = "present" if row["present"] else "absent"
            L.append(f"  {row['feature']:<12} {sign}{row['shap']:>7.3f}   ({state})")
    L += [
        "",
        thin,
        "6. Files written",
        thin,
        f"  report.xlsx         Excel: Consolidated 1/0 matrix + detail sheets",
        f"  report.txt          this report",
        f"  report.md           Markdown copy of this report",
        f"  summary.tsv         one-line summary",
        f"  prediction.json     full machine-readable record",
        f"  marker_calls.tsv    per-marker BLAST calls",
        f"  shap_waterfall.png  SHAP plot",
        f"  kleborate.tsv       Kleborate table (if run)",
        f"  mlst.txt            Seemann mlst (if run)",
        f"  abricate_*.tsv      ABRicate databases (if run)",
        f"  (batch) all_samples.xlsx + consolidated_presence.tsv at outdir root",
    ]
    if warnings:
        L += ["", "Warnings:"]
        for w in warnings:
            L.append(f"  ! {w}")
    L += [bar, ""]
    txt = "\n".join(L)
    txt_path = outdir / "report.txt"
    txt_path.write_text(txt)

    # ---- markdown ----
    md: list[str] = [
        f"# KlebX report: `{result['sample']}`",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Call** | **{result['call']}** |",
        f"| **P(hvKp)** | **{result['p_hvkp']:.4f}** |",
        f"| Markers present | {result['n_markers_present']} / {len(result['markers'])} |",
        f"| ST | {st} |",
        f"| FASTA | `{result['fasta']}` |",
        "",
        "## 1. Marker panel",
        "",
        "BLASTN identity ≥ 80% (iucD ≥ 75%), subject coverage ≥ 60%.",
        "",
        "| Gene | Present | Identity % | Coverage % | Bitscore |",
        "|---|:---:|---:|---:|---:|",
    ]
    for m in markers:
        if m.get("present"):
            md.append(
                f"| `{m['feature']}` | yes | {m['pident']:.1f} | {m['qcov']:.1f} | {m['bitscore']:.0f} |"
            )
        else:
            md.append(f"| `{m['feature']}` | no | — | — | — |")
    md += ["", "## 2. MLST", ""]
    if mlst.get("available") and mlst.get("ST"):
        md.append(f"- Seemann mlst: **ST {mlst['ST']}** (scheme `{mlst.get('scheme', 'klebsiella')}`)")
        if mlst.get("alleles"):
            md.append(f"- Alleles: `{' '.join(mlst['alleles'])}`")
    elif mlst.get("available"):
        md.append(f"- Seemann mlst: no ST (`{_fmt(mlst.get('error'), 'no call')}`)")
    else:
        md.append(f"- Seemann mlst: not installed")
    if kb.get("ST"):
        md.append(f"- Kleborate ST: `{_fmt(kb.get('ST'))}`")
    md += ["", "## 3. Kleborate", ""]
    if kb.get("available") and not kb.get("error"):
        md += [
            f"| Field | Value |",
            f"|---|---|",
            f"| Species | {_fmt(kb.get('species'))} |",
            f"| ST | {_fmt(kb.get('ST'))} |",
            f"| Virulence score | {_fmt(kb.get('virulence_score'))} |",
            f"| Resistance score | {_fmt(kb.get('resistance_score'))} |",
            f"| Aerobactin | {_fmt(kb.get('Aerobactin'))} |",
            f"| Yersiniabactin | {_fmt(kb.get('Yersiniabactin'))} |",
            f"| Salmochelin | {_fmt(kb.get('Salmochelin'))} |",
            f"| RmpADC | {_fmt(kb.get('RmpADC'))} |",
            f"| rmpA2 | {_fmt(kb.get('rmpA2'))} |",
            f"| K locus | {_fmt(kb.get('K_locus'))} |",
            f"| Carbapenemase | {_fmt(kb.get('Bla_Carb_acquired'))} |",
            f"| ESBL | {_fmt(kb.get('Bla_ESBL_acquired'))} |",
            "",
            f"Full table: [`kleborate.tsv`](kleborate.tsv)",
        ]
    else:
        md.append(f"Kleborate not available: {_fmt(kb.get('error') or kb.get('note'))}")
    md += ["", "## 4. ABRicate", ""]
    if ab.get("available"):
        md += ["| Database | n genes | Preview |", "|---|---:|---|"]
        for db in ("ncbi", "card", "vfdb", "plasmidfinder"):
            info = (ab.get("databases") or {}).get(db) or {}
            genes = info.get("genes") or []
            preview = ", ".join(f"`{g}`" for g in genes[:8])
            if len(genes) > 8:
                preview += f" … (+{len(genes) - 8})"
            md.append(f"| {db} | {info.get('n', 0)} | {preview or '—'} |")
    else:
        md.append(f"ABRicate not available: {_fmt(ab.get('note'))}")
    md += [
        "",
        "## 5. SHAP",
        "",
        f"![SHAP waterfall](shap_waterfall.png)",
        "",
    ]
    if ranked:
        md += ["| Gene | SHAP | State |", "|---|---:|---|"]
        for row in ranked[:8]:
            md.append(
                f"| `{row['feature']}` | {row['shap']:+.3f} | {'present' if row['present'] else 'absent'} |"
            )
    if warnings:
        md += ["## Warnings", ""]
        for w in warnings:
            md.append(f"- {w}")
        md.append("")
    md_path = outdir / "report.md"
    md_path.write_text("\n".join(md))
    return txt_path, md_path


MLST_LOCI = ["gapA", "infB", "mdh", "pgi", "phoE", "rpoB", "tonB"]
KLEBORATE_VIR_FIELDS = ["Aerobactin", "Yersiniabactin", "Salmochelin", "Colibactin", "RmpADC", "rmpA2"]
KLEBORATE_AMR_FIELDS = [
    "AGly_acquired",
    "Col_acquired",
    "Fcyn_acquired",
    "Flq_acquired",
    "Gly_acquired",
    "MLS_acquired",
    "Phe_acquired",
    "Rif_acquired",
    "Sul_acquired",
    "Tet_acquired",
    "Tgc_acquired",
    "Tmt_acquired",
    "Bla_acquired",
    "Bla_inhR_acquired",
    "Bla_ESBL_acquired",
    "Bla_ESBL_inhR_acquired",
    "Bla_Carb_acquired",
]
ABRICATE_DBS = ("ncbi", "card", "vfdb", "plasmidfinder")


def _box_border():
    from openpyxl.styles import Border, Side

    side = Side(style="medium", color="000000")
    return Border(left=side, right=side, top=side, bottom=side)


def _thin_border():
    from openpyxl.styles import Border, Side

    side = Side(style="thin", color="000000")
    return Border(left=side, right=side, top=side, bottom=side)


def _style_header(ws, row=1):
    """Plain bold black headers in a clear black box — no fill colours."""
    from openpyxl.styles import Alignment, Font, PatternFill

    font = Font(bold=True, color="000000", name="Calibri", size=11)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    no_fill = PatternFill(fill_type=None)
    box = _box_border()
    for cell in ws[row]:
        if cell.value is None and cell.column > ws.max_column:
            continue
        cell.fill = no_fill
        cell.font = font
        cell.alignment = center
        cell.border = box


def _autosize(ws, max_width=48):
    for col in ws.columns:
        letter = col[0].column_letter
        width = 10
        for cell in col:
            if cell.value is None:
                continue
            width = max(width, min(max_width, len(str(cell.value)) + 2))
        ws.column_dimensions[letter].width = width


def _write_kv_sheet(ws, rows: list[tuple[str, object]]):
    ws.append(["Field", "Value"])
    for k, v in rows:
        ws.append([k, "" if v is None else v])
    _style_header(ws)
    _autosize(ws)


def _clean_gene_token(token: str) -> str | None:
    """Normalize Kleborate/ABRicate gene tokens for column names."""
    if token is None:
        return None
    s = str(token).strip()
    if not s or s in {"-", "NA", "na", "None"}:
        return None
    # drop ICE / descriptive clauses after ';' already split; clean suffixes
    s = s.split()[0] if s.startswith(("ybt", "iuc", "iro", "clb")) and " " in s else s
    s = s.rstrip("^?*")
    s = re.sub(r"\.v\d+\*?$", "", s)
    s = s.rstrip("^?*")
    return s or None


def _split_kleborate_genes(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s in {"-", "NA", "na"}:
        return []
    out = []
    for part in re.split(r"[;]", s):
        g = _clean_gene_token(part.strip())
        if g:
            out.append(g)
    return out


def _kleborate_vir_present(value) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    s = str(value).strip()
    return 0 if (not s or s in {"-", "NA", "na", "None"}) else 1


def _parse_mlst_loci(mlst: dict) -> dict[str, int]:
    present = {loc: 0 for loc in MLST_LOCI}
    alleles = mlst.get("alleles") or []
    for a in alleles:
        m = re.match(r"^([A-Za-z0-9_]+)\(([^)]*)\)$", str(a).strip())
        if not m:
            continue
        loc, allele = m.group(1), m.group(2).strip()
        if loc in present and allele not in {"", "-", "?"}:
            present[loc] = 1
    # also accept raw line if alleles empty
    if not any(present.values()) and mlst.get("raw"):
        for loc in MLST_LOCI:
            m = re.search(rf"\b{re.escape(loc)}\(([^)]*)\)", str(mlst["raw"]))
            if m and m.group(1).strip() not in {"", "-", "?"}:
                present[loc] = 1
    return present


def presence_profile(result: dict) -> dict[str, dict[str, int]]:
    """
    Category -> {gene: 0/1} for one isolate.
    Categories: Markers, MLST, Kleborate_virulence, Kleborate_AMR, ABRicate_*.
    """
    profile: dict[str, dict[str, int]] = {
        "Markers": {},
        "MLST": {loc: 0 for loc in MLST_LOCI},
        "Kleborate_virulence": {f: 0 for f in KLEBORATE_VIR_FIELDS},
        "Kleborate_AMR": {},
        "ABRicate_ncbi": {},
        "ABRicate_card": {},
        "ABRicate_vfdb": {},
        "ABRicate_plasmidfinder": {},
    }

    for m in result.get("markers") or []:
        profile["Markers"][str(m["feature"])] = 1 if m.get("present") else 0

    mlst = result.get("mlst") or {}
    profile["MLST"].update(_parse_mlst_loci(mlst))

    kb = result.get("kleborate") or {}
    if kb.get("available") and not kb.get("error"):
        for f in KLEBORATE_VIR_FIELDS:
            profile["Kleborate_virulence"][f] = _kleborate_vir_present(kb.get(f))
        for f in KLEBORATE_AMR_FIELDS:
            for g in _split_kleborate_genes(kb.get(f)):
                profile["Kleborate_AMR"][g] = 1

    ab = result.get("abricate") or {}
    if ab.get("available"):
        for db in ABRICATE_DBS:
            info = (ab.get("databases") or {}).get(db) or {}
            for g in info.get("genes") or []:
                g2 = _clean_gene_token(g) or str(g)
                profile[f"ABRicate_{db}"][g2] = 1
    return profile


def _category_column_plan(profiles: list[dict[str, dict[str, int]]]) -> list[tuple[str, list[str]]]:
    """Ordered (category, gene_list) with stable marker/MLST order and union of batch genes."""
    meta = json.loads((DATA / "model_meta.json").read_text())
    marker_genes = list(meta["features"])

    def union(cat: str, preferred: list[str] | None = None) -> list[str]:
        seen: dict[str, None] = {}
        if preferred:
            for g in preferred:
                seen[g] = None
        for p in profiles:
            for g in sorted(p.get(cat, {}).keys()):
                seen[g] = None
        return list(seen.keys())

    plan = [
        ("Markers", union("Markers", marker_genes)),
        ("MLST", union("MLST", MLST_LOCI)),
        ("Kleborate_virulence", union("Kleborate_virulence", KLEBORATE_VIR_FIELDS)),
        ("Kleborate_AMR", union("Kleborate_AMR")),
    ]
    for db in ABRICATE_DBS:
        plan.append((f"ABRicate_{db}", union(f"ABRicate_{db}")))
    # drop empty categories except Markers/MLST/virulence
    keep = []
    for cat, genes in plan:
        if genes or cat in {"Markers", "MLST", "Kleborate_virulence"}:
            keep.append((cat, genes))
    return keep


def write_consolidated_sheet(wb, results: list[dict], sheet_name: str = "Consolidated"):
    """
    Two-row header (no colours, no freeze):
      row1 = bold category spans (Markers | MLST | Kleborate_... | ABRicate_...)
      row2 = bold gene sub-columns
    Body: one row per isolate, values 1/0, thin black box borders.
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    profiles = [presence_profile(r) for r in results]
    plan = _category_column_plan(profiles)

    ws = wb.create_sheet(sheet_name, 0)
    ws.cell(1, 1, "Sample")
    ws.cell(2, 1, "isolate")
    ws.cell(1, 2, "Sample")
    ws.cell(2, 2, "call")
    ws.cell(1, 3, "Sample")
    ws.cell(2, 3, "P_hvKp")
    ws.cell(1, 4, "Sample")
    ws.cell(2, 4, "ST")

    col = 5
    spans: list[tuple[str, int, int]] = []
    gene_cols: list[tuple[str, str]] = []
    for cat, genes in plan:
        if not genes:
            continue
        start = col
        for g in genes:
            ws.cell(2, col, g)
            gene_cols.append((cat, g))
            col += 1
        end = col - 1
        for c in range(start, end + 1):
            ws.cell(1, c, cat)
        spans.append((cat, start, end))

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)
    ws.cell(1, 1, "Sample")
    for cat, start, end in spans:
        if end > start:
            ws.merge_cells(start_row=1, start_column=start, end_row=1, end_column=end)

    bold = Font(bold=True, color="000000", name="Calibri", size=11)
    bold_cat = Font(bold=True, color="000000", name="Calibri", size=12)
    body = Font(bold=False, color="000000", name="Calibri", size=10)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    no_fill = PatternFill(fill_type=None)
    box = _box_border()
    thin = _thin_border()
    max_col = 4 + len(gene_cols)

    # header rows: bold text + medium black box, no colour fill
    for r in (1, 2):
        for c in range(1, max_col + 1):
            cell = ws.cell(r, c)
            cell.fill = no_fill
            cell.font = bold_cat if r == 1 else bold
            cell.alignment = center
            cell.border = box

    # data rows
    for result, prof in zip(results, profiles):
        row_vals = [
            Path(str(result.get("sample", ""))).stem,
            result.get("call"),
            round(float(result.get("p_hvkp", 0)), 6),
            _st_display(result),
        ]
        for cat, gene in gene_cols:
            row_vals.append(int(prof.get(cat, {}).get(gene, 0)))
        ws.append(row_vals)

    for r in range(3, ws.max_row + 1):
        for c in range(1, max_col + 1):
            cell = ws.cell(r, c)
            cell.fill = no_fill
            cell.font = body
            cell.alignment = center
            cell.border = thin

    # no freeze panes — plain scroll
    ws.freeze_panes = None
    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 32
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 10
    for i in range(5, max_col + 1):
        ws.column_dimensions[get_column_letter(i)].width = max(
            8, min(18, len(str(ws.cell(2, i).value or "")) + 2)
        )
    return ws


def write_excel_report(result: dict, outdir: Path) -> Path:
    """Per-isolate workbook: Consolidated (1/0) first, then detail sheets. No Summary/Notes."""
    from openpyxl import Workbook

    markers = result.get("markers") or []
    shap = result.get("shap") or {}
    ranked = shap.get("ranked") or []
    kb = result.get("kleborate") or {}
    mlst = result.get("mlst") or {}
    ab = result.get("abricate") or {}

    wb = Workbook()
    # remove default sheet; Consolidated will be inserted at 0
    default = wb.active
    wb.remove(default)
    write_consolidated_sheet(wb, [result], sheet_name="Consolidated")

    # ---- Markers (detail) ----
    ws = wb.create_sheet("Markers")
    ws.append(["Gene", "Present_1_0", "Identity_%", "Coverage_%", "Evalue", "Bitscore"])
    for m in markers:
        ws.append(
            [
                m.get("feature"),
                1 if m.get("present") else 0,
                None if not m.get("present") else round(float(m["pident"]), 3),
                None if not m.get("present") else round(float(m["qcov"]), 3),
                None if not m.get("present") else m.get("evalue"),
                None if not m.get("present") else round(float(m["bitscore"]), 1),
            ]
        )
    _style_header(ws)
    _autosize(ws)

    # ---- SHAP ----
    ws = wb.create_sheet("SHAP")
    ws.append(["Rank", "Gene", "Present_1_0", "SHAP_value", "Direction"])
    for i, row in enumerate(ranked, 1):
        val = float(row["shap"])
        ws.append(
            [
                i,
                row["feature"],
                1 if row["present"] else 0,
                round(val, 6),
                "raises hvKp" if val >= 0 else "lowers hvKp",
            ]
        )
    ws.append([])
    ws.append(["base_value", shap.get("base_value")])
    ws.append(["prediction_margin", shap.get("prediction_margin")])
    _style_header(ws)
    _autosize(ws)

    # ---- MLST ----
    ws = wb.create_sheet("MLST")
    allele_str = " ".join(mlst.get("alleles") or []) if mlst.get("alleles") else ""
    loci = _parse_mlst_loci(mlst)
    rows = [
        ("Seemann mlst available", mlst.get("available")),
        ("Scheme", mlst.get("scheme")),
        ("ST", mlst.get("ST")),
        ("Alleles", allele_str or None),
        ("Kleborate ST", kb.get("ST")),
    ]
    for loc in MLST_LOCI:
        rows.append((f"MLST_{loc}_present", loci.get(loc, 0)))
    _write_kv_sheet(ws, rows)

    # ---- Kleborate ----
    ws = wb.create_sheet("Kleborate")
    if kb.get("available") and not kb.get("error"):
        skip = {"available", "path", "binary", "error", "cmd"}
        key_order = [
            "species",
            "ST",
            "virulence_score",
            "resistance_score",
            *KLEBORATE_VIR_FIELDS,
            "K_locus",
            "wzi",
            "Bla_Carb_acquired",
            "Bla_ESBL_acquired",
            "Bla_acquired",
            "num_resistance_genes",
            "num_resistance_classes",
        ]
        rows = [(k, kb.get(k)) for k in key_order]
        rows.append(("--- full Kleborate row below ---", ""))
        for k, v in kb.items():
            if k in skip or k in key_order:
                continue
            rows.append((k, v))
        _write_kv_sheet(ws, rows)
    else:
        _write_kv_sheet(
            ws,
            [("available", kb.get("available")), ("error", kb.get("error")), ("note", kb.get("note"))],
        )

    # ---- ABRicate summary ----
    ws = wb.create_sheet("ABRicate_summary")
    ws.append(["Database", "n_genes", "Genes"])
    if ab.get("available"):
        for db in ABRICATE_DBS:
            info = (ab.get("databases") or {}).get(db) or {}
            genes = info.get("genes") or []
            ws.append([db, info.get("n", 0), ", ".join(genes)])
    else:
        ws.append(["abricate", 0, _fmt(ab.get("note"))])
    _style_header(ws)
    _autosize(ws, max_width=80)

    for db in ABRICATE_DBS:
        tsv = outdir / f"abricate_{db}.tsv"
        ws = wb.create_sheet(f"ABRicate_{db}"[:31])
        if tsv.exists() and tsv.stat().st_size > 0:
            lines = [ln for ln in tsv.read_text().splitlines() if ln.strip()]
            if not lines:
                ws.append(["No hits"])
            else:
                hdr = lines[0].lstrip("#").split("\t")
                ws.append(hdr)
                start = 1 if lines[0].startswith("#") or "FILE" in hdr or "GENE" in hdr else 0
                for ln in lines[start:]:
                    if ln.startswith("#"):
                        continue
                    ws.append(ln.split("\t"))
                _style_header(ws)
                _autosize(ws, max_width=40)
        else:
            info = (ab.get("databases") or {}).get(db) or {}
            ws.append(["GENE", "Present_1_0"])
            for g in info.get("genes") or []:
                ws.append([g, 1])
            if not info.get("genes"):
                ws.append([info.get("error") or "No hits / not run", 0])
            _style_header(ws)

    out = outdir / "report.xlsx"
    wb.save(out)
    return out


def write_batch_xlsx(results: list[dict], root: Path) -> Path | None:
    """Batch workbook: Consolidated 1/0 matrix (all isolates) + Calls table."""
    if not results:
        return None
    from openpyxl import Workbook

    wb = Workbook()
    default = wb.active
    wb.remove(default)
    write_consolidated_sheet(wb, results, sheet_name="Consolidated")

    ws = wb.create_sheet("Calls", 1)
    ws.append(
        [
            "isolate",
            "sample_file",
            "call",
            "P_hvKp",
            "n_markers_present",
            "ST",
            "species",
            "kleborate_virulence_score",
            "aerobactin",
            "K_locus",
            "report_xlsx",
        ]
    )
    for r in results:
        kb = r.get("kleborate") or {}
        ws.append(
            [
                Path(str(r.get("sample", ""))).stem,
                r.get("sample"),
                r.get("call"),
                round(float(r.get("p_hvkp", 0)), 6),
                r.get("n_markers_present"),
                _st_display(r),
                kb.get("species"),
                kb.get("virulence_score"),
                kb.get("Aerobactin"),
                kb.get("K_locus"),
                r.get("report_xlsx"),
            ]
        )
    _style_header(ws)
    _autosize(ws, max_width=60)

    out = root / "all_samples.xlsx"
    wb.save(out)
    # also a TSV of the consolidated matrix for scripting
    _write_consolidated_tsv(results, root / "consolidated_presence.tsv")
    return out


def _write_consolidated_tsv(results: list[dict], path: Path) -> None:
    profiles = [presence_profile(r) for r in results]
    plan = _category_column_plan(profiles)
    gene_cols: list[tuple[str, str]] = []
    for cat, genes in plan:
        for g in genes:
            gene_cols.append((cat, g))
    headers = ["isolate", "call", "P_hvKp", "ST"] + [f"{c}|{g}" for c, g in gene_cols]
    lines = ["\t".join(headers)]
    for result, prof in zip(results, profiles):
        row = [
            Path(str(result.get("sample", ""))).stem,
            str(result.get("call")),
            f"{float(result.get('p_hvkp', 0)):.6f}",
            _st_display(result),
        ]
        for cat, gene in gene_cols:
            row.append(str(int(prof.get(cat, {}).get(gene, 0))))
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


def write_all_samples_xlsx(rows: list[dict], root: Path) -> Path | None:
    """Backward-compatible name; prefer write_batch_xlsx(results). """
    if not rows:
        return None
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Calls"
    headers = list(rows[0].keys())
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    _style_header(ws)
    _autosize(ws, max_width=60)
    out = root / "all_samples_calls.xlsx"
    wb.save(out)
    return out


def collect_fastas(fasta_args: list[str], directory: str | None, list_file: str | None) -> list[Path]:
    """Resolve -f / -d / --list into a de-duplicated FASTA path list (batch-ready)."""
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path):
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    for fa in fasta_args or []:
        add(Path(fa))
    if directory:
        d = Path(directory)
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        for pat in ("*.fasta", "*.fa", "*.fna", "*.fas", "*.FASTA", "*.FA", "*.FNA"):
            for p in sorted(d.glob(pat)):
                add(p)
    if list_file:
        lf = Path(list_file)
        if not lf.exists():
            raise SystemExit(f"List file not found: {lf}")
        for line in lf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            add(Path(line))
    return out


def predict_one(fasta: Path, outdir: Path, threads: int = 4) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    warnings = validate_fasta(fasta)
    meta = json.loads((DATA / "model_meta.json").read_text())
    features = meta["features"]
    model = joblib.load(DATA / "hv_xgb.joblib")

    hits = blast_markers(fasta, outdir, threads=threads)
    x, det = presence_vector(hits, features)
    det.to_csv(outdir / "marker_calls.tsv", sep="\t", index=False)

    prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    call = classify(prob)
    shap_stats = shap_waterfall(
        model,
        x,
        features,
        outdir / "shap_waterfall.png",
        title=f"{fasta.name}  P(hvKp)={prob:.3f}  {call}",
    )

    result = {
        "sample": fasta.name,
        "fasta": str(fasta.resolve()),
        "p_hvkp": prob,
        "call": call,
        "threshold": meta["threshold"],
        "n_markers_present": int(x.sum()),
        "markers": det.to_dict(orient="records"),
        "shap": shap_stats,
        "kleborate": run_kleborate(fasta, outdir),
        "mlst": run_mlst(fasta, outdir),
        "abricate": run_abricate(fasta, outdir),
        "warnings": warnings,
        "rules": {
            "blast_min_pident": MIN_PIDENT,
            "blast_min_qcov": MIN_QCOV,
            "call_thresholds": {"high": 0.8, "moderate": 0.5, "equivocal": 0.2},
        },
    }
    kb = result["kleborate"]
    if isinstance(kb, dict) and kb.get("ST") and not result["mlst"].get("ST"):
        result["mlst"]["ST_from_kleborate"] = kb.get("ST")
        result["mlst"]["species_from_kleborate"] = kb.get("species")

    (outdir / "prediction.json").write_text(json.dumps(result, indent=2, default=str))

    present = [r["feature"] for r in result["markers"] if r["present"]]
    st = _st_display(result)
    vir = kb.get("virulence_score") if isinstance(kb, dict) else None
    iuc = kb.get("Aerobactin") if isinstance(kb, dict) else None
    kl = kb.get("K_locus") if isinstance(kb, dict) else None
    (outdir / "summary.tsv").write_text(
        "sample\tp_hvkp\tcall\tmarkers_present\tn_markers\tST\tkleborate_virulence_score\taerobactin\tK_locus\n"
        f"{fasta.name}\t{prob:.4f}\t{call}\t{','.join(present) or 'none'}\t{int(x.sum())}\t{st}\t{_fmt(vir, '')}\t{_fmt(iuc, '')}\t{_fmt(kl, '')}\n"
    )

    report_txt, report_md = write_report(result, outdir)
    report_xlsx = write_excel_report(result, outdir)
    result["report_txt"] = str(report_txt)
    result["report_md"] = str(report_md)
    result["report_xlsx"] = str(report_xlsx)
    return result


def _predict_job(payload: tuple[str, str, int]) -> dict:
    fasta, outdir, threads = payload
    try:
        return predict_one(Path(fasta), Path(outdir), threads=threads)
    except Exception as e:
        return {"_error": str(e), "sample": Path(fasta).name, "fasta": str(fasta)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Predict hvKp from genome FASTA(s) (BLAST markers + XGBoost + SHAP; "
            "Kleborate / Seemann mlst / ABRicate if installed). "
            "Writes report.xlsx with a Consolidated 1/0 gene matrix, plus "
            "KlebX.html for the FASTA files in this run."
        )
    )
    ap.add_argument("-f", "--fasta", nargs="+", default=[], help="One or more genome FASTA files")
    ap.add_argument("-d", "--dir", default=None, help="Batch: directory of *.fasta / *.fa / *.fna genomes")
    ap.add_argument("--list", dest="list_file", default=None, help="Batch: text file with one FASTA path per line")
    ap.add_argument("-o", "--outdir", default="hvkp_out", help="Output directory (default: hvkp_out)")
    ap.add_argument(
        "--tools",
        action="store_true",
        help="Check tools and install any that are missing. With no FASTA, exit after the check",
    )
    ap.add_argument(
        "--skip-setup",
        action="store_true",
        help="Check tools but do not download them",
    )
    ap.add_argument("--quiet", action="store_true", help="Do not print the full report to stdout")
    ap.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="BLASTN threads per genome (default: 4)",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Genomes to process at once (default: 1). Each job runs BLAST, Kleborate, mlst and ABRicate.",
    )
    args = ap.parse_args(argv)

    tools = ensure_tools(install=not args.skip_setup)
    if args.tools and not (args.fasta or args.dir or args.list_file):
        return 0 if all(tools.values()) else 1

    fastas = collect_fastas(args.fasta, args.dir, args.list_file)
    if not fastas:
        raise SystemExit("provide -f FASTA ..., -d DIR, --list FILE, or --tools")
    if not tools["blastn"]:
        raise SystemExit("blastn is required. Run KlebX --tools to install it.")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    threads = max(1, int(args.threads))
    jobs = max(1, int(args.jobs))
    print(f"[KlebX] batch size: {len(fastas)} genome(s) → {root}  threads={threads}  jobs={jobs}", flush=True)

    results: list[dict] = []
    failed: list[dict] = []
    call_rows: list[dict] = []
    pending: list[tuple[int, Path, Path]] = []
    ordered: dict[int, dict] = {}
    for i, fasta in enumerate(fastas):
        if not fasta.exists():
            print(f"[KlebX] missing: {fasta}", file=sys.stderr)
            failed.append({"sample": fasta.name, "fasta": str(fasta), "error": "file missing"})
            continue
        sub = root / fasta.stem
        print(f"[KlebX] queued {fasta} → {sub}", flush=True)
        pending.append((i, fasta, sub))

    def _take(idx: int, rec: dict) -> None:
        if rec.get("_error"):
            print(f"[KlebX] ERROR {rec.get('sample')}: {rec['_error']}", file=sys.stderr, flush=True)
            failed.append({"sample": rec.get("sample", ""), "fasta": rec.get("fasta", ""), "error": rec["_error"]})
            return
        ordered[idx] = rec
        print(
            f"       P(hvKp)={rec['p_hvkp']:.3f}  {rec['call']}  "
            f"markers={rec['n_markers_present']}/{len(rec['markers'])}  "
            f"ST={_st_display(rec)}  {rec['sample']}",
            flush=True,
        )
        print(f"       report: {rec['report_txt']}", flush=True)
        print(f"       excel:  {rec['report_xlsx']}", flush=True)
        if jobs == 1 and not args.quiet:
            print(flush=True)
            print(Path(rec["report_txt"]).read_text(), flush=True)

    if jobs == 1:
        for idx, fasta, sub in pending:
            _take(idx, _predict_job((str(fasta), str(sub), threads)))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {
                pool.submit(_predict_job, (str(fasta), str(sub), threads)): idx
                for idx, fasta, sub in pending
            }
            for fut in as_completed(futs):
                _take(futs[fut], fut.result())

    for idx in sorted(ordered):
        rec = ordered[idx]
        kb = rec["kleborate"] if isinstance(rec["kleborate"], dict) else {}
        results.append(rec)
        call_rows.append(
            {
                "sample": rec["sample"],
                "p_hvkp": rec["p_hvkp"],
                "call": rec["call"],
                "n_markers_present": rec["n_markers_present"],
                "ST": _st_display(rec),
                "kleborate_virulence_score": kb.get("virulence_score"),
                "aerobactin": kb.get("Aerobactin"),
                "K_locus": kb.get("K_locus"),
                "species": kb.get("species"),
                "report_txt": rec["report_txt"],
                "report_xlsx": rec["report_xlsx"],
                "shap_png": str(Path(rec["report_txt"]).parent / "shap_waterfall.png"),
            }
        )
    if results:
        pd.DataFrame(call_rows).to_csv(root / "all_samples.tsv", sep="\t", index=False)
        xlsx = write_batch_xlsx(results, root)
        print(f"[KlebX] wrote {root / 'all_samples.tsv'}  ({len(results)} sample(s))", flush=True)
        if xlsx:
            print(f"[KlebX] wrote {xlsx}  (Consolidated 1/0 matrix)", flush=True)
            print(f"[KlebX] wrote {root / 'consolidated_presence.tsv'}", flush=True)
    if results or failed:
        from KlebX.surveillance_report import write_surveillance_report

        tool_bits = ", ".join(f"{n}={p or 'not found'}" for n, p in tools.items())
        html_path, xlsx_surv = write_surveillance_report(
            results,
            root,
            failed=failed,
            run_info={
                "input": ", ".join(str(f) for f in fastas),
                "outdir": str(root),
                "tools": tool_bits,
                "status": "completed" if not failed else "completed with failures",
            },
        )
        print(f"[KlebX] surveillance HTML: {html_path}", flush=True)
        print(f"[KlebX] surveillance Excel: {xlsx_surv}", flush=True)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
