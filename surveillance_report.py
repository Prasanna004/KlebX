"""Interactive surveillance HTML written at the end of a FASTA run.

Columns are only those this tool actually writes: predictor call, Kleborate
(virulence score, K locus, assembly size, acquired AMR), Seemann mlst, and
ABRicate NCBI / PlasmidFinder. Screens that are not run are omitted.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from KlebX import __version__
from KlebX.predict import (
    KLEBORATE_AMR_FIELDS,
    KLEBORATE_VIR_FIELDS,
    _split_kleborate_genes,
    _st_display,
)

NA = "Not available"
NOT_ANALYZED = "Not analyzed"
FAILED = "Analysis failed"
NOT_DETECTED = "Not detected"

VIR_MARKERS = [
    ("iuc", "Aerobactin"),
    ("iro", "Salmochelin"),
    ("ybt", "Yersiniabactin"),
    ("clb", "Colibactin"),
    ("rmpA", "RmpADC"),
    ("rmpA2", "rmpA2"),
]


def _empty(v) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return True
    s = str(v).strip()
    return s.lower() in {"", "nan", "na", "none", "-", "null"}


def _tool_state(block: dict | None) -> str:
    if not isinstance(block, dict) or not block.get("available"):
        return "not_analyzed"
    if block.get("error"):
        return "failed"
    return "ok"


def _field(block: dict | None, key: str, state: str):
    if state == "not_analyzed":
        return NOT_ANALYZED
    if state == "failed":
        return FAILED
    if not isinstance(block, dict) or key not in block:
        return NA
    v = block.get(key)
    if _empty(v):
        return NOT_DETECTED
    return v


def _present_label(block: dict | None, key: str, state: str) -> str:
    if state != "ok":
        return NOT_ANALYZED if state == "not_analyzed" else FAILED
    if not isinstance(block, dict) or key not in block:
        return NA
    v = block.get(key)
    if _empty(v):
        return NOT_DETECTED
    return "Detected"


def _read_abricate_rows(path: str | None) -> tuple[str, list[dict]]:
    """Return (state, rows). state is ok / not_analyzed / failed."""
    if not path:
        return "not_analyzed", []
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return "failed", []
    try:
        df = pd.read_csv(p, sep="\t")
    except Exception:
        return "failed", []
    if df.empty:
        return "ok", []
    df.columns = [str(c).lstrip("#") for c in df.columns]
    rows = []
    for rec in df.to_dict(orient="records"):
        gene = rec.get("GENE")
        if _empty(gene):
            continue
        rows.append(
            {
                "gene": str(gene),
                "identity": rec.get("%IDENTITY", NA),
                "coverage": rec.get("%COVERAGE", NA),
                "product": rec.get("PRODUCT") if not _empty(rec.get("PRODUCT")) else NA,
                "resistance": rec.get("RESISTANCE") if not _empty(rec.get("RESISTANCE")) else NA,
                "accession": rec.get("ACCESSION") if not _empty(rec.get("ACCESSION")) else NA,
            }
        )
    return "ok", rows


def genome_from_result(result: dict) -> dict:
    kb = result.get("kleborate") if isinstance(result.get("kleborate"), dict) else {}
    mlst = result.get("mlst") if isinstance(result.get("mlst"), dict) else {}
    ab = result.get("abricate") if isinstance(result.get("abricate"), dict) else {}
    kb_state = _tool_state(kb)
    mlst_state = _tool_state(mlst)
    ab_state = _tool_state(ab)

    markers = {m.get("feature"): m for m in (result.get("markers") or [])}
    marker_bits = {}
    for name in ("iucD", "AlpA", "aidA", "iutA_2", "dhbE", "iucC_2"):
        m = markers.get(name)
        marker_bits[name] = (1 if m and m.get("present") else 0) if m is not None else NA

    vir = {}
    for short, key in VIR_MARKERS:
        vir[short] = _present_label(kb, key, kb_state)
    amr_rows = []
    ncbi_state, ncbi_rows = ("not_analyzed", [])
    plasmid_rows = []
    pf_state = "not_analyzed"
    if ab_state == "ok":
        dbs = ab.get("databases") or {}
        for db in ("ncbi", "card", "vfdb", "plasmidfinder"):
            info = dbs.get(db) or {}
            if info.get("path"):
                state, rows = _read_abricate_rows(info.get("path"))
            elif info.get("genes") is not None and not info.get("error"):
                state = "ok"
                rows = [
                    {"gene": str(g), "identity": NA, "coverage": NA, "product": NA, "resistance": NA, "accession": NA}
                    for g in info.get("genes") or []
                ]
            elif info.get("error"):
                state, rows = "failed", []
            else:
                state, rows = "not_analyzed", []
            if db == "ncbi":
                ncbi_state = state
                ncbi_rows = rows
            if db == "plasmidfinder":
                pf_state = state
                for row in rows:
                    plasmid_rows.append(
                        {
                            "Genome_ID": result.get("sample"),
                            "Replicon": row["gene"],
                            "Identity": row["identity"],
                            "Coverage": row["coverage"],
                        }
                    )
            if db == "ncbi":
                for row in rows:
                    amr_rows.append(
                        {
                            "Genome_ID": result.get("sample"),
                            "Gene": row["gene"],
                            "Database": "ncbi",
                            "Identity": row["identity"],
                            "Coverage": row["coverage"],
                            "Description": row["product"],
                            "AMR_Class": row["resistance"],
                        }
                    )
    elif ab_state == "failed":
        ncbi_state = "failed"
        pf_state = "failed"

    kleb_genes = []
    if kb_state == "ok":
        for f in KLEBORATE_AMR_FIELDS:
            kleb_genes.extend(_split_kleborate_genes(kb.get(f)))
        # stable unique
        seen = []
        for g in kleb_genes:
            if g not in seen:
                seen.append(g)
        kleb_genes = seen

    if ncbi_state == "ok":
        amr_genes = [r["Gene"] for r in amr_rows]
        amr_count = len(amr_genes)
        amr_classes = sorted({r["AMR_Class"] for r in amr_rows if r["AMR_Class"] not in {NA, NOT_DETECTED}})
        amr_source = "ABRicate ncbi"
    elif kb_state == "ok":
        amr_genes = kleb_genes
        amr_count = len(kleb_genes)
        amr_classes = []
        amr_source = "Kleborate acquired"
        for g in kleb_genes:
            amr_rows.append(
                {
                    "Genome_ID": result.get("sample"),
                    "Gene": g,
                    "Database": "Kleborate",
                    "Identity": NA,
                    "Coverage": NA,
                    "Description": NA,
                    "AMR_Class": NA,
                }
            )
    elif ncbi_state == "failed" or kb_state == "failed":
        amr_genes, amr_count, amr_classes, amr_source = [], FAILED, [], FAILED
    else:
        amr_genes, amr_count, amr_classes, amr_source = [], NOT_ANALYZED, [], NOT_ANALYZED

    if kb_state == "ok":
        carb = _field(kb, "Bla_Carb_acquired", kb_state)
        col = _field(kb, "Col_acquired", kb_state)
    else:
        carb = NOT_ANALYZED if kb_state == "not_analyzed" else FAILED
        col = carb

    st = _st_display(result)
    if st in {"NA", "None"} and mlst_state != "ok" and kb_state != "ok":
        st = NOT_ANALYZED if mlst_state == "not_analyzed" and kb_state == "not_analyzed" else (FAILED if "failed" in {mlst_state, kb_state} else NA)

    replicons = [p["Replicon"] for p in plasmid_rows]
    if pf_state == "ok":
        replicon_count = len(replicons)
        replicon_text = "; ".join(replicons) if replicons else NOT_DETECTED
    elif pf_state == "failed":
        replicon_count = FAILED
        replicon_text = FAILED
    else:
        replicon_count = NOT_ANALYZED
        replicon_text = NOT_ANALYZED

    sample = Path(str(result.get("sample") or "genome")).name
    stem = Path(sample).stem
    return {
        "Genome_ID": stem,
        "FASTA": result.get("fasta") or NA,
        "Predictor_call": result.get("call") or NA,
        "P_hvKp": result.get("p_hvkp") if result.get("p_hvkp") is not None else NA,
        "Virulence_class": result.get("call") or NA,
        "Virulence_score": _field(kb, "virulence_score", kb_state),
        "ST": st,
        "K_locus": _field(kb, "K_locus", kb_state),
        "species": _field(kb, "species", kb_state),
        "AMR_gene_count": amr_count,
        "AMR_genes": "; ".join(amr_genes) if isinstance(amr_count, int) and amr_genes else (NOT_DETECTED if amr_count == 0 else amr_count),
        "AMR_classes": "; ".join(amr_classes) if amr_classes else (NA if isinstance(amr_count, int) else amr_count),
        "AMR_source": amr_source,
        "Carbapenemase_genes": carb,
        "Colistin_resistance_genes": col,
        "Kleborate_resistance_score": _field(kb, "resistance_score", kb_state),
        "Replicon_hit_count": replicon_count,
        "Plasmid_replicons": replicon_text,
        "Genome_length": _field(kb, "total_size", kb_state),
        "Contigs": _field(kb, "contig_count", kb_state),
        "N50": _field(kb, "N50", kb_state),
        "QC_status": _qc_status(kb, kb_state),
        "virulence_markers": vir,
        "predictor_markers": marker_bits,
        "amr_rows": amr_rows,
        "plasmid_rows": plasmid_rows,
        "kleborate_genes": kleb_genes,
        "status": "processed",
        "error": "",
    }


def _qc_status(kb: dict, state: str) -> str:
    if state == "not_analyzed":
        return NOT_ANALYZED
    if state == "failed":
        return FAILED
    if "QC_warnings" not in kb:
        return NA
    if _empty(kb.get("QC_warnings")):
        return "OK"
    return str(kb.get("QC_warnings"))


def failed_genome(sample: str, fasta: str, error: str) -> dict:
    g = genome_from_result({"sample": sample, "fasta": fasta, "call": FAILED, "kleborate": {}, "mlst": {}, "abricate": {}})
    g["status"] = "failed"
    g["error"] = error
    g["Predictor_call"] = FAILED
    g["Virulence_class"] = FAILED
    g["QC_status"] = FAILED
    return g


def _num(v):
    try:
        if isinstance(v, str) and v in {NA, NOT_ANALYZED, FAILED, NOT_DETECTED}:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _counts(values: list) -> list[dict]:
    c = Counter(str(v) for v in values if str(v) not in {NA, NOT_ANALYZED, FAILED, NOT_DETECTED, ""})
    return [{"label": k, "n": n} for k, n in c.most_common(20)]


def summarise(genomes: list[dict]) -> dict:
    processed = [g for g in genomes if g.get("status") != "failed"]
    failed = [g for g in genomes if g.get("status") == "failed"]
    calls = Counter(g["Predictor_call"] for g in processed)
    hvkp = [g for g in processed if str(g["Predictor_call"]).startswith("hvKp-like")]
    non = [g for g in processed if g["Predictor_call"] == "non-hvKp-like"]
    equiv = [g for g in processed if g["Predictor_call"] == "equivocal"]

    def with_amr(rows):
        return [g for g in rows if isinstance(g["AMR_gene_count"], int) and g["AMR_gene_count"] > 0]

    def carb_pos(rows):
        return [
            g
            for g in rows
            if str(g["Carbapenemase_genes"]) not in {NA, NOT_ANALYZED, FAILED, NOT_DETECTED}
        ]

    def replicon_pos(rows):
        return [g for g in rows if isinstance(g["Replicon_hit_count"], int) and g["Replicon_hit_count"] > 0]

    def st_assigned(rows):
        return [g for g in rows if str(g["ST"]) not in {NA, NOT_ANALYZED, FAILED, NOT_DETECTED}]

    amr_counts = []
    for g in processed:
        if isinstance(g["AMR_gene_count"], int):
            amr_counts.append(g["AMR_gene_count"])
    gene_counter = Counter()
    class_counter = Counter()
    for g in processed:
        for row in g["amr_rows"]:
            gene_counter[row["Gene"]] += 1
            if row["AMR_Class"] not in {NA, NOT_DETECTED}:
                for part in str(row["AMR_Class"]).split(";"):
                    part = part.strip()
                    if part:
                        class_counter[part] += 1
    rep_counter = Counter()
    for g in processed:
        for p in g["plasmid_rows"]:
            rep_counter[p["Replicon"]] += 1
    vir_counter = Counter()
    for g in processed:
        for name, state in g["virulence_markers"].items():
            if state == "Detected":
                vir_counter[name] += 1

    def stats(xs):
        if not xs:
            return {"n": 0, "mean": NA, "median": NA, "max": NA}
        s = pd.Series(xs, dtype=float)
        return {"n": int(len(xs)), "mean": float(s.mean()), "median": float(s.median()), "max": float(s.max())}

    n_proc = len(processed)
    return {
        "n_input": len(genomes),
        "n_processed": n_proc,
        "n_failed": len(failed),
        "n_hvkp_like": len(hvkp),
        "n_non": len(non),
        "n_equivocal": len(equiv),
        "hvkp_pct": (100.0 * len(hvkp) / n_proc) if n_proc else NA,
        "n_amr_genomes": len(with_amr(processed)),
        "n_carb": len(carb_pos(processed)),
        "n_replicon_genomes": len(replicon_pos(processed)),
        "n_st": len(st_assigned(processed)),
        "n_unique_st": len({g["ST"] for g in st_assigned(processed)}),
        "calls": calls,
        "amr_burden": stats(amr_counts),
        "top_genes": gene_counter.most_common(10),
        "top_classes": class_counter.most_common(10),
        "top_replicons": rep_counter.most_common(10),
        "top_st": _counts([g["ST"] for g in processed])[:10],
        "top_vir": vir_counter.most_common(10),
        "ckp": NA,
        "excluded": NA,
        "mdr": NA,
        "high_priority": NA,
        "convergent_defined": False,
    }


def _bar_svg(pairs: list[tuple[str, float]], title: str, width=720) -> str:
    if not pairs:
        return f'<div class="panel"><div class="chart-title">{title}</div><p class="small">None in this run</p></div>'
    vals = [v for _, v in pairs]
    peak = max(vals) or 1
    h = 36 + 28 * len(pairs)
    rows = []
    for i, (lab, v) in enumerate(pairs):
        y = 32 + i * 28
        w = 20 + (width - 280) * (v / peak)
        rows.append(
            f'<text x="8" y="{y+14}" font-size="12">{lab}</text>'
            f'<rect x="180" y="{y}" width="{w:.1f}" height="18"></rect>'
            f'<text x="{184+w:.1f}" y="{y+14}" font-size="12">{v:g}</text>'
        )
    body = "".join(rows)
    return (
        f'<div class="panel amr-chart"><svg viewBox="0 0 {width} {h}" role="img" aria-label="{title}">'
        f'<text x="8" y="18" class="chart-title">{title} (n={len(pairs)} bars)</text>{body}</svg></div>'
    )


def _call_bar(pairs: list[tuple[str, float]]) -> str:
    total = sum(v for _, v in pairs) or 1
    colors = {
        "hvKp-like (high)": "#0f766e",
        "hvKp-like (moderate)": "#14b8a6",
        "equivocal": "#d97706",
        "non-hvKp-like": "#64748b",
    }
    segs = []
    legend = []
    for lab, v in pairs:
        pct = 100.0 * v / total
        color = colors.get(lab, "#334155")
        segs.append(f'<span style="width:{pct:.2f}%;background:{color}" title="{lab}: {v:g}"></span>')
        legend.append(
            f'<span class="legend"><i style="background:{color}"></i>{lab} <b>{v:g}</b> ({pct:.1f}%)</span>'
        )
    return (
        '<div class="panel">'
        '<div class="chart-title">Call composition</div>'
        f'<div class="stack" role="img" aria-label="Call composition">{"".join(segs)}</div>'
        f'<div class="legend-row">{"".join(legend)}</div></div>'
    )


def _table(headers: list[str], rows: list[dict], table_id: str) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for r in rows:
        tds = "".join(f"<td>{'' if r.get(h) is None else r.get(h)}</td>" for h in headers)
        body.append(f"<tr>{tds}</tr>")
    return (
        f'<div class="table-wrap"><table class="data" id="{table_id}"><thead><tr>{th}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def render_html(genomes: list[dict], run_info: dict) -> str:
    summary = summarise(genomes)
    marker_order = ["iucD", "AlpA", "aidA", "iutA_2", "dhbE", "iucC_2"]
    payload = []
    for g in genomes:
        row = {k: v for k, v in g.items() if k not in {"amr_rows", "plasmid_rows", "predictor_markers", "virulence_markers"}}
        bits = g.get("predictor_markers") or {}
        row.update(bits)
        row["Markers_present"] = sum(1 for name in marker_order if bits.get(name) == 1) if bits else NA
        payload.append(row)
    data_json = json.dumps(payload, default=str)


    def card(label, value):
        if isinstance(value, float):
            value = f"{value:.4g}"
        return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'

    pct = summary["hvkp_pct"]
    pct_s = "Not available" if pct == NA else f"{pct:.1f}%"
    burden = summary["amr_burden"]

    call_pairs = [(k, v) for k, v in summary["calls"].most_common()]
    gene_pairs = summary["top_genes"]
    class_pairs = summary["top_classes"]

    marker_genes = ["iucD", "AlpA", "aidA", "iutA_2", "dhbE", "iucC_2"]
    ml_rows = []
    for g in genomes:
        bits = g.get("predictor_markers") or {}
        present = sum(1 for name in marker_genes if bits.get(name) == 1)
        ml_rows.append({
            "Genome_ID": g["Genome_ID"],
            "Call": g["Virulence_class"],
            "P_hvKp": g["P_hvKp"],
            "Markers_present": present if bits else NA,
            **{name: bits.get(name, NA) for name in marker_genes},
        })

    notice = (
        "Counts are the FASTA files in this run. "
        "The hvKp call is the six-gene XGBoost model (mean absolute SHAP at least 0.30). "
        "AMR genes are ABRicate NCBI, or Kleborate acquired genes when NCBI was not run."
    )

    cards = "".join([
        card("Total FASTA in this run", summary["n_input"]),
        card("Successfully processed", summary["n_processed"]),
        card("Analysis failed", summary["n_failed"]),
        card("hvKp-like calls", summary["n_hvkp_like"]),
        card("non-hvKp-like calls", summary["n_non"]),
        card("Equivocal calls", summary["n_equivocal"]),
        card("hvKp-like percentage", pct_s),
        card("Genomes with AMR genes", summary["n_amr_genomes"]),
        card("Carbapenemase reported", summary["n_carb"]),
    ])

    amr_stats = (
        f"AMR burden uses one source per genome (ABRicate NCBI when that table exists, otherwise Kleborate acquired genes). "
        f"Genomes with a numeric burden: n={burden['n']}. "
        f"Mean={burden['mean'] if not isinstance(burden['mean'], float) else round(burden['mean'], 3)}, "
        f"median={burden['median'] if not isinstance(burden['median'], float) else round(burden['median'], 3)}, "
        f"max={burden['max']}."
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KlebX - Genomic Profiling &amp; Phenotype Prediction Report</title>
<style>
:root {{font-family:Inter,Arial,sans-serif; color:#17324d; background:#eef7fc; --blue:#095e97; --blue-dark:#245f8a; --blue-soft:#e5f3fb;}}
body {{margin:0}} header {{background:linear-gradient(135deg,#245f8a,#4f9bc5);color:white;padding:22px 4vw;position:sticky;top:0;z-index:2}}
nav {{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:12px}}
nav a {{color:white;text-decoration:none;font-size:13px}}
main {{max-width:1500px;margin:auto;padding:24px 4vw 60px}}
h1 {{margin:0 0 6px;font-size:28px}} h2 {{margin-top:36px}}
.notice {{padding:14px 18px;background:#e5f3fb;border-left:5px solid var(--blue);border-radius:8px}}
.grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:14px}}
.card {{background:#fff;border-radius:10px;padding:16px;box-shadow:0 2px 8px #2f80b715;border:1px solid #dcecf5}}
.label {{font-size:12px;color:#667}} .value {{font-size:22px;font-weight:700;margin-top:6px;word-break:break-word}}
.panel {{background:#fff;border-radius:10px;padding:16px;margin-top:14px;overflow:auto;border:1px solid #dcecf5}}
.data {{border-collapse:collapse;width:100%;font-size:13px}}
.data th,.data td {{padding:8px;border-bottom:1px solid #e5e7eb;text-align:left;white-space:nowrap}}
.data th {{background:#e5f3fb;position:sticky;top:0;cursor:pointer}}
svg {{width:100%;height:auto}} svg rect {{fill:#1372a5}} .chart-title {{font-weight:700}}
.small {{color:#667;font-size:13px}} footer {{padding:24px 4vw 40px;color:#667}}
.controls {{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}}
input,select {{padding:6px 8px;border:1px solid #ccd;border-radius:6px}}
button {{padding:6px 10px;border:0;border-radius:6px;background:#2f80b7;color:white;cursor:pointer}}
.amr-charts{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}.amr-chart{{margin-top:0;padding:10px}}.amr-chart svg{{max-height:270px}}
.table-wrap {{background:#fff;border:1px solid #dcecf5;border-radius:10px;margin-top:14px;max-height:520px;overflow:auto}}
@media(max-width:900px){{.amr-charts{{grid-template-columns:1fr}}}}
@media(max-width:700px) {{h1{{font-size:22px}}}}
</style></head>
<body>
<header>
<h1>KlebX</h1>
<div class="small" style="color:#d9e2ec">
An Integrated Machine Learning and Genomic Profiling Tool for Phenotype Prediction in <i>Klebsiella pneumoniae</i>
</div>
<nav>
<a href="#about">About</a><a href="#overview">Summary</a><a href="#model">Prediction Model</a><a href="#amr">AMR</a><a href="#convergence">Co-occurrence</a><a href="#genomes">Genomes</a>
</nav></header>
<main>
<section id="about"><h2>About KlebX</h2>
<div class="notice">
<b>KlebX</b> is an integrated machine learning and genomic profiling tool
for phenotype prediction in <i>Klebsiella pneumoniae</i>. It combines
machine-learning-based hvKp-like prediction with antimicrobial resistance
profiling, virulence-associated markers, sequence typing, K-locus
characterization, plasmid replicon detection, genome quality metrics and
resistance-virulence co-occurrence analysis.
</div>
</section>
<section id="overview"><h2>1. Executive summary</h2>
<div class="grid">{cards}</div>
</section>
<section id="ml"><h2>2. hvKp-like / non-hvKp-like classification (six-gene model)</h2>
<p class="small">Call and P(hvKp) come from the six genes whose mean absolute SHAP was at least 0.30: iucD, AlpA, aidA, iutA_2, dhbE, and iucC_2. A marker is 1 when BLASTN finds it (identity at least 80%, iucD at least 75%, coverage at least 60%). hvKp-like (high) is P at least 0.8, hvKp-like (moderate) is P at least 0.5, equivocal is P at least 0.2, otherwise non-hvKp-like.</p>
<div class="grid">{card("hvKp-like", summary["n_hvkp_like"])}{card("non-hvKp-like", summary["n_non"])}{card("Equivocal", summary["n_equivocal"])}</div>
{_table(["Genome_ID","Call","P_hvKp","Markers_present",*marker_genes], ml_rows, "ml-table")}
</section>
<section id="model"><h2>Prediction Model</h2>
<div class="panel">
<table class="data">
<tr><th>Model</th><td>XGBoost</td></tr>
<tr><th>Features</th><td>{len(marker_genes)} genomic markers</td></tr>
<tr><th>Input</th><td>Genome FASTA</td></tr>
<tr><th>Marker detection</th><td>BLASTN</td></tr>
<tr><th>Default identity threshold</th><td>&ge;80%</td></tr>
<tr><th><i>iucD</i> identity threshold</th><td>&ge;75%</td></tr>
<tr><th>Coverage threshold</th><td>&ge;60%</td></tr>
<tr><th>High hvKp-like</th><td>P &ge; 0.80</td></tr>
<tr><th>Moderate hvKp-like</th><td>P &ge; 0.50</td></tr>
<tr><th>Equivocal</th><td>P &ge; 0.20</td></tr>
<tr><th>Non-hvKp-like</th><td>P &lt; 0.20</td></tr>
</table>
</div>
</section>
<section id="amr"><h2>3. AMR summary</h2>
<p class="small">{amr_stats} Carbapenemase is Kleborate Bla_Carb_acquired. Colistin is Kleborate Col_acquired.</p>
<div class="amr-charts">
{_bar_svg(gene_pairs, "Top AMR genes")}
{_bar_svg(class_pairs, "AMR classes")}
</div>
{_table(["Genome_ID","AMR_gene_count","AMR_genes","AMR_classes","Carbapenemase_genes","Colistin_resistance_genes"], [g for g in genomes], "amr-summary")}
</section>
<section id="convergence"><h2>4. Resistance-virulence co-occurrence</h2>
<p class="small">hvKp-like genomes from the six-gene model, with the AMR genes found in the same genome.</p>
<div class="grid">{card("hvKp-like", summary["n_hvkp_like"])}{card("hvKp-like with AMR genes", len([g for g in genomes if str(g['Predictor_call']).startswith('hvKp-like') and isinstance(g['AMR_gene_count'], int) and g['AMR_gene_count']>0]))}{card("hvKp-like with carbapenemase", len([g for g in genomes if str(g['Predictor_call']).startswith('hvKp-like') and str(g['Carbapenemase_genes']) not in {NA, NOT_ANALYZED, FAILED, NOT_DETECTED}]))}</div>
<div class="controls">
<label>Call <select id="f-call"><option value="">Any</option></select></label>
<label>Carbapenemase <select id="f-carb"><option value="">Any</option><option>Set</option><option>Not detected</option><option>Not analyzed</option></select></label>
<label>ST <select id="f-st"><option value="">Any</option></select></label>
<label>Min AMR genes <input id="f-amr" type="number" min="0" placeholder="any"></label>
</div>
<div class="panel" style="max-height:480px;overflow:auto"><table class="data" id="conv-table"></table></div>
</section>
<section id="genomes"><h2>5. Genome-level results</h2>
<div class="controls">
<input id="q" placeholder="Search Genome_ID" />
<select id="g-call"><option value="">Any call</option></select>
<select id="g-amr"><option value="">AMR any</option><option value="pos">AMR genes &gt; 0</option><option value="zero">AMR genes = 0</option></select>
<select id="g-carb"><option value="">Carbapenemase any</option><option value="set">Field set</option><option value="nd">Not detected</option></select>
<button id="export" type="button">Export filtered CSV</button>
</div>
<div class="panel" style="max-height:640px;overflow:auto"><table class="data" id="genome-table"></table></div>
</section>
</main>
<footer>
<section id="reproducibility">
<h2>Reproducibility Information</h2>
<div class="panel">
<table class="data">
<tr><th>Software</th><td>KlebX</td></tr>
<tr><th>Version</th><td>{run_info.get("version", __version__)}</td></tr>
<tr><th>Analysis date</th><td>{run_info.get("date", "")}</td></tr>
<tr><th>Input type</th><td>FASTA</td></tr>
<tr><th>Genomes processed</th><td>{summary["n_input"]}</td></tr>
<tr><th>Pipeline status</th><td>{run_info.get("status", "Completed")}</td></tr>
<tr><th>Prediction model</th><td>XGBoost using {len(marker_genes)} genomic markers</td></tr>
<tr><th>AMR analysis</th><td>Genome-level resistance determinant profiling</td></tr>
</table>
</div>
</section>
<p>
<b>KlebX v{run_info.get("version", __version__)}</b><br>
Genomic Surveillance &amp; Prediction Platform for
<i>Klebsiella pneumoniae</i>
</p>
<p>
<b>Developed by:</b> Prasanna Kumar Selvam
<b>Corresponding author:</b> Karthick Vasudevan
</p>
<p>
<b>Copyright &copy; 2026 Institute of Bioinformatics, Bangalore.</b><br>
</p>
<p>
<b>Recommended citation:</b><br>
Selvam PK, Mohan S, Vasudevan K. <i>KlebX: An Integrated Machine Learning and Genomic Profiling Tool for Phenotype Prediction in Klebsiella pneumoniae</i>. Version {run_info.get("version", __version__)}. 2026.
</p>
<p class="small">
This report was generated automatically by KlebX.
Results are computational predictions and should be interpreted
within the scope and limitations of the underlying models,
databases and input data. Third-party software, databases and
datasets remain subject to their respective licenses and terms of use.
</p>
</footer>
<script>
const GENOMES = {data_json};

const GENOME_COLS = ["Genome_ID","Virulence_class","P_hvKp","Markers_present","iucD","AlpA","aidA","iutA_2","dhbE","iucC_2","AMR_gene_count","AMR_genes","AMR_classes","Carbapenemase_genes","Colistin_resistance_genes"];
const NA_SET = new Set(["Not available","Not analyzed","Analysis failed","Not detected"]);
function esc(v) {{ return String(v ?? "").replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c])); }}
function cell(v) {{
  const s = String(v ?? "");
  if (s.startsWith("hvKp-like")) return '<span style="color:#0f766e;font-weight:650">'+esc(s)+'</span>';
  if (s === "non-hvKp-like") return '<span style="color:#64748b;font-weight:650">'+esc(s)+'</span>';
  if (s === "equivocal") return '<span style="color:#d97706;font-weight:650">'+esc(s)+'</span>';
  return esc(s);
}}
function table(el, cols, rows) {{
  el.innerHTML = "<thead><tr>" + cols.map(c => "<th>"+c+"</th>").join("") + "</tr></thead><tbody>" +
    rows.map(r => "<tr>" + cols.map(c => "<td>"+cell(r[c])+"</td>").join("") + "</tr>").join("") + "</tbody>";
  [...el.querySelectorAll("th")].forEach((th, i) => th.onclick = () => {{
    const key = cols[i];
    rows.sort((a,b) => String(a[key]).localeCompare(String(b[key]), undefined, {{numeric:true}}));
    table(el, cols, rows);
  }});
}}
function fillSelect(id, values) {{
  const s = document.getElementById(id);
  if (!s) return;
  const cur = s.value;
  const opts = [...new Set(values.filter(v => v && !NA_SET.has(String(v))))].sort();
  s.innerHTML = s.options[0].outerHTML + opts.map(v => "<option>"+esc(v)+"</option>").join("");
  s.value = cur;
}}
function carbState(g) {{
  const v = String(g.Carbapenemase_genes);
  if (v === "Not detected") return "nd";
  if (NA_SET.has(v)) return "na";
  return "set";
}}
function passes(g) {{
  const q = document.getElementById("q").value.trim().toLowerCase();
  if (q && !String(g.Genome_ID).toLowerCase().includes(q)) return false;
  const c = document.getElementById("g-call").value;
  if (c && g.Virulence_class !== c && g.Predictor_call !== c) return false;
  const amr = document.getElementById("g-amr").value;
  if (amr === "pos" && !(Number(g.AMR_gene_count) > 0)) return false;
  if (amr === "zero" && Number(g.AMR_gene_count) !== 0) return false;
  const carb = document.getElementById("g-carb").value;
  if (carb === "set" && carbState(g) !== "set") return false;
  if (carb === "nd" && carbState(g) !== "nd") return false;
  return true;
}}
let view = GENOMES.slice();
function drawGenomes() {{
  view = GENOMES.filter(passes);
  table(document.getElementById("genome-table"), GENOME_COLS, view);
}}
["q","g-call","g-amr","g-carb"].forEach(id => document.getElementById(id).addEventListener("input", drawGenomes));
document.getElementById("export").onclick = () => {{
  const cols = GENOME_COLS;
  const lines = [cols.join(",")].concat(view.map(r => cols.map(c => '"'+String(r[c]??"").replace(/"/g,'""')+'"').join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")], {{type:"text/csv"}}));
  a.download = "KlebX_genomes_filtered.csv";
  a.click();
}};
fillSelect("g-call", GENOMES.map(g => g.Predictor_call));
fillSelect("f-call", GENOMES.map(g => g.Predictor_call));
fillSelect("f-st", GENOMES.map(g => g.ST));
function convRows() {{
  const call = document.getElementById("f-call").value;
  const st = document.getElementById("f-st").value;
  const carb = document.getElementById("f-carb").value;
  const min = document.getElementById("f-amr").value;
  return GENOMES.filter(g => String(g.Predictor_call).startsWith("hvKp-like")).filter(g => {{
    if (call && g.Predictor_call !== call) return false;
    if (st && String(g.ST) !== st) return false;
    if (carb === "Set" && carbState(g) !== "set") return false;
    if (carb === "Not detected" && carbState(g) !== "nd") return false;
    if (carb === "Not analyzed" && carbState(g) !== "na") return false;
    if (min !== "" && !(Number(g.AMR_gene_count) >= Number(min))) return false;
    return true;
  }});
}}
function drawConv() {{
  table(document.getElementById("conv-table"), ["Genome_ID","Virulence_class","P_hvKp","Markers_present","AMR_gene_count","AMR_genes","Carbapenemase_genes","Colistin_resistance_genes"], convRows());
}}
["f-call","f-carb","f-st","f-amr"].forEach(id => document.getElementById(id).addEventListener("input", drawConv));
drawGenomes(); drawConv();
</script>
</body></html>
"""
    return html


def write_excel(genomes: list[dict], path: Path) -> None:
    marker_genes = ["iucD", "AlpA", "aidA", "iutA_2", "dhbE", "iucC_2"]
    ml_rows = []
    genome_rows = []
    for g in genomes:
        bits = g.get("predictor_markers") or {}
        present = sum(1 for name in marker_genes if bits.get(name) == 1) if bits else NA
        ml_rows.append({
            "Genome_ID": g["Genome_ID"],
            "Call": g["Virulence_class"],
            "P_hvKp": g["P_hvKp"],
            "Markers_present": present,
            **{name: bits.get(name, NA) for name in marker_genes},
        })
        genome_rows.append({
            "Genome_ID": g["Genome_ID"],
            "Call": g["Virulence_class"],
            "P_hvKp": g["P_hvKp"],
            "Markers_present": present,
            **{name: bits.get(name, NA) for name in marker_genes},
            "AMR_gene_count": g.get("AMR_gene_count", NA),
            "AMR_genes": g.get("AMR_genes", NA),
            "AMR_classes": g.get("AMR_classes", NA),
            "Carbapenemase_genes": g.get("Carbapenemase_genes", NA),
            "Colistin_resistance_genes": g.get("Colistin_resistance_genes", NA),
        })
    gdf = pd.DataFrame(genome_rows)
    amr_sum = gdf[["Genome_ID", "AMR_gene_count", "AMR_genes", "AMR_classes", "Carbapenemase_genes", "Colistin_resistance_genes"]]
    conv = gdf[gdf["Call"].astype(str).str.startswith("hvKp-like")] if len(gdf) else gdf
    s = summarise(genomes)
    summary_df = pd.DataFrame([
        {"Item": "Total FASTA", "Value": s["n_input"]},
        {"Item": "Successfully processed", "Value": s["n_processed"]},
        {"Item": "Analysis failed", "Value": s["n_failed"]},
        {"Item": "hvKp-like", "Value": s["n_hvkp_like"]},
        {"Item": "non-hvKp-like", "Value": s["n_non"]},
        {"Item": "Equivocal", "Value": s["n_equivocal"]},
        {"Item": "Genomes with AMR genes", "Value": s["n_amr_genomes"]},
        {"Item": "Carbapenemase reported", "Value": s["n_carb"]},
    ])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="Executive_summary", index=False)
        pd.DataFrame(ml_rows).to_excel(xw, sheet_name="ML_12gene", index=False)
        amr_sum.to_excel(xw, sheet_name="AMR_Summary", index=False)
        conv.to_excel(xw, sheet_name="Co_occurrence", index=False)
        gdf.to_excel(xw, sheet_name="Genome_Results", index=False)


def validation_text(genomes: list[dict], html_path: Path, xlsx_path: Path) -> str:
    s = summarise(genomes)
    ids = [g["Genome_ID"] for g in genomes]
    dup = len(ids) - len(set(ids))
    lines = [
        "========================================",
        "HVKp GENOMIC SURVEILLANCE REPORT",
        "========================================",
        f"Total input genomes: {s['n_input']}",
        f"Successfully processed: {s['n_processed']}",
        f"hvKp-like: {s['n_hvkp_like']}",
        f"non-hvKp-like: {s['n_non']}",
        f"Equivocal: {s['n_equivocal']}",
        f"Analysis failed: {s['n_failed']}",
        "",
        f"MLST assigned: {s['n_st']}",
        f"AMR detected: {s['n_amr_genomes']}",
        f"Carbapenemase field set: {s['n_carb']}",
        f"Plasmid replicons detected: {s['n_replicon_genomes']}",
        "",
        f"Unique Genome_IDs: {len(set(ids))}",
        f"Duplicate Genome_IDs: {dup}",
        "Missing Genome_IDs: 0",
        "",
        f"HTML report: {html_path}",
        f"Excel report: {xlsx_path}",
        "========================================",
    ]
    return "\n".join(lines)


def write_surveillance_report(results: list[dict], outdir: Path, failed: list[dict] | None = None, run_info: dict | None = None) -> tuple[Path, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    genomes = [genome_from_result(r) for r in results]
    for f in failed or []:
        genomes.append(failed_genome(f.get("sample", "unknown"), f.get("fasta", ""), f.get("error", "")))
    info = {
        "date": str(date.today()),
        "input": "FASTA paths passed to KlebX",
        "outdir": str(outdir),
        "version": __version__,
        "tools": "",
        "status": "completed" if genomes else "no genomes",
    }
    if run_info:
        info.update(run_info)
    html_path = outdir / "KlebX.html"
    xlsx_path = outdir / "KlebX.xlsx"
    html_path.write_text(render_html(genomes, info), encoding="utf-8")
    write_excel(genomes, xlsx_path)
    text = validation_text(genomes, html_path, xlsx_path)
    print(text, flush=True)
    return html_path, xlsx_path
