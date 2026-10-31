# KlebX

Predict **hypervirulent *Klebsiella pneumoniae*** (hvKp) from a genome FASTA.

The tool BLASTs six genes whose mean absolute SHAP was at least 0.30 in the 12-gene model: `iucD`, `AlpA`, `aidA`, `iutA_2`, `dhbE`, and `iucC_2`. It scores a compact XGBoost model, writes a **SHAP waterfall**, and writes a **clear human-readable report** (`report.txt` + `report.md`).

The first `KlebX` command creates a conda environment named `hvkp-tools` and installs `blast`, `kleborate`, `mlst`, `abricate`, and `mash`, including their databases. NumPy, pandas, scikit-learn, XGBoost, SHAP, and the other model libraries are installed with pip, so conda does not solve the whole bioconda index for them. Every later command checks the programs, then scores the FASTA. A system copy already on `PATH` is not used in place of this environment.

| Binary | Conda package | What you get |
|---|---|---|
| `blastn` | `blast` | Six-gene marker search |
| `kleborate` | `kleborate` | Species, ST, virulence score, K locus, acquired AMR |
| `mlst` | `mlst` (Torsten Seemann) | Pasteur 7-gene ST |
| `abricate` | `abricate` | NCBI, CARD, VFDB, PlasmidFinder |

## Install

Conda or Mamba must already be installed. From a checkout:

```bash
pip install -e .
KlebX -f genome.fasta -o hvkp_out
```

`pip install -e .` installs the launcher into the current Python. The first `KlebX -f` command creates `hvkp-tools` if it is missing, installs KlebX into that environment, checks that BLAST, Kleborate, mlst, and ABRicate run, then scores the FASTA. Every later command does the same check and then scores. `KlebX --tools` only checks and installs, then exits. Pass `--skip-setup` to refuse the download.

You do not need to activate `hvkp-tools`. The `KlebX` command switches into that environment itself.

## Predict from a FASTA (single or batch)

```bash
# single genome
KlebX -f genome.fasta -o hvkp_out

# batch: several files
KlebX -f isolate1.fna isolate2.fna isolate3.fna -o hvkp_out

# batch: all FASTA/FA/FNA in a directory
KlebX -d /path/to/genomes -o hvkp_out

# batch: paths listed in a text file (one per line)
KlebX --list genomes.txt -o hvkp_out

# quiet (still writes files; no full report on stdout)
KlebX -d /path/to/genomes -o hvkp_out --quiet

# 8 genomes at once, 4 BLASTN threads inside each genome
KlebX -d /path/to/genomes -o hvkp_out -j 8 -t 4 --quiet
```

`-t` / `--threads` is the BLASTN thread count for one genome (default 4). `-j` / `--jobs` is how many genomes run at the same time (default 1). Each job also runs Kleborate, mlst and ABRicate when those tools are installed.

### Primary Excel report

| File | Content |
|---|---|
| **`report.xlsx`** (per sample) | **`Consolidated`** sheet first: bold category headers (Markers / MLST / Kleborate_virulence / Kleborate_AMR / ABRicate_ncbi / card / vfdb / plasmidfinder) with gene sub-columns and **1 = present, 0 = absent**. Detail sheets: Markers, SHAP, MLST, Kleborate, ABRicate_*. No Summary/Notes sheets. |
| **`all_samples.xlsx`** (batch root) | Same **Consolidated** 1/0 matrix with **one row per isolate** (union of all genes in the batch) + **Calls** sheet |
| **`consolidated_presence.tsv`** | Tab-delimited copy of the Consolidated matrix |
| **`report.txt` / `report.md`** | Human-readable text / Markdown report |
| **`KlebX.html`** (batch root) | Interactive KlebX report for the FASTA files in this run. |
| **`KlebX.xlsx`** | Same run, as tables |

The full report is also printed to the terminal unless you pass `--quiet`.

### Other outputs (per sample, under `hvkp_out/<stem>/`)

| File | Content |
|---|---|
| `summary.tsv` | One-line summary (P, call, markers, ST, virulence score, aerobactin, K locus) |
| `prediction.json` | Full machine-readable record including SHAP values |
| `marker_calls.tsv` | Per-gene BLAST identity / coverage |
| `shap_waterfall.png` | SHAP explanation of this genome |
| `kleborate.tsv` | Kleborate table (if installed) |
| `mlst.txt` | Seemann mlst (if installed) |
| `abricate_*.tsv` | ABRicate DBs (if installed) |

`hvkp_out/all_samples.tsv` concatenates one row per genome (with path to each `report.txt`).

### Example report header

```
==============================================================================
KlebX report
==============================================================================
Sample:          ERR10471622.fasta
Call:            hvKp-like (high)
P(hvKp):         0.9960
Markers present: 5 / 6
ST (best):       231
...
```

### How the ML call is made

1. `blastn` vs `KlebX/data/markers.fna` (one representative CDS per Panaroo cluster).
2. A marker is **present** if identity ≥ 80% and subject coverage ≥ 60%. `iucD` is called at ≥ 75% so both aerobactin lineages are detected: iuc 1 (classic ST23) and iuc 5 (the reference CDS) share about 79% nucleotide identity.
3. XGBoost on the six markers returns P(hvKp).
4. SHAP attributes that probability to each marker.
5. A human-readable **`report.txt` / `report.md`** is written with MLST, Kleborate and ABRicate sections.

Calls: `hvKp-like (high)` P≥0.8 · `hvKp-like (moderate)` P≥0.5 · `equivocal` P≥0.2 · `non-hvKp-like` otherwise.

On the study held-out set this six-gene model matches the 12-gene classification: AUROC 0.9998, sensitivity 0.987 and specificity 1.000 (295 true positives, 0 false positives, 4 false negatives). Refit SHAP is `iucD` 4.25, `aidA` 1.59, `AlpA` 1.51, `dhbE` 0.91, `iutA_2` 0.49, `iucC_2` 0.27. **`iucD` alone has AUROC 0.977**, sensitivity 1.000 and 36 false positives.

### Self-check

```bash
KlebX -f KlebX/data/markers.fna -o hvkp_out_selftest --quiet
# expect: markers present and Call hvKp-like (high)
```

## Marker genes

`iucD`, `AlpA`, `aidA`, `iutA_2`, `dhbE`, `iucC_2`

`ccmM_1`, `vapC_1`, `iucA_1`, `vap`, `def_4`, and `iucB` are not scored. Held-out AUROC is 0.9998, the same as the 12-gene model. The `dhbE` marker is the representative sequence from the Panaroo cluster `pchD~~~acs_6~~~dhbE`. Prokka calls that sequence `dhbE`, a 2,3-dihydroxybenzoate-AMP ligase. It is not a verified pyochelin `pchD` gene.

## Repository layout

```
KlebX/
  README.md
  KlebX/                 # Python package + marker DB + model
  manuscript/
    manuscript.md / .docx
    tables/  figures/
  examples/
```

## Citation

See `manuscript/manuscript.md`. Analysis: Panaroo pangenome of 3,600 isolates, Scoary and PC-corrected GWAS, seven classifiers (logistic regression, linear SVM, random forest, extra trees, histogram gradient boosting, LightGBM, XGBoost) plus SHAP. BLAST names for unnamed clusters required at least 80% protein identity. The FASTA tool scores the six-gene XGBoost model. Kleborate `iuc 5` is an aerobactin lineage, not a marker count.

## License

MIT
