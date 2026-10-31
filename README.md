# KlebX
### Genome-based machine learning for hypervirulence-associated marker detection in *Klebsiella pneumoniae*

KlebX is a genome-based bioinformatics tool that predicts **hypervirulent *Klebsiella pneumoniae* (hvKp)-associated genomic signatures** from FASTA assemblies using a compact XGBoost classifier. It integrates BLASTN-based marker detection, SHAP-based model interpretation, and supplementary genomic characterisation to generate comprehensive, user-friendly reports.

---

## Key Features

- **Machine-learning prediction:** Packaged XGBoost classifier using six genomic markers.
- **Genome-based screening:** BLASTN detection of marker sequences from FASTA assemblies.
- **Explainable predictions:** SHAP-based feature attribution and waterfall visualisation.
- **Genomic characterisation:** Integration of Kleborate, MLST and ABRicate.
- **Batch processing:** Analyse individual genomes or multiple FASTA files simultaneously.
- **Automated setup:** Creates a dedicated Conda environment and installs required tools.
- **Comprehensive outputs:** HTML dashboard, Excel workbooks, TSV files and human-readable reports.
- **Reproducible workflow:** Packaged model, marker reference sequences and configuration files.

> **Important:** KlebX predicts agreement with a genomic classification framework. Its output is not a clinically calibrated probability of hypervirulence and does not replace clinical, epidemiological or experimental assessment.

---

## Installation

### Prerequisites

- Linux operating system
- Conda or Mamba
- Python and pip

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/YOUR_USERNAME/KlebX.git
cd KlebX
```

Install KlebX:

```bash
pip install -e .
```

### Automated environment setup

The first execution of KlebX creates a dedicated Conda environment named `hvkp-tools` and installs the required genomic analysis tools and databases.

```bash
KlebX --tools
```

This installs and checks:

| Tool | Purpose |
|---|---|
| BLAST+ (`blast`) | Marker sequence detection |
| Kleborate | Species identification, sequence type, virulence and AMR characterisation |
| MLST | Seven-locus sequence typing |
| ABRicate | Screening against NCBI, CARD, VFDB and PlasmidFinder databases |

Python dependencies, including NumPy, pandas, scikit-learn, XGBoost and SHAP, are installed within the dedicated environment.

**Note:** You do not need to activate `hvkp-tools` manually. KlebX automatically executes within this environment.

To prevent automatic installation:

```bash
KlebX --skip-setup
```

---

## Usage

### Single-genome prediction

```bash
KlebX -f genome.fasta -o hvkp_out
```

### Multiple genomes

```bash
KlebX -f isolate1.fna isolate2.fna isolate3.fna -o hvkp_out
```

### Batch analysis of a directory

```bash
KlebX -d /path/to/genomes -o hvkp_out
```

### Batch analysis using a file list

Provide a text file containing one FASTA path per line:

```bash
KlebX --list genomes.txt -o hvkp_out
```

### Quiet mode

```bash
KlebX -d /path/to/genomes -o hvkp_out --quiet
```

### Parallel processing

```bash
KlebX -d /path/to/genomes -o hvkp_out -j 8 -t 4 --quiet
```

| Parameter | Description | Default |
|---|---|---|
| `-f` | Input FASTA file(s) | — |
| `-d` | Input directory | — |
| `--list` | Text file containing FASTA paths | — |
| `-o` | Output directory | — |
| `-t`, `--threads` | BLASTN threads per genome | 4 |
| `-j`, `--jobs` | Number of genomes processed simultaneously | 1 |
| `--quiet` | Suppress detailed terminal reports | False |
| `--tools` | Install and check required tools | — |
| `--skip-setup` | Disable automated environment setup | False |

---

## Prediction Workflow

KlebX follows an integrated genome-screening and machine-learning workflow:

1. **Marker detection:** BLASTN searches input genome assemblies against reference sequences in `data/markers.fna`.
2. **Marker identification:** Hits are evaluated using nucleotide identity and subject coverage thresholds.
3. **Feature encoding:** Detected markers are converted into a binary presence/absence vector.
4. **Machine-learning prediction:** The six-marker vector is supplied to the packaged XGBoost classifier.
5. **Model interpretation:** SHAP calculates feature-level contributions and generates a waterfall plot.
6. **Supplementary analysis:** Kleborate, MLST and ABRicate characterise additional genomic features.
7. **Report generation:** Results are exported as structured data, human-readable reports and interactive visualisations.

### Marker detection thresholds

| Parameter | Threshold |
|---|---|
| General nucleotide identity | ≥80% |
| Subject-sequence coverage | ≥60% |
| *iucD* nucleotide identity | ≥75% |

The relaxed *iucD* threshold accommodates sequence variation between aerobactin-associated lineages represented in the reference data.

---

## Machine-Learning Model

The packaged classifier uses six binary genomic features:

| Marker | Description |
|---|---|
| `iucD` | Aerobactin-associated marker |
| `AlpA` | Representative-sequence-derived annotation |
| `aidA` | Associated genomic marker |
| `iutA_2` | Aerobactin-associated marker |
| `dhbE` | Composite Panaroo cluster-derived feature |
| `iucC_2` | Aerobactin-associated marker |

The model is stored in `data/hv_xgb.joblib`, with feature ordering and configuration specified in `data/model_meta.json`.

### Classification categories

| Predicted category | Probability threshold |
|---|---|
| High hvKp-like | P ≥ 0.8 |
| Moderate hvKp-like | 0.5 ≤ P < 0.8 |
| Equivocal | 0.2 ≤ P < 0.5 |
| Non-hvKp-like | P < 0.2 |

These categories are software-defined reporting thresholds and are **not clinically calibrated probability estimates**.

### Model performance

On the held-out test dataset:

| Metric | Performance |
|---|---:|
| AUROC | 0.9998 |
| Sensitivity | 98.7% |
| Specificity | 100.0% |
| True positives | 295 |
| False positives | 0 |
| True negatives | 781 |
| False negatives | 4 |

The six-marker classifier demonstrated performance comparable to the original 12-marker classification model within the study dataset.

### SHAP interpretation

Mean absolute SHAP values for the six-marker model:

| Feature | Mean absolute SHAP |
|---|---:|
| `iucD` | 4.25 |
| `aidA` | 1.59 |
| `AlpA` | 1.51 |
| `dhbE` | 0.91 |
| `iutA_2` | 0.49 |
| `iucC_2` | 0.27 |

SHAP values represent contributions to model predictions and should not be interpreted as causal biological effects.

**Annotation note:** `dhbE` corresponds to the composite Panaroo cluster `pchD~~~acs_6~~~dhbE`, not a confirmed standalone gene. The `AlpA` annotation is based on representative-sequence similarity.

---

## Output Files

### Primary reports

| File | Description |
|---|---|
| `report.xlsx` | Per-sample Excel workbook containing consolidated marker and genomic characterisation results |
| `all_samples.xlsx` | Batch-level consolidated presence/absence matrix and prediction calls |
| `consolidated_presence.tsv` | Tab-delimited consolidated genomic feature matrix |
| `report.txt` | Human-readable prediction report |
| `report.md` | Markdown-formatted prediction report |
| `KlebX.html` | Interactive batch-level HTML dashboard |
| `KlebX.xlsx` | Batch-level Excel report |

### Per-sample outputs

Each sample is analysed in a separate directory:

```text
hvkp_out/
├── all_samples.tsv
├── all_samples.xlsx
├├── KlebX.html
├── KlebX.xlsx
└── sample_name/
    ├── summary.tsv
    ├── prediction.json
    ├── marker_calls.tsv
    ├── shap_waterfall.png
    ├── report.txt
    ├── report.md
    ├── kleborate.tsv
    ├── mlst.txt
    └── abricate_*.tsv
```

| File | Description |
|---|---|
| `summary.tsv` | Prediction summary, probability, markers, ST and genomic characteristics |
| `prediction.json` | Machine-readable prediction and SHAP information |
| `marker_calls.tsv` | BLASTN identity, coverage and marker detection results |
| `shap_waterfall.png` | SHAP waterfall visualisation |
| `kleborate.tsv` | Kleborate output |
| `mlst.txt` | Sequence typing results |
| `abricate_*.tsv` | Database-specific ABRicate screening results |

### Consolidated Excel matrix

The `Consolidated` worksheet contains genomic feature categories, including:

- Markers
- MLST
- Kleborate virulence
- Kleborate AMR
- ABRicate NCBI
- CARD
- VFDB
- PlasmidFinder

Feature columns use binary encoding:

- `1` — Present
- `0` — Absent

The batch-level workbook contains one row per isolate and a separate `Calls` worksheet.

---

## Self-test

Run the packaged marker sequences as a test input:

```bash
KlebX -f KlebX/data/markers.fna -o hvkp_out_selftest --quiet
```

The test should identify marker sequences and return a high hvKp-like prediction.

---

## Repository Structure

```text
KlebX/
├── README.md
├── KlebX/
│   ├── __init__.py
│   ├── __main__.py
│   ├── bootstrap.py
│   ├── predict.py
│   ├── surveillance_report.py
│   └── data/
│       ├── hv_xgb.joblib
│       ├── markers.fna
│       ├── markers.tsv
│       ├── model_meta.json
│       └── README.md
├── examples/
└── manuscript/
```

---

## Scientific Background

KlebX was developed using a pangenome-based analysis of 3,600 *K. pneumoniae* genomes, comprising 997 hvKp-labelled and 2,603 cKp-labelled isolates.

The development workflow integrated:

- Panaroo-based pangenome reconstruction
- Population-structure assessment
- PCA-based residualised gene association analysis
- Scoary-based association screening
- Sequence-based annotation of candidate genomic features
- Comparative evaluation of seven machine-learning classifiers
- SHAP-based feature interpretation
- Compact-marker selection and model implementation

The six-marker classifier was packaged to enable genome-level screening without requiring users to reconstruct the complete pangenome.

---

## Limitations

- Predictions reflect agreement with the genomic classification used during model development.
- The probability categories are not clinically calibrated.
- Independent clinical and experimental validation is required.
- Population structure and relatedness may influence genomic associations.
- Some marker annotations are cluster-derived and require further validation.

KlebX is intended for **genomic research, comparative genomics and exploratory surveillance**, not as a standalone clinical diagnostic system.

---

## Citation

If you use KlebX in your research, please cite the associated manuscript:

> Prasanna Selvam, et al. *KlebX: A Genome-Based Machine-Learning Framework for Hypervirulence-Associated Marker Detection in Klebsiella pneumoniae.* [Manuscript details to be updated upon publication.]

The manuscript and supplementary materials will be made available through this repository.

---

## License

This project is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

We acknowledge the developers and maintainers of BLAST+, Panaroo, Kleborate, MLST, ABRicate, XGBoost and SHAP, whose open-source tools support the KlebX workflow.
