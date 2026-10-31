# Model assets

| File | Role |
|---|---|
| `markers.fna` | One representative Prokka CDS per marker (from the Panaroo cluster) |
| `markers.n*` | BLASTN database (`makeblastdb -in markers.fna -dbtype nucl`) |
| `markers.tsv` | Feature metadata (cluster, isolate, locus, length) |
| `hv_xgb.joblib` | XGBoost classifier on six genes: iucD, AlpA, aidA, iutA_2, dhbE, iucC_2 |
| `model_meta.json` | Feature order, threshold, held-out metrics |

Held-out metrics of this compact model (same 1,080-isolate test split as the paper): AUROC 0.9998, sensitivity 0.987, specificity 1.000.
