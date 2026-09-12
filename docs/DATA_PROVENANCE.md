# Data Provenance and RAVDESS Duplicate Audit

The four corpora are downloaded through the Kaggle handles in `config.yaml` and parsed into one harmonized manifest.

## Canonical corpora

| Corpus | Canonical size | Protocol |
|---|---:|---|
| CREMA-D | 7,442 clips | six known emotions |
| RAVDESS | 1,440 speech files | six known + calm/surprise as target semantic unknowns |
| TESS | 2,800 | six known + pleasant surprise mapped to protocol label `surprise` |
| SAVEE | 480 | six known + surprise as target semantic unknown |

## Post-run audit

The documented v2 result package contained **2,880 RAVDESS manifest rows** because the downloaded tree exposed two mirrored copies of the same canonical 1,440 speech files. The duplicate rows do not represent additional speakers or recordings.

Because the outer evaluation holds out entire corpora, mirrored RAVDESS files did **not** cross the held-out target/source corpus boundary. However, when RAVDESS participated as a source corpus, identical files could be assigned to both source-training and source-validation subsets. This can reweight optimization and may influence early stopping and post-hoc calibration.

For that reason, the numerical artifacts under `results/` are preserved strictly as provenance of the documented development run. They are **not** relabeled as results from the corrected pipeline.

## v2.1 duplicate-content safeguard

The current manifest builder computes a SHA-256 content hash for every audio file and removes exact within-corpus duplicates before any train/validation/test split or feature-cache construction. Duplicate rows are written to `manifest_duplicates.csv` for auditability.

This safeguard means a fresh v2.1 run will not recreate the mirrored-RAVDESS manifest used by the archived result tables.

## Confirmatory rerun

Before treating the numerical estimates as final archival results, the professor-reviewed manuscript recommends a complete rerun using:

- the duplicate-safe v2.1 manifest;
- all four leave-one-dataset-out folds;
- seeds 42, 52, and 62;
- the same architecture and declared hyperparameters;
- disjoint source-side model-selection and calibration data where feasible;
- matched pre- and post-temperature-scaling ECE;
- no target-domain tuning.

Any future result table generated from that confirmatory protocol should be clearly distinguished from the archived development-run tables currently under `results/`.
