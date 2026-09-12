# Result Artifacts

This directory contains curated publication-facing artifacts from the **documented DIMAP-C v2 development run**.

## Important provenance boundary

These files predate two later safeguards highlighted in the professor-reviewed manuscript:

1. the v2.1 content-hash duplicate guard for mirrored RAVDESS audio; and
2. the proposed separation of source model-selection data from source calibration data.

The target corpus was excluded from training, early stopping, temperature fitting, conformal calibration, and risk-threshold fitting in the documented run. However, the same source-validation partition was reused for early stopping and post-hoc calibration. The archived result tables are therefore retained for traceability and should **not** be described as results of the stronger confirmatory protocol.

## Tables

- `tables/paper_main_results.csv` — three-seed full-model summary by held-out corpus.
- `tables/results_summary.csv` — all recorded full/ablation runs.
- `tables/ablation_results.csv` — representative ablations and baselines.
- `tables/dataset_summary.csv` — manifest counts from the documented development run.

The ECE values in the archived tables are **post-temperature-scaling** values. A matched pre-scaling ECE series was not retained, so the archived artifacts do not establish a numerical calibration improvement from temperature scaling.

## Figures

`figures/` contains generated confusion matrices, calibration plots, risk-coverage curves, unknown-emotion ROC curves, training curves, architecture/pipeline illustrations, and the representative ablation figure.

## HTML report

Open `final_report.html` locally to view the original automatically generated development-run report. Its figure links resolve to the sibling `figures/` directory.

## Confirmatory rerun recommended for final archival claims

A future confirmatory run should use:

- the duplicate-safe v2.1 manifest;
- all four leave-one-dataset-out folds;
- seeds 42, 52, and 62;
- disjoint source training / model-selection / calibration partitions where feasible;
- matched pre- and post-temperature-scaling ECE;
- preferably paired multi-seed ablations.

See `../docs/DATA_PROVENANCE.md`, `../docs/experiment_protocol.md`, and `../docs/REPRODUCIBILITY.md`.
