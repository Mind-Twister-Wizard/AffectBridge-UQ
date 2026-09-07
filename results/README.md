# Result Artifacts

This directory contains the curated publication-facing artifacts from the documented completed AffectBridge-UQ / DIMAP-C v2 experiment.

## Tables

- `tables/paper_main_results.csv` — three-seed full-model summary by held-out corpus.
- `tables/results_summary.csv` — all recorded full/ablation runs.
- `tables/ablation_results.csv` — representative ablations and baselines.
- `tables/dataset_summary.csv` — harmonized manifest counts from the documented run.

## Figures

`figures/` contains the generated confusion matrices, calibration plots, risk-coverage curves, unknown-emotion ROC curves, training curves, architecture, pipeline and ablation figure.

## HTML report

Open `final_report.html` locally to view the original automatically generated experiment report. Its figure links resolve to the sibling `figures/` directory.

## Provenance note

These results correspond to the documented completed v2 experiment. The current v2.1 code adds duplicate-content guarding after a post-run audit identified mirrored RAVDESS paths. See `../docs/DATA_PROVENANCE.md`.
