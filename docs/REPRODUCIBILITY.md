# Reproducibility

## Archived development run

The curated `results/` directory corresponds to the documented DIMAP-C v2 development run.

- Outer evaluation: four leave-one-dataset-out folds.
- Full-model seeds: `42`, `52`, `62`.
- Target corpus: excluded from training, early stopping, temperature fitting, conformal calibration, and risk-threshold fitting.
- Model selection: mean source-domain macro-F1.
- Source validation: constructed separately within each source corpus.
- Speaker grouping: used where feasible; TESS used a stratified utterance-level fallback because it has only two speakers.
- Calibration: the same source-validation partition was reused for early stopping and temperature/conformal/risk calibration.
- Feature encoder: frozen `microsoft/wavlm-base-plus`.
- Selected layers: 4, 8, 12.
- Full-model result reporting: mean ± SD across the three fixed seeds.

The archived run predates the duplicate-content guard. A post-run audit identified mirrored RAVDESS paths; see `DATA_PROVENANCE.md`.

## Current v2.1 safeguards

- TESS `ps` / pleasant-surprise parsing is explicit.
- Exact duplicate audio is removed by content hash before split construction.
- Rejected and duplicate manifest rows are saved for audit.
- Feature caching uses a stable content-derived manifest signature.
- Compatible completed runs can be resumed through configuration digests.

## Confirmatory protocol recommended by the professor review

For final archival numerical claims, rerun all four folds and three seeds with:

1. duplicate-safe manifest construction;
2. disjoint source training, model-selection validation, and calibration partitions where feasible;
3. untouched target-corpus evaluation;
4. matched pre- and post-temperature-scaling ECE;
5. paired multi-seed ablations for the principal components;
6. computational profiling if practical (peak GPU memory, training time per fold, inference time per utterance, trainable parameter count, cache size).

The exact split fractions should be declared in the new run metadata. A nominal `70/15/15` source train/validation/calibration split is only an example; class and speaker support must be preserved.

## Main command

```bash
python run_all.py --download --train --report
```

## Smoke test

```bash
python run_all.py --demo --report --epochs 3 --ablation-scope none
```

## Full ablations

```bash
python run_all.py --download --train --report --ablation-scope full
```

## Stored artifacts

The pipeline stores configuration metadata, feature-cache metadata, checkpoints, history, calibration parameters, predictions, run metrics, result tables, figures, and a generated HTML report. New executions write to `outputs/`; curated historical artifacts remain under `results/`.

## Interpretation

Do not select the best target seed. Do not tune weights, temperature, conformal thresholds, or shift-risk parameters using target labels. The archived ECE values are post-temperature-scaling values; because the development run did not preserve a matched pre-scaling ECE series, no calibration-improvement claim should be made from those values alone.
