# Reproducibility

## Main protocol

- Outer evaluation: four leave-one-dataset-out folds.
- Full-model seeds: `42`, `52`, `62`.
- Model selection: mean source-domain macro-F1.
- Target corpus: excluded from training, early stopping, temperature fitting, conformal calibration and risk-threshold fitting.
- Feature encoder: frozen `microsoft/wavlm-base-plus`.
- Selected hidden layers: 4, 8, 12.
- Feature cache: content-derived signature with preprocessing metadata.
- Resume: compatible completed runs are skipped automatically.

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

## Stored result artifacts

The curated `results/` directory contains publication-facing CSV tables and figures from the documented completed run. New executions write to `outputs/`, which is ignored by Git except for `.gitkeep`.
