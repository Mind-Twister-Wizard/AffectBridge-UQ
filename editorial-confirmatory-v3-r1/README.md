# Corrected editorial confirmatory v3-r1 — October 2026

This directory documents the **new, deduplicated, independently source-calibrated** evaluation used in the revised manuscript *AffectBridge-UQ: Uncertainty-Aware Domain-Generalized Cross-Corpus Speech Emotion Recognition with Selective Human-Review Deferral*.

**This is not the historical v2 development experiment.** Keep the [original result archive](../results/README.md) separate. The editor requested a four-fold, three-seed confirmatory evaluation, independent selection/calibration partitions and paired component ablations. The saved local output and offline audit reported 42/42 run records and no missing run artifacts.

## Confirmatory performance

| Held-out dataset | Accuracy | Macro-F1 (mean ± SD) | UAR | Post-scale ECE | Empirical set coverage | Semantic-unknown AUROC |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D | 0.3609 | 0.3313 ± 0.0669 | 0.3556 | 0.4645 | 0.4113 | undefined |
| RAVDESS | 0.5513 | 0.5359 ± 0.0452 | 0.5769 | 0.2579 | 0.7033 | 0.5696 |
| SAVEE | 0.5905 | 0.5659 ± 0.0227 | 0.5810 | 0.1062 | 0.7500 | 0.4640 |
| TESS | 0.4579 | 0.4084 ± 0.0350 | 0.4579 | 0.3148 | 0.6411 | 0.5346 |

Across the four targets, **unweighted mean macro-F1 = 0.4604**. All set coverages were below 0.90; unknown-class performance was near chance; and no human-reviewed operational outcomes were evaluated. Numerical evidence must not be portrayed as a guarantee of machine empathy or clinical safety.

## Files

- [Main four-fold aggregate summary](results/summary.csv)
- [12 individual held-out seed metrics](results/full_model_seeds.csv)
- [Paired three-seed ablation minus full-model deltas](results/paired_ablation_deltas.csv)
- [Exact experimental protocol and code status](PROTOCOL.md)
- [Dataset duplicate and label-conflict audit](DATA_AUDIT.md)

**Release boundary:** These small text artifacts are public. The **v3-r1 model-changing source patch and reproducibility launcher are now public** in [source/](source/README.md). The full per-utterance predictions, 42 large checkpoints, and private path-bearing run artifacts remain with the authors and are not silently represented by the historical v2 directory. Follow corpus licenses when downloading source data.

## Exact public source release

The [confirmatory source directory](source/README.md) publishes the scientific patch, separated train/selection/calibration split implementation, one-click Windows entry point, result validator, and source-level tests. See the source README for environment and execution instructions. The user's completed experiment results do not have to be rerun to inspect this source.
