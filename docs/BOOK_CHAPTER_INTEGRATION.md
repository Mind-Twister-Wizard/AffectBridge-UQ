# Book Chapter Integration

Repository URL:

**https://github.com/Mind-Twister-Wizard/AffectBridge-UQ**

## Recommended code-and-data availability statement

> **Code and Data Availability.** The implementation and reproducibility materials for AffectBridge-UQ v2.1.0 are available at https://github.com/Mind-Twister-Wizard/AffectBridge-UQ. The repository contains source code, configuration files, training, calibration and ablation scripts, result tables, publication figures, label harmonization, WavLM feature caching, duplicate-content checks, and reproducibility materials. Public datasets should be obtained from their original providers or Kaggle mirrors under the applicable licenses.

## Important provenance wording

The repository contains archived numerical results from the documented development run. Those results predate the v2.1 duplicate-content guard and reused one source-validation partition for both early stopping and post-hoc calibration. They should therefore be described as **documented development-run results**, not as results of the corrected confirmatory protocol.

A confirmatory archival rerun should use the duplicate-safe manifest, disjoint source-side model-selection/calibration partitions where feasible, the same four folds and three seeds, and matched pre/post temperature-scaling ECE.

## Suggested repository citation

Raj, A., Al-Saeedi, B. A. K., Vaidya, P., & Rana, A. (2026). *AffectBridge-UQ: Uncertainty-Aware Domain-Generalized Cross-Corpus Speech Emotion Recognition with Selective Human-Review Deferral* (Version 2.1.0) [Computer software]. https://github.com/Mind-Twister-Wizard/AffectBridge-UQ

## BibTeX

```bibtex
@software{affectbridge_uq_2026,
  author  = {Alok Raj and Bashar Aswad Kokaz Al-Saeedi and Pankaj Vaidya and Anurag Rana},
  title   = {AffectBridge-UQ: Uncertainty-Aware Domain-Generalized Cross-Corpus Speech Emotion Recognition with Selective Human-Review Deferral},
  year    = {2026},
  version = {2.1.0},
  url     = {https://github.com/Mind-Twister-Wizard/AffectBridge-UQ},
  license = {MIT}
}
```

After creating a Zenodo release, its DOI can be added to both the manuscript and `CITATION.cff` without changing the software content.
