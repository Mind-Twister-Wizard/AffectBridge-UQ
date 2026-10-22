# AffectBridge-UQ v3-r1 — publicly released confirmatory code

This directory provides all **scientific model-changing source corrections** used in the October 2026 confirmatory experiment and a reproduction launcher. It is a *versioned overlay* applied to the complete historical upstream source pinned at commit `6c043d12e40afce2790e8f11ad67a52153ad20ac`, rather than a misleading replacement of older source files and results. The public patch is source-identical to the author's experiment copy (blob SHA verified), while the launcher and offline result-checker include post-experiment path-handling improvements.

## Contents

| File | Purpose |
|---|---|
| [patches/apply_editorial_patch.py](patches/apply_editorial_patch.py) | Fail-closed edits to `src/train.py`, `src/features.py` and `src/reporting.py` (independent calibration, three-seed paired ablations, ECE reporting, resumable checkpoints) |
| [patches/splits.py](patches/splits.py) | Source training/selection/calibration separation, explicit utterance fallback for speaker-sparse TESS, held-out target exclusion |
| [scripts/bootstrap_and_run.ps1](scripts/bootstrap_and_run.ps1) | Windows dependency bootstrap, downloads pinned historical base, installs patch, reruns experiment and validates |
| [RUN_FINAL_EXPERIMENT.bat](RUN_FINAL_EXPERIMENT.bat) | Windows launcher for a **fresh separate** reproduction |
| [scripts/prepare_config.py](scripts/prepare_config.py) | Three-seed config used for v3-r1; upstream `load_config` routes actual artifacts to `ENGINE/outputs` |
| [scripts/check_and_package.py](scripts/check_and_package.py) | Fail-closed post-run verification and paired result exports; output path corrected from historical launcher |
| [tests/test_protocol.py](tests/test_protocol.py) | Synthetic split tests for 4 folds × 3 seeds and invalid split fractions |

## Reproduce from source

1. Clone or download the public repository. Enter `editorial-confirmatory-v3-r1/source` in a **clean** directory. Do not run this over the author's already-completed local experiment tree.
2. On Windows 11 with a suitable NVIDIA GPU and Python 3.10–3.12 installed, double-click `RUN_FINAL_EXPERIMENT.bat`. It creates a private `.venv`, fetches base source at commit `6c043d12e40afce2790e8f11ad67a52153ad20ac`, applies the pinned source patch, installs dependencies, downloads configured corpora, and runs the 42 training/evaluation jobs. Windows, PyTorch, Hugging Face and Kaggle downloads must be reachable; dataset authentication or license acceptance may be needed.
3. For the independent partition tests alone, install `pytest numpy pandas scikit-learn` and run `python -m pytest -q tests/test_protocol.py`. These tests require no audio or GPU and do not constitute a full training reproduction.
4. Final experimental outputs are in `ENGINE/outputs/`; curated summary metrics from the author's original runs are in [../results/](../results/). The launcher was fixed to validate the actual upstream output directory rather than `outputs_editorial_confirmatory`.

## Integrity and caveats

- The source patch pinned by the author is publicly visible at [apply_editorial_patch.py](patches/apply_editorial_patch.py) and [splits.py](patches/splits.py). Original base source remains available via the upstream commit.
- The model-changing patch has not been silently altered to improve the manuscript's observed scores. Launcher and post-run validation fixes do **not** change numerical training code.
- The 42-run experiment was completed on the author's laptop and its summary artifacts were audited. Independent full GPU reproduction was **not** conducted as part of this GitHub publication, and numerical identity is not guaranteed across different dataset mirrors and environments.
- Complete output/checkpoint artifacts and licensed source audio are not part of the Git repository. Do not upload Kaggle API keys, pretrained weight caches, absolute private paths or licensed audio.
- Known limitation: CREMA-D contained three hash-identical recordings with conflicting labels; RAVDESS included mirrored source directories. Speaker-sparse TESS uses a documented source utterance-stratified fallback.
- Under target shift, observed conformal coverage is empirical, not guaranteed; semantic unknown AUROC was near chance. No human-review trial was conducted.

**Historical v2**: Do not cite the top-level [results/](../../results/README.md) directory for confirmatory performance; the [new result tables](../results/summary.csv) correspond to this v3-r1 protocol.
