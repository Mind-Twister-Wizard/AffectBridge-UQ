# Package validation status

## Software validation

Validation performed on the packaged implementation included:

- Python syntax compilation: **passed** (`python -m compileall`).
- Automated smoke tests: **4 passed**.
  - DIMAP-C forward/output shapes.
  - FP16-safe supervised contrastive loss.
  - CREMA-D/RAVDESS/TESS/SAVEE label parser checks, including TESS `ps` → protocol label `surprise`.
  - Temperature/conformal/hybrid-risk calibration produces finite outputs.
- Synthetic end-to-end pipeline: **passed** for training, calibration, target evaluation, CSV output, figures, and HTML reporting.
- Synthetic separable-data sanity check: the full model reached perfect source-validation and held-out known-emotion classification after training, confirming that the optimization path can learn rather than remaining dominated by EMA initialization.

## Real-data status

A complete real-data DIMAP-C v2 development run was later executed and its curated result artifacts are stored under `results/`. Those archived results use four leave-one-dataset-out folds and three full-model seeds.

A subsequent audit identified mirrored RAVDESS audio in the development-run manifest. The current v2.1 code removes exact within-corpus duplicates by content hash before split construction. The archived numerical results are retained for provenance and are not retroactively relabeled as deduplicated results.

The professor-reviewed manuscript also notes that the development run reused one source-validation partition for both model selection and post-hoc calibration. A stronger confirmatory rerun should separate these source-side roles where feasible and should report matched pre- and post-temperature-scaling ECE.

See:

- `docs/DATA_PROVENANCE.md`
- `docs/experiment_protocol.md`
- `docs/REPRODUCIBILITY.md`
- `docs/PROFESSOR_REVIEW_ALIGNMENT.md`
