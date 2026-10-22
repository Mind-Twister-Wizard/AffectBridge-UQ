# Editorial confirmatory v3-r1 — experimental protocol

**Scope:** This is the corrected experiment referenced by the revised AffectBridge-UQ chapter; do not conflate it with top-level v2 code/results.

- Held-out corpus: in turn CREMA-D, RAVDESS, SAVEE, TESS. All target recordings are test-only, including the held-out semantic-unknown labels.
- Source training and model-selection and post-hoc calibration partitions are separate for each fold and seed. Source speaker-group separation is attempted where feasible; two-speaker TESS uses a recorded stratified utterance-level source fallback.
- Known emotion classes: anger, disgust, fear, happiness, sadness, neutral. RAVDESS calm and surprise, TESS pleasant surprise, and SAVEE surprise are held-out semantic unknowns for their corresponding targets.
- Deduplication: within-corpus SHA-256 before splitting, preserving metadata about discarded duplicate rows and conflicting emotion labels.
- Prespecified seeds: 42, 52, 62. Full model: 4 × 3 = 12 runs. Five controlled ablations on CREMA-D and RAVDESS across three seeds: 5 × 2 × 3 = 30 runs.
- Source-calibrated temperature scaling, classwise nonconformity quantiles (alpha=0.10), and a signed hybrid shift-risk score are fitted without using target labels.
- Frozen `microsoft/wavlm-base-plus` layers 4/8/12; eight segments with mean and standard-deviation statistics; 192-dimensional adapter; PyTorch 2.6.0+cu124 on an RTX 3050 Ti laptop GPU.
- Results are descriptive. Target conformal coverage is not guaranteed under shift, and no experiment tested whether human review improves decisions.

## Software provenance and honest reproduction boundary

The local corrected experiment was run using pinned upstream commit `6c043d12e40afce2790e8f11ad67a52153ad20ac` together with the v3-r1 editorial patch, not unmodified top-level v2 scripts. The exact model-changing patch and the Windows reproduction entry point are now published in [source/](source/README.md). The full per-run checkpoints and path-bearing artifacts are retained by the authors. Public code plus aggregate summaries do not constitute an independently completed 42-run replication, but reviewers can inspect and reproduce the experimental procedure with permitted datasets and supported hardware.

No licensed raw audio or large model checkpoints are included. Do not run `setup_and_run.bat` and cite its outputs as the corrected v3-r1 study without installing and verifying the protocol patch.
