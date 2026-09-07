# AffectBridge-UQ v2 experiment protocol

## Primary task

The primary experiment is **strict leave-one-dataset-out (LODO) cross-corpus speech emotion recognition** over four corpora: CREMA-D, RAVDESS, TESS, and SAVEE. The six shared known labels are anger, disgust, fear, happiness, sadness, and neutral. Calm and surprise are retained as semantic unknowns and are never merged into neutral.

For each outer fold, the held-out target corpus is excluded from training, early stopping, temperature fitting, conformal calibration, and risk-score calibration. This is a domain-generalization protocol, not target-domain adaptation.

## Why v2 changes the exploratory implementation

The exploratory report showed large cross-corpus imbalance and uneven generalization, with the strongest failures on TESS. v2 therefore targets the causes rather than simply increasing epochs:

1. **Dataset/parser audit.** TESS `ps` / pleasant-surprise naming is parsed explicitly; unrecognized files are rejected instead of being silently treated as unknown emotion.
2. **Multi-layer WavLM evidence.** Segment-level mean and standard deviation statistics are concatenated from WavLM hidden layers 4, 8, and 12 rather than relying only on the final layer.
3. **Corpus-class balanced sampling.** Large corpora such as CREMA-D no longer dominate optimization.
4. **Hierarchical prototypes.** Each emotion has one global prototype plus bounded corpus-specific residual prototypes. This preserves multiple prototypes while regularizing them toward a common emotion geometry.
5. **Progressive invariance.** Prototype fusion and gradient reversal are introduced gradually rather than at full strength from epoch 1.
6. **Class-conditional source alignment.** Same-emotion centroids from different source corpora are aligned without forcing unlike emotions together.
7. **Cross-corpus same-emotion mixup.** Source features from the same class but different corpora are mixed to create domain-interpolated training examples.
8. **Boundary-entropy pseudo-OOD.** Different-emotion, different-corpus mixtures are encouraged to have high-entropy predictions. No real calm/surprise sample is exposed to the six-class learner.
9. **Temperature scaling before conformal prediction.** Source validation logits are temperature-scaled before prediction sets are formed.
10. **Hybrid unknown score.** Open-set risk combines low confidence, normalized entropy, global-prototype distance, and prototype dispersion. Thresholding is source-domain conservative.

## Splits and model selection

Source validation is created separately within every source corpus so early stopping is not dominated by the largest corpus. Speaker-grouped validation is used when the source corpus has enough speakers. For very small-speaker corpora such as TESS, a stratified utterance-level source-validation fallback is used to avoid discarding half of the corpus merely for validation. The held-out target corpus remains completely untouched regardless of this source-validation choice.

Early stopping uses the **mean macro-F1 across source validation corpora**, not a pooled score dominated by sample count.

## Repeated runs

The default paper run evaluates the full DIMAP-C v2 model with three fixed seeds: 42, 52, and 62. The report writes per-seed metrics plus mean, standard deviation, and approximate 95% confidence intervals per held-out corpus.

Representative ablations and baselines are run with one fixed seed by default to keep the one-click laptop run practical. `--ablation-scope full` runs every ablation on every held-out corpus.

## Baselines and ablations

The package compares the full method against:

- `source_only_adapter`: frozen WavLM + temporal adapter + classifier only;
- `DANN_adapter`: source-only adapter plus corpus adversarial learning;
- `no_prototypes`;
- `global_prototype`;
- `no_domain_adversarial`;
- `no_speaker_adversarial`;
- `no_cross_corpus_mixup`;
- `no_conformal_deferral`.

## Metrics

Known-emotion metrics:

- Accuracy
- Macro-F1
- UAR / balanced accuracy
- Negative log-likelihood
- Brier score
- Expected calibration error (ECE)
- Prediction-set coverage
- Singleton rate
- Auto-accept coverage
- Selective risk and selective macro-F1

Open-set metrics when the target corpus contains semantic unknowns:

- Unknown AUROC
- Unknown AUPR
- FPR@95%TPR

CREMA-D has no calm/surprise class in the standard label set, so unknown-emotion AUROC is expected to be undefined for that target fold.

## Statistical reporting

Do not choose the best target seed. Report all three full-model seeds and their aggregate statistics. Do not tune hyperparameters, temperature, conformal thresholds, or unknown-risk weights against held-out target labels.

## Interpretation boundary

The model estimates **displayed vocal affect** in acted/elicited corpora. It should not be described as a direct measurement of private emotion, mental health, or diagnosis. Conformal coverage is source-calibrated; under distribution shift, target coverage is measured empirically rather than claimed as guaranteed.
