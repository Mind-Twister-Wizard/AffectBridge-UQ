# Professor-Review Alignment

This note records how the repository documentation was aligned with the professor-reviewed manuscript revision.

## Incorporated clarifications

- Revised manuscript title: **AffectBridge-UQ: Uncertainty-Aware Domain-Generalized Cross-Corpus Speech Emotion Recognition with Selective Human-Review Deferral**.
- Affiliation normalized to **Shoolini University**, Solan, Himachal Pradesh, India.
- Predictive uncertainty, distributional shift, and semantic novelty are explicitly distinguished.
- Contribution claims are phrased conservatively; the repository does not claim state-of-the-art performance or validated deployment.
- WavLM layers 4/8/12 are described as lower/middle/higher design snapshots, with no claim that this combination is empirically optimal.
- The hierarchical prototype formulation is explicitly distinguished from local utterance-level prototype mapping.
- Standard conformal validity is distinguished from empirical target-domain behavior under corpus shift.
- Label harmonization is documented explicitly, including TESS **pleasant surprise** (`ps`) → protocol label `surprise`.
- RAVDESS mirror duplication is described as a development-run provenance issue rather than silently corrected in historical results.
- The shared source-validation role in the documented run is disclosed: it was used for both early stopping and post-hoc calibration.
- Method equations are defined in `ARCHITECTURE.md`, including the adapter embedding, hierarchical prototypes, prototype fusion, full loss coefficients, exact conformal quantile convention, signed shift score, and mathematical 98th-percentile threshold.
- Same-emotion mixup and boundary pseudo-OOD generation are documented algorithmically.
- The fixed shift-score weights `(1.00, 0.55, 0.85, 0.20)` are described as source-side engineering coefficients, not target-tuned optima.
- Negative signed shift scores are explicitly permitted and not clipped.
- No live deployment or user study is claimed.
- Archived ECE values are labeled as **post-temperature-scaling** only; no pre/post calibration improvement claim is made because matched pre-scaling ECE was not retained.

## Items that require new experiments rather than documentation edits

The professor review identified several items that cannot be resolved by rewriting historical results:

1. **Complete deduplicated rerun** — four LODO folds × three seeds using the duplicate-safe RAVDESS manifest.
2. **Disjoint source-side model selection and calibration** — separate optimization, early-stopping, and calibration data where feasible.
3. **Pre- vs post-temperature ECE** — retain and report both values in the confirmatory run.
4. **Layer-selection ablation** — compare layers 4/8/12 against individual layers or alternative combinations if the final chapter wishes to claim that the selection is empirically optimal.
5. **Paired multi-seed ablations** — recommended for stronger component-level inference.
6. **Computational profiling** — peak GPU memory, training time per fold, inference time per utterance, trainable parameter count, and feature-cache size if practical.

Until those experiments are completed, the curated `results/` directory is explicitly described as a **documented development run** and retained for traceability.

## Relevant files

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/experiment_protocol.md`
- `docs/DATA_PROVENANCE.md`
- `docs/REPRODUCIBILITY.md`
- `docs/ETHICS_AND_LIMITATIONS.md`
- `results/README.md`
- `CITATION.cff`
