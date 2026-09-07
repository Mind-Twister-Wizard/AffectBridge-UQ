# Performance upgrade notes: exploratory v1 -> DIMAP-C v2

The existing exploratory run is useful because it exposes exactly where the original implementation is weak. The full-model macro-F1 values were approximately 0.281 (CREMA-D), 0.491 (RAVDESS), 0.086 (TESS), and 0.457 (SAVEE). The full model also underperformed the `no_prototypes` ablation on CREMA-D, target ECE was high on TESS, and unknown-emotion AUROC was below chance on several corpora.

v2 does **not** try to make those numbers look better by changing the test set or using target labels. Instead, it fixes model/training weaknesses that can legitimately improve cross-corpus transfer.

## Highest-impact changes

### 1. Corpus domination is removed
The original source pool is highly imbalanced by corpus. A pooled random loader therefore lets a large corpus dominate every epoch. v2 uses inverse corpus-class frequency sampling so each source corpus and emotion receives meaningful optimization weight.

### 2. Multi-prototypes are regularized rather than free-floating
The original corpus prototypes were independent trainable tensors. That can preserve corpus idiosyncrasies instead of bridging them. v2 uses a shared global emotion prototype plus a bounded corpus residual, with explicit consistency and alignment losses.

### 3. Prototypes and gradient reversal are progressive
The original model applies randomly initialized prototypes and domain reversal immediately. v2 warms up the emotion space and progressively increases both influences.

### 4. The representation is richer without full WavLM fine-tuning
One frozen WavLM forward now retains statistics from layers 4, 8, and 12. This gives the lightweight adapter access to low/mid/high-level SSL information while keeping memory and training time suitable for a 4 GB GPU.

### 5. Domain interpolation is source-only
Same-emotion samples from different source corpora are mixed in feature space. This directly trains the classifier on points between source domains and does not require target data.

### 6. Open-set calibration no longer depends on prototype dispersion alone
The v1 unknown score was a simple confidence-plus-dispersion heuristic and produced weak AUROC. v2 combines four source-calibrated signals and reports AUROC, AUPR, and FPR@95%TPR.

### 7. Calibration is explicitly optimized
A scalar temperature is fitted on source validation logits before conformal sets are constructed. This is intended to address the very high target ECE observed in the exploratory report without touching target labels.

### 8. A parser/cache integrity problem is removed
TESS pleasant surprise is explicitly recognized, unparsed audio is rejected, and feature-cache validity is based on a content signature rather than the timestamp of a newly rewritten manifest. Therefore the same unchanged dataset is not needlessly re-encoded on every rerun.

## What v2 can and cannot promise

The architecture is materially stronger and the package is designed to attack the observed failure modes, but no code change can honestly guarantee a particular target macro-F1 before the real experiment is run. Treat the next full run as the first paper-quality v2 experiment, not as a guaranteed score.
