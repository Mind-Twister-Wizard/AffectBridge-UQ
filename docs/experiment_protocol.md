# AffectBridge-UQ v2 Experiment Protocol

## Primary task

The study evaluates **leave-one-dataset-out (LODO) cross-corpus speech emotion recognition** over CREMA-D, RAVDESS, TESS, and SAVEE. The six shared known labels are:

`anger, disgust, fear, happiness, sadness, neutral`.

Calm and surprise are not merged into neutral. When present in a held-out target corpus they are treated as semantic unknowns. For TESS, the source label **pleasant surprise** (often encoded as `ps`) is explicitly parsed and normalized to the protocol label `surprise`.

## Label mapping

| Corpus | Known-label mapping | Semantic-unknown mapping |
|---|---|---|
| CREMA-D | `ANG→anger`; `DIS→disgust`; `FEA→fear`; `HAP→happiness`; `SAD→sadness`; `NEU→neutral` | none |
| RAVDESS | `05→anger`; `07→disgust`; `06→fear`; `03→happiness`; `04→sadness`; `01→neutral` | `02→calm`; `08→surprise` |
| TESS | `angry→anger`; `disgust→disgust`; `fear→fear`; `happy→happiness`; `sad→sadness`; `neutral→neutral` | `ps / pleasant surprise→surprise` |
| SAVEE | `a→anger`; `d→disgust`; `f→fear`; `h→happiness`; `sa→sadness`; `n→neutral` | `su→surprise` |

## Documented development-run protocol

For outer fold `k`, corpus `D_k` is reserved as the target test corpus and the other three corpora form the source pool. In the documented v2 development run:

- the target corpus was excluded from optimization, early stopping, temperature fitting, conformal calibration, and risk-threshold fitting;
- source validation was created separately within each source corpus;
- speaker-grouped validation was used when feasible;
- TESS used a stratified utterance-level fallback because it contains only two speakers;
- early stopping used the **mean macro-F1 across source validation corpora**;
- the same source-validation partition was also reused for temperature fitting and conformal/risk calibration;
- full-model runs used seeds `42`, `52`, and `62`.

The target corpus therefore remained unseen until final evaluation, but source model selection and post-hoc calibration were not separated. The reported numerical results are retained as the documented development-run results under this shared-validation protocol.

## RAVDESS provenance issue

A post-run audit found two mirrored copies of the canonical RAVDESS speech-file set, giving 2,880 manifest rows for 1,440 unique speech files. Because the outer split is at corpus level, identical RAVDESS files did not cross the target/source corpus boundary. However, when RAVDESS served as a source corpus, duplicate files could be allocated to both source-training and source-validation subsets, reweighting training and potentially influencing selection/calibration.

The v2.1 manifest builder now computes content hashes and removes exact within-corpus duplicates **before any split is constructed**. The archived result tables are not retroactively relabeled as deduplicated results.

## Confirmatory protocol recommended for archival publication

The professor-reviewed manuscript recommends a confirmatory rerun with the corrected manifest and **disjoint source-side partitions**:

1. **Training partition** — parameter optimization.
2. **Model-selection validation partition** — early stopping and hyperparameter selection.
3. **Calibration partition** — temperature scaling, classwise conformal quantiles, and shift-risk threshold fitting.
4. **Held-out target corpus** — final evaluation only.

Speaker grouping should be used whenever feasible. For corpora with too few speakers to support three disjoint speaker groups, the fallback should be explicitly reported and kept source-side only.

A nominal example such as 70% training / 15% validation / 15% calibration is acceptable only if implemented without compromising class support; exact split fractions should be reported with the rerun.

The confirmatory rerun should preserve:

- all four LODO folds;
- seeds `42`, `52`, and `62`;
- the same model architecture and hyperparameters unless a change is explicitly declared;
- no target-domain tuning;
- matched **pre-temperature vs post-temperature ECE**;
- preferably paired multi-seed ablations.

## WavLM representation

Hidden layers 4, 8, and 12 are used as lower-, middle-, and higher-level snapshots. Each layer is divided into eight temporal segments; mean and standard deviation statistics are retained per segment. With WavLM-base hidden size 768, three layers produce `768 × 2 × 3 = 4,608` values per segment.

No dedicated layer-selection ablation has been completed, so the `4/8/12` combination is a design choice, not an empirically proven optimum.

## Source-only feature-space augmentation

### Same-emotion cross-corpus mixup

For an eligible sample `i`, choose `j` with the same class and a different source corpus. Retain approximately half of eligible pairs, draw `λ ~ Beta(0.6,0.6)`, rescale `λ` to `[0.2,0.8]`, and mix cached feature tensors:

`x_mix = λx_i + (1-λ)x_j`.

The shared emotion label is retained. The mixup cross-entropy term has weight `0.18`.

### Boundary pseudo-OOD

Choose `j` with a different emotion and different source corpus. Retain approximately half of eligible pairs, draw `λ ~ Beta(4,4)`, rescale it to `[0.3,0.7]`, and form:

`x_ood = λx_i + (1-λ)x_j`.

No class label is assigned. The prediction is optimized toward uniformity with `0.025 × KL(p || Uniform)`. Supervised losses remain active on the original batch, preventing uniform collapse on in-distribution data.

## Calibration and conformal prediction

Temperature scaling is fitted on source-validation logits by minimizing negative log-likelihood. For class `c`, the nonconformity score is `a_i = 1-p_T(c|x_i)` for calibration samples with `y_i=c`.

For `n_c` class-specific scores and `α=0.10`, the implementation uses:

`τ_c = min{1, ceil((n_c+1)(1-α))/n_c}`

and the NumPy **higher** empirical quantile convention:

`q_c = Q_higher(τ_c)`.

The prediction set is `Γ(x) = {c : p_T(c|x) >= 1-q_c}`.

Standard split-conformal finite-sample validity depends on exchangeability. Under a deliberately shifted target corpus, target coverage is therefore reported empirically rather than claimed as guaranteed.

## Signed shift-risk score

The score combines:

- `1 - maximum calibrated probability`;
- normalized entropy;
- nearest global-prototype distance;
- cross-corpus prototype dispersion.

Each component is standardized using its source-validation median and robust MAD scale. Fixed weights are `(1.00, 0.55, 0.85, 0.20)`. They are source-side engineering coefficients held constant across folds and seeds; no sensitivity analysis or target tuning is claimed.

Negative total scores are allowed. Scores are not clipped or transformed. With source corpora as calibration groups, the threshold is the maximum 98th percentile across source groups:

`τ_R = max_d Q_0.98({R_i : corpus_i=d})`.

## Metrics

Known-emotion metrics:

- accuracy;
- macro-F1;
- UAR / balanced accuracy;
- NLL;
- Brier score;
- ECE;
- prediction-set coverage and average set size;
- singleton rate;
- auto-accept coverage;
- selective risk and selective macro-F1.

Open-set metrics when semantic unknowns exist:

- unknown AUROC;
- unknown AUPR;
- FPR@95%TPR.

CREMA-D has no protocol unknown class, so unknown-emotion AUROC is undefined for that target fold.

## Reporting boundary

The model estimates **displayed vocal affect** in English acted/elicited corpora. It should not be presented as a direct measurement of private emotion, intention, sincerity, mental-health state, or diagnosis. No live deployment or user study was conducted.
