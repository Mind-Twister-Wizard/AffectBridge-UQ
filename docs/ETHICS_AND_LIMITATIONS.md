# Ethics, Limitations and Intended Use

AffectBridge-UQ estimates **displayed vocal affect** from acted/elicited emotional speech. It should not be presented as a direct measurement of private emotion, intention, sincerity, mental-health state, or diagnosis.

## Intended research use

Appropriate research contexts include:

- cross-corpus SER benchmarking;
- uncertainty-aware selective prediction;
- source-only domain-generalization studies;
- human-deferral policy research;
- low-stakes affect-aware interface research;
- reproducibility studies for calibration and conformal prediction under shift.

## No deployment claim

No live deployment or user study was conducted. The three-way policy illustrated in the manuscript—auto-accept, prediction set, or human review—is a **conceptual deployment framework**, not a validated operational workflow.

## Not intended for

- medical or mental-health diagnosis;
- employment, insurance, credit, or law-enforcement decisions;
- lie detection;
- covert emotional surveillance;
- high-impact automated decisions about individuals.

## Key limitations

- All evaluated corpora are English-language and primarily acted/elicited; external validity to spontaneous conversational affect is unknown.
- Corpus identity is confounded with speaker demographics, especially because TESS contains two female speakers and SAVEE four male speakers. The current design cannot cleanly separate corpus shift from demographic shift.
- The documented development run contains the disclosed mirrored-RAVDESS provenance issue. The v2.1 code removes exact duplicate content before splitting, but the archived numerical tables were not produced by that corrected manifest.
- The development run reused source-validation data for both model selection and post-hoc calibration. This avoids target leakage but couples selection and calibration; a confirmatory rerun should use disjoint source partitions where feasible.
- Only three full-model seeds were used.
- Default ablations are representative single-run comparisons using different seeds and are not paired significance tests.
- WavLM layers 4, 8, and 12 were selected as lower/middle/higher snapshots without a dedicated layer-selection ablation.
- Semantic-unknown AUROC is near chance on some targets, so the semantic-novelty detector is exploratory rather than solved open-set SER.
- Source-calibrated conformal coverage does not automatically transfer to shifted targets; target coverage is empirical under corpus shift.
- Archived ECE values are post-temperature-scaling only. Without matched pre-scaling ECE, the development run does not establish that temperature scaling numerically improved calibration.
- Human affect is mixed, contextual, culturally shaped, and not exhausted by the six categorical labels used here.

## Safeguard principle

Predicted vocal affect should remain a fallible contextual cue rather than an objective fact about a person. Uncertainty must remain visible to downstream decision logic, dataset licenses and consent conditions must be respected, and emotion predictions should never be the sole basis for sensitive decisions.
