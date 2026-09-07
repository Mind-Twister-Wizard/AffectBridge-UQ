# Ethics, Limitations and Intended Use

AffectBridge-UQ estimates **displayed vocal affect** from acted/elicited emotional speech. It should not be presented as a direct measurement of private emotion, intention, sincerity, mental-health state or diagnosis.

## Intended use

Suitable research contexts include:

- cross-corpus SER benchmarking;
- affect-aware conversational interfaces;
- uncertainty-aware selective prediction;
- human-deferral design studies;
- reproducibility studies in domain generalization.

## Not intended for

- medical or mental-health diagnosis;
- employment, insurance, credit or law-enforcement decisions;
- lie detection;
- covert emotional surveillance;
- high-impact automated decisions about individuals.

## Key limitations

- all evaluated corpora are English-language and mostly acted/elicited;
- target-domain class behavior remains uneven;
- semantic-unknown AUROC is near chance on some targets;
- source-calibrated conformal coverage does not automatically transfer to shifted targets;
- default ablations are representative single-run comparisons;
- the documented v2 result package contains the disclosed RAVDESS mirror-duplication issue described in `DATA_PROVENANCE.md`.
