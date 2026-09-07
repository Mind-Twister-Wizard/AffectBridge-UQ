# Data Provenance and RAVDESS Duplicate Audit

The four corpora are downloaded through the Kaggle handles in `config.yaml` and parsed into one harmonized manifest.

## Canonical corpora

| Corpus | Canonical size | Protocol |
|---|---:|---|
| CREMA-D | 7,442 clips | six known emotions |
| RAVDESS | 1,440 speech files | six known + calm/surprise as target semantic unknowns |
| TESS | 2,800 | six known + surprise as target semantic unknown |
| SAVEE | 480 | six known + surprise as target semantic unknown |

## Post-run audit

The documented v2 result package contained **2,880 RAVDESS manifest rows** because the downloaded directory tree exposed two mirrored copies of the same speech-file set. This was a path-level duplication, not an additional set of speakers or recordings.

The numerical artifacts under `results/` are retained as the provenance of that completed experiment. They are **not retroactively relabeled as deduplicated results**. In source folds where RAVDESS participated in training, duplicated rows can modestly reweight optimization.

## v2.1 safeguard

The current code release adds **content-based duplicate guarding** during manifest construction so mirror paths do not enter future training manifests.

A future archival rerun should use the duplicate-safe v2.1 manifest if the study is extended or final numerical claims are revalidated.
