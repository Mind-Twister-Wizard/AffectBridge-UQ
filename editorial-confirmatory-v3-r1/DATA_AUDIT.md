# Deduplicated data provenance for the confirmatory experiment

| Corpus | Downloaded path entries | SHA-256 duplicate rows dropped | SHA-256-unique retained |
|---|---:|---:|---:|
| CREMA-D | 7,442 | 3 | 7,439 |
| RAVDESS | 2,880 | 1,441 | 1,439 |
| TESS | 2,800 | 0 | 2,800 |
| SAVEE | 480 | 0 | 480 |
| **Total** | **13,602** | **1,444** | **12,158** |

The downloaded RAVDESS mirror had duplicated directory copies of nominal speech recordings and an additional repeated happiness recording with different take filenames. CREMA-D had three byte-identical pairs with **conflicting labels** (happiness/neutral, disgust/sadness, anger/fear). The deterministic hash guard retained one per conflict; it did **not** adjudicate the correct underlying emotion. These are limitations, not resolved annotation disagreements. The authors retain the sample-level manifest and hash audit; no original audio files are redistributed.

**Manuscript-reported manifest SHA-256:** `c861744099d045691568cd29d4b41c14c4d25af5dc946f929b54cd3ca4e3c845` (for verification against the author's original manifest export; no claim of regenerating that exact digest from this README).
