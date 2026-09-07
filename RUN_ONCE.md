# Run this version once

## Recommended upgrade from your current AffectBridge-UQ folder

1. Keep your current `data/` directory and `.venv/` if they already work.
2. Copy/merge the files from this package over the current `AffectBridge-UQ` folder.
3. Double-click `setup_and_run.bat`.

The first v2 run creates a **new multi-layer WavLM cache** because v2 intentionally uses layers 4, 8, and 12. That one-time re-encoding is required for the stronger representation. After it exists, the cache uses a stable content signature and will be reused.

Training is resumable. Every completed fold/seed has a configuration digest. If the process stops later, rerunning the launcher skips completed compatible runs instead of training them again.

## What to open at the end

- `outputs/final_report.html` - complete experiment report
- `outputs/paper_main_results.csv` - mean, SD, and 95% CI across the three full-model seeds
- `outputs/results_summary.csv` - all seed-level metrics
- `outputs/ablation_results.csv` - baselines and ablations
- `outputs/figures/` - publication figures

## Important

Do not compare the new v2 numbers to the old report until the whole v2 pipeline completes. The feature representation, sampler, prototype geometry, calibration, and unknown score are all different.
