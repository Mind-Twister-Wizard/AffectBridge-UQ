# Contributing

Contributions that improve reproducibility, domain-generalization evaluation, calibration, open-set detection or documentation are welcome.

1. Create a feature branch.
2. Keep changes focused and documented.
3. Run `python -m compileall src run_all.py`.
4. Run the smoke test when dependencies are available.
5. Do not commit datasets, credentials, cached WavLM features, checkpoints or private data.
6. For metric changes, explain whether old and new results are directly comparable.
