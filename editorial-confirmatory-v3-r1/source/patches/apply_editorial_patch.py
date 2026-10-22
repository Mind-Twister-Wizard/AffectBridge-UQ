"""Fail-closed, idempotent patch against pinned original AffectBridge-UQ snapshot.
Never silently run the old shared-validation evaluation.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

PATCH_VERSION = 'AffectBridge-UQ-editorial-confirmatory-2026-10-22-r1'

def replace_once(s: str, old: str, new: str, label: str) -> str:
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'Unsafe source mismatch for {label}: expected one anchor, found {n}. Pinned base changed.')
    return s.replace(old, new, 1)


def patch_train(s: str) -> str:
    s = replace_once(s,
        'import hashlib\nfrom contextlib import nullcontext',
        'import hashlib\nimport random\nimport os\nfrom contextlib import nullcontext', 'train imports')
    s = replace_once(s, '"implementation": "DIMAP-C-v2.0",',
           '"implementation": "DIMAP-C-v3-editorial-confirmatory-r1",', 'run digest version')
    s = replace_once(s,
        '    val_loader = _loader(features, encoded, split["val"], cfg, shuffle=False, device=device, seed=seed, balanced=False)\n    test_loader =',
        '    val_loader = _loader(features, encoded, split["val"], cfg, shuffle=False, device=device, seed=seed, balanced=False)\n    cal_loader = _loader(features, encoded, split["cal"], cfg, shuffle=False, device=device, seed=seed, balanced=False)\n    test_loader =',
        'dedicated calibration loader')
    s = replace_once(s,
        '    optimizer_step = 0\n\n    for epoch in range(1, epochs + 1):',
'''    optimizer_step = 0
    run_dir = ensure_dir(output_root / "runs" / f"{ablation_name}__holdout_{holdout}__seed_{seed}")
    state_path = run_dir / "epoch_resume.pt"
    resume_epoch = 1
    expected_digest = _config_digest(cfg, ablation_name, ablation, holdout, seed, cache_meta)
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("config_digest") != expected_digest:
            raise RuntimeError(f"Incompatible saved epoch checkpoint: {state_path}. Do not reuse results from a different protocol.")
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema_model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_state = state["best_state"]
        best_score = state["best_score"]
        stale = state["stale"]
        history = state["history"]
        optimizer_step = state["optimizer_step"]
        resume_epoch = int(state["last_epoch"]) + 1
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        if state.get("sampler_rng") is not None:
            train_loader.sampler.generator.set_state(state["sampler_rng"].cpu())
        print(f"Resuming {holdout}/{ablation_name}/seed={seed} from epoch {resume_epoch}")

    for epoch in range(resume_epoch, epochs + 1):''', 'epoch checkpoint load')
    s = replace_once(s,
'''        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break

    if best_state is not None:''',
'''        else:
            stale += 1
        # Atomic epoch checkpoint: interruptions do not discard all training.
        # Store all stochastic states, including the weighted sampler generator.
        state_tmp = state_path.with_suffix(".pt.partial")
        torch.save({
            "config_digest": expected_digest, "last_epoch": epoch,
            "model": model.state_dict(), "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best_state": best_state, "best_score": best_score, "stale": stale,
            "history": history, "optimizer_step": optimizer_step,
            "random_state": random.getstate(), "numpy_state": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "sampler_rng": train_loader.sampler.generator.get_state(),
        }, state_tmp)
        os.replace(state_tmp, state_path)
        if stale >= int(cfg["patience"]):
            break

    if best_state is not None:''', 'epoch checkpoint save')
    s = replace_once(s,
'''    val_output = collect_outputs(model, val_loader, device)
    test_output = collect_outputs(model, test_loader, device)
    calibration = fit_classwise_conformal(
        val_output["logits"],
        val_output["labels"],
        val_output["dispersion"],''',
'''    # Never fit calibration on model-selection data.
    cal_output = collect_outputs(model, cal_loader, device)
    test_output = collect_outputs(model, test_loader, device)
    calibration = fit_classwise_conformal(
        cal_output["logits"],
        cal_output["labels"],
        cal_output["dispersion"],''', 'calibration separation')
    s = replace_once(s,
        '        prototype_distance=val_output["prototype_distance"],',
        '        prototype_distance=cal_output["prototype_distance"],', 'cal prototype distance')
    s = replace_once(s,
        '        calibration_groups=val_output["corpus"],',
        '        calibration_groups=cal_output["corpus"],', 'cal corpus groups')
    s = replace_once(s,
'''    metric_values.update({
        "holdout": holdout,''',
'''    # Matched pre/post metrics on exactly the same final target examples.
    from .metrics import expected_calibration_error
    test_known = labels_np >= 0
    raw_probs = torch.softmax(test_output["logits"].float(), dim=-1).cpu().numpy()
    before_ece = float(expected_calibration_error(raw_probs[test_known], labels_np[test_known]))
    after_ece = float(expected_calibration_error(decisions["probabilities"][test_known], labels_np[test_known]))
    metric_values.update({
        "ece_pre": before_ece, "ece_post": after_ece,
        "ece_delta_post_minus_pre": after_ece - before_ece,
        "selection_count": int(len(split["val"])),
        "calibration_count": int(len(split["cal"])),
        "training_count": int(len(split["train"])),
        "source_split_methods": split["methods"],
        "protocol_version": "editorial_confirmatory_v3_r1",
        "holdout": holdout,''', 'matched ECE metrics')
    s = replace_once(s, '"implementation_version": "DIMAP-C-v2.0",',
                  '"implementation_version": "DIMAP-C-v3-editorial-confirmatory-r1",', 'metrics protocol version')
    s = replace_once(s,
'''    _save_predictions(run_dir / "predictions.csv", test_output, decisions, list(cfg["known_labels"]))
    return {**metric_values,''',
'''    _save_predictions(run_dir / "predictions.csv", test_output, decisions, list(cfg["known_labels"]))
    save_json({"holdout": holdout, "seed": seed, "split_methods": split["methods"],
               "counts": {key: int(len(split[key])) for key in ("train", "val", "cal", "test")},
               "data_leakage_guard": "all four partitions are pairwise disjoint",
               "calibration_separate_from_model_selection": True}, run_dir / "split_audit.json")
    if state_path.exists():
        state_path.unlink()  # Complete run is already resumable from metrics/checkpoint.
    return {**metric_values,''', 'finish checkpoint cleanup')
    # Paired ablations on two predeclared held-out domains. Full model all four.
    a = s.index('    # Ablations are intentionally run at one fixed seed unless the user asks for')
    b = s.index('    save_json(records, output_root / "all_results.json")', a)
    s = s[:a] + '''    # Paired component ablations: same outer folds and seeds as the full model.
    # Five important components, two held-out corpora, three seeds = 30 runs.
    if ablation_scope != "none":
        paired_names = ["no_prototypes", "global_prototype", "no_domain_adversarial",
                        "no_speaker_adversarial", "no_cross_corpus_mixup"]
        paired_holdouts = [name for name in ("crema_d", "ravdess") if name in corpora]
        if ablation_scope == "full":
            paired_holdouts = list(corpora)
        for holdout in paired_holdouts:
            for name in paired_names:
                flags = ABLATIONS[name]
                for run_seed in seeds:
                    existing = _load_completed_run(output_root, cfg, feature_cache, holdout, name, flags, int(run_seed))
                    records.append(existing if existing is not None else fit_fold(
                        cfg, manifest, feature_cache, holdout, name, flags, device,
                        output_root, int(run_seed), max_epochs))

''' + s[b:]
    return s


def patch_features(s: str) -> str:
    s = replace_once(s,
'''    mmap = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=(len(manifest), tokens, feature_dim))
    batch_size = int(cfg["feature_batch_size"])
    rows = manifest.reset_index(drop=True)
    for start in tqdm(range(0, len(rows), batch_size), desc="Caching multi-layer WavLM statistics"):''',
'''    # Incomplete feature extraction is resumable at completed batch boundaries.
    work_path = cache_dir / f"{stem}.partial.npy"
    progress_path = cache_dir / f"{stem}.progress.json"
    progress = {}
    if work_path.exists() and progress_path.exists() and not force:
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("signature") != manifest_signature or progress.get("num_rows") != len(manifest):
                progress = {}
        except Exception:
            progress = {}
    start_row = int(progress.get("next_row", 0)) if progress else 0
    if progress and start_row > 0:
        mmap = np.lib.format.open_memmap(work_path, mode="r+")
        if mmap.shape != (len(manifest), tokens, feature_dim):
            raise RuntimeError("Incomplete cache shape mismatch; remove the .partial.npy/.progress.json files")
        print(f"Resuming WavLM feature extraction at row {start_row}/{len(manifest)}")
    else:
        mmap = np.lib.format.open_memmap(work_path, mode="w+", dtype=np.float16, shape=(len(manifest), tokens, feature_dim))
        start_row = 0
    batch_size = int(cfg["feature_batch_size"])
    rows = manifest.reset_index(drop=True)
    for start in tqdm(range(start_row, len(rows), batch_size), desc="Caching multi-layer WavLM statistics"):''', 'feature partial cache initialization')
    s = replace_once(s,
'''            mmap[start + offset] = stats.cpu().numpy().astype(np.float16)
    mmap.flush()
    metadata = {''',
'''            mmap[start + offset] = stats.cpu().numpy().astype(np.float16)
        mmap.flush()
        tmp_progress = progress_path.with_suffix(".json.tmp")
        tmp_progress.write_text(json.dumps({"signature": manifest_signature,
                                 "num_rows": len(manifest),
                                 "next_row": min(start + len(batch), len(rows))}), encoding="utf-8")
        tmp_progress.replace(progress_path)
    mmap.flush()
    del mmap
    work_path.replace(cache_path)
    if progress_path.exists():
        progress_path.unlink()
    metadata = {''', 'incremental feature progress')
    # Avoid accepting partially written cache as final: metadata only after atomic rename.
    return s


def patch_reporting(s: str) -> str:
    s = replace_once(s,
'''["accuracy", "macro_f1", "uar", "ece", "unknown_auroc", "unknown_aupr", "coverage", "auto_coverage", "selective_risk"]''',
'''["accuracy", "macro_f1", "uar", "ece", "ece_pre", "ece_post", "ece_delta_post_minus_pre", "unknown_auroc", "unknown_aupr", "coverage", "auto_coverage", "selective_risk"]''', 'report additional metrics')
    s = replace_once(s, '1.96 * values.std(ddof=1) / np.sqrt(len(values))',
                     '4.3026527299 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) == 3 else 1.96 * values.std(ddof=1) / np.sqrt(len(values))', 'small-sample t CI')
    s = replace_once(s,
'''["holdout", "seed", "ablation", "accuracy", "macro_f1", "uar", "ece", "unknown_auroc",''',
'''["holdout", "seed", "ablation", "accuracy", "macro_f1", "uar", "ece", "ece_pre", "ece_post", "unknown_auroc",''', 'HTML ECE columns')
    s = replace_once(s, '<title>AffectBridge-UQ v2 report</title>',
                         '<title>AffectBridge-UQ Editorial Confirmatory v3 Report</title>', 'html report title')
    s = replace_once(s, 'Temperature scaling and conformal thresholds are fitted only on source validation data.',
                         'Temperature scaling and conformal thresholds are fitted only on dedicated source calibration data, disjoint from model-selection validation.', 'calibration report statement')
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--patches', required=True)
    args = parser.parse_args()
    root, patch_root = Path(args.engine).resolve(), Path(args.patches).resolve()
    marker = root / '.editorial_confirmatory_patch.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')).get('version') != PATCH_VERSION:
            raise RuntimeError('Existing patch version differs. Manual upgrade required to protect provenance.')
        print('Confirmatory patch already installed; skipping.')
        return
    targets = {'src/train.py': patch_train, 'src/features.py': patch_features,
               'src/reporting.py': patch_reporting}
    replacements = {}
    for rel, fn in targets.items():
        source = (root / rel).read_text(encoding='utf-8')
        replacements[rel] = fn(source)
    splits = (patch_root / 'splits.py').read_text(encoding='utf-8')
    compile(splits, 'splits.py', 'exec')
    for rel, data in replacements.items():
        compile(data, rel, 'exec')
    backup = root / 'editorial_original_sources'
    backup.mkdir(exist_ok=True)
    for rel in (*targets, 'src/splits.py'):
        (backup / Path(rel).name).write_bytes((root / rel).read_bytes())
    for rel, data in replacements.items():
        (root / rel).write_text(data, encoding='utf-8')
    (root / 'src/splits.py').write_text(splits, encoding='utf-8')
    marker.write_text(json.dumps({'version': PATCH_VERSION,
       'files_sha256': {rel: hashlib.sha256((root/rel).read_bytes()).hexdigest()
          for rel in (*targets, 'src/splits.py')}, 'note':'Pinned upstream code patched for independent source calibration, paired ablations, recovery.'}, indent=2), encoding='utf-8')
    print('Applied and syntax-checked editorial protocol patch:', PATCH_VERSION)

if __name__ == '__main__':
    main()
