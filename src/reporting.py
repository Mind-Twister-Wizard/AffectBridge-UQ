from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from .config import load_config
from .utils import ensure_dir, save_json

LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "neutral"]
PALETTE = {"blue": "#1f5f8b", "navy": "#0b2745", "gold": "#c18a00", "teal": "#008b8b", "red": "#ad3b32"}


def _figure_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / "figures")


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_training(history: list[dict[str, float]], title: str, path: Path) -> None:
    frame = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(frame["epoch"], frame["train_loss"], marker="o", color=PALETTE["navy"], label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    metric = "val_domain_macro_f1" if "val_domain_macro_f1" in frame else "val_macro_f1"
    ax2.plot(frame["epoch"], frame[metric], marker="s", color=PALETTE["gold"], label="source-domain validation macro-F1")
    ax2.set_ylabel("Validation macro-F1")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_calibration(predictions: pd.DataFrame, title: str, path: Path) -> None:
    known = predictions[predictions["true_label_id"] >= 0]
    if known.empty:
        return
    probs = known[[f"prob_{label}" for label in LABELS]].to_numpy()
    y = known["true_label_id"].to_numpy()
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, 11)
    xs, ys = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            xs.append(confidence[mask].mean())
            ys.append(correct[mask].mean())
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    ax.plot(xs, ys, "o-", color=PALETTE["teal"], label="temperature-scaled DIMAP-C")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_risk_coverage(predictions: pd.DataFrame, title: str, path: Path) -> None:
    known = predictions[predictions["true_label_id"] >= 0].copy()
    if known.empty:
        return
    rank_key = "unknown_score" if "unknown_score" in known else "max_probability"
    ascending = rank_key == "unknown_score"
    known = known.sort_values(rank_key, ascending=ascending)
    correctness = (known["prediction_id"].to_numpy() == known["true_label_id"].to_numpy()).astype(float)
    coverage = np.arange(1, len(known) + 1) / len(known)
    risk = 1 - np.cumsum(correctness) / np.arange(1, len(known) + 1)
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    ax.plot(coverage, risk, color=PALETTE["red"], linewidth=2)
    ax.set(xlabel="Coverage", ylabel="Selective risk", title=title, ylim=(0, 1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_unknown_roc(predictions: pd.DataFrame, title: str, path: Path) -> None:
    unknown = (predictions["true_label_id"].to_numpy() < 0).astype(int)
    if unknown.min() == unknown.max():
        return
    if "unknown_score" in predictions:
        score = predictions["unknown_score"].to_numpy()
    else:
        probabilities = predictions[[f"prob_{label}" for label in LABELS]].to_numpy()
        score = 1 - probabilities.max(axis=1)
    fpr, tpr, _ = roc_curve(unknown, score)
    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    ax.plot(fpr, tpr, color=PALETTE["blue"], linewidth=2, label="hybrid source-calibrated risk score")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_ablation(ablation_frame: pd.DataFrame, path: Path) -> None:
    if ablation_frame.empty or "ablation" not in ablation_frame or "macro_f1" not in ablation_frame:
        return
    grouped = ablation_frame.groupby("ablation", as_index=False)[["macro_f1", "ece"]].mean(numeric_only=True)
    x = np.arange(len(grouped))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - width / 2, grouped["macro_f1"], width, label="macro-F1", color=PALETTE["blue"])
    ax.bar(x + width / 2, grouped["ece"], width, label="ECE", color=PALETTE["gold"])
    ax.set_xticks(x, grouped["ablation"], rotation=28, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean score")
    ax.set_title("DIMAP-C v2 ablation study")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 8.2))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    boxes = [
        (0.25, 3.25, 1.5, 1.0, "16-kHz\nwaveform", "#d9edf7"),
        (2.0, 3.25, 1.9, 1.0, "Silence trim +\nfrozen WavLM", "#d9edf7"),
        (4.15, 3.25, 2.0, 1.0, "Layers 4/8/12\nsegment mean+std", "#e8f4e8"),
        (6.4, 3.25, 1.9, 1.0, "Dual-scale\ntemporal adapter", "#e8f4e8"),
        (8.6, 5.65, 2.1, 1.0, "Progressive corpus +\nspeaker adversarial", "#fff1cc"),
        (8.6, 3.25, 2.1, 1.0, "Global + corpus\nresidual prototypes", "#f7dfeb"),
        (8.6, 0.85, 2.1, 1.0, "Emotion classifier +\nadaptive prototype gate", "#f7dfeb"),
        (11.05, 4.45, 2.75, 1.0, "Temperature scaling +\nclasswise conformal sets", "#ffe6cc"),
        (11.05, 2.35, 2.75, 1.0, "Hybrid shift risk:\nconfidence + entropy +\nprototype distance", "#ffe6cc"),
        (11.05, 0.25, 2.75, 1.0, "Auto-accept | set |\nhuman review", "#ffe6cc"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=PALETTE["navy"], linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.6, weight="bold", color=PALETTE["navy"])
    arrows = [
        ((1.75, 3.75), (2.0, 3.75)), ((3.9, 3.75), (4.15, 3.75)), ((6.15, 3.75), (6.4, 3.75)),
        ((8.3, 3.75), (8.6, 3.75)), ((8.3, 3.75), (8.6, 6.15)), ((8.3, 3.75), (8.6, 1.35)),
        ((10.7, 3.75), (11.05, 4.95)), ((10.7, 3.75), (11.05, 2.85)), ((10.7, 1.35), (11.05, 0.75)),
        ((12.4, 4.45), (12.4, 3.35)), ((12.4, 2.35), (12.4, 1.25)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": PALETTE["navy"], "lw": 1.5})
    ax.text(7.1, 7.75, "AffectBridge-UQ / DIMAP-C v2", ha="center", fontsize=16, weight="bold", color=PALETTE["navy"])
    ax.text(7.1, 7.20, "Corpus-class balanced sampling + same-emotion cross-corpus mixup + boundary-entropy pseudo-OOD", ha="center", fontsize=10.2, color=PALETTE["red"])
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    labels = ["Download\nKaggle", "Audit + harmonize\nlabels", "Cache multi-layer\nWavLM once", "Balanced LODO\ntraining", "3-seed\nevaluation", "Calibrate +\nopen-set test", "Paper tables +\nfigures + report"]
    colors = ["#d9edf7", "#e8f4e8", "#e8f4e8", "#f7dfeb", "#f7dfeb", "#ffe6cc", "#fff1cc"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.18 + i * 1.82
        ax.add_patch(plt.Rectangle((x, 1.65), 1.40, 1.15, facecolor=color, edgecolor=PALETTE["navy"], linewidth=1.5))
        ax.text(x + 0.70, 2.225, label, ha="center", va="center", fontsize=9.2, weight="bold")
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + 1.75, 2.225), xytext=(x + 1.40, 2.225), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": PALETTE["navy"]})
    ax.text(6.5, 3.85, "AffectBridge-UQ automated paper experiment flow", ha="center", fontsize=15, weight="bold", color=PALETTE["navy"])
    ax.text(6.5, 0.80, "Held-out target corpus is not used for training, model selection, temperature fitting, or conformal calibration", ha="center", fontsize=10, color=PALETTE["teal"])
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No results were generated."
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float):
                cells.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _aggregate_full(full_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in ["accuracy", "macro_f1", "uar", "ece", "unknown_auroc", "unknown_aupr", "coverage", "auto_coverage", "selective_risk"] if c in full_frame]
    rows: list[dict[str, Any]] = []
    for holdout, group in full_frame.groupby("holdout"):
        row: dict[str, Any] = {"holdout": holdout, "n_seeds": int(group["seed"].nunique()) if "seed" in group else len(group)}
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else float("nan")
            row[f"{metric}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0 if len(values) == 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(cfg: dict[str, Any], manifest: pd.DataFrame, records: list[dict[str, Any]], output_root: str | Path) -> Path:
    output_root = ensure_dir(output_root)
    fig_dir = _figure_dir(output_root)
    full_records = [r for r in records if r.get("ablation") == "full_DIMAP-C"]
    ablation_records = [r for r in records if r.get("ablation") != "full_DIMAP-C"]
    records_frame = pd.DataFrame([{k: v for k, v in record.items() if not isinstance(v, (list, dict))} for record in records])
    records_frame.to_csv(output_root / "results_summary.csv", index=False)
    full_frame = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in full_records])
    abl_frame = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in ablation_records])
    aggregate = _aggregate_full(full_frame) if not full_frame.empty else pd.DataFrame()
    aggregate.to_csv(output_root / "paper_main_results.csv", index=False)
    (output_root / "paper_main_results.md").write_text(_markdown_table(aggregate), encoding="utf-8")
    (output_root / "results_summary.md").write_text(_markdown_table(full_frame), encoding="utf-8")
    (output_root / "ablation_results.md").write_text(_markdown_table(abl_frame), encoding="utf-8")
    if not abl_frame.empty:
        abl_frame.to_csv(output_root / "ablation_results.csv", index=False)
    plot_ablation(abl_frame, fig_dir / "ablation_comparison.png")
    manifest.groupby(["corpus", "label"], dropna=False).size().reset_index(name="count").to_csv(output_root / "dataset_summary.csv", index=False)
    draw_architecture(fig_dir / "architecture.png")
    draw_flowchart(fig_dir / "pipeline_flowchart.png")

    aggregate_cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    # One representative seed per target for per-fold figures; all seeds remain
    # in CSV/JSON tables and confidence intervals.
    representative: dict[str, dict[str, Any]] = {}
    for record in full_records:
        holdout = str(record["holdout"])
        if holdout not in representative or int(record.get("seed", 10**9)) < int(representative[holdout].get("seed", 10**9)):
            representative[holdout] = record
        aggregate_cm += np.asarray(record.get("confusion_matrix", np.zeros_like(aggregate_cm)))
    for holdout, record in representative.items():
        run_dir = Path(record["run_dir"])
        history_path = run_dir / "history.json"
        prediction_path = run_dir / "predictions.csv"
        if history_path.exists():
            plot_training(json.loads(history_path.read_text(encoding="utf-8")), f"Training: held-out {holdout}", fig_dir / f"training_{holdout}.png")
        if prediction_path.exists():
            predictions = pd.read_csv(prediction_path)
            plot_calibration(predictions, f"Calibration: held-out {holdout}", fig_dir / f"calibration_{holdout}.png")
            plot_risk_coverage(predictions, f"Risk-coverage: held-out {holdout}", fig_dir / f"risk_coverage_{holdout}.png")
            plot_unknown_roc(predictions, f"Unknown-emotion ROC: held-out {holdout}", fig_dir / f"unknown_roc_{holdout}.png")
            plot_confusion(np.asarray(record["confusion_matrix"]), f"Confusion matrix: held-out {holdout}", fig_dir / f"confusion_{holdout}.png")
    plot_confusion(aggregate_cm, "Aggregate known-emotion confusion matrix across seeds", fig_dir / "confusion_aggregate.png")

    save_json({"records": records, "config": cfg}, output_root / "run_manifest.json")
    metric_cols = [c for c in ["holdout", "seed", "ablation", "accuracy", "macro_f1", "uar", "ece", "unknown_auroc", "unknown_aupr", "coverage", "auto_coverage", "selective_risk", "temperature"] if c in records_frame]
    metric_html = records_frame[metric_cols].to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)) if not records_frame.empty else "<p>No results.</p>"
    aggregate_html = aggregate.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)) if not aggregate.empty else "<p>No aggregate results.</p>"
    dataset_html = manifest.groupby(["corpus", "label"], dropna=False).size().reset_index(name="count").to_html(index=False)
    figure_tags = []
    for image in sorted(fig_dir.glob("*.png")):
        figure_tags.append(f'<figure><img src="figures/{html.escape(image.name)}" alt="{html.escape(image.stem)}"><figcaption>{html.escape(image.stem.replace("_", " "))}</figcaption></figure>')
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>AffectBridge-UQ v2 report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1220px;margin:2rem auto;color:#17324d;line-height:1.45}}h1,h2{{color:#0b4774}}table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.86rem}}th,td{{border:1px solid #b7c5d1;padding:.4rem;text-align:left}}th{{background:#1f5f8b;color:white}}figure{{display:inline-block;vertical-align:top;width:46%;margin:1rem}}figure img{{width:100%;border:1px solid #ccd6df}}figcaption{{text-align:center;color:#526b7f;font-size:.9rem}}.callout{{background:#edf5fa;border-left:5px solid #1f5f8b;padding:1rem}}</style></head>
<body><h1>AffectBridge-UQ / DIMAP-C v2 experiment report</h1>
<p class='callout'><b>Protocol boundary:</b> the held-out target corpus is not used for model training, early stopping, temperature fitting, conformal calibration, or risk-score calibration. Calm/surprise remain excluded from the six-class emotion learner. The implementation estimates displayed vocal affect; it is not a diagnostic measure of private emotional state.</p>
<h2>Dataset manifest</h2>{dataset_html}
<h2>Paper main table: mean, SD, and 95% CI across seeds</h2>{aggregate_html}
<h2>All runs</h2>{metric_html}
<h2>Figures</h2>{''.join(figure_tags)}
<h2>Reproducibility notes</h2><ul>
<li>Four leave-one-dataset-out folds are used when all corpora are present.</li>
<li>Multi-layer WavLM features are cached once using a stable content signature, so an unchanged dataset is not needlessly re-encoded.</li>
<li>Source training uses corpus-class balancing and source-only cross-corpus feature mixup; target utterances are never used for adaptation.</li>
<li>Temperature scaling and conformal thresholds are fitted only on source validation data.</li>
<li>Nominal conformal guarantees require exchangeability; target-corpus coverage is therefore reported empirically rather than claimed as guaranteed.</li>
</ul></body></html>"""
    report_path = output_root / "final_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tables, figures, diagrams, and an HTML report.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(Path(cfg["metadata_dir"]) / "manifest.csv")
    records = json.loads((Path(cfg["output_dir"]) / "all_results.json").read_text(encoding="utf-8"))
    print(write_report(cfg, manifest, records, cfg["output_dir"]))


if __name__ == "__main__":
    main()
