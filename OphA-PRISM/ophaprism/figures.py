from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from .common import private_output, ValidationError, identifiers, require_columns, configuration, save_json, binary
from .metrics import auc_details, wilson, decision_rows


def plot_predictions(source, destination, task, bootstrap=2000, seed=42):
    """Empirical figures and aggregate CSVs; never construct curves from scalar AUCs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = pd.read_csv(source, dtype={"patient_id": str})
    require_columns(data, ["patient_id", "y_true", "probability"])
    if "task" in data:
        data = data.loc[data.task == task]
    identifiers(data)
    y, p = binary(data.y_true).astype(int).to_numpy(), data.probability.to_numpy(float)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValidationError("complete_two_class_probability_file_required_for_plot")
    destination = private_output(destination)
    grid = np.linspace(0, 1, 101)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    # Bootstrap bands condition on fixed saved patient probabilities.
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    roc_samples = []
    for _ in range(bootstrap):
        ix = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        a, b, _ = roc_curve(y[ix], p[ix])
        roc_samples.append(np.interp(grid, a, b))
    bands = np.quantile(roc_samples, [.025, .975], axis=0) if roc_samples else None
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 450, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].plot(fpr, tpr, color="#176B87", label=f"AUC {auc_details(y, p)['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="0.65", lw=1)
    if bands is not None:
        axes[0].fill_between(grid, bands[0], bands[1], color="#176B87", alpha=.16)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[1].plot(recall, precision, color="#B55B26", label=f"Average precision {average_precision_score(y, p):.3f}")
    axes[1].axhline(y.mean(), color="0.65", linestyle="--", lw=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(frameon=False)
    # Explicit bins and Wilson uncertainty; do not label these a smoothed calibration curve.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 6)))
    bins = np.searchsorted(edges[1:-1], p, side="left")
    calibration = []
    for i in np.unique(bins):
        k = bins == i
        w = wilson(int(y[k].sum()), int(k.sum()))
        calibration.append({"mean_probability": float(p[k].mean()), "observed": float(y[k].mean()),
                            "n": int(k.sum()), "events": int(y[k].sum()), "lower": w["ci95"][0], "upper": w["ci95"][1]})
    cal = pd.DataFrame(calibration)
    axes[2].errorbar(cal.mean_probability, cal.observed,
        yerr=np.vstack([cal.observed-cal.lower, cal.upper-cal.observed]), fmt="o-", color="#416A59", capsize=3)
    axes[2].plot([0, 1], [0, 1], "--", color="0.65", lw=1)
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed proportion", xlim=(0, 1), ylim=(0, 1))
    for letter, ax in zip("ABC", axes):
        ax.text(-.12, 1.04, letter, transform=ax.transAxes, fontweight="bold")
    fig.suptitle(f"{task} · {len(y)} patients · {int(y.sum())} events")
    fig.savefig(destination / f"{task}_diagnostics.pdf")
    fig.savefig(destination / f"{task}_diagnostics.png")
    plt.close(fig)
    # Exact ROC/PR axes are aggregate coordinates. Individual probability thresholds are omitted.
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(destination / f"{task}_roc_coordinates.csv", index=False)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(destination / f"{task}_pr_coordinates.csv", index=False)
    cal.to_csv(destination / f"{task}_calibration_bins.csv", index=False)
    if bands is not None:
        pd.DataFrame({"fpr_grid": grid, "lower": bands[0], "upper": bands[1]}).to_csv(
            destination / f"{task}_roc_bands.csv", index=False)
    thresholds = configuration()["thresholds"][task]
    pd.DataFrame(decision_rows(y, p, thresholds)).to_csv(destination / f"{task}_decision_curve.csv", index=False)
    save_json(destination / f"{task}_figure_provenance.json", {
        "n": len(y), "events": int(y.sum()), "bootstrap": bootstrap,
        "interval_scope": "fixed-prediction patient-stratified percentile bands",
        "calibration": "five quantile bins with descriptive Wilson intervals; not smoothed calibration",
        "source": "actual saved patient predictions; no scalar-metric reconstruction",
        "publication_status": "empirical diagnostic panel; final manuscript layout requires author review"})
