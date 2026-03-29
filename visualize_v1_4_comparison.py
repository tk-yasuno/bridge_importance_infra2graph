"""
Create a compact visual summary for v1.4 experiment comparison.

Input:
  output/v1_4_experiment_comparison/v1_4_experiment_comparison.csv
Output:
  output/v1_4_experiment_comparison/v1_4_edge_info_noise_summary.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    csv_path = Path("output/v1_4_experiment_comparison/v1_4_experiment_comparison.csv")
    out_path = Path("output/v1_4_experiment_comparison/v1_4_edge_info_noise_summary.png")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing comparison CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Keep plotting labels compact and readable.
    label_map = {
        "Exp-1_baseline_graph": "Exp-1 baseline",
        "Exp-2_knn3": "Exp-2 kNN",
        "Exp-3_knn3_edge_attr": "Exp-3 edge_attr",
        "Exp-3_knn3_edge_attr_s300": "Exp-3 s300",
        "Exp-3_knn3_edge_attr_s500": "Exp-3 s500",
        "Exp-4_exp3_plus_quantile_weight": "Exp-4 strongQ",
        "Exp-4_exp3_s500_quantile_weak": "Exp-4 weakQ",
        "Exp-4_lite_knn3_noedgeattr_quantile_weak": "Exp-4-lite",
    }
    df["label"] = df["experiment"].map(label_map).fillna(df["experiment"])

    # Rank by R2 descending for a clear top-to-bottom comparison.
    plot_df = df.sort_values("r2", ascending=False).reset_index(drop=True)

    best_label = "Exp-4-lite"
    colors = ["#f28e2b" if lbl == best_label else "#4e79a7" for lbl in plot_df["label"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: R2
    ax = axes[0, 0]
    bars = ax.bar(plot_df["label"], plot_df["r2"], color=colors)
    ax.set_title("R2 Comparison (higher is better)")
    ax.set_ylabel("R2")
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, plot_df["r2"]):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Top-20 Recall
    ax = axes[0, 1]
    bars = ax.bar(plot_df["label"], plot_df["top20_recall"], color=colors)
    ax.set_title("Top-20 Recall (higher is better)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.45, 0.75)
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, plot_df["top20_recall"]):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Panel 3: RMSE / MAE scatter
    ax = axes[1, 0]
    for _, row in plot_df.iterrows():
        is_best = row["label"] == best_label
        ax.scatter(row["rmse"], row["mae"], s=120 if is_best else 80, c="#f28e2b" if is_best else "#4e79a7")
        ax.text(row["rmse"] + 0.01, row["mae"] + 0.005, row["label"], fontsize=9)
    ax.set_title("Error Plane (lower-left is better)")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)

    # Panel 4: Delta vs Exp-1 baseline
    ax = axes[1, 1]
    x = np.arange(len(plot_df))
    width = 0.38
    b1 = ax.bar(x - width / 2, plot_df["delta_r2_vs_exp1"], width, label="ΔR2", color="#59a14f")
    b2 = ax.bar(x + width / 2, plot_df["delta_top20_recall_vs_exp1"], width, label="ΔTop20 Recall", color="#e15759")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Improvement vs Exp-1 Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=30)
    ax.legend()
    for b in b1:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, v + (0.005 if v >= 0 else -0.01), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for b in b2:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, v + (0.005 if v >= 0 else -0.01), f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("v1.4 Edge Strategy Study: Edge Information vs Noise", fontsize=16, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
