"""
v1.4 実験比較スクリプト

Exp-1: baseline graph edge
Exp-2: kNN edge (k=3)
Exp-3: kNN edge + distance-weighted edge_attr
Exp-4: Exp-3 + quantile weighted loss
"""

from pathlib import Path
import argparse
import pandas as pd


DEFAULT_EXPERIMENTS = {
    "Exp-1_baseline_graph": "output/hgnn_training_v1_4_exp1_baseline/test_metrics.csv",
    "Exp-2_knn3": "output/hgnn_training_v1_4_exp2_knn3/test_metrics.csv",
    "Exp-3_knn3_edge_attr": "output/hgnn_training_v1_4_exp3_knn3_edgeattr/test_metrics.csv",
    "Exp-3_knn3_edge_attr_s300": "output/hgnn_training_v1_4_exp3_knn3_edgeattr_s300/test_metrics.csv",
    "Exp-3_knn3_edge_attr_s500": "output/hgnn_training_v1_4_exp3_knn3_edgeattr_s500/test_metrics.csv",
    "Exp-4_exp3_plus_quantile_weight": "output/hgnn_training_v1_4_exp4_knn3_edgeattr_quantile/test_metrics.csv",
    "Exp-4_exp3_s500_quantile_weak": "output/hgnn_training_v1_4_exp4_knn3_edgeattr_s500_quantile_weak/test_metrics.csv",
    "Exp-4_lite_knn3_noedgeattr_quantile_weak": "output/hgnn_training_v1_4_exp4lite_knn3_noedgeattr_quantile_weak/test_metrics.csv",
}

KEEP_COLS = [
    "r2", "mae", "rmse", "top20_recall",
    "best_epoch", "best_time_sec", "total_epochs", "total_time_sec", "peak_gpu_memory_mb"
]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """tabulate不要の簡易Markdownテーブル変換"""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_metrics(path: Path):
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    out = {}
    for c in KEEP_COLS:
        out[c] = float(row[c]) if c in row else None
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare v1.4 experiments")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/v1_4_experiment_comparison",
        help="Output directory for comparison table"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, p in DEFAULT_EXPERIMENTS.items():
        metrics = load_metrics(Path(p))
        if metrics is None:
            print(f"[WARN] Missing: {p}")
            continue
        row = {"experiment": name, **metrics}
        rows.append(row)

    if not rows:
        raise FileNotFoundError("No experiment metrics found.")

    df = pd.DataFrame(rows)

    # 主指標で並び替え（Top-20 Recall優先、次にR²）
    if "top20_recall" in df.columns and "r2" in df.columns:
        df = df.sort_values(["top20_recall", "r2"], ascending=False)

    # baselineとの差分（Exp-1）
    base = df[df["experiment"] == "Exp-1_baseline_graph"]
    if len(base) > 0:
        base = base.iloc[0]
        df["delta_r2_vs_exp1"] = df["r2"] - base["r2"]
        df["delta_mae_vs_exp1"] = df["mae"] - base["mae"]
        df["delta_rmse_vs_exp1"] = df["rmse"] - base["rmse"]
        df["delta_top20_recall_vs_exp1"] = df["top20_recall"] - base["top20_recall"]

    csv_path = out_dir / "v1_4_experiment_comparison.csv"
    md_path = out_dir / "v1_4_experiment_comparison.md"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# v1.4 Experiment Comparison\n\n")
        f.write(dataframe_to_markdown(df))
        f.write("\n")

    print("=== v1.4 experiment comparison ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
