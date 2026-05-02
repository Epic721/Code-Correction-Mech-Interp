"""
plot_probe_results.py - creates report-ready plots from probe sweep results.

Outputs:
- outputs/figures/f1_by_layer.png
- outputs/figures/accuracy_by_layer.png
- outputs/figures/top_results.csv

Usage:
    python plot_probe_results.py
    python plot_probe_results.py --input outputs/probes/sweep_results.json --top-k 10
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find sweep results at {path}")
    rows = json.loads(path.read_text())
    if not rows:
        raise ValueError("Sweep results file is empty")
    return rows


def split_by_probe(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["probe"], []).append(r)

    for probe in grouped:
        grouped[probe] = sorted(grouped[probe], key=lambda x: x["layer"])
    return grouped


def plot_metric_by_layer(grouped, metric, out_path: Path):
    plt.figure(figsize=(10, 5.5))

    style_map = {
        "linear": {"marker": "o", "linewidth": 2.2},
        "mlp": {"marker": "s", "linewidth": 2.2},
    }

    for probe_name, rows in sorted(grouped.items()):
        xs = [r["layer"] for r in rows]
        ys = [r[metric] for r in rows]
        style = style_map.get(probe_name, {"marker": "o", "linewidth": 2.0})
        plt.plot(xs, ys, label=probe_name.upper(), **style)

    plt.xlabel("Layer")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} Across Layers")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_top_results(rows, out_csv: Path, top_k: int):
    top_rows = sorted(rows, key=lambda x: x["f1"], reverse=True)[:top_k]
    fieldnames = ["layer", "probe", "accuracy", "precision", "recall", "f1"]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in top_rows:
            writer.writerow({k: r[k] for k in fieldnames})


def print_summary(rows):
    best = max(rows, key=lambda x: x["f1"])
    print("Best overall result")
    print(
        f"  layer={best['layer']} probe={best['probe']} "
        f"acc={best['accuracy']:.4f} precision={best['precision']:.4f} "
        f"recall={best['recall']:.4f} f1={best['f1']:.4f}"
    )

    probes = sorted({r["probe"] for r in rows})
    print("Best by probe type")
    for p in probes:
        best_p = max((r for r in rows if r["probe"] == p), key=lambda x: x["f1"])
        print(
            f"  {p}: layer={best_p['layer']} "
            f"acc={best_p['accuracy']:.4f} f1={best_p['f1']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot probe sweep results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/probes/sweep_results.json"),
        help="Path to sweep_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for generated figures",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Rows to keep in top-results CSV")
    args = parser.parse_args()

    rows = load_results(args.input)
    grouped = split_by_probe(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    f1_path = args.out_dir / "f1_by_layer.png"
    acc_path = args.out_dir / "accuracy_by_layer.png"
    top_csv = args.out_dir / "top_results.csv"

    plot_metric_by_layer(grouped, "f1", f1_path)
    plot_metric_by_layer(grouped, "accuracy", acc_path)
    save_top_results(rows, top_csv, args.top_k)
    print_summary(rows)

    print("\nWrote:")
    print(f"  {f1_path}")
    print(f"  {acc_path}")
    print(f"  {top_csv}")


if __name__ == "__main__":
    main()
