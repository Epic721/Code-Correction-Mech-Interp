"""
generate_additional_figures.py

Creates additional milestone-ready figures beyond the existing F1/accuracy plots.

Outputs:
- outputs/figures/precision_recall_by_layer.png
- outputs/figures/error_type_distribution.png
- outputs/figures/top5_f1_bar.png
- outputs/figures/error_type_counts.csv
"""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def normalize_error(error_text: str) -> str:
    if not error_text:
        return "None"
    if error_text.startswith("timeout"):
        return "Timeout"
    return error_text.split(":", 1)[0]


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path):
    return json.loads(path.read_text())


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_precision_recall_by_layer(sweep_rows, out_path: Path):
    grouped = {}
    for r in sweep_rows:
        grouped.setdefault(r["probe"], []).append(r)

    plt.figure(figsize=(10, 5.6))

    styles = {
        "linear": {"linestyle": "--", "marker": "o"},
        "mlp": {"linestyle": "-", "marker": "s"},
    }

    for probe, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda x: x["layer"])
        layers = [r["layer"] for r in rows]
        prec = [r["precision"] for r in rows]
        rec = [r["recall"] for r in rows]
        style = styles.get(probe, {"linestyle": "-", "marker": "o"})

        plt.plot(
            layers,
            prec,
            label=f"{probe.upper()} precision",
            alpha=0.95,
            linewidth=2.0,
            **style,
        )
        plt.plot(
            layers,
            rec,
            label=f"{probe.upper()} recall",
            alpha=0.85,
            linewidth=2.0,
            linestyle=":" if style["linestyle"] == "-" else "-.",
            marker=style["marker"],
        )

    plt.ylim(0.0, 1.0)
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title("Precision and Recall Across Layers")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_error_distribution(generation_rows, out_png: Path, out_csv: Path):
    failed = [r for r in generation_rows if not r.get("passed", False)]
    counts = Counter(normalize_error(r.get("error", "")) for r in failed)

    keep = counts.most_common(8)
    labels = [k for k, _ in keep]
    values = [v for _, v in keep]

    plt.figure(figsize=(10, 5.2))
    bars = plt.bar(labels, values)
    plt.title("Failure Error Type Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "count"])
        for k, v in counts.most_common():
            writer.writerow([k, v])


def plot_top5_f1(sweep_rows, out_png: Path):
    top5 = sorted(sweep_rows, key=lambda r: r["f1"], reverse=True)[:5]
    labels = [f"L{r['layer']} {r['probe']}" for r in top5]
    values = [r["f1"] for r in top5]

    plt.figure(figsize=(8.5, 4.8))
    bars = plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1")
    plt.title("Top-5 Probe Configurations by F1")
    plt.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    root = Path(".")
    sweep_path = root / "outputs/probes/sweep_results.json"
    gen_path = root / "outputs/generations.jsonl"
    out_dir = root / "outputs/figures"
    ensure_dir(out_dir)

    sweep_rows = load_json(sweep_path)
    gen_rows = load_jsonl(gen_path)

    plot_precision_recall_by_layer(sweep_rows, out_dir / "precision_recall_by_layer.png")
    plot_error_distribution(
        gen_rows,
        out_dir / "error_type_distribution.png",
        out_dir / "error_type_counts.csv",
    )
    plot_top5_f1(sweep_rows, out_dir / "top5_f1_bar.png")

    print("Wrote:")
    print(out_dir / "precision_recall_by_layer.png")
    print(out_dir / "error_type_distribution.png")
    print(out_dir / "top5_f1_bar.png")
    print(out_dir / "error_type_counts.csv")


if __name__ == "__main__":
    main()
