"""
plot_steering_results.py - rebuild aggregate metrics + plots after a
steering sweep is complete (potentially across multiple sbatch jobs).

reads per_problem.jsonl (the source of truth, immune to duplicate
aggregate rows) and the original generations.jsonl baseline, then:
  1. computes correction/corruption/pass@1 per (method, layer, alpha, mode)
  2. writes a clean, deduped results_deduped.jsonl
  3. regenerates the pareto and pass@1 figures

usage:
    python plot_steering_results.py
    python plot_steering_results.py --datasets mbpp
    python plot_steering_results.py --output-dir outputs_local
"""

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# headless matplotlib for cluster + ssh sessions
import matplotlib
matplotlib.use("Agg")

# reuse the plot fns from eval_steering so the visual style stays consistent
from eval_steering import (
    plot_pareto, plot_pass_at_1, compute_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("plot_steering")


def load_baseline(path: Path) -> dict:
    """task_id -> bool, pulled from the canonical generations.jsonl"""
    if not path.exists():
        log.warning(f"baseline missing at {path}")
        return {}
    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out[r["task_id"]] = bool(r["passed"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def load_per_problem(path: Path) -> dict:
    """
    returns dict: (method, layer, alpha, mode) -> {task_id: bool}.

    on duplicate (sweep_point, task_id) keys, the LATEST row wins. that
    matches the resume semantics: a newer run's pass/fail supersedes an
    older one for the same problem (e.g. if you reran with a different
    max_new_tokens setting).
    """
    if not path.exists():
        return {}
    rows: dict = defaultdict(dict)
    n_lines = 0
    n_kept = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                key = (
                    r["method"],
                    int(r["layer"]),
                    round(float(r["alpha"]), 6),
                    r["mode"],
                )
            except (KeyError, ValueError, TypeError):
                continue
            rows[key][r["task_id"]] = bool(r["passed"])
            n_kept += 1
    log.info(
        f"  read {n_lines} lines from per_problem.jsonl, "
        f"kept {n_kept} valid rows across {len(rows)} sweep points"
    )
    return rows


def aggregate(rows: dict, baseline: dict) -> list[dict]:
    """one summary row per (method, layer, alpha, mode), nan-safe"""
    out: list[dict] = []
    for (method, layer, alpha, mode), steered in sorted(rows.items()):
        if not steered:
            continue
        m = compute_metrics(steered, baseline)
        if m["n_total"] == 0:
            log.warning(
                f"no overlap with baseline for "
                f"{method} L{layer} alpha={alpha} {mode} -- skipping"
            )
            continue
        out.append({
            "method": method,
            "layer": layer,
            "alpha": alpha,
            "mode": mode,
            **m,
        })
    return out


def write_deduped_results(results: list[dict], path: Path):
    """clean, sorted, one-row-per-sweep-point jsonl for downstream tools"""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"  wrote deduped results -> {path}")


def summarize(results: list[dict], baseline_pass_rate: float):
    """print a quick status table so the user can eyeball results"""
    if not results:
        log.info("  (no results to summarize)")
        return

    log.info("")
    log.info(
        f"  {'method':<8} {'L':>3} {'alpha':>6} {'mode':<11} "
        f"{'pass@1':>7} {'corr':>6} {'corrupt':>8} {'n':>5}"
    )
    log.info("  " + "-" * 60)
    for r in sorted(
        results,
        key=lambda r: (r["method"], r["layer"], r["mode"], r["alpha"]),
    ):
        marker = "*" if r["pass_at_1"] > baseline_pass_rate else " "
        log.info(
            f"{marker} {r['method']:<8} {r['layer']:>3} "
            f"{r['alpha']:>6.2f} {r['mode']:<11} "
            f"{r['pass_at_1']:>7.3f} "
            f"{r['correction_rate']:>6.3f} "
            f"{r['corruption_rate']:>8.3f} "
            f"{r['n_total']:>5d}"
        )
    log.info(f"  baseline pass@1 = {baseline_pass_rate:.3f}")
    log.info("  '*' = beats baseline pass@1")


def process_dataset(dataset: str, output_dir: Path):
    log.info(f"\n=== {dataset} ===")
    ds_root = output_dir / dataset
    steering_dir = ds_root / "steering"
    per_problem_path = steering_dir / "per_problem.jsonl"
    baseline_path = ds_root / "generations.jsonl"

    if not per_problem_path.exists():
        log.warning(f"  no steering data at {per_problem_path}, skipping")
        return

    baseline = load_baseline(baseline_path)
    rows = load_per_problem(per_problem_path)
    log.info(
        f"  baseline: {len(baseline)} problems  "
        f"({sum(baseline.values())} pass / "
        f"{len(baseline) - sum(baseline.values())} fail)"
    )

    results = aggregate(rows, baseline)
    if not results:
        log.warning("  no aggregable results, skipping plots")
        return

    write_deduped_results(results, steering_dir / "results_deduped.jsonl")

    baseline_pass_rate = (
        sum(baseline.values()) / max(len(baseline), 1)
    )
    summarize(results, baseline_pass_rate)

    plot_pareto(
        results,
        steering_dir / "pareto_curve.png",
        dataset,
    )
    plot_pass_at_1(
        results, baseline_pass_rate,
        steering_dir / "pass_at_1_curve.png",
        dataset,
    )


def main():
    ap = argparse.ArgumentParser(
        description="rebuild steering aggregate metrics + plots from "
                    "per_problem.jsonl (post-sweep cleanup)"
    )
    ap.add_argument(
        "--datasets", default="mbpp,humaneval",
        help="comma-separated dataset names to process",
    )
    ap.add_argument(
        "--output-dir", default="outputs",
        help="root dir containing <dataset>/steering/per_problem.jsonl",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        log.error(f"output_dir does not exist: {output_dir.resolve()}")
        return

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        process_dataset(ds, output_dir)

    log.info("\ndone.")


if __name__ == "__main__":
    main()
