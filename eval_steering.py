"""
eval_steering.py - master harness for the phase 2 steering experiment.

bridges everything that came before:
  - the trained vectors from outputs/vectors/ (caa, sae, learned)
  - HookedTransformer for inference-time intervention
  - the multiprocessing sandbox from eval_sandbox for pass/fail labels
  - the existing generations.jsonl from baseline as the comparison anchor

universal steering rule (consistent across CAA, SAE, learned):
    x' = x - alpha * v
where v always points toward failure. so positive alpha pushes the model
AWAY from failure. alpha=0 is the no-intervention sanity check.

intervention modes (ablation toggle):
  surgical    -> subtract once, at the final prompt token (prefill only)
  continuous  -> also subtract at every generated token during decode

for each (method, layer, alpha, mode) we generate completions for every
problem in the dataset, run them through the sandbox, and compute:
  - correction rate: % of baseline-failed problems that now pass
  - corruption rate: % of baseline-passed problems that now fail
  - pass@1: overall pass rate

outputs:
  outputs/<dataset>/steering/per_problem.jsonl   (full provenance, resumeable)
  outputs/<dataset>/steering/results.jsonl       (one row per sweep point)
  outputs/<dataset>/steering/pareto_curve.png    (correction vs corruption)
  outputs/<dataset>/steering/pass_at_1_curve.png (pass@1 vs alpha)
  outputs/<dataset>/steering/steering_summary.json

usage:
    python eval_steering.py                                   # full sweep, both datasets
    python eval_steering.py --datasets mbpp                   # mbpp only
    python eval_steering.py --methods caa,learned             # skip sae
    python eval_steering.py --layers 25 --alphas 0,1,4        # focal subset
    python eval_steering.py --max-problems 20 --local         # smoke test on cpu
"""

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch

# headless backend BEFORE importing pyplot -- matters on slurm where there's
# no display and importing pyplot first would crash on a missing $DISPLAY
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transformer_lens import HookedTransformer

from config import (
    get_config, setup_logging, log_config, PipelineConfig,
    SUPPORTED_DATASETS, SUPPORTED_INTERVENTION_MODES,
)
from eval_sandbox import (
    get_adapter, run_in_sandbox, extract_code,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_tl_model(cfg: PipelineConfig) -> HookedTransformer:
    """
    same loading approach as cache_activations / train_learned_vector --
    from_pretrained_no_processing for bf16/fp16. keeping the residual
    stream basis identical to the one our cached activations + vectors
    were built in is the whole reason this hookable substrate works.
    """
    log.info(f"loading {cfg.model_name} via transformerlens")

    use_no_processing = cfg.torch_dtype in (torch.float16, torch.bfloat16)
    loader = (
        HookedTransformer.from_pretrained_no_processing
        if use_no_processing
        else HookedTransformer.from_pretrained
    )

    from_pretrained_kwargs = {}
    if cfg.hf_token:
        from_pretrained_kwargs["token"] = cfg.hf_token

    model = loader(
        cfg.model_name,
        device=cfg.device,
        dtype=cfg.torch_dtype,
        **from_pretrained_kwargs,
    )
    model.eval()
    model.requires_grad_(False)

    log.info(
        f"  loaded: n_layers={model.cfg.n_layers}  "
        f"d_model={model.cfg.d_model}"
    )
    return model


# ---------------------------------------------------------------------------
# vector i/o
# ---------------------------------------------------------------------------

def vector_path_for(cfg: PipelineConfig, method: str, layer: int) -> Path:
    return cfg.vectors_path / f"{method}_layer_{layer:02d}.pt"


def load_vector(path: Path) -> tuple[torch.Tensor, dict]:
    """returns (vector [d_model] in float32 on cpu, metadata dict)"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    v = payload["vector"].float()
    meta = {k: v_ for k, v_ in payload.items() if k != "vector"}
    return v, meta


def discover_vectors(
    cfg: PipelineConfig, methods: list[str], layers: list[int],
) -> list[tuple[str, int, Path]]:
    """
    returns the (method, layer, path) tuples for every requested combo
    that actually exists on disk. silently skips combos w/ no saved
    vector so a partial vector run doesnt fail this script.
    """
    found = []
    for method in methods:
        for layer in layers:
            p = vector_path_for(cfg, method, layer)
            if p.exists():
                found.append((method, layer, p))
            else:
                log.info(f"  no vector at {p.name}, skipping {method}/L{layer}")
    return found


# ---------------------------------------------------------------------------
# the steering hook
# ---------------------------------------------------------------------------

def make_steering_hook(mode: str, alpha: float, v: torch.Tensor):
    """
    returns a forward hook for blocks.{layer}.hook_resid_post that
    implements x' = x - alpha * v according to the chosen mode.

    timing detection:
        activation shape [B, T, D]. with use_past_kv_cache=True in
        transformerlens.generate, we get:
            T > 1   ->  prefill pass (entire prompt at once)
            T == 1  ->  decode pass (one new token, kv cache filled)

    surgical: subtract once during prefill at position -1 (final prompt
        token). during decode the hook fires but no-ops -- but the kv
        cache stored during prefill ALREADY reflects the perturbation
        (since the modified residual propagates through subsequent layers
        and into K/V projections), so attention from generated tokens to
        the final-prompt position uses the steered representation.

    continuous: subtract during prefill at position -1 AND during every
        decode step at position -1 (which is the only position when
        T==1). each new token's residual at this layer gets perturbed.

    note: alpha=0 still clones the activation but the result is
    numerically identical to no-hook, modulo bf16 noise on the clone.
    that residual drift is what we cross-check against the hf baseline.
    """
    if mode not in SUPPORTED_INTERVENTION_MODES:
        raise ValueError(f"unknown intervention_mode {mode!r}")

    def hook_fn(activation, hook_obj):
        B, T, D = activation.shape
        delta = (alpha * v).to(dtype=activation.dtype, device=activation.device)

        if T > 1:
            # prefill: surgical injection at the final prompt token
            out = activation.clone()
            out[:, -1, :] = out[:, -1, :] - delta
            return out
        # decode (T == 1)
        if mode == "continuous":
            return activation - delta.view(1, 1, D)
        return activation

    return hook_fn


# ---------------------------------------------------------------------------
# steered generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def steered_generate(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    mode: str,
    alpha: float,
    v: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """
    runs model.generate w/ the steering hook active for ONE problem.
    returns just the generated text (everything after the prompt tokens).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"

    # tokenize prompt (no extra bos -- chat template already wrote one in)
    input_ids = model.to_tokens(prompt, prepend_bos=False)
    prompt_len = input_ids.shape[1]

    eos_id = (
        model.tokenizer.eos_token_id
        if model.tokenizer.eos_token_id is not None
        else None
    )

    hook_fn = make_steering_hook(mode, alpha, v)

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        out_tokens = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy, matches eval_sandbox temp=0
            use_past_kv_cache=True,     # so prefill vs decode shapes work as expected
            stop_at_eos=eos_id is not None,
            eos_token_id=eos_id,
            return_type="tokens",
            verbose=False,
        )

    new_tokens = out_tokens[0, prompt_len:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# baseline loading
# ---------------------------------------------------------------------------

def load_baseline_passed(cfg: PipelineConfig) -> dict:
    """
    pulls task_id -> bool from the existing generations.jsonl. this is
    the HF-pipeline baseline produced by eval_sandbox; we use it as the
    anchor for correction/corruption metrics.
    """
    path = cfg.generations_path
    if not path.exists():
        raise FileNotFoundError(
            f"baseline generations not found at {path}. run "
            f"`python eval_sandbox.py --dataset {cfg.dataset}` first."
        )

    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["task_id"]] = bool(rec["passed"])
            except (json.JSONDecodeError, KeyError):
                continue

    log.info(
        f"loaded baseline: {len(out)} problems  "
        f"({sum(out.values())} pass / {len(out) - sum(out.values())} fail)"
    )
    return out


# ---------------------------------------------------------------------------
# per-problem result file (resume-friendly)
# ---------------------------------------------------------------------------

def _result_key(method: str, layer: int, alpha: float, mode: str, task_id) -> tuple:
    """
    key used both in the resume-set and as part of per-row records.
    rounding alpha avoids float precision mismatch on resume.
    """
    return (method, int(layer), round(float(alpha), 6), mode, str(task_id))


def load_done_set(per_problem_path: Path) -> set:
    if not per_problem_path.exists():
        return set()
    done = set()
    with open(per_problem_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add(_result_key(
                    r["method"], r["layer"], r["alpha"], r["mode"],
                    r["task_id"],
                ))
            except (json.JSONDecodeError, KeyError):
                log.warning(
                    f"skipping malformed line {i} in {per_problem_path}"
                )
    return done


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(steered: dict, baseline: dict) -> dict:
    """
    steered:  task_id -> bool   (this run, w/ steering applied)
    baseline: task_id -> bool   (the HF-pipeline baseline)

    correction rate = #(baseline_fail and steered_pass) / #(baseline_fail)
    corruption rate = #(baseline_pass and steered_fail) / #(baseline_pass)
    pass@1          = #(steered_pass) / #(total covered)

    we restrict to task_ids present in BOTH dicts so partial runs dont
    skew the denominators.
    """
    common = set(steered) & set(baseline)
    if not common:
        return {
            "n_total": 0, "n_baseline_pass": 0, "n_baseline_fail": 0,
            "n_steered_pass": 0, "n_corrected": 0, "n_corrupted": 0,
            "correction_rate": float("nan"),
            "corruption_rate": float("nan"),
            "pass_at_1": float("nan"),
            "baseline_pass_at_1": float("nan"),
        }

    base_fail = {tid for tid in common if not baseline[tid]}
    base_pass = {tid for tid in common if baseline[tid]}

    n_corrected = sum(1 for tid in base_fail if steered[tid])
    n_corrupted = sum(1 for tid in base_pass if not steered[tid])
    n_steered_pass = sum(1 for tid in common if steered[tid])

    return {
        "n_total": len(common),
        "n_baseline_pass": len(base_pass),
        "n_baseline_fail": len(base_fail),
        "n_steered_pass": n_steered_pass,
        "n_corrected": n_corrected,
        "n_corrupted": n_corrupted,
        "correction_rate": n_corrected / max(len(base_fail), 1),
        "corruption_rate": n_corrupted / max(len(base_pass), 1),
        "pass_at_1": n_steered_pass / len(common),
        "baseline_pass_at_1": len(base_pass) / len(common),
    }


# ---------------------------------------------------------------------------
# single sweep point: (method, layer, alpha, mode) over all problems
# ---------------------------------------------------------------------------

def run_sweep_point(
    model, problems, adapter, baseline_passed,
    method: str, layer: int, alpha: float, mode: str,
    vector: torch.Tensor, vector_meta: dict,
    cfg: PipelineConfig, max_new_tokens: int,
    per_problem_fp, done_set: set,
    log_every: int = 25,
) -> dict:
    """
    iterates every problem, generates a steered completion, runs it
    through the sandbox, writes the per-problem record to disk, and
    returns the aggregate metrics for this sweep point.
    """
    log.info(
        f"--> sweep point: method={method:>7s}  L{layer:>2d}  "
        f"alpha={alpha:>5.2f}  mode={mode:<10s}  "
        f"|v|={float(vector.norm()):.3f}"
    )
    point_start = time.time()
    steered_passed: dict = {}

    n_skipped = 0
    for i, problem in enumerate(problems):
        tid = adapter.task_id(problem)
        key = _result_key(method, layer, alpha, mode, tid)
        if key in done_set:
            # already done in a previous run -- pull its pass/fail from disk
            n_skipped += 1
            continue

        prompt = adapter.build_prompt(problem, model.tokenizer)
        try:
            raw_gen = steered_generate(
                model, prompt, layer, mode, alpha, vector, max_new_tokens,
            )
        except Exception as e:
            log.warning(
                f"generation failed for {tid}: "
                f"{type(e).__name__}: {e} -- recording as fail"
            )
            raw_gen = ""

        code = extract_code(raw_gen)
        script = adapter.assemble_test_script(problem, code)
        result = run_in_sandbox(script, timeout=cfg.exec_timeout_sec)
        passed = bool(result["passed"])

        steered_passed[tid] = passed

        record = {
            "method": method,
            "layer": int(layer),
            "alpha": round(float(alpha), 6),
            "mode": mode,
            "task_id": tid,
            "passed": passed,
            "baseline_passed": baseline_passed.get(tid),
            "error": result.get("error"),
            # truncate for log size; full traceability sits in generations.jsonl
            "raw_generation": raw_gen[:2000],
            "extracted_code": code[:2000],
        }
        per_problem_fp.write(json.dumps(record) + "\n")
        per_problem_fp.flush()

        done_set.add(key)

        if (i + 1) % log_every == 0:
            elapsed = time.time() - point_start
            rate = (i + 1 - n_skipped) / max(elapsed, 1e-6)
            log.info(
                f"    [{i + 1}/{len(problems)}]  "
                f"{sum(steered_passed.values())}/{len(steered_passed)} pass "
                f"({rate:.2f} prob/s)"
            )

    # if we resumed, we need to also rebuild steered_passed from disk so the
    # aggregate metrics are correct. cheap re-scan since it's just one file.
    if n_skipped > 0:
        steered_passed = _hydrate_steered_from_file(
            per_problem_fp.name, method, layer, alpha, mode,
        )

    metrics = compute_metrics(steered_passed, baseline_passed)
    elapsed = time.time() - point_start
    log.info(
        f"    done in {elapsed:.1f}s  "
        f"pass@1={metrics['pass_at_1']:.3f}  "
        f"correction={metrics['correction_rate']:.3f}  "
        f"corruption={metrics['corruption_rate']:.3f}"
    )

    out = {
        "method": method,
        "layer": int(layer),
        "alpha": round(float(alpha), 6),
        "mode": mode,
        "vector_norm": float(vector.norm()),
        "vector_meta": {
            k: vector_meta.get(k) for k in (
                "method", "layer", "sign_convention",
            ) if k in vector_meta
        },
        "elapsed_seconds": elapsed,
        **metrics,
    }
    return out


def _hydrate_steered_from_file(
    path: str, method: str, layer: int, alpha: float, mode: str,
) -> dict:
    """
    reread per_problem.jsonl and pull just the rows matching this sweep
    point. used after a partial-resume run to recover the full
    {task_id -> pass/fail} dict for metrics computation.
    """
    target = (method, int(layer), round(float(alpha), 6), mode)
    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                r.get("method") == target[0]
                and int(r.get("layer", -1)) == target[1]
                and round(float(r.get("alpha", -1)), 6) == target[2]
                and r.get("mode") == target[3]
            ):
                out[r["task_id"]] = bool(r["passed"])
    return out


# ---------------------------------------------------------------------------
# full sweep across all (method, layer, alpha, mode) combos for one dataset
# ---------------------------------------------------------------------------

def run_full_sweep(
    cfg: PipelineConfig,
    methods: list[str], layers: list[int],
    alphas: list[float], modes: list[str],
    max_new_tokens: int, max_problems: Optional[int] = None,
    model: Optional[HookedTransformer] = None,
) -> dict:
    """
    runs the entire sweep for one dataset and returns the aggregated
    summary. model is passed in (rather than loaded inside) so the same
    instance can be reused across mbpp + humaneval in one job.
    """
    log.info("=" * 72)
    log.info(f"steering eval -- dataset: {cfg.dataset}")
    log.info("=" * 72)

    adapter = get_adapter(cfg.dataset)
    ds = adapter.load(cfg)
    if max_problems is not None:
        ds = ds.select(range(min(max_problems, len(ds))))
    problems = list(ds)
    log.info(f"  {len(problems)} problems")

    baseline_passed = load_baseline_passed(cfg)

    available = discover_vectors(cfg, methods, layers)
    if not available:
        log.warning("no vectors found for this sweep -- skipping dataset")
        return {"dataset": cfg.dataset, "results": [], "warning": "no vectors"}
    log.info(f"  {len(available)} (method, layer) pairs to evaluate")

    if model is None:
        model = load_tl_model(cfg)

    out_dir = cfg.steering_path
    per_problem_path = out_dir / "per_problem.jsonl"
    results_path = out_dir / "results.jsonl"

    done_set = load_done_set(per_problem_path)
    if done_set:
        log.info(f"  resuming: {len(done_set)} per-problem rows already on disk")

    sweep_results: list[dict] = []
    total_points = len(available) * len(alphas) * len(modes)
    point_idx = 0

    with open(per_problem_path, "a", encoding="utf-8") as per_fp, \
         open(results_path, "a", encoding="utf-8") as res_fp:

        for method, layer, vec_path in available:
            v, vmeta = load_vector(vec_path)
            v = v.to(cfg.device)

            for mode in modes:
                for alpha in alphas:
                    point_idx += 1
                    log.info(
                        f"\n[{point_idx}/{total_points}] "
                        f"{method} L{layer} alpha={alpha} {mode}"
                    )
                    row = run_sweep_point(
                        model, problems, adapter, baseline_passed,
                        method, layer, alpha, mode, v, vmeta,
                        cfg, max_new_tokens,
                        per_problem_fp=per_fp, done_set=done_set,
                    )
                    sweep_results.append(row)
                    res_fp.write(json.dumps(row) + "\n")
                    res_fp.flush()

                    if cfg.device == "cuda":
                        torch.cuda.empty_cache()

            del v
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "dataset": cfg.dataset,
        "n_problems": len(problems),
        "baseline_pass_rate": (
            sum(baseline_passed.values()) / max(len(baseline_passed), 1)
        ),
        "config": {
            "methods": methods,
            "layers": layers,
            "alphas": alphas,
            "modes": modes,
            "max_new_tokens": max_new_tokens,
            "model_name": cfg.model_name,
            "intervention_rule": "x' = x - alpha * v",
            "sign_convention": "v_points_toward_failure",
        },
        "results": sweep_results,
    }

    summary_path = out_dir / "steering_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nsteering summary -> {summary_path}")

    # cross-check tl@alpha=0 vs hf baseline (numerical drift sanity)
    _log_tl_vs_hf_drift(sweep_results, baseline_passed)

    return summary


def _log_tl_vs_hf_drift(results: list[dict], baseline: dict):
    """
    at alpha=0 the steering hook adds zero, so any disagreement between
    our TL-pipeline pass/fail and the HF-pipeline baseline is purely
    numerical drift between the two stacks (different attn impls, dtype
    paths, etc). worth surfacing so the report can comment on it.
    """
    zeros = [r for r in results if r["alpha"] == 0.0]
    if not zeros:
        return

    log.info("\n--- tl@alpha=0 vs hf baseline drift check ---")
    for r in zeros:
        # at alpha=0 correction_rate is the share of baseline-failed
        # problems where TL happens to flip them to pass; corruption_rate
        # is the converse. both should be close to 0.
        n_disagree = r["n_corrected"] + r["n_corrupted"]
        log.info(
            f"  {r['method']} L{r['layer']} {r['mode']:10s}  "
            f"disagree on {n_disagree}/{r['n_total']} tasks  "
            f"(correction={r['correction_rate']:.3f} "
            f"corruption={r['corruption_rate']:.3f})"
        )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "caa":     "tab:blue",
    "sae":     "tab:orange",
    "learned": "tab:green",
}
_MODE_STYLES = {
    "surgical":   "-",
    "continuous": "--",
}
_LAYER_MARKERS = {12: "o", 20: "s", 25: "^"}


def _color_for(method: str) -> str:
    return _METHOD_COLORS.get(method, "tab:gray")


def _ls_for(mode: str) -> str:
    return _MODE_STYLES.get(mode, ":")


def _marker_for(layer: int) -> str:
    return _LAYER_MARKERS.get(int(layer), "x")


def plot_pareto(results: list[dict], out_path: Path, dataset: str):
    """correction (y) vs corruption (x), one curve per (method, layer, mode)"""
    fig, ax = plt.subplots(figsize=(8, 6))

    groups: dict = {}
    for r in results:
        if not (r["correction_rate"] == r["correction_rate"]):  # nan-safe
            continue
        key = (r["method"], r["layer"], r["mode"])
        groups.setdefault(key, []).append(r)

    for (method, layer, mode), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["alpha"])
        xs = [r["corruption_rate"] for r in rs]
        ys = [r["correction_rate"] for r in rs]
        ax.plot(
            xs, ys,
            linestyle=_ls_for(mode),
            color=_color_for(method),
            marker=_marker_for(layer),
            markersize=6,
            linewidth=1.7,
            label=f"{method} L{layer} {mode}",
            alpha=0.9,
        )
        # annotate alpha at each point so the curve direction is readable
        for r in rs:
            ax.annotate(
                f"α={r['alpha']:g}",
                (r["corruption_rate"], r["correction_rate"]),
                fontsize=6,
                alpha=0.6,
                xytext=(3, 3),
                textcoords="offset points",
            )

    # diagonal y=x reference -- above it = net positive, below = net negative
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=":",
            alpha=0.4, linewidth=1, label="break-even (y = x)")

    ax.set_xlabel("Corruption Rate (% baseline-pass that now fail)")
    ax.set_ylabel("Correction Rate (% baseline-fail that now pass)")
    ax.set_title(f"Steering Pareto -- {dataset}")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  pareto plot -> {out_path}")


def plot_pass_at_1(
    results: list[dict], baseline_pass_rate: float,
    out_path: Path, dataset: str,
):
    """pass@1 (y) vs alpha (x), one curve per (method, layer, mode)"""
    fig, ax = plt.subplots(figsize=(8, 6))

    groups: dict = {}
    for r in results:
        if not (r["pass_at_1"] == r["pass_at_1"]):
            continue
        key = (r["method"], r["layer"], r["mode"])
        groups.setdefault(key, []).append(r)

    for (method, layer, mode), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["alpha"])
        xs = [r["alpha"] for r in rs]
        ys = [r["pass_at_1"] for r in rs]
        ax.plot(
            xs, ys,
            linestyle=_ls_for(mode),
            color=_color_for(method),
            marker=_marker_for(layer),
            markersize=6,
            linewidth=1.7,
            label=f"{method} L{layer} {mode}",
            alpha=0.9,
        )

    ax.axhline(
        baseline_pass_rate, color="black", linestyle=":",
        linewidth=1.2,
        label=f"hf baseline ({baseline_pass_rate:.3f})",
    )

    ax.set_xlabel("Steering Strength α")
    ax.set_ylabel("Pass@1")
    ax.set_title(f"Pass@1 vs Steering Strength -- {dataset}")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  pass@1 plot -> {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_csv_str(s: str | None) -> list[str] | None:
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_int(s: str | None) -> list[int] | None:
    if s is None:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_float(s: str | None) -> list[float] | None:
    if s is None:
        return None
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "evaluate steering vectors (caa, sae, learned) on mbpp and "
            "humaneval. sweeps alpha and intervention mode, computes "
            "correction/corruption rates against the existing hf baseline."
        )
    )
    parser.add_argument(
        "--datasets", default=None,
        help=f"comma-separated; default: {','.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument(
        "--methods", default="caa,sae,learned",
        help="comma-separated subset of {caa,sae,learned}",
    )
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated layers (default: cfg.target_layers)",
    )
    parser.add_argument(
        "--alphas", default="0,0.5,1,2,4,8",
        help="comma-separated alpha values (alpha=0 is the sanity check)",
    )
    parser.add_argument(
        "--modes", default="surgical,continuous",
        help=f"comma-separated subset of {SUPPORTED_INTERVENTION_MODES}",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="cap problems per dataset (handy for debugging the harness)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="max tokens to generate per problem",
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="skip generating the matplotlib figures",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file", default=None,
        help="optional path to also tee logs into",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=args.log_file)

    datasets = _parse_csv_str(args.datasets) or list(SUPPORTED_DATASETS)
    methods = _parse_csv_str(args.methods) or ["caa", "sae", "learned"]
    layers_override = _parse_csv_int(args.layers)
    alphas = _parse_csv_float(args.alphas) or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    modes = _parse_csv_str(args.modes) or list(SUPPORTED_INTERVENTION_MODES)

    log.info(f"datasets : {datasets}")
    log.info(f"methods  : {methods}")
    log.info(f"alphas   : {alphas}")
    log.info(f"modes    : {modes}")

    # build one config per dataset, plus a "primary" config for shared model load
    primary_cfg = get_config(local=args.local, dataset=datasets[0])
    log_config(primary_cfg, log)

    layers = layers_override or primary_cfg.target_layers

    # one model load, reused across datasets to amortize the ~5GB+ load cost
    model = load_tl_model(primary_cfg)

    all_summaries = {}
    for ds_name in datasets:
        cfg = get_config(local=args.local, dataset=ds_name)

        try:
            summary = run_full_sweep(
                cfg, methods, layers, alphas, modes,
                max_new_tokens=args.max_new_tokens,
                max_problems=args.max_problems,
                model=model,
            )
        except FileNotFoundError as e:
            log.error(f"skipping {ds_name}: {e}")
            continue

        all_summaries[ds_name] = summary

        if not args.no_plot and summary["results"]:
            plot_pareto(
                summary["results"],
                cfg.steering_path / "pareto_curve.png",
                ds_name,
            )
            plot_pass_at_1(
                summary["results"], summary["baseline_pass_rate"],
                cfg.steering_path / "pass_at_1_curve.png",
                ds_name,
            )

    # cross-dataset summary in primary cfg's vectors_path -- handy for the
    # report since it's one file with everything
    cross_path = primary_cfg.output_dir / "steering_cross_dataset_summary.json"
    with open(cross_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"\ncross-dataset summary -> {cross_path}")


if __name__ == "__main__":
    main()
