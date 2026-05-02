"""
train_probes.py - trains linear and mlp probes on cached residual stream
activations to detect whether the model's generation will fail or pass.

phase 2 update: replaced the single 80/20 split w/ stratified 5-fold CV,
and added AUROC + Balanced Accuracy alongside the milestone metrics so
our reported numbers stop being misleading at 73/27 class imbalance.

evaluation protocol per (layer, probe_type):
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - inside each fold: fit normalizer on train partition only,
    train a fresh probe via gradient descent, evaluate on val partition
  - aggregate metrics across folds as mean ± sample std (ddof=1)
  - save the single best-AUROC fold's probe weights for downstream use

labels: fail=1, pass=0  (we're training an error detector)

usage:
    python train_probes.py --local
    python train_probes.py
    python train_probes.py --epochs 200 --lr 5e-4 --n-folds 5
"""

import json
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)

from config import get_config, setup_logging, log_config, PipelineConfig

log = logging.getLogger(__name__)


# names of the metrics we report. order matters for the summary table.
METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "auroc",
)


# ---------------------------------------------------------------------------
# probe architectures (unchanged from milestone)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """single linear layer -- tests if error signal is linearly separable"""

    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class MLPProbe(nn.Module):
    """2-layer mlp -- can pick up nonlinear structure the linear cant"""

    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# data loading + per-fold preprocessing
# ---------------------------------------------------------------------------

def load_layer_data(cfg, layer):
    """loads cached activations + labels for a single layer"""
    path = cfg.activations_path / f"layer_{layer:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no activation file at {path}")

    data = torch.load(path, weights_only=False, map_location="cpu")
    return data["activations"], data["labels"], data["d_model"]


def fit_normalizer(train_acts):
    """
    compute zero-mean unit-var stats from the TRAIN partition only.
    this is the no-leakage line: stats never see the val partition.
    """
    mean = train_acts.mean(dim=0)
    std = train_acts.std(dim=0).clamp(min=1e-8)
    return mean, std


def apply_normalizer(acts, mean, std):
    """apply pre-fit stats to whatever partition we're transforming"""
    return (acts - mean) / std


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    accuracy / precision / recall / f1 (threshold = 0 on raw logits)
    + balanced_accuracy (TPR + TNR)/2 -- robust to class imbalance
    + auroc -- rank-based, threshold-independent

    auroc is computed directly from logits since it only cares about
    rank ordering (a monotonic transform like sigmoid wouldnt change it).
    """
    logits_np = logits.detach().cpu().float().numpy()
    labels_np = labels.detach().cpu().int().numpy()
    preds_np = (logits_np > 0).astype(int)

    # under stratified k-fold both classes should always be present in
    # both partitions, but if N is tiny (eg gpt2 local mode) it can break
    has_both_classes = (labels_np.min() == 0) and (labels_np.max() == 1)

    return {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "precision": float(
            precision_score(labels_np, preds_np, zero_division=0)
        ),
        "recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "f1": float(f1_score(labels_np, preds_np, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_np, preds_np)
        ),
        "auroc": (
            float(roc_auc_score(labels_np, logits_np))
            if has_both_classes
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# training loop (per fold)
# ---------------------------------------------------------------------------

def train_probe_one_fold(
    probe, train_acts, train_labels, val_acts, val_labels,
    epochs=100, lr=1e-3, weight_decay=0.01, batch_size=64,
    device="cpu", log_every=50,
):
    """
    trains a single probe on one CV fold. standard pytorch training loop
    w/ BCEWithLogitsLoss + AdamW. tracks best val loss and restores those
    weights at the end.

    returns (probe, val_logits, best_val_loss).
    """
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(train_acts, train_labels)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = val_acts.to(device)
    val_y = val_labels.to(device)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # -- train --
        probe.train()
        running_loss = 0.0
        n_seen = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = probe(bx)
            loss = criterion(logits, by)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        train_loss = running_loss / max(n_seen, 1)

        # -- val --
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_x)
            val_loss = criterion(val_logits, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.cpu().clone() for k, v in probe.state_dict().items()
            }

        # epoch logs are debug-only since 5 folds * 26 layers * 2 types
        # would otherwise spam ~500 INFO lines per sweep
        if epoch % log_every == 0 or epoch == epochs:
            log.debug(
                f"      ep {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

    # restore the checkpoint w/ lowest val loss
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe = probe.to(device)
    probe.eval()

    with torch.no_grad():
        val_logits = probe(val_x)

    return probe, val_logits, best_val_loss


# ---------------------------------------------------------------------------
# fold-level aggregation
# ---------------------------------------------------------------------------

def aggregate_folds(per_fold_metrics: list[dict]) -> tuple[dict, dict]:
    """
    take a list of per-fold metric dicts, return (mean, std) dicts.
    std is sample std (ddof=1) which is what you want for CV reporting.
    nan-resilient in case a fold's auroc is undefined.
    """
    mean: dict = {}
    std: dict = {}
    for name in METRIC_NAMES:
        vals = np.array([m[name] for m in per_fold_metrics], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 1:
            mean[name] = float(np.mean(vals))
            std[name] = float(np.std(vals, ddof=1))
        elif len(vals) == 1:
            mean[name] = float(vals[0])
            std[name] = 0.0
        else:
            mean[name] = float("nan")
            std[name] = float("nan")
    return mean, std


# ---------------------------------------------------------------------------
# stratified k-fold CV per (layer, probe_type)
# ---------------------------------------------------------------------------

def cross_validate_probe(
    ProbeClass, probe_kwargs, acts, labels,
    n_folds=5, seed=42, epochs=100, lr=1e-3, weight_decay=0.01,
    batch_size=64, device="cpu",
):
    """
    runs stratified K-fold CV for one probe arch on one layer's activations.

    returns a tuple (per_fold_records, best_fold_idx, best_state, best_norm).
    "best" here = highest val AUROC -- threshold-independent so it's the
    cleanest single-number criterion for the imbalanced case.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels_np = labels.cpu().numpy().astype(int)

    per_fold: list[dict] = []
    best_auroc = -float("inf")
    best_fold_idx = -1
    best_state = None
    best_norm = None  # (mean, std) tensors from the winning fold

    fold_iter = skf.split(np.zeros(len(labels)), labels_np)
    for fold_idx, (tr_idx, val_idx) in enumerate(fold_iter):
        tr_acts = acts[tr_idx]
        tr_labels = labels[tr_idx]
        val_acts = acts[val_idx]
        val_labels = labels[val_idx]

        # NO-LEAKAGE: norm stats fit on train partition only
        norm_mean, norm_std = fit_normalizer(tr_acts)
        tr_acts_n = apply_normalizer(tr_acts, norm_mean, norm_std)
        val_acts_n = apply_normalizer(val_acts, norm_mean, norm_std)

        # fresh probe per fold so weights from prior folds dont leak in
        probe = ProbeClass(**probe_kwargs)
        probe, val_logits, best_val_loss = train_probe_one_fold(
            probe, tr_acts_n, tr_labels, val_acts_n, val_labels,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, device=device,
        )

        m = compute_metrics(val_logits, val_labels)
        log.info(
            f"    fold {fold_idx + 1}/{n_folds}  "
            f"acc={m['accuracy']:.3f}  "
            f"f1={m['f1']:.3f}  "
            f"bal={m['balanced_accuracy']:.3f}  "
            f"auroc={m['auroc']:.3f}"
        )

        per_fold.append({
            "fold": fold_idx,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "best_val_loss": float(best_val_loss),
            **m,
        })

        # track the fold whose probe we'll persist for downstream use
        if not np.isnan(m["auroc"]) and m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            best_fold_idx = fold_idx
            best_state = {
                k: v.cpu().clone() for k, v in probe.state_dict().items()
            }
            best_norm = (norm_mean.cpu().clone(), norm_std.cpu().clone())

    return per_fold, best_fold_idx, best_state, best_norm


# ---------------------------------------------------------------------------
# per-layer sweep
# ---------------------------------------------------------------------------

def run_sweep(
    cfg: PipelineConfig,
    n_folds: int = 5,
    seed: int = 42,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    batch_size: int = 64,
    hidden_dim: int = 256,
):
    """
    trains both probe types on every cached layer w/ K-fold CV.
    saves the best-fold probe weights + per-fold metrics + aggregate
    mean/std for each (layer, probe_type) into the dataset's probes/ dir.
    """
    all_results = []

    for layer in cfg.probe_layers:
        layer_path = cfg.activations_path / f"layer_{layer:02d}.pt"
        if not layer_path.exists():
            log.warning(f"skipping layer {layer} -- no activation file")
            continue

        acts, labels, d_model = load_layer_data(cfg, layer)

        # check we actually have both classes -- gpt2 local mode usually
        # fails every problem so all labels = 1. stratified split would
        # blow up in this case, easier to skip cleanly.
        labels_int = labels.int()
        n_pos = int((labels_int == 1).sum())
        n_neg = int((labels_int == 0).sum())
        if n_pos == 0 or n_neg == 0:
            log.warning(
                f"layer {layer}: only one class present "
                f"(pos={n_pos}, neg={n_neg}) -- skipping"
            )
            continue
        if n_pos < n_folds or n_neg < n_folds:
            log.warning(
                f"layer {layer}: too few minority samples "
                f"(pos={n_pos}, neg={n_neg}) for {n_folds}-fold CV -- skipping"
            )
            continue

        fail_rate = float(labels.mean())
        log.info("-" * 72)
        log.info(
            f"layer {layer:2d}: N={len(labels)} "
            f"(fail rate: {fail_rate:.1%})  d_model={d_model}"
        )

        probe_specs = [
            ("linear", LinearProbe, {"d_model": d_model}),
            ("mlp", MLPProbe, {"d_model": d_model, "hidden_dim": hidden_dim}),
        ]

        for probe_name, ProbeClass, pkwargs in probe_specs:
            log.info(f"  training {probe_name} probe ({n_folds}-fold CV)...")

            per_fold, best_fold_idx, best_state, best_norm = cross_validate_probe(
                ProbeClass, pkwargs, acts, labels,
                n_folds=n_folds, seed=seed,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, device=cfg.device,
            )

            mean, std = aggregate_folds(per_fold)

            log.info(
                f"  >> {probe_name:>6} | "
                f"acc={mean['accuracy']:.3f}±{std['accuracy']:.3f}  "
                f"bal={mean['balanced_accuracy']:.3f}±{std['balanced_accuracy']:.3f}  "
                f"f1={mean['f1']:.3f}±{std['f1']:.3f}  "
                f"auroc={mean['auroc']:.3f}±{std['auroc']:.3f}"
            )

            # save the best-fold probe weights + the norm stats from THAT
            # specific fold so anyone reloading it can apply the exact
            # transform the probe was trained against
            if best_state is not None and best_norm is not None:
                norm_mean, norm_std_t = best_norm
                save_payload = {
                    "probe_state_dict": best_state,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std_t,
                    "layer": layer,
                    "probe_type": probe_name,
                    "d_model": d_model,
                    "hidden_dim": hidden_dim if probe_name == "mlp" else None,
                    "best_fold_idx": best_fold_idx,
                    "metrics_per_fold": per_fold,
                    "metrics_mean": mean,
                    "metrics_std": std,
                    "cv_config": {
                        "n_folds": n_folds,
                        "seed": seed,
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                    },
                }
                probe_path = cfg.probes_path / f"layer_{layer:02d}_{probe_name}.pt"
                torch.save(save_payload, probe_path)

            all_results.append({
                "layer": layer,
                "probe": probe_name,
                "n_folds": n_folds,
                "best_fold_idx": best_fold_idx,
                "per_fold": per_fold,
                "mean": mean,
                "std": std,
            })

    # save sweep summary -- top-level dict so the run config travels w/
    # the results, makes the json self-documenting
    summary = {
        "config": {
            "dataset": cfg.dataset,
            "model_name": cfg.model_name,
            "probe_layers": cfg.probe_layers,
            "n_folds": n_folds,
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
        },
        "results": all_results,
    }
    results_path = cfg.probes_path / "sweep_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nsweep results saved to {results_path}")

    _print_summary(all_results, n_folds)
    return all_results


# ---------------------------------------------------------------------------
# pretty summary
# ---------------------------------------------------------------------------

def _print_summary(results, n_folds):
    if not results:
        log.warning("no results to summarize")
        return

    log.info("")
    log.info("=" * 96)
    log.info(f"summary: per-layer mean ± std over {n_folds}-fold CV")
    log.info("-" * 96)
    header = (
        f"{'layer':>5} {'probe':>7}  "
        f"{'acc':>13} {'bal_acc':>13} "
        f"{'f1':>13} {'auroc':>13}"
    )
    log.info(header)
    log.info("-" * 96)
    for r in results:
        m, s = r["mean"], r["std"]
        log.info(
            f"{r['layer']:5d} {r['probe']:>7}  "
            f"{m['accuracy']:.3f}±{s['accuracy']:.3f}  "
            f"{m['balanced_accuracy']:.3f}±{s['balanced_accuracy']:.3f}  "
            f"{m['f1']:.3f}±{s['f1']:.3f}  "
            f"{m['auroc']:.3f}±{s['auroc']:.3f}"
        )
    log.info("=" * 96)

    # best by AUROC (threshold-independent, so its our primary criterion now)
    valid = [r for r in results if not np.isnan(r["mean"]["auroc"])]
    if valid:
        best = max(valid, key=lambda x: x["mean"]["auroc"])
        m, s = best["mean"], best["std"]
        log.info(
            f"best by AUROC: layer {best['layer']} {best['probe']}  "
            f"(auroc={m['auroc']:.3f}±{s['auroc']:.3f}, "
            f"bal_acc={m['balanced_accuracy']:.3f}±{s['balanced_accuracy']:.3f}, "
            f"f1={m['f1']:.3f}±{s['f1']:.3f})"
        )

    # majority-class baseline -- printed alongside so its impossible to
    # accidentally read the f1 numbers without the imbalance context
    log.info("")
    log.info("majority-class baseline (always-predict-fail) for context:")
    log.info("  acc=0.730  prec=0.730  rec=1.000  f1=0.844  bal_acc=0.500  auroc=0.500")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "train linear + mlp probes across all cached layers w/ "
            "stratified k-fold CV and imbalance-aware metrics"
        )
    )
    parser.add_argument(
        "--dataset", default="mbpp", choices=["mbpp", "humaneval"],
        help="probes are trained on mbpp activations; humaneval is for "
             "completeness if you cache activations there too",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="use gpt2 + outputs_local for testing",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="number of stratified CV folds (default 5)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="hidden size for the mlp probe",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file", default=None,
        help="optional path to also tee logs into (handy for slurm runs)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=args.log_file)
    cfg = get_config(local=args.local, dataset=args.dataset)

    if args.dataset != "mbpp":
        log.warning(
            "running probe sweep on a non-mbpp dataset. our project plan "
            "trains probes on mbpp only -- only do this if you've also "
            "run cache_activations against this dataset."
        )

    log_config(cfg, log)
    run_sweep(
        cfg,
        n_folds=args.n_folds,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
