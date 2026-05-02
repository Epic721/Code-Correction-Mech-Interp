"""
train_probes.py - trains linear and mlp probes on cached residual stream
activations to detect whether the model's generation will fail or pass.

sweeps across all layers defined in config to find where the "error"
signal is most linearly (or nonlinearly) accessible. this directly
answers the grader's question about early vs late layer intervention.

labels: fail=1, pass=0  (we're training an error detector)

usage:
    python train_probes.py --local
    python train_probes.py
    python train_probes.py --epochs 200 --lr 5e-4
"""

import json
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import get_config, setup_logging, PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# probe architectures
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
# data loading + preprocessing
# ---------------------------------------------------------------------------

def load_layer_data(cfg, layer):
    """loads cached activations + labels for a single layer"""
    path = cfg.activations_path / f"layer_{layer:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no activation file at {path}")

    data = torch.load(path, weights_only=False, map_location="cpu")
    return data["activations"], data["labels"], data["d_model"]


def make_splits(acts, labels, train_frac=0.8, seed=42):
    """deterministic train/val split using a fixed seed"""
    n = len(acts)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)

    n_train = int(train_frac * n)
    tr_idx = perm[:n_train]
    val_idx = perm[n_train:]

    return (
        acts[tr_idx], labels[tr_idx],
        acts[val_idx], labels[val_idx],
    )


def normalize(train_acts, val_acts):
    """zero-mean unit-var normalization, fit on train only"""
    mean = train_acts.mean(dim=0)
    std = train_acts.std(dim=0).clamp(min=1e-8)
    return (
        (train_acts - mean) / std,
        (val_acts - mean) / std,
        mean, std,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits, labels):
    """accuracy / precision / recall / f1 from raw logits (threshold=0)"""
    preds = (logits > 0).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def train_probe(probe, train_acts, train_labels, val_acts, val_labels,
                epochs=100, lr=1e-3, weight_decay=0.01, batch_size=64,
                device="cpu", log_every=25):
    """
    standard pytorch training loop w/ BCEWithLogitsLoss + AdamW.
    tracks best val loss and restores those weights at the end.
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

        if epoch % log_every == 0 or epoch == epochs:
            m = compute_metrics(val_logits, val_y)
            log.info(
                f"  ep {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={m['accuracy']:.3f}  f1={m['f1']:.3f}"
            )

    # restore the checkpoint w/ lowest val loss
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe = probe.to(device)
    probe.eval()
    return probe


# ---------------------------------------------------------------------------
# per-layer sweep
# ---------------------------------------------------------------------------

def run_sweep(cfg, epochs=100, lr=1e-3, weight_decay=0.01,
              batch_size=64, train_frac=0.8, seed=42, hidden_dim=256):
    """
    trains both probe types on every layer and logs all the metrics.
    saves individual probe weights + a summary json for the report.
    """
    all_results = []

    for layer in cfg.probe_layers:
        layer_path = cfg.activations_path / f"layer_{layer:02d}.pt"
        if not layer_path.exists():
            log.warning(f"skipping layer {layer} -- no activation file")
            continue

        acts, labels, d_model = load_layer_data(cfg, layer)
        tr_acts, tr_labels, val_acts, val_labels = make_splits(
            acts, labels, train_frac=train_frac, seed=seed
        )

        # quick class balance check (useful context for interpreting metrics)
        fail_rate = tr_labels.mean().item()
        log.info(
            f"layer {layer:2d}: {len(tr_labels)} train / "
            f"{len(val_labels)} val  "
            f"(fail rate: {fail_rate:.1%})"
        )

        # normalize features -- fit on train, apply to both
        tr_acts_n, val_acts_n, norm_mean, norm_std = normalize(
            tr_acts, val_acts
        )

        probe_specs = [
            ("linear", LinearProbe, {"d_model": d_model}),
            ("mlp", MLPProbe, {"d_model": d_model, "hidden_dim": hidden_dim}),
        ]

        for probe_name, ProbeClass, pkwargs in probe_specs:
            log.info(f"  training {probe_name} probe...")

            probe = ProbeClass(**pkwargs)
            probe = train_probe(
                probe, tr_acts_n, tr_labels, val_acts_n, val_labels,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, device=cfg.device,
            )

            # final metrics on val set
            with torch.no_grad():
                val_logits = probe(val_acts_n.to(cfg.device))
            metrics = compute_metrics(val_logits, val_labels.to(cfg.device))

            log.info(
                f"  >> {probe_name:>6}  "
                f"acc={metrics['accuracy']:.3f}  "
                f"prec={metrics['precision']:.3f}  "
                f"rec={metrics['recall']:.3f}  "
                f"f1={metrics['f1']:.3f}"
            )

            # save probe weights + normalization stats so we can reload later
            save_payload = {
                "probe_state_dict": {
                    k: v.cpu() for k, v in probe.state_dict().items()
                },
                "norm_mean": norm_mean,
                "norm_std": norm_std,
                "layer": layer,
                "probe_type": probe_name,
                "d_model": d_model,
                "hidden_dim": hidden_dim if probe_name == "mlp" else None,
                "metrics": metrics,
            }
            probe_path = cfg.probes_path / f"layer_{layer:02d}_{probe_name}.pt"
            torch.save(save_payload, probe_path)

            all_results.append({
                "layer": layer,
                "probe": probe_name,
                **metrics,
            })

    # save sweep summary for the report
    results_path = cfg.probes_path / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nsweep results saved to {results_path}")

    _print_summary(all_results)

    return all_results


def _print_summary(results):
    """logs a clean summary table"""
    if not results:
        log.warning("no results to summarize")
        return

    log.info("")
    header = f"{'layer':>5} {'probe':>7} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7}"
    log.info(header)
    log.info("-" * len(header))

    for r in results:
        log.info(
            f"{r['layer']:5d} {r['probe']:>7} "
            f"{r['accuracy']:7.3f} {r['precision']:7.3f} "
            f"{r['recall']:7.3f} {r['f1']:7.3f}"
        )

    best = max(results, key=lambda x: x["f1"])
    log.info("-" * len(header))
    log.info(
        f"best: layer {best['layer']} {best['probe']} "
        f"(f1={best['f1']:.3f})"
    )


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="train linear + mlp probes across all cached layers"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="use local config + outputs",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="hidden size for the mlp probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = get_config(local=args.local)

    log.info(f"device={cfg.device}  probe layers={cfg.probe_layers}")

    run_sweep(
        cfg,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
