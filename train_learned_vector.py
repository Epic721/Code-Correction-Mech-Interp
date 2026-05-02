"""
train_learned_vector.py - learn a steering vector v_learn at each target
layer via gradient descent on a frozen Gemma-2-2b-it.

phase 2, module 4. this is the supervised counterpart to vector_extraction:
instead of computing v from cached activation statistics (CAA / SAE), we
optimize a single trainable parameter via the standard CLM loss.

setup:
  - load gemma-2-2b-it through HookedTransformer
  - freeze every weight: model.requires_grad_(False)
  - v_learn = nn.Parameter(zeros(d_model))   (the only trainable thing)
  - register a forward hook at blocks.{layer}.hook_resid_post that
    subtracts v_learn from each example's final-prompt-token position
    (and ONLY that position) during teacher-forced training
  - train with AdamW on (prompt, reference_code) pairs from MBPP train split,
    masking prompt + padding positions out of the cross-entropy loss

sign convention: v_learn is trained such that subtracting it at the final
prompt token reduces CLM loss against the canonical solution. so v_learn
points TOWARD failure (consistent w/ CAA + SAE), and the universal
inference rule x' = x - alpha * v applies as-is.

usage:
    python train_learned_vector.py --layers 12,20,25
    python train_learned_vector.py --epochs 20 --lr 1e-2
    python train_learned_vector.py --local      # cpu smoke test on gpt2
"""

import json
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformer_lens import HookedTransformer

from config import get_config, setup_logging, log_config, PipelineConfig
from eval_sandbox import MBPPAdapter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_tl_model(cfg: PipelineConfig) -> HookedTransformer:
    """
    same loading approach as cache_activations -- from_pretrained_no_processing
    for bf16/fp16 to keep the residual stream in raw HF basis (matches what we
    cached + what eval_steering will hook into) and avoid the memory spike
    that the standard processing path causes
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

    log.info(
        f"  loaded: n_layers={model.cfg.n_layers}  "
        f"d_model={model.cfg.d_model}  "
        f"d_vocab={model.cfg.d_vocab}"
    )
    return model


# ---------------------------------------------------------------------------
# training data
# ---------------------------------------------------------------------------

def build_training_pair(
    adapter: MBPPAdapter, problem, tokenizer, max_length: int = 1024,
):
    """
    for one MBPP train problem, build (full_ids, labels, prompt_end_idx).

    full_ids = tokens(prompt) ++ tokens(reference_code)
    labels   = -100 at prompt positions, actual ids at code positions
    prompt_end_idx = position of the last prompt token (where v_learn fires)

    we wrap the reference code in markdown fences because the model's natural
    generation distribution at inference is markdown-wrapped (we confirmed
    this in milestone outputs). training on raw code would conflate two
    objectives: suppressing fences AND learning correct logic.
    """
    prompt = adapter.build_prompt(problem, tokenizer)

    code_text = (problem.get("code") or "").strip()
    if not code_text:
        raise ValueError("problem has empty code field")

    reference = f"```python\n{code_text}\n```"

    # add_special_tokens=False because the chat template already wrote BOS
    # into the prompt string, and the reference code has no special tokens
    p_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    c_ids = tokenizer(
        reference, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]

    full_ids = torch.cat([p_ids, c_ids])
    if len(full_ids) > max_length:
        # rare; trim trailing code rather than the prompt so we never
        # corrupt the prompt_end_idx
        full_ids = full_ids[:max_length]

    prompt_end_idx = len(p_ids) - 1
    code_start = len(p_ids)
    code_end = min(len(full_ids), len(p_ids) + len(c_ids))

    labels = torch.full_like(full_ids, -100)
    labels[code_start:code_end] = full_ids[code_start:code_end]

    return full_ids, labels, prompt_end_idx


def load_training_data(
    cfg: PipelineConfig, tokenizer, val_frac: float = 0.1, seed: int = 42,
):
    """
    load mbpp train split, build (full_ids, labels, prompt_end_idx) tuples,
    return (train, val) lists.
    """
    log.info("loading mbpp train split for v_learn training")
    adapter = MBPPAdapter()

    # smaller cap for local mode so the smoke test finishes quickly
    split = "train"
    ds = load_dataset("google-research-datasets/mbpp", split=split)
    if cfg.num_problems is not None:
        ds = ds.select(range(min(cfg.num_problems, len(ds))))
    log.info(f"  {len(ds)} problems in mbpp {split} split")

    pairs = []
    skipped = 0
    for problem in ds:
        try:
            pairs.append(build_training_pair(adapter, problem, tokenizer))
        except Exception as e:
            log.warning(
                f"skipping task {problem.get('task_id', '?')}: {e}"
            )
            skipped += 1

    log.info(f"  built {len(pairs)} training pairs ({skipped} skipped)")

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=rng).tolist()
    n_val = max(1, int(val_frac * len(pairs))) if val_frac > 0 else 0
    val_idx_set = set(perm[:n_val])
    train = [p for i, p in enumerate(pairs) if i not in val_idx_set]
    val = [p for i, p in enumerate(pairs) if i in val_idx_set]

    log.info(f"  split: {len(train)} train / {len(val)} val")

    # sanity sample so we know the prompt+code wrapping looks right
    if pairs:
        sample_full, _, sample_pe = pairs[0]
        log.info(
            f"  sample: total_len={len(sample_full)}  "
            f"prompt_end_idx={sample_pe}"
        )

    return train, val


def collate_batch(items, pad_token_id: int, device: str):
    """
    right-pad to max length in the batch. since attention is causal,
    real tokens never see padding tokens (theyre to the right and
    "future" w.r.t. the causal mask), so we dont strictly need an
    attention_mask -- but we pass one anyway for robustness against
    any future TL changes.
    """
    max_len = max(len(it[0]) for it in items)
    B = len(items)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    prompt_end = torch.zeros(B, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, (full_ids, lbl, p_end) in enumerate(items):
        L = len(full_ids)
        input_ids[i, :L] = full_ids
        labels[i, :L] = lbl
        prompt_end[i] = p_end
        attention_mask[i, :L] = 1

    return (
        input_ids.to(device),
        labels.to(device),
        prompt_end.to(device),
        attention_mask.to(device),
    )


def shuffle_iter(items, batch_size: int, seed: int):
    """yield mini-batches of `items` in a freshly-shuffled order"""
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(items), generator=rng).tolist()
    for i in range(0, len(perm), batch_size):
        yield [items[j] for j in perm[i:i + batch_size]]


# ---------------------------------------------------------------------------
# the steering hook (training-time, surgical mode)
# ---------------------------------------------------------------------------

def make_v_learn_hook(prompt_end_indices: torch.Tensor, v_learn: nn.Parameter):
    """
    forward hook for blocks.{layer}.hook_resid_post that subtracts v_learn
    from each example's prompt_end_idx position (and ONLY that position).

    implementation note:
    we build a [B, T] mask via scatter_, broadcast to [B, T, D], then
    return activation - mask * v_learn. this is fully functional (no
    in-place ops, no clone) so autograd is happy and the gradient path
    v_learn -> mask*v -> downstream layers -> loss is unbroken.

    sign convention:
    we SUBTRACT v_learn during training, so the trained vector points
    toward failure. at inference, the universal rule x' = x - alpha * v
    applies the same direction at any chosen alpha.
    """

    def hook_fn(activation, hook_obj):
        v_cast = v_learn.to(activation.dtype)
        B, T, D = activation.shape

        # mask[b, t] = 1 iff t == prompt_end_indices[b], else 0
        mask = torch.zeros(
            (B, T), dtype=activation.dtype, device=activation.device
        )
        mask.scatter_(1, prompt_end_indices.unsqueeze(1), 1.0)

        return activation - mask.unsqueeze(-1) * v_cast.view(1, 1, D)

    return hook_fn


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    standard causal LM cross-entropy. shifts labels by 1 so logits at
    position t predicts labels at position t+1. ignore_index=-100 silently
    masks out prompt + padding positions, leaving loss on code only.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# per-layer training
# ---------------------------------------------------------------------------

def evaluate_val(
    model, v_learn, val_data, layer, cfg, pad_token_id, batch_size,
) -> float:
    """compute mean CLM loss on val set without taking gradient steps"""
    if not val_data:
        return float("nan")

    hook_name = f"blocks.{layer}.hook_resid_post"
    losses = []
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i + batch_size]
            input_ids, labels, p_end, attn_mask = collate_batch(
                batch, pad_token_id, cfg.device
            )

            model.reset_hooks()
            model.add_hook(hook_name, make_v_learn_hook(p_end, v_learn))
            logits = model(input_ids, attention_mask=attn_mask)
            model.reset_hooks()

            losses.append(causal_lm_loss(logits, labels).item())

    return sum(losses) / max(len(losses), 1)


def train_one_layer(
    model, train_data, val_data, layer, cfg,
    epochs: int = 10, lr: float = 5e-3,
    weight_decay: float = 0.0, batch_size: int = 4, seed: int = 42,
):
    """
    optimize v_learn at the given layer. fresh zero init each call so
    the sweep across multiple layers doesnt have any cross-layer leakage.

    returns (v_learn, history_dict, best_val_loss).
    """
    d_model = model.cfg.d_model
    v_learn = nn.Parameter(
        torch.zeros(d_model, device=cfg.device, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        [v_learn], lr=lr, weight_decay=weight_decay
    )

    pad_token_id = (
        model.tokenizer.pad_token_id
        if model.tokenizer.pad_token_id is not None
        else model.tokenizer.eos_token_id
    )
    hook_name = f"blocks.{layer}.hook_resid_post"

    # baseline val loss (alpha=0 effectively, since v_learn is zeros)
    baseline_val = evaluate_val(
        model, v_learn, val_data, layer, cfg, pad_token_id, batch_size
    )
    log.info(f"  baseline val_loss (v=0): {baseline_val:.4f}")

    history = {"train_loss": [], "val_loss": [], "v_norm": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()  # no-op since model is frozen, but flips dropout if any
        train_losses = []

        for batch in shuffle_iter(train_data, batch_size, seed=seed * epoch):
            input_ids, labels, p_end, attn_mask = collate_batch(
                batch, pad_token_id, cfg.device
            )

            model.reset_hooks()
            model.add_hook(hook_name, make_v_learn_hook(p_end, v_learn))
            logits = model(input_ids, attention_mask=attn_mask)
            loss = causal_lm_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.reset_hooks()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = evaluate_val(
            model, v_learn, val_data, layer, cfg, pad_token_id, batch_size
        )
        v_norm = float(v_learn.norm().item())
        ep_time = time.time() - ep_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["v_norm"].append(v_norm)

        log.info(
            f"  epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"|v|={v_norm:.4f}  "
            f"({ep_time:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = v_learn.detach().cpu().clone()

    # restore best-val-loss checkpoint
    if best_state is not None:
        v_learn.data.copy_(best_state.to(v_learn.device))

    log.info(
        f"  done. best_val_loss={best_val_loss:.4f}  "
        f"(baseline: {baseline_val:.4f}, "
        f"delta: {best_val_loss - baseline_val:+.4f})"
    )

    history["baseline_val_loss"] = baseline_val
    return v_learn, history, best_val_loss


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_vector(path, vector: torch.Tensor, **metadata):
    payload = {
        "vector": vector.cpu().float(),
        "norm": float(vector.norm()),
        **metadata,
    }
    torch.save(payload, path)
    log.info(f"  saved -> {path}  (norm={payload['norm']:.3f})")


# ---------------------------------------------------------------------------
# main runner
# ---------------------------------------------------------------------------

def run_training(
    cfg: PipelineConfig, layers: list[int],
    epochs: int, lr: float, weight_decay: float, batch_size: int, seed: int,
):
    model = load_tl_model(cfg)
    model.requires_grad_(False)

    # sanity: confirm all model params are frozen
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"model frozen: {n_trainable}/{n_params} model params trainable "
        f"({n_trainable / n_params:.4%}) -- should be 0/N"
    )

    train_data, val_data = load_training_data(
        cfg, model.tokenizer, val_frac=0.1, seed=seed
    )

    full_summary = {
        "config": {
            "dataset": cfg.dataset,
            "model_name": cfg.model_name,
            "layers": layers,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
            "training_split": "mbpp_train",
            "n_train_examples": len(train_data),
            "n_val_examples": len(val_data),
            "sign_convention": "v_points_toward_failure (x' = x - alpha * v)",
        },
        "results": [],
    }

    for layer in layers:
        log.info("=" * 72)
        log.info(f"training v_learn at layer {layer}")
        log.info("=" * 72)

        layer_start = time.time()
        v, history, best_val = train_one_layer(
            model, train_data, val_data, layer, cfg,
            epochs=epochs, lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size, seed=seed,
        )
        layer_elapsed = time.time() - layer_start

        out_path = cfg.vectors_path / f"learned_layer_{layer:02d}.pt"
        save_vector(
            out_path, v.detach(),
            method="learned",
            layer=layer,
            sign_convention="v_points_toward_failure",
            diag={
                "training_split": "mbpp_train",
                "n_train_examples": len(train_data),
                "n_val_examples": len(val_data),
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "best_val_loss": best_val,
                "baseline_val_loss": history["baseline_val_loss"],
                "loss_history": {
                    "train_loss": history["train_loss"],
                    "val_loss": history["val_loss"],
                    "v_norm": history["v_norm"],
                },
                "layer_train_seconds": layer_elapsed,
            },
        )

        full_summary["results"].append({
            "layer": layer,
            "norm": float(v.norm()),
            "best_val_loss": best_val,
            "baseline_val_loss": history["baseline_val_loss"],
            "delta_val_loss": best_val - history["baseline_val_loss"],
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "history": {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "v_norm": history["v_norm"],
            },
            "elapsed_seconds": layer_elapsed,
        })

        # release optimizer state + parameter graph between layers
        del v
        torch.cuda.empty_cache() if cfg.device == "cuda" else None

    summary_path = cfg.vectors_path / "learned_vectors_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    log.info(f"\nlearned vectors summary -> {summary_path}")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _parse_layer_csv(s: str | None) -> list[int] | None:
    if s is None:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "train one steering vector per target layer via gradient "
            "descent against MBPP reference code (frozen LLM)"
        )
    )
    parser.add_argument(
        "--local", action="store_true",
        help="local cpu smoke test on gpt2 + outputs_local",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="comma-separated target layers (default: cfg.target_layers)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--lr", type=float, default=5e-3,
        help="learning rate for AdamW. higher than typical LLM finetuning "
             "since v_learn has only d_model parameters",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="drop to 2 or 1 if you OOM on the cluster",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file", default=None,
        help="optional path to also tee logs into (handy for slurm)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=args.log_file)
    cfg = get_config(local=args.local, dataset="mbpp")

    layers = _parse_layer_csv(args.layers) or cfg.target_layers
    log.info(f"training learned vectors at layers: {layers}")
    log_config(cfg, log)

    run_training(
        cfg, layers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
