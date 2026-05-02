"""
cache_activations.py - runs the original prompts through the model
via transformerlens and caches the residual stream activations at
every layer we want to probe.

the key idea: we grab the resid_post vector at the *final prompt token*
(i.e. the position right before generation would start). thats where
the model's "intent" is most concentrated.

saves one .pt file per layer so train_probes.py can load just what it
needs without eating all our ram.

usage:
    python cache_activations.py --local
    python cache_activations.py
"""

import json
import logging
import argparse

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from config import get_config, setup_logging, PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_generations(cfg):
    """reads the labeled generations.jsonl from the eval sandbox step"""
    path = cfg.generations_path
    if not path.exists():
        raise FileNotFoundError(
            f"no generations file at {path} -- run eval_sandbox.py first"
        )

    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n_pass = sum(r["passed"] for r in records)
    n_fail = len(records) - n_pass
    log.info(f"loaded {len(records)} generations ({n_pass} pass, {n_fail} fail)")

    if n_pass == 0 or n_fail == 0:
        log.warning(
            "all samples have the same label -- probes wont learn anything "
            "useful. this is expected w/ gpt2 in local mode tho"
        )

    return records


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_tl_model(cfg):
    """loads the model through transformerlens for activation caching"""
    log.info(f"loading {cfg.model_name} via transformerlens")

    # TransformerLens can briefly spike memory when loading + processing
    # reduced-precision models via from_pretrained. The no-processing path
    # avoids that higher peak and is usually safer on shared cluster nodes.
    use_no_processing = cfg.torch_dtype in (torch.float16, torch.bfloat16)
    loader = (
        HookedTransformer.from_pretrained_no_processing
        if use_no_processing
        else HookedTransformer.from_pretrained
    )

    model = loader(
        cfg.model_name,
        device=cfg.device,
        dtype=cfg.torch_dtype,
    )

    log.info(
        f"loaded: {model.cfg.n_layers} layers, "
        f"d_model={model.cfg.d_model}, "
        f"d_vocab={model.cfg.d_vocab}"
    )
    return model


# ---------------------------------------------------------------------------
# activation caching
# ---------------------------------------------------------------------------

def _make_resid_filter(layers):
    """builds a names_filter that only keeps resid_post for target layers"""
    hooks = {f"blocks.{l}.hook_resid_post" for l in layers}
    return lambda name: name in hooks


def validate_layers(cfg, model):
    """makes sure the requested probe layers actually exist in the model"""
    n_layers = model.cfg.n_layers
    bad = [l for l in cfg.probe_layers if l < 0 or l >= n_layers]
    if bad:
        raise ValueError(
            f"probe_layers {bad} are out of range -- "
            f"{cfg.model_name} only has {n_layers} layers (0-{n_layers - 1})"
        )


def cache_all_activations(cfg):
    """
    main loop -- tokenizes each prompt, runs a forward pass through
    transformerlens, and pulls the resid_post activation at the last
    token position for every requested layer
    """
    records = load_generations(cfg)
    model = load_tl_model(cfg)
    validate_layers(cfg, model)

    names_filter = _make_resid_filter(cfg.probe_layers)

    per_layer = {l: [] for l in cfg.probe_layers}
    task_ids = []
    labels = []

    for rec in tqdm(records, desc="caching activations"):
        prompt = rec["prompt"]

        # prepend_bos=False because the chat template already includes
        # <bos> for gemma. for gpt2 local mode this just means no extra
        # bos token which is fine
        tokens = model.to_tokens(prompt, prepend_bos=False)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=names_filter
            )

        # activation at the final prompt token -- shape [d_model]
        # this is the position right before the model would start generating
        for layer in cfg.probe_layers:
            act = cache["resid_post", layer][0, -1, :]
            per_layer[layer].append(act.cpu().float())

        task_ids.append(rec["task_id"])
        labels.append(0.0 if rec["passed"] else 1.0)  # fail=1, pass=0

        del cache
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    # save one .pt per layer
    labels_t = torch.tensor(labels, dtype=torch.float32)

    for layer in cfg.probe_layers:
        acts_t = torch.stack(per_layer[layer])  # [N, d_model]

        payload = {
            "task_ids": task_ids,
            "labels": labels_t,
            "activations": acts_t,
            "layer": layer,
            "model_name": cfg.model_name,
            "d_model": int(acts_t.shape[1]),
        }

        out_path = cfg.activations_path / f"layer_{layer:02d}.pt"
        torch.save(payload, out_path)
        log.info(f"layer {layer:2d}: {tuple(acts_t.shape)} -> {out_path}")

    log.info(
        f"done -- cached {len(records)} prompts x "
        f"{len(cfg.probe_layers)} layers"
    )


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="cache residual stream activations for probing"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="use gpt2 + local outputs for testing",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = get_config(local=args.local)

    log.info(f"model={cfg.model_name}  device={cfg.device}  dtype={cfg.dtype}")
    log.info(f"caching layers: {cfg.probe_layers}")

    cache_all_activations(cfg)


if __name__ == "__main__":
    main()
