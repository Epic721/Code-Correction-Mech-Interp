"""
vector_extraction.py - extract unsupervised steering vectors from the
cached MBPP activations.

phase 2, module 3. produces two candidate vectors per layer that we'll
later sweep through eval_steering.py:

  CAA (Contrastive Activation Addition):
    v_CAA = mu_fail - mu_pass
    cheap, computed for every cached layer, totally model-free.

  SAE (Sparse Autoencoder, via saelens + Gemma Scope):
    1. encode each cached activation with the layer's pretrained SAE
    2. for each feature, compute Welch's t-statistic between the
       fail and pass populations
    3. pick i* = argmax t  (feature firing most strongly on failures)
    4. v_SAE = W_dec[i*] -- the direction this feature adds to the
       residual stream when it fires

both methods produce vectors that point TOWARD failure per our agreed
sign convention, so the universal inference rule x' = x - alpha * v
subtracts the error direction at steering time.

vectors are saved to outputs/vectors/ -- this dir is dataset-agnostic
since the same vectors get evaluated on both mbpp and humaneval downstream.

usage:
    python vector_extraction.py                              # default: all CAA layers + SAE at 12, 20
    python vector_extraction.py --no-sae                     # CAA only (no GPU needed)
    python vector_extraction.py --sae-layers 12,20,25        # custom SAE layer set
    python vector_extraction.py --sae-width 32k              # bigger Gemma Scope SAE
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

from config import get_config, setup_logging, log_config, PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_layer_activations(cfg: PipelineConfig, layer: int):
    """returns (acts [N, d_model], labels [N], task_ids list)"""
    path = cfg.activations_path / f"layer_{layer:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no cached activations at {path}")

    data = torch.load(path, weights_only=False, map_location="cpu")
    return data["activations"], data["labels"], data.get("task_ids", [])


def has_layer_activations(cfg: PipelineConfig, layer: int) -> bool:
    return (cfg.activations_path / f"layer_{layer:02d}.pt").exists()


def discover_cached_layers(cfg: PipelineConfig) -> list[int]:
    """find every layer that has a cached activations file in this dataset"""
    layers: list[int] = []
    for p in sorted(cfg.activations_path.glob("layer_*.pt")):
        try:
            layers.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return layers


# ---------------------------------------------------------------------------
# CAA: contrastive activation addition
# ---------------------------------------------------------------------------

def extract_caa_vector(acts: torch.Tensor, labels: torch.Tensor):
    """
    v_CAA = mu_fail - mu_pass.

    by our agreed sign convention v points toward failure, so subtracting
    alpha*v at inference pushes the activation away from the failure mean
    and back toward the pass mean.
    """
    fail_mask = labels == 1
    pass_mask = labels == 0

    n_fail = int(fail_mask.sum())
    n_pass = int(pass_mask.sum())

    if n_fail == 0 or n_pass == 0:
        raise ValueError(
            f"need both classes for CAA, got n_fail={n_fail}, n_pass={n_pass}"
        )

    acts_f = acts.float()
    mu_fail = acts_f[fail_mask].mean(dim=0)
    mu_pass = acts_f[pass_mask].mean(dim=0)
    v_caa = mu_fail - mu_pass

    diag = {
        "n_fail": n_fail,
        "n_pass": n_pass,
        "mu_fail_norm": float(mu_fail.norm()),
        "mu_pass_norm": float(mu_pass.norm()),
        "v_caa_norm": float(v_caa.norm()),
        # context for calibrating the alpha sweep later: typical activation
        # norms at this layer give us a sense of what "1 unit of v" means
        "act_mean_norm": float(acts_f.norm(dim=1).mean()),
        "act_std_norm": float(acts_f.norm(dim=1).std()),
    }
    return v_caa, diag


# ---------------------------------------------------------------------------
# SAE: Gemma Scope feature extraction
# ---------------------------------------------------------------------------

def load_gemma_scope_sae(
    layer: int,
    release: str = "gemma-scope-2b-pt-res-canonical",
    sae_id_template: str = "layer_{layer}/width_16k/canonical",
    device: str = "cuda",
):
    """
    load a Gemma Scope SAE for hook_resid_post at the given layer.
    requires saelens >= 4.0 and an internet connection on first call;
    subsequent calls hit the local hf cache.

    important: Gemma Scope's residual SAEs are trained on hook_resid_post,
    which is the same hook we cached activations from. if you ever swap
    in a different release that targets resid_pre or attn_out, the
    decoder directions will not be in the right basis.
    """
    try:
        from sae_lens import SAE
    except ImportError as e:
        raise ImportError(
            "saelens is required for SAE vector extraction. "
            "install w/ `pip install sae-lens`."
        ) from e

    sae_id = sae_id_template.format(layer=layer)
    log.info(f"loading SAE  release={release}  sae_id={sae_id}  device={device}")

    sae, _cfg_dict, _sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device,
    )
    sae.eval()

    log.info(
        f"  loaded: d_in={sae.cfg.d_in}  d_sae={sae.cfg.d_sae}  "
        f"hook={sae.cfg.hook_name}"
    )
    return sae


def sae_encode_batched(
    sae, acts: torch.Tensor, batch_size: int = 64
) -> torch.Tensor:
    """
    forward acts through the SAE encoder in batches. returns feature
    activations of shape [N, d_sae] on cpu / float32 so the downstream
    t-stat math is in a stable dtype.
    """
    sae_device = sae.W_enc.device
    sae_dtype = sae.W_enc.dtype

    feats = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            batch = acts[i:i + batch_size].to(sae_device, dtype=sae_dtype)
            f = sae.encode(batch)
            feats.append(f.detach().cpu().float())
    return torch.cat(feats, dim=0)


def welch_t_per_feature(
    f_fail: torch.Tensor, f_pass: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Welch's t-statistic per feature:

        t_i = (mean_i_fail - mean_i_pass)
              / sqrt(var_i_fail / n_fail + var_i_pass / n_pass)

    positive t = feature fires more on failures, which is what we want.

    robustness:
      - clamp the SE so we never divide by 0
      - features that never fire on either side return t = 0 so they cant
        win argmax (else sparse autoencoder = lots of dead features)
      - any residual nan / inf gets zeroed via nan_to_num
    """
    n_fail = f_fail.shape[0]
    n_pass = f_pass.shape[0]

    mean_fail = f_fail.mean(dim=0)
    mean_pass = f_pass.mean(dim=0)

    # ddof=1 (Bessel-corrected) for proper t-stat. requires N > 1.
    var_fail = (
        f_fail.var(dim=0, unbiased=True) if n_fail > 1
        else torch.zeros_like(mean_fail)
    )
    var_pass = (
        f_pass.var(dim=0, unbiased=True) if n_pass > 1
        else torch.zeros_like(mean_pass)
    )

    se = torch.sqrt(
        (var_fail / max(n_fail, 1) + var_pass / max(n_pass, 1)).clamp(min=eps)
    )
    t = (mean_fail - mean_pass) / se

    fired_anywhere = (
        (f_fail.abs().max(dim=0).values > 0)
        | (f_pass.abs().max(dim=0).values > 0)
    )
    t = torch.where(fired_anywhere, t, torch.zeros_like(t))
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def extract_sae_vector(
    sae, acts: torch.Tensor, labels: torch.Tensor,
    batch_size: int = 64, top_k_diagnostic: int = 10,
):
    """
    full SAE pipeline:
      1. encode every cached activation through the SAE -> features [N, d_sae]
      2. Welch t-stat per feature comparing fail vs pass
      3. i* = argmax t  (feature fires most strongly on failures)
      4. v_SAE = W_dec[i*] in residual stream space

    returns (v_sae [d_model], diag_dict, t_full [d_sae]).
    """
    feats = sae_encode_batched(sae, acts, batch_size=batch_size)

    fail_mask = labels == 1
    pass_mask = labels == 0
    f_fail = feats[fail_mask]
    f_pass = feats[pass_mask]

    t = welch_t_per_feature(f_fail, f_pass)
    top_idx = int(t.argmax().item())

    # decoder row: saelens layout is W_dec[d_sae, d_in], so W_dec[i] gives
    # the [d_in]-dim direction that feature i adds when it fires with mag 1
    v_sae = sae.W_dec[top_idx].detach().cpu().float()

    # top-k by t-stat -- super useful for the report. we can inspect these
    # features qualitatively (eg via Neuronpedia) and discuss what the
    # model's "top error feature" actually represents
    topk_idx = torch.topk(t, k=min(top_k_diagnostic, t.numel())).indices
    topk_records = []
    for idx in topk_idx.tolist():
        topk_records.append({
            "feature_idx": int(idx),
            "t_stat": float(t[idx]),
            "mean_fail": float(f_fail[:, idx].mean()),
            "mean_pass": float(f_pass[:, idx].mean()),
            "fire_rate_fail": float((f_fail[:, idx] > 0).float().mean()),
            "fire_rate_pass": float((f_pass[:, idx] > 0).float().mean()),
        })

    n_features_active = int(((feats.abs() > 0).any(dim=0)).sum())

    diag = {
        "top_feature_idx": top_idx,
        "top_t_stat": float(t[top_idx]),
        "top_mean_fail": float(f_fail[:, top_idx].mean()),
        "top_mean_pass": float(f_pass[:, top_idx].mean()),
        "top_fire_rate_fail": float((f_fail[:, top_idx] > 0).float().mean()),
        "top_fire_rate_pass": float((f_pass[:, top_idx] > 0).float().mean()),
        "v_sae_norm": float(v_sae.norm()),
        "n_fail": int(fail_mask.sum()),
        "n_pass": int(pass_mask.sum()),
        "n_features_total": int(feats.shape[1]),
        "n_features_active": n_features_active,
        "top_k_features": topk_records,
    }
    return v_sae, diag, t


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_vector(path: Path, vector: torch.Tensor, **metadata):
    """save a vector + arbitrary metadata as a torch payload"""
    payload = {
        "vector": vector.cpu().float(),
        "norm": float(vector.norm()),
        **metadata,
    }
    torch.save(payload, path)
    log.info(f"  saved -> {path}  (norm={payload['norm']:.3f})")


# ---------------------------------------------------------------------------
# pipeline runners
# ---------------------------------------------------------------------------

def run_caa_extraction(cfg: PipelineConfig, layers: list[int]) -> list[dict]:
    log.info("=" * 72)
    log.info(f"CAA extraction over {len(layers)} layers")
    log.info("=" * 72)

    summary: list[dict] = []
    for layer in layers:
        if not has_layer_activations(cfg, layer):
            log.warning(f"layer {layer:2d}: no cached activations, skipping")
            continue

        acts, labels, _ = load_layer_activations(cfg, layer)
        v, diag = extract_caa_vector(acts, labels)

        log.info(
            f"layer {layer:2d}  |v_CAA|={diag['v_caa_norm']:.3f}  "
            f"|act|≈{diag['act_mean_norm']:.1f}±{diag['act_std_norm']:.1f}  "
            f"n_fail={diag['n_fail']}  n_pass={diag['n_pass']}"
        )

        out_path = cfg.vectors_path / f"caa_layer_{layer:02d}.pt"
        save_vector(
            out_path, v,
            method="caa",
            layer=layer,
            sign_convention="v_points_toward_failure",
            diag=diag,
        )
        summary.append({"method": "caa", "layer": layer, **diag})

    return summary


def run_sae_extraction(
    cfg: PipelineConfig,
    layers: list[int],
    release: str = "gemma-scope-2b-pt-res-canonical",
    sae_id_template: str = "layer_{layer}/width_16k/canonical",
    batch_size: int = 64,
) -> list[dict]:
    log.info("=" * 72)
    log.info(f"SAE extraction over {len(layers)} layers ({release})")
    log.info("=" * 72)

    summary: list[dict] = []
    for layer in layers:
        if not has_layer_activations(cfg, layer):
            log.warning(f"layer {layer:2d}: no cached activations, skipping")
            continue

        log.info("")
        log.info("-" * 60)
        log.info(f"layer {layer:2d}")
        log.info("-" * 60)

        # one SAE load per layer; they're ~300MB each at width_16k so we
        # release after each layer to avoid stacking memory
        sae = None
        try:
            sae = load_gemma_scope_sae(
                layer, release=release,
                sae_id_template=sae_id_template,
                device=cfg.device,
            )
        except Exception as e:
            log.error(f"  failed to load SAE for layer {layer}: {e}")
            continue

        try:
            acts, labels, _ = load_layer_activations(cfg, layer)
            v, diag, _t_full = extract_sae_vector(
                sae, acts, labels, batch_size=batch_size,
            )

            log.info(
                f"  top feature  idx={diag['top_feature_idx']}  "
                f"t={diag['top_t_stat']:.2f}  "
                f"fail_mean={diag['top_mean_fail']:.3f}  "
                f"pass_mean={diag['top_mean_pass']:.3f}"
            )
            log.info(
                f"  fire rates   fail={diag['top_fire_rate_fail']:.1%}  "
                f"pass={diag['top_fire_rate_pass']:.1%}"
            )
            log.info(
                f"  active features  {diag['n_features_active']:>5} / "
                f"{diag['n_features_total']}"
            )
            log.info(f"  |v_SAE|={diag['v_sae_norm']:.3f}")
            log.info("  top-5 features by t-stat:")
            for rec in diag["top_k_features"][:5]:
                log.info(
                    f"    feat {rec['feature_idx']:>5}  "
                    f"t={rec['t_stat']:6.2f}  "
                    f"fail={rec['mean_fail']:7.3f} ({rec['fire_rate_fail']:.0%})  "
                    f"pass={rec['mean_pass']:7.3f} ({rec['fire_rate_pass']:.0%})"
                )

            out_path = cfg.vectors_path / f"sae_layer_{layer:02d}.pt"
            save_vector(
                out_path, v,
                method="sae",
                layer=layer,
                sign_convention="v_points_toward_failure",
                release=release,
                sae_id=sae_id_template.format(layer=layer),
                d_sae=int(sae.cfg.d_sae),
                d_in=int(sae.cfg.d_in),
                diag=diag,
            )
            summary.append({"method": "sae", "layer": layer, **diag})
        finally:
            del sae
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    return summary


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
            "extract steering vectors (CAA + SAE) from cached MBPP activations"
        )
    )
    parser.add_argument(
        "--local", action="store_true",
        help="local mode (gpt2, outputs_local) -- SAE wont actually work "
             "for gpt2 but CAA will, useful for smoke tests",
    )
    parser.add_argument(
        "--dataset", default="mbpp", choices=["mbpp", "humaneval"],
        help="cached-activation dataset to extract from "
             "(should always be mbpp per project plan)",
    )
    parser.add_argument(
        "--caa-layers", type=str, default=None,
        help="comma-separated layer ids for CAA extraction "
             "(default: every cached layer found on disk)",
    )
    parser.add_argument(
        "--sae-layers", type=str, default="12,20",
        help="comma-separated layer ids for SAE extraction (default: 12,20)",
    )
    parser.add_argument(
        "--sae-release", default="gemma-scope-2b-pt-res-canonical",
        help="saelens release name for the Gemma Scope SAEs to use",
    )
    parser.add_argument(
        "--sae-width", default="16k",
        help="Gemma Scope width to load (16k / 32k / 65k). "
             "ignored if --sae-id-template is set explicitly",
    )
    parser.add_argument(
        "--sae-id-template", default=None,
        help="full sae_id format string with {layer} placeholder. "
             "if omitted, derived from --sae-width",
    )
    parser.add_argument(
        "--sae-batch-size", type=int, default=64,
        help="batch size for SAE encoding (drop if you OOM)",
    )
    parser.add_argument(
        "--no-caa", action="store_true",
        help="skip CAA extraction",
    )
    parser.add_argument(
        "--no-sae", action="store_true",
        help="skip SAE extraction (use this if saelens isnt installed)",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=args.log_file)
    cfg = get_config(local=args.local, dataset=args.dataset)

    if args.dataset != "mbpp":
        log.warning(
            "extracting vectors from a non-mbpp dataset. our project plan "
            "trains all vectors on mbpp activations only -- this is unusual."
        )

    log_config(cfg, log)
    log.info(f"vectors output dir: {cfg.vectors_path.resolve()}")

    cached = discover_cached_layers(cfg)
    if not cached:
        log.error(
            f"no cached activations found at {cfg.activations_path}. "
            "run cache_activations.py first."
        )
        return
    log.info(f"found cached activations for {len(cached)} layers: {cached}")

    caa_layers = _parse_layer_csv(args.caa_layers) or cached
    sae_layers = _parse_layer_csv(args.sae_layers) or []

    sae_id_template = args.sae_id_template
    if sae_id_template is None:
        sae_id_template = f"layer_{{layer}}/width_{args.sae_width}/canonical"

    full_summary = {
        "config": {
            "dataset": cfg.dataset,
            "model_name": cfg.model_name,
            "caa_layers": caa_layers,
            "sae_layers": sae_layers,
            "sae_release": args.sae_release,
            "sae_id_template": sae_id_template,
            "sae_width": args.sae_width,
            "sign_convention": "v_points_toward_failure (x' = x - alpha * v)",
        },
        "caa": [],
        "sae": [],
    }

    if not args.no_caa:
        full_summary["caa"] = run_caa_extraction(cfg, caa_layers)
    else:
        log.info("skipping CAA extraction (--no-caa)")

    if not args.no_sae:
        full_summary["sae"] = run_sae_extraction(
            cfg, sae_layers,
            release=args.sae_release,
            sae_id_template=sae_id_template,
            batch_size=args.sae_batch_size,
        )
    else:
        log.info("skipping SAE extraction (--no-sae)")

    # default=str catches anything weird (eg numpy scalars) so the json
    # write never blows up the run after we've already done the work
    summary_path = cfg.vectors_path / "vectors_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, default=str)
    log.info(f"\nfull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
