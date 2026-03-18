"""
config.py - central config for the code correction mech interp project.
handles model selection, hf auth, paths, and local-vs-cluster toggling.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """all the knobs for the pipeline in one place"""

    # model
    model_name: str = "google/gemma-2-2b-it"
    hf_token: str | None = None

    # layers we want to cache / probe (gemma-2-2b has 26 transformer blocks)
    probe_layers: list[int] = field(default_factory=lambda: list(range(26)))

    # generation
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0 = greedy

    # sandbox
    exec_timeout_sec: int = 10

    # dataset
    mbpp_split: str = "test"
    num_problems: int | None = None  # none = all 500

    # output paths
    output_dir: Path = Path("outputs")
    generations_file: str = "generations.jsonl"
    activations_dir: str = "activations"
    probes_dir: str = "probes"

    # hardware
    device: str = "auto"
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype, torch.float32)

        # grab hf token from env if caller didn't pass one in
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def generations_path(self) -> Path:
        return self.output_dir / self.generations_file

    @property
    def activations_path(self) -> Path:
        p = self.output_dir / self.activations_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def probes_path(self) -> Path:
        p = self.output_dir / self.probes_dir
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_config(local: bool = False) -> PipelineConfig:
    """
    returns a config for either local dev or the pace ice cluster.
    local mode swaps in gpt2 so you can test the whole pipeline
    end-to-end without needing a gpu or hf gated access.
    """
    if local:
        log.info("local mode -- swapping in gpt2 for quick testing")
        return PipelineConfig(
            model_name="gpt2",
            hf_token=None,
            max_new_tokens=128,
            num_problems=10,
            output_dir=Path("outputs_local"),
            probe_layers=list(range(12)),  # gpt2 = 12 layers
            dtype="float32",               # gpt2 doesn't vibe w/ bf16 on cpu
        )

    # cluster mode -- gemma-2 is gated so we absolutely need the token
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is not set. gemma-2 is a gated model -- you need to "
            "grab your token from https://huggingface.co/settings/tokens "
            "and export HF_TOKEN=<token> before running."
        )

    log.info("cluster mode -- using gemma-2-2b-it")
    return PipelineConfig(hf_token=token)


def setup_logging(level: str = "INFO"):
    """sets up logging w/ timestamps so cluster logs are actually readable"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
