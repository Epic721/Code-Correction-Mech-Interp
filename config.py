"""
config.py - central config for the code correction mech interp project.
handles model selection, hf auth, paths, and local-vs-cluster toggling.

phase 2 update: dataset-aware paths so mbpp and humaneval artifacts dont
clobber each other. the new layout is:

    outputs/
      mbpp/
        generations.jsonl
        activations/layer_XX.pt
        probes/layer_XX_*.pt
        steering/...           <- phase 2 steering eval results
      humaneval/
        generations.jsonl
        ...
      vectors/                 <- extracted from mbpp, reused on humaneval
        caa_layer_XX.pt
        sae_layer_XX.pt
        learned_layer_XX.pt

if youre coming from milestone outputs (outputs/generations.jsonl etc),
move those into outputs/mbpp/ before running anything new. eval_sandbox
will print a hint at startup if it detects legacy files.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

log = logging.getLogger(__name__)


SUPPORTED_DATASETS = ("mbpp", "humaneval")
SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")
SUPPORTED_INTERVENTION_MODES = ("surgical", "continuous")


@dataclass
class PipelineConfig:
    """all the knobs for the pipeline in one place"""

    # model
    model_name: str = "google/gemma-2-2b-it"
    hf_token: str | None = None

    # dataset
    dataset: str = "mbpp"
    mbpp_split: str = "test"
    humaneval_split: str = "test"
    num_problems: int | None = None  # none = use the full split

    # layers we want to cache / probe (gemma-2-2b has 26 transformer blocks)
    probe_layers: list[int] = field(default_factory=lambda: list(range(26)))

    # generation
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0 = greedy

    # sandbox
    exec_timeout_sec: int = 10

    # steering (used by phase 2 scripts; kept here so theres one source of truth)
    # universal rule: x' = x - alpha * v, where v always points toward failure.
    intervention_mode: str = "surgical"  # one of SUPPORTED_INTERVENTION_MODES
    target_layers: list[int] = field(default_factory=lambda: [12, 20, 25])
    alpha_sweep: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    )

    # output paths
    output_dir: Path = Path("outputs")
    generations_file: str = "generations.jsonl"
    activations_dir: str = "activations"
    probes_dir: str = "probes"
    vectors_dir: str = "vectors"
    steering_dir: str = "steering"

    # hardware
    device: str = "auto"
    dtype: str = "bfloat16"

    # logging
    log_file: str | None = None  # optional path to also tee logs to a file

    def __post_init__(self):
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got {self.dataset!r}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {SUPPORTED_DTYPES}, got {self.dtype!r}"
            )
        if self.intervention_mode not in SUPPORTED_INTERVENTION_MODES:
            raise ValueError(
                f"intervention_mode must be one of {SUPPORTED_INTERVENTION_MODES}, "
                f"got {self.intervention_mode!r}"
            )

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
        }[self.dtype]

        # grab hf token from env if caller didnt pass one in
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # eagerly create the dataset subdir so resume logic doesnt face race
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    # -- paths --

    @property
    def dataset_path(self) -> Path:
        """root dir for everything tied to the current dataset"""
        return self.output_dir / self.dataset

    @property
    def generations_path(self) -> Path:
        return self.dataset_path / self.generations_file

    @property
    def activations_path(self) -> Path:
        p = self.dataset_path / self.activations_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def probes_path(self) -> Path:
        p = self.dataset_path / self.probes_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vectors_path(self) -> Path:
        # vectors are extracted once from mbpp activations and re-used across
        # both datasets, so they sit at outputs/vectors/ rather than under
        # any specific dataset subdir
        p = self.output_dir / self.vectors_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def steering_path(self) -> Path:
        # steering eval is dataset-specific (we evaluate the same vectors on
        # both mbpp and humaneval), so this nests under the dataset
        p = self.dataset_path / self.steering_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def legacy_generations_path(self) -> Path:
        """where the milestone code used to put mbpp generations"""
        return self.output_dir / self.generations_file


def get_config(
    local: bool = False,
    dataset: str = "mbpp",
    **overrides,
) -> PipelineConfig:
    """
    factory for a pipeline config. local mode swaps in gpt2 so you can
    smoke-test the whole thing on cpu without needing gpu or hf access.
    cluster mode requires HF_TOKEN since gemma-2 is gated.
    """
    if local:
        log.info("local mode -- swapping in gpt2 for quick testing")
        defaults = dict(
            model_name="gpt2",
            hf_token=None,
            dataset=dataset,
            max_new_tokens=128,
            num_problems=10,
            output_dir=Path("outputs_local"),
            probe_layers=list(range(12)),  # gpt2 has 12 layers
            target_layers=[5, 8, 11],      # rough analogues for steering tests
            dtype="float32",               # gpt2 doesnt vibe w/ bf16 on cpu
        )
        defaults.update(overrides)
        return PipelineConfig(**defaults)

    # cluster mode -- gemma-2 is gated so we absolutely need the token
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is not set. gemma-2 is a gated model -- you need to "
            "grab your token from https://huggingface.co/settings/tokens "
            "and export HF_TOKEN=<token> before running."
        )

    log.info(f"cluster mode -- using {overrides.get('model_name', 'gemma-2-2b-it')}")
    return PipelineConfig(hf_token=token, dataset=dataset, **overrides)


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """
    sets up logging w/ timestamps so cluster logs are actually readable.
    if log_file is given we also tee everything to disk -- useful for slurm
    runs where stdout might get truncated or split across nodes.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # override any preexisting handlers from libs we import
    )


def log_config(cfg: PipelineConfig, logger: logging.Logger | None = None):
    """
    dump the config to logs at startup so a slurm run is self-documenting --
    if anything weird shows up later you can read the log header to see
    exactly what config produced it
    """
    out = logger or log
    out.info("=" * 72)
    out.info("pipeline config")
    out.info("-" * 72)
    out.info(f"  model         : {cfg.model_name}")
    out.info(f"  device        : {cfg.device}")
    out.info(f"  dtype         : {cfg.dtype}")
    out.info(f"  hf_token      : {'set' if cfg.hf_token else 'NOT SET'}")
    out.info(f"  dataset       : {cfg.dataset}")
    out.info(f"  num_problems  : {cfg.num_problems if cfg.num_problems else 'all'}")
    out.info(f"  max_new_tok   : {cfg.max_new_tokens}")
    out.info(f"  temperature   : {cfg.temperature}")
    out.info(f"  timeout (sec) : {cfg.exec_timeout_sec}")
    out.info(f"  probe_layers  : {cfg.probe_layers}")
    out.info(f"  target_layers : {cfg.target_layers}")
    out.info(f"  alpha_sweep   : {cfg.alpha_sweep}")
    out.info(f"  intervention  : {cfg.intervention_mode}")
    out.info(f"  output_dir    : {cfg.output_dir.resolve()}")
    out.info(f"  dataset_path  : {cfg.dataset_path.resolve()}")
    out.info("=" * 72)
