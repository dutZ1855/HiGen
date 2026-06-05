"""
Configuration objects shared by the RL-driven nnsmith fuzzer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _default_operator_pool() -> List[str]:
    return [
        "core.NCHWConv2d",
        "core.ReLU",
        "core.MaxPool2d",
        "core.Add",
        "core.Mul",
        "core.Softmax",
        "core.BatchNorm2d",
        "core.AvgPool2d",
        "core.Slice",
        "core.Concat2",
    ]


def _default_dtype_pool() -> List[str]:
    return ["float16", "float32", "float64", "int32", "int64", "bool"]


def _default_rank_choices() -> List[int]:
    return list(range(1, 5))


@dataclass
class DimensionPoolConfig:
    """User controllable dimension pools."""

    operator_pool: List[str] = field(default_factory=_default_operator_pool)
    dtype_pool: List[str] = field(default_factory=_default_dtype_pool)
    rank_pool: List[int] = field(default_factory=_default_rank_choices)
    node_range: Tuple[int, int] = (2, 16)
    timeout_range: Tuple[int, int] = (2000, 20000)
    max_elem_exp_range: Tuple[int, int] = (8, 16)  # 2**exp


@dataclass
class RLHyperParams:
    """High level PPO/SAC hyperparameters."""

    big_epoch_num: int = 20
    small_epochs_per_big: int = 8
    gamma: float = 0.99
    learning_rate: float = 3e-4
    buffer_size: int = 1024


@dataclass
class RewardWeights:
    # lower-layer weights
    vuln: float = 0.4
    valid: float = 0.3
    diversity: float = 0.3

    # upper-layer weights
    upper_vuln: float = 0.5
    upper_valid: float = 0.25
    upper_div_comb: float = 0.15
    upper_eff: float = 0.1


@dataclass
class DiversityRewardConfig:
    """Diversity reward hyperparameters (prompt-based)."""

    # Lower (per-step): novelty + token gain
    window_size: int = 100  # suggested 50~200
    alpha: float = 0.7  # novelty weight
    beta_invalid: float = 0.1  # down-weight diversity when valid=False

    # Upper (per-big-epoch): entropy + new combo coverage
    upper_lambda: float = 0.6  # entropy weight


@dataclass
class VulnAdaptiveRewardConfig:
    """Adaptive vulnerability reward (prompt-based).

    Replace fixed vuln bases (crash=15/div=8) with an adaptive vulnBase in [0,1].
    The env maintains EMAs of crash/div rates and vulnRaw moments, then maps:
      vulnRaw -> z-score -> sigmoid(z) = vulnBase.
    """

    enabled: bool = True

    # EMA for event rates p(crash), p(divergence)
    ema_alpha: float = 0.02  # suggested 0.01~0.05
    init_p_crash: float = 0.05
    init_p_div: float = 0.05
    eps: float = 1e-6

    # EMA for vulnRaw normalization (z-score)
    norm_alpha: float = 0.02
    init_vuln_mean: float = 1.0
    init_vuln_var: float = 0.0
    std_eps: float = 1e-6



@dataclass
class PathConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    nnsmith_root: Path = (Path(__file__).resolve().parents[1] / "nnsmith-main").resolve()
    run_root: Path = Path(__file__).resolve().parents[1] / "rl_runs_ort"
    bug_root: Optional[Path] = None

    def __post_init__(self):
        if self.bug_root is None:
            self.bug_root = self.run_root / "bug_cases"


@dataclass
class CompilerFuzzConfig:
    dimension_pool: DimensionPoolConfig = field(default_factory=DimensionPoolConfig)
    rl: RLHyperParams = field(default_factory=RLHyperParams)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    diversity: DiversityRewardConfig = field(default_factory=DiversityRewardConfig)
    vuln_adapt: VulnAdaptiveRewardConfig = field(default_factory=VulnAdaptiveRewardConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    backend_opts: Dict[str, str] = field(
        default_factory=lambda: {"backend.type": "onnxruntime", "backend.target": "cpu"}
    )
    enable_tvm_check: bool = False
    tvm_run_timeout_s: int = 180
    # Differential testing: use only oracle inputs and compare outputs across multiple backends
    diff_backends: List[str] = field(default_factory=list)  # e.g. ["ort", "ov", "tvm"]
    diff_rtol: float = 1e-3
    diff_atol: float = 1e-3
    # Differential testing device selection: affects whether ORT/TVM/OV use CPU or GPU
    # Recommended values: "cpu" or "gpu" (None means follow testing.py defaults)
    diff_device: Optional[str] = None
    compiler: str = "ort"  # Compiler type: "ort", "tvm", "ov" - used for CPU/GPU/PyTorch comparison

    def ensure_run_root(self) -> Path:
        self.paths.run_root.mkdir(parents=True, exist_ok=True)
        if self.paths.bug_root is None:
            self.paths.bug_root = self.paths.run_root / "bug_cases"
        self.paths.bug_root.mkdir(parents=True, exist_ok=True)
        return self.paths.run_root

