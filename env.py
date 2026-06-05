"""Compiler fuzzing reinforcement-learning environment module.

This module implements the RL environment used for compiler fuzzing. It
handles:
- dimension selection (operator types, dtypes, tensor ranks, etc.)
- parameter generation (model size, timeouts, etc.)
- coordinating nnsmith test runs
- reward calculation and history tracking

The environment is designed to integrate with algorithms such as PPO/SAC
to optimize fuzzing strategies.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from .config import CompilerFuzzConfig, DimensionPoolConfig
from .reward import LowerMetrics, UpperMetrics, compute_lower_reward, compute_upper_reward
from .utils.testing import GenerationParams, run_generation_and_test


METHODS = ["symbolic", "symbolic-cinit", "concolic", "single-io"]


@dataclass
class HistoryEntry:
    """History entry for a single test execution.

    Records a complete execution trace for analysis and learning.

    Attributes:
        dimensions: Selected dimension combination (e.g. operator types, dtypes).
        params: Generated parameter configuration.
        reward: Computed reward value.
        crashed: Whether the run crashed.
        divergence: Whether outputs diverged.
        valid: Whether the test result is considered valid.
        duration: Execution time in seconds.
    """
    dimensions: Tuple[str, ...]
    params: Dict[str, float]
    reward: float
    crashed: bool
    divergence: bool
    valid: bool
    duration: float


@dataclass
class CompilerFuzzEnvState:
    """Environment state for the compiler fuzzing environment.

    Maintains current state information such as epoch counters and current
    dimension selection.

    Attributes:
        big_epoch_idx: Index of the outer (dimension-selection) epoch.
        small_epoch_idx: Index of the inner (parameter-optimization) epoch.
        current_dimensions: Currently selected dimension combination.
        remaining_small_epochs: Number of remaining inner epochs.
        last_reward: Last observed reward.
    """
    big_epoch_idx: int = 0
    small_epoch_idx: int = 0
    current_dimensions: Tuple[str, ...] = field(default_factory=tuple)
    remaining_small_epochs: int = 0
    last_reward: float = 0.0


class CompilerFuzzEnv:
    """Reinforcement-learning environment for compiler fuzzing.

    This class implements a Gym-like interface suitable for use with PPO/SAC
    algorithms. The objective is to learn dimension selections and parameter
    configurations that maximize compiler bug discovery efficiency.

    The environment maintains two hierarchical epoch levels:
    - outer epoch: select test dimensions (operator type, dtype, etc.)
    - inner epoch: optimize concrete parameters under fixed dimensions

    Observation space: current epoch state and historical performance metrics
    Action space: dimension selection or parameter adjustments
    Reward: composite metric based on bug discovery and testing efficiency
    """

    def __init__(self, config: Optional[CompilerFuzzConfig] = None):
        self.config = config or CompilerFuzzConfig()
        self.dimension_keys = ("operator", "dtype", "rank", "max_nodes", "timeout", "max_elem")
        self.state = CompilerFuzzEnvState(
            remaining_small_epochs=self.config.rl.small_epochs_per_big
        )
        self.history: List[HistoryEntry] = []
        self.current_metrics: List[LowerMetrics] = []
        self.dimension_pool = self.config.dimension_pool
        self.config.paths.run_root.mkdir(parents=True, exist_ok=True)
        # Diversity tracking (prompt-based)
        self._div_window: Deque[Set[str]] = deque(maxlen=int(self.config.diversity.window_size))
        self._token_count: Dict[str, int] = {}
        # Upper-layer diversity tracking (prompt-based)
        # NOTE: Scheme A: z is the *dimension-set* (not concrete value combo).
        # Represent z as a sorted tuple of dimension keys, e.g. ("dtype","operator","rank").
        self._epoch_dimsets: List[Tuple[str, ...]] = []
        self._global_dimsets: Set[Tuple[str, ...]] = set()

        # --- Adaptive vulnerability reward (prompt-based) ---
        self._ema_p_crash = float(self.config.vuln_adapt.init_p_crash)
        self._ema_p_div = float(self.config.vuln_adapt.init_p_div)
        self._ema_vuln_mean = float(self.config.vuln_adapt.init_vuln_mean)
        self._ema_vuln_var = float(self.config.vuln_adapt.init_vuln_var)

    # -- Dimension selection -------------------------------------------------
    def sample_dimension_combo(self, action_vector: Optional[Sequence[float]] = None) -> Tuple[str, ...]:
        """
        Select a subset of dimensions based on an action vector (if provided).
        Falls back to random selection.
        """
        dims = list(self.dimension_keys)
        if action_vector is not None and len(action_vector) == len(dims):
            weights = np.array(action_vector, dtype=np.float32)
            order = np.argsort(weights)[::-1]
            count = max(2, int(np.clip(len(dims) * float(np.mean((weights + 1) / 2)), 2, len(dims))))
            picked = [dims[i] for i in order[:count]]
        else:
            picked = random.sample(dims, k=random.randint(2, len(dims)))
        self.state.current_dimensions = tuple(sorted(picked))
        return self.state.current_dimensions

    # -- Small epoch ---------------------------------------------------------
    def step_small_epoch(self, param_vector: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one configuration generation + model test round.
        """
        gen_params, param_dict = self._vector_to_generation(param_vector)
        run_name = f"big_{self.state.big_epoch_idx}_small_{self.state.small_epoch_idx}"
        result = run_generation_and_test(
            gen_params,
            nnsmith_root=self.config.paths.nnsmith_root,
            run_root=self.config.paths.run_root,
            bug_root=self.config.paths.bug_root,
            run_name=run_name,
            enable_tvm_check=self.config.enable_tvm_check,
            tvm_timeout_s=self.config.tvm_run_timeout_s,
            diff_backends=self.config.diff_backends,
            diff_rtol=self.config.diff_rtol,
            diff_atol=self.config.diff_atol,
            diff_device=self.config.diff_device,
            compiler=self.config.compiler,
        )

        # --- Adaptive vulnerability base (lower-layer): crash/div rarity + normalization ---
        vuln_debug = self._update_and_compute_vuln_adapt(
            crashed=bool(result.crashed),
            divergence=bool((not result.crashed) and (not result.success) and result.divergence),
        )

        # --- Diversity (lower-layer): novelty + token gain (prompt-based) ---
        tokens = self._tokenize_param_dict(param_dict)
        novelty, gain, div_raw, div_mapped = self._compute_lower_diversity(tokens, valid=result.valid)
        # Update window AFTER computing novelty/gain (so gain counts "new to window").
        self._update_div_window(tokens)
        # Upper diversity (Scheme A) uses dimension-set z at each step.
        # (Within one big-epoch, z is typically constant; we still append per-step for a uniform interface.)
        self._epoch_dimsets.append(tuple(self.state.current_dimensions))

        metrics = LowerMetrics(
            crashed=result.crashed,
            divergence=not result.success and result.divergence,
            valid=result.valid,
            vuln_raw=float(vuln_debug["vulnRaw"]),
            vuln_base=float(vuln_debug["vulnBase"]),
            novelty=float(novelty),
            token_gain=float(gain),
            div_raw=float(div_raw),
            div_mapped=float(div_mapped),
        )
        reward = compute_lower_reward(metrics, self.config.rewards)
        self.current_metrics.append(metrics)
        self.history.append(
            HistoryEntry(
                dimensions=self.state.current_dimensions,
                params=param_dict,
                reward=reward,
                crashed=result.crashed,
                divergence=metrics.divergence,
                valid=result.valid,
                duration=result.duration_s,
            )
        )

        self.state.small_epoch_idx += 1
        self.state.remaining_small_epochs = (
            self.config.rl.small_epochs_per_big - self.state.small_epoch_idx
        )
        self.state.last_reward = reward

        obs = self._build_observation()
        done = self.state.small_epoch_idx >= self.config.rl.small_epochs_per_big
        info = {
            "result": result,
            "metrics": metrics,
            "dimensions": self.state.current_dimensions,
            "params": param_dict,
            "vuln_adapt": vuln_debug,
            "diversity": {
                "tokens": sorted(tokens),
                "novelty": float(novelty),
                "token_gain": float(gain),
                "div_raw": float(div_raw),
                "div_mapped": float(div_mapped),
            },
        }
        return obs, reward, done, info

    # -- Big epoch -----------------------------------------------------------
    def step_big_epoch(self) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Aggregate the small epoch results and produce the upper-layer reward.
        """
        upper_metrics = self._aggregate_metrics()
        upper_reward = compute_upper_reward(
            upper_metrics, self.config.rewards, self.config.rl.small_epochs_per_big
        )
        self.state.big_epoch_idx += 1
        self.state.small_epoch_idx = 0
        self.state.remaining_small_epochs = self.config.rl.small_epochs_per_big
        self.state.last_reward = upper_reward
        self.current_metrics.clear()
        # reset per-epoch z list
        self._epoch_dimsets.clear()

        obs = self._build_observation()
        done = self.state.big_epoch_idx >= self.config.rl.big_epoch_num
        info = {"upper_metrics": upper_metrics}
        return obs, upper_reward, done, info

    # -- Helpers -------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.state = CompilerFuzzEnvState(
            remaining_small_epochs=self.config.rl.small_epochs_per_big
        )
        self.current_metrics.clear()
        self.history.clear()
        self._div_window.clear()
        self._token_count.clear()
        self._epoch_dimsets.clear()
        self._global_dimsets.clear()
        # reset adaptive vuln EMAs
        self._ema_p_crash = float(self.config.vuln_adapt.init_p_crash)
        self._ema_p_div = float(self.config.vuln_adapt.init_p_div)
        self._ema_vuln_mean = float(self.config.vuln_adapt.init_vuln_mean)
        self._ema_vuln_var = float(self.config.vuln_adapt.init_vuln_var)
        return self._build_observation()

    def _build_observation(self) -> np.ndarray:
        progress = [
            self.state.big_epoch_idx / max(self.config.rl.big_epoch_num, 1),
            self.state.small_epoch_idx / max(self.config.rl.small_epochs_per_big, 1),
            self.state.remaining_small_epochs / max(self.config.rl.small_epochs_per_big, 1),
            self.state.last_reward,
        ]
        dim_vector = [
            1.0 if key in self.state.current_dimensions else 0.0
            for key in self.dimension_keys
        ]
        return np.array(progress + dim_vector, dtype=np.float32)

    def _vector_to_generation(self, action_vec: np.ndarray) -> Tuple[GenerationParams, Dict[str, float]]:
        pool: DimensionPoolConfig = self.dimension_pool
        vec = np.tanh(action_vec).astype(float)
        max_nodes = int(self._scale(vec[0], pool.node_range))
        timeout_ms = int(self._scale(vec[1], pool.timeout_range))
        max_elem_exp = int(self._scale(vec[2], pool.max_elem_exp_range))
        method = METHODS[int(self._normalize(vec[3]) * (len(METHODS) - 1))]
        vulops = bool(vec[4] > 0)
        # Force-disable grad_check to avoid numerical checks during torch.export.
        grad_check = False

        include_ops = (
            self._sample_from_pool(pool.operator_pool, vec[6], max_pick=min(3, len(pool.operator_pool)))
            if "operator" in self.state.current_dimensions
            else None
        )
        dtype_choices = (
            self._sample_from_pool(pool.dtype_pool, vec[7], max_pick=min(3, len(pool.dtype_pool)))
            if "dtype" in self.state.current_dimensions
            else None
        )
        rank_choices = (
            self._sample_from_pool(pool.rank_pool, vec[8], max_pick=min(2, len(pool.rank_pool)))
            if "rank" in self.state.current_dimensions
            else None
        )

        param_dict = {
            "max_nodes": max_nodes,
            "timeout_ms": timeout_ms,
            "max_elem_exp": max_elem_exp,
            "method": method,
            "vulops": float(vulops),
            "grad_check": float(grad_check),
            "include": include_ops or [],
            "dtype_choices": dtype_choices or [],
            "rank_choices": rank_choices or [],
        }
        return GenerationParams(
            max_nodes=max_nodes,
            timeout_ms=timeout_ms,
            method=method,
            seed=random.getrandbits(32),
            max_elem_per_tensor=2**max_elem_exp,
            vulops=vulops,
            grad_check=grad_check,
            rank_choices=rank_choices,
            dtype_choices=dtype_choices,
            include=include_ops,
            exclude=None,
        ), param_dict

    def _scale(self, value: float, rng: Tuple[int, int]) -> float:
        norm = self._normalize(value)
        return rng[0] + norm * (rng[1] - rng[0])

    @staticmethod
    def _normalize(value: float) -> float:
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _sample_from_pool(pool: Sequence, seed_val: float, max_pick: int) -> Optional[List]:
        if not pool or max_pick <= 0:
            return None
        rng = random.Random(int((seed_val + 2) * 1e6))
        count = rng.randint(1, max_pick)
        return rng.sample(list(pool), k=count)

    # -------------------- Diversity helpers (prompt-based) --------------------
    def _bucketize(self, name: str, value: float) -> str:
        """Bucketize continuous scalars into coarse tokens."""
        if name == "max_nodes":
            v = int(value)
            if v <= 3:
                return "max_nodes=1-3"
            if v <= 6:
                return "max_nodes=4-6"
            if v <= 10:
                return "max_nodes=7-10"
            return "max_nodes=11+"
        if name == "timeout_ms":
            s = float(value) / 1000.0
            if s < 1.0:
                return "timeout=<1s"
            if s < 3.0:
                return "timeout=1-3s"
            if s < 10.0:
                return "timeout=3-10s"
            return "timeout=>10s"
        if name == "max_elem":
            v = float(value)
            if v < 1e3:
                return "max_elem=<1e3"
            if v < 1e5:
                return "max_elem=1e3-1e5"
            if v < 1e7:
                return "max_elem=1e5-1e7"
            return "max_elem=>1e7"
        return f"{name}=UNK"

    def _tokenize_param_dict(self, params: Dict[str, float]) -> Set[str]:
        """Tokenize a configuration into a set of discrete tokens S(x_t)."""
        tokens: Set[str] = set()
        method = str(params.get("method", ""))
        if method:
            tokens.add(f"method={method}")
        vulops = bool(params.get("vulops", 0.0) > 0)
        tokens.add(f"vulops={int(vulops)}")

        # ranks/dtypes/include ops are lists in param_dict
        ranks = params.get("rank_choices", []) or []
        if ranks:
            ranks_s = ",".join(str(int(r)) for r in sorted(ranks))
            tokens.add(f"rank_choices={ranks_s}")
        dtypes = params.get("dtype_choices", []) or []
        if dtypes:
            dtypes_s = ",".join(sorted(str(d) for d in dtypes))
            tokens.add(f"dtype_choices={dtypes_s}")
        ops = params.get("include", []) or []
        if ops:
            # join first K to avoid token explosion
            ops_s = "|".join(sorted(str(o) for o in ops)[:3])
            tokens.add(f"include_ops={ops_s}")

        # bucketize scalars
        if "max_nodes" in params:
            tokens.add(self._bucketize("max_nodes", float(params["max_nodes"])))
        if "timeout_ms" in params:
            tokens.add(self._bucketize("timeout_ms", float(params["timeout_ms"])))
        # max_elem is derived from exp
        max_elem = 2 ** int(params.get("max_elem_exp", 0)) if "max_elem_exp" in params else None
        if max_elem is not None:
            tokens.add(self._bucketize("max_elem", float(max_elem)))

        return tokens

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        uni = len(a | b)
        return float(inter / uni) if uni else 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(x))))

    def _update_and_compute_vuln_adapt(self, *, crashed: bool, divergence: bool) -> Dict[str, float]:
        """Update EMA state and compute adaptive vulnerability reward components.

        Implements the prompt scheme:
        - Maintain EMA of p(crash), p(div) per step.
        - Compute rarity weights: w = log1p(1 / (p + eps)) (log-inverse, but numerically safer).
        - vulnRaw = w_crash if crash else w_div if divergence else 0.
        - Normalize vulnRaw via EMA mean/var -> z-score -> sigmoid(z) = vulnBase in (0,1).
        """
        cfg = self.config.vuln_adapt
        if not bool(cfg.enabled):
            return {
                "ema_p_crash": 0.0,
                "ema_p_div": 0.0,
                "w_crash": 0.0,
                "w_div": 0.0,
                "vulnRaw": 0.0,
                "vulnBase": 0.0,
                "ema_vuln_mean": 0.0,
                "ema_vuln_var": 0.0,
                "z": 0.0,
            }

        # B1: event indicators (crash takes precedence over divergence)
        i_crash = 1.0 if crashed else 0.0
        i_div = 1.0 if (divergence and (not crashed)) else 0.0

        # B2: update event-rate EMAs
        a = float(cfg.ema_alpha)
        self._ema_p_crash = (1.0 - a) * float(self._ema_p_crash) + a * float(i_crash)
        self._ema_p_div = (1.0 - a) * float(self._ema_p_div) + a * float(i_div)

        # C1: rarity weights (log-inverse)
        eps = float(cfg.eps)
        w_crash = float(np.log1p(1.0 / (float(self._ema_p_crash) + eps)))
        w_div = float(np.log1p(1.0 / (float(self._ema_p_div) + eps)))

        # C2: vulnRaw
        if crashed:
            vuln_raw = w_crash
        elif divergence:
            vuln_raw = w_div
        else:
            vuln_raw = 0.0

        # D1: normalize to stable scale via EMA z-score, then sigmoid
        a2 = float(cfg.norm_alpha)
        # update mean (EMA)
        mean_prev = float(self._ema_vuln_mean)
        mean_new = (1.0 - a2) * mean_prev + a2 * float(vuln_raw)
        self._ema_vuln_mean = mean_new
        # update variance (EMA of squared deviation)
        dev = float(vuln_raw) - mean_new
        self._ema_vuln_var = (1.0 - a2) * float(self._ema_vuln_var) + a2 * (dev * dev)
        std = float(np.sqrt(float(self._ema_vuln_var) + float(cfg.std_eps)))
        z = (float(vuln_raw) - float(mean_new)) / std if std > 0 else 0.0
        vuln_base = self._sigmoid(z)

        return {
            "ema_p_crash": float(self._ema_p_crash),
            "ema_p_div": float(self._ema_p_div),
            "w_crash": float(w_crash),
            "w_div": float(w_div),
            "vulnRaw": float(vuln_raw),
            "vulnBase": float(vuln_base),
            "ema_vuln_mean": float(self._ema_vuln_mean),
            "ema_vuln_var": float(self._ema_vuln_var),
            "z": float(z),
        }

    def _compute_lower_diversity(self, tokens: Set[str], *, valid: bool) -> Tuple[float, float, float, float]:
        """Compute novelty + token gain and map into [-2,+3]."""
        W = list(self._div_window)
        if not W:
            novelty = 1.0
        else:
            sim_max = max(self._jaccard(tokens, prev) for prev in W)
            novelty = 1.0 - float(sim_max)
        # token gain: fraction of tokens unseen in window
        if not tokens:
            gain = 0.0
        else:
            unseen = sum(1 for t in tokens if self._token_count.get(t, 0) == 0)
            gain = float(unseen / max(len(tokens), 1))
        alpha = float(self.config.diversity.alpha)
        div_raw = alpha * float(novelty) + (1.0 - alpha) * float(gain)
        # valid gating
        if not valid:
            div_raw *= float(self.config.diversity.beta_invalid)
        # map [0,1] -> [-2, +3] (5-wide range)
        div_mapped = -2.0 + 5.0 * float(np.clip(div_raw, 0.0, 1.0))
        return float(novelty), float(gain), float(div_raw), float(div_mapped)

    def _update_div_window(self, tokens: Set[str]) -> None:
        """Push tokens into window and maintain token_count in O(|S|)."""
        if self._div_window.maxlen and len(self._div_window) >= int(self._div_window.maxlen):
            old = self._div_window.popleft()
            for t in old:
                c = self._token_count.get(t, 0) - 1
                if c <= 0:
                    self._token_count.pop(t, None)
                else:
                    self._token_count[t] = c
        self._div_window.append(tokens)
        for t in tokens:
            self._token_count[t] = self._token_count.get(t, 0) + 1

    def _aggregate_metrics(self) -> UpperMetrics:
        crash_count = sum(1 for m in self.current_metrics if m.crashed)
        divergence_count = sum(1 for m in self.current_metrics if m.divergence)
        valid_ratio = (
            sum(1 for m in self.current_metrics if m.valid) / max(len(self.current_metrics), 1)
        )
        recent_entries = self.history[-self.config.rl.small_epochs_per_big :]
        avg_duration = (
            float(np.mean([entry.duration for entry in recent_entries])) if recent_entries else 0.0
        )

        # --- Upper diversity (Scheme A): entropy + new dimset coverage ---
        # z is the dimension-set (sorted tuple of dimension keys).
        dimsets_epoch = list(self._epoch_dimsets)
        dimsets_set = set(dimsets_epoch)
        if not dimsets_epoch or len(dimsets_set) <= 1:
            entropy_norm = 0.0
        else:
            # distribution over z in this epoch
            counts: Dict[Tuple[str, ...], int] = {}
            for z in dimsets_epoch:
                counts[z] = counts.get(z, 0) + 1
            total = sum(counts.values())
            ps = np.array([c / total for c in counts.values()], dtype=np.float64)
            H = float(-(ps * np.log(ps + 1e-12)).sum())
            entropy_norm = float(H / (np.log(len(ps) + 1e-12)))
            entropy_norm = float(np.clip(entropy_norm, 0.0, 1.0))

        new = dimsets_set - self._global_dimsets
        cover = float(len(new) / max(len(dimsets_set), 1))
        self._global_dimsets |= dimsets_set
        lam = float(self.config.diversity.upper_lambda)
        up_div_raw = lam * float(entropy_norm) + (1.0 - lam) * float(cover)
        # map [0,1] -> [-5, +8] (13-wide range)
        up_div_mapped = -5.0 + 13.0 * float(np.clip(up_div_raw, 0.0, 1.0))

        return UpperMetrics(
            crash_count=crash_count,
            divergence_count=divergence_count,
            valid_ratio=valid_ratio,
            avg_duration_s=avg_duration,
            entropy_norm=float(entropy_norm),
            new_combo_cover=float(cover),
            up_div_raw=float(up_div_raw),
            up_div_mapped=float(up_div_mapped),
        )

    def _parameter_repeat_ratio(self) -> float:
        if not self.history:
            return 0.0
        recent = self.history[-self.config.rl.small_epochs_per_big :]
        combos = [entry.dimensions for entry in recent]
        unique = len(list(dict.fromkeys(combos)))
        return 1.0 - unique / max(len(combos), 1)

