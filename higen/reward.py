"""
Reward functions for the hierarchical RL loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from .config import RewardWeights


@dataclass
class LowerMetrics:
    crashed: bool = False
    divergence: bool = False
    valid: bool = True
    # Adaptive vulnerability reward (prompt-based): in [0,1]
    vuln_raw: float = 0.0
    vuln_base: float = 0.0
    # Diversity (prompt-based): novelty + token gain
    novelty: float = 0.0
    token_gain: float = 0.0
    div_raw: float = 0.0  # in [0,1]
    div_mapped: float = 0.0  # mapped to [-2, +3] before applying weights.diversity


@dataclass
class UpperMetrics:
    crash_count: int = 0
    divergence_count: int = 0
    valid_ratio: float = 1.0
    avg_duration_s: float = 0.0
    vulnerability_score_threshold: float = 10.0
    # Upper diversity (Scheme A): entropy + new-z coverage, where z is the *dimension-set*
    # (a sorted tuple of dimension keys, e.g. ("dtype","operator","rank")).
    entropy_norm: float = 0.0  # in [0,1]
    new_combo_cover: float = 0.0  # in [0,1] (kept name for backward-compat; semantics: new z coverage)
    up_div_raw: float = 0.0  # in [0,1]
    up_div_mapped: float = 0.0  # mapped to [-5, +8] before applying weights.upper_div_comb


def compute_lower_reward(metrics: LowerMetrics, weights: RewardWeights) -> float:
    # NOTE: vulnerability term is now adaptive (computed in env) and normalized to [0,1].
    rv = weights.vuln * float(metrics.vuln_base)

    valid_reward = 5.0 if metrics.valid else -10.0
    rv += weights.valid * valid_reward

    # Diversity reward is computed in env (novelty + token gain), mapped into [-2,+3].
    rv += weights.diversity * float(metrics.div_mapped)
    return rv


def compute_upper_reward(
    aggregated_metrics: UpperMetrics, weights: RewardWeights, small_epochs: int
) -> float:
    vuln_reward = (
        (aggregated_metrics.crash_count * 20.0 + aggregated_metrics.divergence_count * 10.0)
        / max(small_epochs, 1)
    )
    vuln_reward *= weights.upper_vuln

    if aggregated_metrics.valid_ratio >= 0.9:
        valid_reward = 6.0
    elif aggregated_metrics.valid_ratio >= 0.7:
        valid_reward = 3.0
    else:
        valid_reward = -3.0
    valid_reward *= weights.upper_valid

    # Upper diversity is computed in env (entropy + new combo coverage), mapped into [-5,+8].
    div_reward = weights.upper_div_comb * float(aggregated_metrics.up_div_mapped)

    if aggregated_metrics.avg_duration_s < aggregated_metrics.vulnerability_score_threshold:
        eff_reward = 4.0
    elif vuln_reward > 10.0:
        eff_reward = 0.0
    else:
        eff_reward = -2.0
    eff_reward *= weights.upper_eff

    return vuln_reward + valid_reward + div_reward + eff_reward

