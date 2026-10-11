"""Paper Table 1 benchmark definitions (PPO, Schulman et al. 2017, section 6.1)."""
from dataclasses import dataclass

ENVIRONMENTS = (
    'Hopper-v1',
    'HalfCheetah-v1',
    'InvertedDoublePendulum-v1',
    # 'InvertedPendulum-v1', 'Reacher-v1', 'Swimmer-v1', 'Walker2d-v1',
)
# Gymnasium v4 is an approximation to the historical v1 benchmark, NOT identical.
MODERN_ENVS = {e: e.replace('-v1', '-v4') for e in ENVIRONMENTS}
SEEDS = (0, 1, 2)
PAPER_STEPS = 1_000_000

@dataclass(frozen=True)
class Objective:
    name: str
    variant: str
    parameter: float
    paper_score: float

OBJECTIVES = (
    Objective('unclipped', 'unclipped', 0.0, -0.39),
    Objective('clip_0.1', 'clip', 0.1, 0.76),
    Objective('clip_0.2', 'clip', 0.2, 0.82),
    Objective('clip_0.3', 'clip', 0.3, 0.70),
    Objective('adaptive_0.003', 'adaptive_kl', 0.003, 0.68),
    Objective('adaptive_0.01', 'adaptive_kl', 0.01, 0.74),
    Objective('adaptive_0.03', 'adaptive_kl', 0.03, 0.71),
    Objective('fixed_0.3', 'fixed_kl', 0.3, 0.62),
    Objective('fixed_1', 'fixed_kl', 1.0, 0.71),
    Objective('fixed_3', 'fixed_kl', 3.0, 0.72),
    Objective('fixed_10', 'fixed_kl', 10.0, 0.69),
)
BY_NAME = {o.name: o for o in OBJECTIVES}
