"""The adversary and the CRN slack protocol.

Strategy family: a PERSISTENT additive deviation d applied to tenant `tenant`'s
report every epoch (others truthful). We search d over a grid; the grid IS the
protocol (the paper states that grid best-response is exhaustive within the modeled
strategy family). We deliberately do NOT train a learned/RL adversary: a grid over a
one-dimensional persistent deviation is exactly the deviation class the theory bounds,
and a learned adversary would confound "the mechanism leaks" with "the attacker is
undertrained".

CRN (common random numbers). For a seed s we pre-draw A[T,n] and q[T,n] ONCE, then
run TRUTH(s) and DEV(s,d) on those identical arrays. The utility difference is then a
paired statistic: everything the two runs share cancels. This is what lets us resolve
"slack is exactly zero" at r = 0 -- with independent runs, the Monte-Carlo noise of the
utilities would be far larger than the effect.

Bias caveat (stated in the paper): slack(s) = max_d [...] is a maximum of noisy
estimates and is therefore upward biased. CRN makes the per-d noise tiny, so the bias
is negligible; we report the full gain-vs-d curve, the seed count and the CI method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .gates import Gate
from .instances import Instance, draw_channel
from .mechanism import RunResult, persistent_deviation, run, truthful
from .traffic import draw_traffic


def deviation_grid(inst: Instance, tenant: int, points: int = 41) -> np.ndarray:
    """Instance A: d in [-0.1, +0.1]. Instance B: +/-0.2 * theta_tenant."""
    if inst.name == "A":
        return np.linspace(-0.1, 0.1, points)
    span = 0.2 * inst.theta[tenant]
    return np.linspace(-span, span, points)


@dataclass
class SlackResult:
    slack: float                 # max over d of the paired utility gain
    best_d: float
    gains: np.ndarray            # [G] gain vs d curve (for V2 / diagnostics)
    grid: np.ndarray
    truth_utility: float
    rho: float
    w_path: np.ndarray


def slack_for_seed(
    inst: Instance,
    gate: Gate,
    seed: int,
    T: int,
    L: int,
    sigma: float,
    tenant: int = 0,
    grid: Optional[Sequence[float]] = None,
    eta: float = 0.1,
    grid_size: int = 41,
    **run_kw,
) -> SlackResult:
    """One CRN-paired slack measurement."""
    rng = np.random.default_rng(seed)
    A = draw_traffic(inst, sigma, rng, T)       # drawn from the TRUE type: a report
    q = draw_channel(inst, rng, T)              # cannot change the traffic (open loop)

    truth = run(inst, gate, A, q, L, sigma, truthful(inst), eta=eta,
                grid_size=grid_size, rng=np.random.default_rng(seed + 1), **run_kw)
    u_truth = float(truth.utility[tenant])

    grid = np.asarray(deviation_grid(inst, tenant) if grid is None else grid, float)
    gains = np.empty(len(grid))
    for j, d in enumerate(grid):
        dev = run(inst, gate, A, q, L, sigma, persistent_deviation(inst, tenant, float(d)),
                  eta=eta, grid_size=grid_size,
                  rng=np.random.default_rng(seed + 1), **run_kw)
        gains[j] = float(dev.utility[tenant]) - u_truth

    j = int(np.argmax(gains))
    return SlackResult(
        slack=float(max(0.0, gains[j])),
        best_d=float(grid[j]),
        gains=gains,
        grid=grid,
        truth_utility=u_truth,
        rho=truth.rho,
        w_path=truth.w_path,
    )
