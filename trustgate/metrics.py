"""W*, regret, slack, rho.

Regret is measured against the best FIXED weight for the TRUE types, evaluated with
truthful inputs on a common channel pool. Using the same pool for every weight makes
the regret curve smooth (common randomness), which matters because the differences we
are resolving are small.

Slack is measured by the CRN protocol: a deviating run and a truthful run consume
*identical* pre-drawn traffic and channel arrays, so their utility difference is the
manipulation effect and nothing else. This mirrors the coupling used in the proofs and
cuts the variance by orders of magnitude -- without it, the O(1/T) effects we need to
resolve at r = 0 would be buried in noise.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy import optimize

from .allocators import allocate
from .instances import Instance, h
from .learner import project, tangent_basis


class WelfareOracle:
    """W(w; theta) with truthful inputs, and its maximizer W*."""

    def __init__(self, inst: Instance, pool_size: int = 6_000, seed: int = 12345):
        self.inst = inst
        rng = np.random.default_rng(seed)
        if inst.channel_states:
            # The pool is COMMON randomness: the same channel draws evaluate every
            # weight, which is what makes the regret curve smooth enough to resolve
            # the small gaps in Fig. 3. 20k draws is ample (s.e. ~ 1/sqrt(2e4)).
            self.pool = rng.choice(np.asarray(inst.channel_states, float),
                                   size=(pool_size, inst.n))
        else:
            self.pool = np.ones((1, inst.n))     # single channel state => exact
        self._cache: dict = {}
        self.w_star, self.W_star = self._optimize()

    def W(self, w: np.ndarray) -> float:
        # Regret needs W(w_k) once per epoch (K ~ 1e3) and RULE_W runs a 60-step
        # bisection over the whole pool, so memoize on the rounded weight: the learner
        # converges, so the great majority of epochs share a weight to 4 decimals.
        w = np.asarray(w, float)
        key = tuple(np.round(w, 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        inst = self.inst
        M = self.pool.shape[0]
        z = np.broadcast_to(inst.theta_arr, (M, inst.n))
        x, _ = allocate(z, w, self.pool, inst)
        val = float(np.mean(np.sum(inst.theta_arr * h(x, self.pool, inst.sat_arr), axis=-1)))
        self._cache[key] = val
        return val

    def _optimize(self) -> Tuple[np.ndarray, float]:
        inst = self.inst
        basis = tangent_basis(inst.n)
        w0 = np.full(inst.n, inst.weight_sum / inst.n)

        def neg(coef):
            w = project(w0 + basis.T @ np.asarray(coef, float), inst)
            return -self.W(w)

        # coarse grid then Nelder-Mead in the (n-1)-dim tangent coordinates
        best, best_val = np.zeros(inst.n - 1), neg(np.zeros(inst.n - 1))
        grid = np.linspace(-1.0, 1.0, 21)
        if inst.n == 2:
            for a in grid:
                v = neg([a])
                if v < best_val:
                    best, best_val = np.array([a]), v
        else:
            for a in grid[::2]:
                for b in grid[::2]:
                    v = neg([a, b])
                    if v < best_val:
                        best, best_val = np.array([a, b]), v
        res = optimize.minimize(neg, best, method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
        w = project(w0 + basis.T @ res.x, inst)
        return w, float(-res.fun)

    def regret(self, w_path: np.ndarray, L: int) -> float:
        """Regret(T) = sum_k L * (W* - W(w_k; theta))."""
        return float(sum(L * (self.W_star - self.W(w)) for w in w_path))


def ci95(x: np.ndarray) -> Tuple[float, float]:
    """Mean and 95% t half-width over seeds."""
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), 0.0
    hw = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(hw)
