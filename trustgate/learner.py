"""Projected online gradient ascent on the weight simplex slice.

The learner maximizes the plug-in welfare of the epoch, using whatever input the
gate hands it:

    What_k(w) = (1/L) sum_{t in E_k} sum_j  itilde_j * h_j( g_j(itilde, w, c_t), c_t )

Note itilde appears TWICE: as the type estimate in the value and as the report
fed to the allocator inside the objective. That is the plug-in estimator of the
true welfare W(w; theta); it is unbiased in itilde when itilde -> theta.

The gradient is taken along an orthonormal (Helmert) basis of the tangent space
{ sum_i dw_i = 0 } of the slice, by central differences on the SAME channel path
(common randomness across the +/- evaluations kills most of the gradient noise).
"""

from __future__ import annotations

import numpy as np

from .allocators import allocate
from .instances import Instance, h


def tangent_basis(n: int) -> np.ndarray:
    """Orthonormal Helmert basis of {v in R^n : sum v = 0}; shape [n-1, n]."""
    rows = []
    for k in range(1, n):
        v = np.zeros(n)
        v[:k] = 1.0 / np.sqrt(k * (k + 1.0))
        v[k] = -k / np.sqrt(k * (k + 1.0))
        rows.append(v)
    return np.array(rows)


def project(w: np.ndarray, inst: Instance) -> np.ndarray:
    """Project onto { sum_i w_i = weight_sum, w_i >= weight_lo }."""
    S, lo, n = inst.weight_sum, inst.weight_lo, inst.n
    w = np.asarray(w, dtype=float).copy()
    for _ in range(n + 1):
        w = w + (S - w.sum()) / n                       # back onto the hyperplane
        below = w < lo - 1e-15
        if not below.any():
            break
        # clamp the violating coordinates and redistribute their deficit
        free = ~below
        if not free.any():
            return np.full(n, S / n)
        w[below] = lo
        deficit = S - w.sum()
        w[free] += deficit / free.sum()
    return w


def plug_in_welfare(w: np.ndarray, itilde: np.ndarray, q: np.ndarray, inst: Instance) -> float:
    """What(w) on the epoch's channel path q[L, n], with learner input itilde[n]."""
    L = q.shape[0]
    z = np.broadcast_to(itilde, (L, inst.n))
    x, _ = allocate(z, w, q, inst)
    return float(np.sum(itilde * h(x, q, inst.sat_arr)) / L)


class OGD:
    """Projected OGD with a fixed step. eta is tuned on TRUTHFUL-run regret only and
    then frozen (plan ground rule 2: never tune against slack)."""

    def __init__(self, inst: Instance, eta: float = 0.1, delta: float = 1e-3,
                 lambda_reg: float = 0.0):
        self.inst = inst
        self.eta = float(eta)
        self.delta = float(delta)
        self.lambda_reg = float(lambda_reg)
        self.basis = tangent_basis(inst.n)
        self.w0 = np.full(inst.n, inst.weight_sum / inst.n)
        self.w = self.w0.copy()

    def objective(self, w, itilde, q) -> float:
        val = plug_in_welfare(w, itilde, q, self.inst)
        if self.lambda_reg:
            val -= 0.5 * self.lambda_reg * float(np.sum((w - self.w0) ** 2))
        return val

    def step(self, itilde: np.ndarray, q: np.ndarray) -> np.ndarray:
        """One epoch update. Returns the NEW weights (used from the next epoch)."""
        g = np.zeros(self.inst.n)
        for v in self.basis:
            wp = self.w + self.delta * v
            wm = self.w - self.delta * v
            d = (self.objective(wp, itilde, q) - self.objective(wm, itilde, q)) / (2 * self.delta)
            g += d * v
        self.w = project(self.w + self.eta * g, self.inst)
        return self.w
