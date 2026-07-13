"""Traffic marks A[t, n] and the sigma calibration.

The traffic mean IS the tenant's type: E[A_{i,t}] = theta_i. That is the whole
point of the paper -- the type is a *measurable* quantity, so a learner can be fed
the measurement instead of the report. sigma controls how noisy that measurement
is, and therefore the price of invariance (Thm 4/4').

Marks are Gamma(shape k_i, scale theta_i/k_i) with k_i = (theta_i/sigma_i)^2, which
has mean theta_i, sd sigma_i and support >= 0. Gamma is sub-exponential rather than
sub-Gaussian; the theory assumes sub-Gaussian tails, and the eval text notes this
(results are insensitive; a truncated-normal robustness line is a P2 check).
"""

from __future__ import annotations

import numpy as np

from .instances import Instance


def sigma_vector(inst: Instance, sigma: float) -> np.ndarray:
    """Instance A: a single absolute sigma. Instance B: a *relative* knob,
    sigma_i = sigma_rel * theta_i."""
    if inst.name == "A":
        return np.full(inst.n, float(sigma))
    return float(sigma) * inst.theta_arr


def draw_traffic(inst: Instance, sigma: float, rng: np.random.Generator, T: int) -> np.ndarray:
    """A[T, n], i.i.d. across slots, mean theta_i, sd sigma_i."""
    theta = inst.theta_arr
    sig = sigma_vector(inst, sigma)
    k = (theta / np.maximum(sig, 1e-12)) ** 2          # Gamma shape
    scale = theta / k
    return rng.gamma(shape=k, scale=scale, size=(T, inst.n))


def epoch_means(A: np.ndarray, L: int) -> np.ndarray:
    """m[K, n]: the per-epoch mean of the marks -- the measurement the gate uses."""
    T, n = A.shape
    K = T // L
    return A[: K * L].reshape(K, L, n).mean(axis=1)


def running_means(m: np.ndarray) -> np.ndarray:
    """mbar[K, n]: the running mean of ALL measurements available when the epoch-k
    update is taken, i.e. epochs 0..k INCLUSIVE.

    Inclusive is both causal and essential. Causal, because the learner updates
    w_{k+1} *after* epoch k has finished, so m_k is already measured. Essential,
    because the alternative (a strictly-past mean) leaves epoch 0 with no anchor and
    tempts a "one epoch of full trust" fallback onto the raw report -- which would
    put a report back into the learner and destroy the exact invariance of Theorem 3
    (V3 catches precisely this: the weight paths stop being bitwise identical).

    With k+1 epochs of data the anchor has sd sigma/sqrt((k+1)L); summing that over
    the horizon is what converts the price of invariance from O(sigma T/sqrt(L)) into
    O(sigma sqrt(T)) -- Theorem 4'.
    """
    csum = np.cumsum(m, axis=0)
    counts = np.arange(1, m.shape[0] + 1)[:, None]
    return csum / counts


def closed_loop_traffic(
    inst: Instance, sigma: float, rng: np.random.Generator, T: int,
    kappa_fb: float, xbar: np.ndarray,
):
    """E6 only. Allocation-dependent arrivals reopen the manipulation channel even
    at r = 0, because the *measurement* then inherits a dependence on the report
    through last slot's allocation. Returns a generator-style callable used by the
    mechanism loop (mean depends on x_{t-1})."""
    theta = inst.theta_arr
    sig = sigma_vector(inst, sigma)

    def mark(t: int, x_prev: np.ndarray) -> np.ndarray:
        mean = theta * (1.0 + kappa_fb * (x_prev - xbar) / inst.budget)
        mean = np.maximum(mean, 0.05 * theta)
        k = (mean / np.maximum(sig, 1e-12)) ** 2
        return rng.gamma(shape=k, scale=mean / k)

    return mark
