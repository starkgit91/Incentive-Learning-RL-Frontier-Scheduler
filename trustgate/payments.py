"""Myerson threshold payments, computed once per epoch.

Within an epoch the weights w_k are FIXED, so the epoch is a static single-parameter
mechanism with allocation rule a_i(.) and the unique DSIC payment

    abar_{i,k}(s) = sum_{t in E_k} h_i( g_i( (s, z_{-i,k}), w_k, c_t ), c_t )
    p_{i,k}       = z_i * abar_{i,k}(z_i) - int_{theta_lo_i}^{z_i} abar_{i,k}(s) ds

Because g is nondecreasing in the own report (allocators.py), abar is nondecreasing,
and monotone + threshold payment = dominant-strategy truthful within the epoch
(Myerson). The integral is done by trapezoid on a G-point grid, so truthfulness is
exact only up to the quadrature error -- we compute that epsilon-DSIC bound and
report it in the paper rather than pretending it is zero.

The grid axis and the slot axis are both vectorized: one call evaluates the
allocator on [G, L, n] at once.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .allocators import allocate
from .instances import Instance, h


def epoch_payment(
    i: int,
    z: np.ndarray,          # [n] reports this epoch
    w: np.ndarray,          # [n] frozen weights this epoch
    q: np.ndarray,          # [L, n] channel path of the epoch
    inst: Instance,
    grid_size: int = 41,
) -> Tuple[float, float]:
    """Return (payment, abar_at_report) for tenant i over one epoch."""
    lo = inst.theta_lo[i]
    zi = float(max(z[i], lo))

    # NOTE: at z_i = theta_lo the payment is NOT zero -- Myerson gives
    #   p(theta_lo) = theta_lo * abar(theta_lo) - 0 = theta_lo * abar(theta_lo),
    # which is exactly what makes the LOWEST type's utility zero. Short-circuiting
    # p = 0 here hands the floor-reporting tenant its allocation for free and
    # manufactures an enormous spurious "manipulation gain" at the grid edge. The
    # degenerate grid below (all points = lo) yields trapz = 0 and reproduces this
    # correctly, so no special case is needed.
    grid = np.linspace(lo, zi, grid_size)                  # [G]
    L = q.shape[0]

    ztrial = np.broadcast_to(z, (grid_size, L, inst.n)).copy()   # [G, L, n]
    ztrial[:, :, i] = grid[:, None]
    qq = np.broadcast_to(q, (grid_size, L, inst.n))

    x, _ = allocate(ztrial, w, qq, inst)                   # [G, L, n]
    hi = h(x, qq, inst.sat_arr)[:, :, i]                   # [G, L]
    abar = hi.sum(axis=1)                                  # [G]  (epoch-summed)

    integral = float(np.trapezoid(abar, grid))
    payment = zi * float(abar[-1]) - integral
    return payment, float(abar[-1])


def epoch_payments_all(
    z: np.ndarray, w: np.ndarray, q: np.ndarray, inst: Instance, grid_size: int = 41
) -> np.ndarray:
    return np.array(
        [epoch_payment(i, z, w, q, inst, grid_size)[0] for i in range(inst.n)],
        dtype=float,
    )


def eps_dsic_bound(inst: Instance, L: int, grid_size: int = 41) -> float:
    """Trapezoid error bound on the threshold integral => the epsilon of epsilon-DSIC.

    |int - trapz| <= (b-a)^3 / (12 G^2) * max|abar''|. abar is an epoch SUM of L
    terms each bounded by vbar = max_i q_max, and its curvature in the own report is
    bounded by the same scale, so a safe, reported bound is

        eps <= vbar * L * (b-a)^3 / (12 * (G-1)^2).
    """
    qmax = max(inst.channel_states) if inst.channel_states else 1.0
    width = max(hi - lo for lo, hi in zip(inst.theta_lo, inst.theta_hi))
    return float(qmax * L * width ** 3 / (12.0 * (grid_size - 1) ** 2))
