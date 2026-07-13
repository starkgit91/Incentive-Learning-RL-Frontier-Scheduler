"""Allocation rules g(z, w, c) -> x.

Both rules are (i) closed/deterministic given (reports, weights, channel) and
(ii) nondecreasing in a tenant's OWN report. Monotonicity is what makes the
Myerson threshold payment in payments.py incentive compatible within an epoch.

RULE_P (proportional, floor-free). Closed form, paper eq. (6):

    x_i = f_i + (B - sum_j f_j) * (w_i z_i q_i) / sum_j (w_j z_j q_j)

CAUTION: RULE_P can never *bind* a floor -- every floored tenant receives f_i plus
a strictly positive share of the surplus. So it cannot be used for any experiment
that sweeps the binding frequency rho; those must use RULE_W. (This fixes an
internal inconsistency in draft v1, which implied eq. (6) throughout.)

RULE_W (weighted-welfare program, floors bind). Solves

    max_x  sum_i w_i z_i q_i (1 - exp(-x_i/s_i))   s.t.  sum_i x_i = B,  x_i >= f_i

whose KKT conditions give waterfilling at level nu = ln(mu):

    a_i(c) = s_i * ln( w_i z_i q_i / s_i )      (unconstrained level at nu = 0)
    x_i(nu) = max( f_i, a_i - s_i * nu )

sum_i x_i(nu) is continuous and nonincreasing in nu, so we bisect on nu. Both the
slot axis and the payment-quadrature grid axis are vectorized, which is what keeps
the Myerson grid (G re-allocations per tenant per epoch) affordable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .instances import Instance


def rule_p(z: np.ndarray, w: np.ndarray, q: np.ndarray, inst: Instance) -> np.ndarray:
    """Proportional rule. z, q broadcast to [..., n]; w is [n] (or [..., n])."""
    f = inst.floors_arr
    score = w * z * q
    denom = np.sum(score, axis=-1, keepdims=True)
    free = inst.budget - float(np.sum(f))
    return f + free * score / np.maximum(denom, 1e-300)


def rule_w(
    z: np.ndarray, w: np.ndarray, q: np.ndarray, inst: Instance, iters: int = 45
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted-welfare waterfilling. Returns (x, binding) with binding[...] a bool
    flag per slot: some floored tenant is held AT its floor by the constraint (i.e.
    its unconstrained level would fall below f_i). That indicator is the rho we
    sweep -- the resource channel of Remark 2."""
    s = inst.sat_arr
    f = inst.floors_arr
    B = inst.budget

    score = np.maximum(w * z * q, 1e-300)
    a = s * np.log(score / s)                      # [..., n]

    # Bracket nu. At nu_hi = max_i(a_i/s_i) every unconstrained level is <= 0, so
    # sum x = sum f <= B. At nu_lo the unconstrained sum already exceeds B.
    nu_hi = np.max(a / s, axis=-1, keepdims=True)
    nu_lo = (np.sum(a, axis=-1, keepdims=True) - B) / np.sum(s) - 1.0
    nu_lo = np.minimum(nu_lo, nu_hi - 1e-9)

    for _ in range(iters):
        nu = 0.5 * (nu_lo + nu_hi)
        x = np.maximum(f, a - s * nu)
        total = np.sum(x, axis=-1, keepdims=True)
        too_much = total > B                        # need a larger nu (less water)
        nu_lo = np.where(too_much, nu, nu_lo)
        nu_hi = np.where(too_much, nu_hi, nu)

    nu = 0.5 * (nu_lo + nu_hi)
    x = np.maximum(f, a - s * nu)

    # Renormalize the tiny bisection residual onto the unfloored coordinates so the
    # budget holds to machine precision without disturbing any binding floor.
    slack = B - np.sum(x, axis=-1, keepdims=True)
    free_mask = (x > f + 1e-12).astype(float)
    nfree = np.maximum(np.sum(free_mask, axis=-1, keepdims=True), 1.0)
    x = x + slack * free_mask / nfree

    unconstrained = a - s * nu
    binding = np.any((f > 0) & (unconstrained < f - 1e-12), axis=-1)
    return x, binding


def rule_pf(z: np.ndarray, w: np.ndarray, q: np.ndarray, inst: Instance):
    """Proportional rule with BINDING floors -- the rule the floored experiments need.

    Why not RULE_W. RULE_W maximizes sum_i w_i z_i q_i (1-e^{-x/s}); at w = 1 that IS
    the (constrained) welfare program, so w = 1 is plug-in optimal for EVERY input and
    the learner has nothing whatsoever to learn. The weight is vacuous, regret is
    identically zero, and with it the entire cross-epoch channel disappears. We found
    this empirically (the learner never left its uniform initialisation) and report it:
    a weight is only a meaningful object when the allocator is a PARAMETRIC family, not
    the welfare maximizer itself.

    RULE_PF keeps RULE_P's parametric proportional form -- so w genuinely tunes the
    allocation -- but clamps shares that fall below a floor and re-splits the remaining
    budget among the unclamped tenants (iterated to a fixed point in <= N passes). A
    floor therefore BINDS exactly when a tenant's proportional share would fall below
    it, which is the rho the paper sweeps. Own-report monotonicity is preserved: raising
    z_i raises tenant i's score and hence its share, and can only unclamp it.
    """
    f = inst.floors_arr
    B = inst.budget
    score = np.maximum(w * z * q, 1e-300)
    active = np.broadcast_to(np.ones_like(f, dtype=bool), score.shape).copy()

    x = np.zeros(score.shape, dtype=float)
    for _ in range(inst.n + 1):
        clamped_mass = np.sum(np.where(active, 0.0, f), axis=-1, keepdims=True)
        rem = B - clamped_mass
        sc = np.where(active, score, 0.0)
        denom = np.sum(sc, axis=-1, keepdims=True)
        share = rem * sc / np.maximum(denom, 1e-300)
        x = np.where(active, share, f)
        below = active & (x < f - 1e-12)
        if not np.any(below):
            break
        active = active & ~below

    binding = np.any((f > 0) & ~active, axis=-1)
    return x, binding


def allocate(z: np.ndarray, w: np.ndarray, q: np.ndarray, inst: Instance):
    """Dispatch on the instance's allocator. Returns (x, binding)."""
    if inst.allocator == "rule_p":
        x = rule_p(z, w, q, inst)
        return x, np.zeros(x.shape[:-1], dtype=bool)   # RULE_P never binds a floor
    if inst.allocator == "rule_pf":
        return rule_pf(z, w, q, inst)
    x, b = rule_w(z, w, q, inst)
    return x, b
