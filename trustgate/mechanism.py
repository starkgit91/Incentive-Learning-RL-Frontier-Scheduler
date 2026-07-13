"""Algorithm 1: the epoch loop.

    for each epoch k:
        w_k          frozen for the whole epoch
        x_t          = g(z_k, w_k, c_t)          <-- RAW REPORTS
        p_{i,k}      = Myerson(z_k, w_k, c_Ek)   <-- RAW REPORTS
        U_{i,k}      = sum_t theta_i h_i(x_t) - p_{i,k}   <-- TRUE type in the value
        m_k          = mean_{t in E_k} A_{i,t}   <-- MEASUREMENT
        itilde_k     = gate(z_k, m_k, mbar_k)    <-- GATED
        w_{k+1}      = Proj( w_k + eta grad What_k(itilde_k, .) )   <-- GATED ONLY

THE WIRING RULE. Allocation and payments consume the RAW report z. Only the learner
consumes the gated input itilde. Crossing these two wires is the single most likely
implementation bug -- it would either (a) destroy within-epoch DSIC (if itilde drove
the allocation the payment prices) or (b) silently reopen the cross-epoch channel
(if z leaked into the learner). V3 is the canary: at r = 0 two runs with *different*
reports on the same seed must produce BITWISE IDENTICAL weight paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .allocators import allocate
from .gates import Gate
from .instances import Instance, h
from .learner import OGD
from .payments import epoch_payments_all
from .traffic import epoch_means, running_means, sigma_vector


@dataclass
class RunResult:
    w_path: np.ndarray            # [K, n] weights used in each epoch
    utility: np.ndarray           # [n] total realized utility over the horizon
    payments: np.ndarray          # [n] total payments
    rho: float                    # fraction of slots with a binding floor
    welfare_path: np.ndarray      # [K] true welfare W(w_k; theta) (filled by metrics)
    K: int
    L: int


def run(
    inst: Instance,
    gate: Gate,
    A: np.ndarray,                # [T, n] pre-drawn traffic (CRN)
    q: np.ndarray,                # [T, n] pre-drawn channel (CRN)
    L: int,
    sigma: float,
    report_fn: Callable[[int], np.ndarray],   # epoch -> reports z_k [n]
    eta: float = 0.1,
    grid_size: int = 41,
    lambda_reg: float = 0.0,
    grad_noise: float = 0.0,      # baseline B2 (DP-gradient): sd of Gaussian noise
    freeze_after: Optional[int] = None,   # baseline B4 (burn-in commit): epoch index
    rng: Optional[np.random.Generator] = None,
    pad_delta: Optional[np.ndarray] = None,   # E4: traffic padding added by tenant
    fine_kappa: float = 0.0,      # baseline B3 (fines): per-unit penalty on |z - m_k|
    static: bool = False,         # baseline B6: never update the weights
    pad_cost: float = 0.0,        # E4: per-unit cost of padding the traffic
    compute_payments: bool = True,   # skip the Myerson quadrature when only the
    learner_factory: Optional[Callable] = None,   # E8: swap OGD for any update rule
) -> RunResult:                      # weight path / rho are needed (regret, calibration)
    T = A.shape[0]
    K = T // L
    theta = inst.theta_arr
    sig = sigma_vector(inst, sigma)

    A_eff = A if pad_delta is None else A + pad_delta       # padding moves the MEASUREMENT
    m = epoch_means(A_eff, L)                               # [K, n]
    mbar = running_means(m)                                 # [K, n]

    # Theorem 3 is agnostic to the update rule: any learner whose inputs carry no
    # report inherits the exact-invariance guarantee (E8 swaps in an RL agent here).
    lrn = OGD(inst, eta=eta, lambda_reg=lambda_reg) if learner_factory is None \
        else learner_factory(inst)
    rng = np.random.default_rng(0) if rng is None else rng

    w_path = np.zeros((K, inst.n))
    util = np.zeros(inst.n)
    pay = np.zeros(inst.n)
    bind_slots = 0

    for k in range(K):
        sl = slice(k * L, (k + 1) * L)
        qk = q[sl]                                          # [L, n]
        zk = np.asarray(report_fn(k), dtype=float)          # [n]
        wk = lrn.w.copy()
        w_path[k] = wk

        # --- allocation + payments: RAW REPORTS ---------------------------------
        zrep = np.broadcast_to(zk, (L, inst.n))
        x, binding = allocate(zrep, wk, qk, inst)           # [L, n]
        bind_slots += int(np.sum(binding))
        pk = (epoch_payments_all(zk, wk, qk, inst, grid_size=grid_size)
              if compute_payments else np.zeros(inst.n))
        # realized utility: TRUE type in the value, reported type in alloc & payment
        util += np.sum(theta * h(x, qk, inst.sat_arr), axis=0) - pk
        pay += pk

        # B3 (fines): a transfer that penalizes a report the measurement does not
        # corroborate. This is the *other* way to use the measurement -- price the
        # discrepancy instead of refusing to learn from it. It works, but only if the
        # operator can commit to a large enough penalty (kappa >= C1), and it taxes
        # honest tenants whose traffic happens to fluctuate. The gate needs neither.
        if fine_kappa > 0.0:
            from .gates import radius_epoch
            rL = radius_epoch(sig, L, inst.n, K)
            util -= fine_kappa * L * np.maximum(np.abs(zk - m[k]) - rL, 0.0)

        # E4: padding is not free -- the tenant pays c_pad per unit of fake traffic it
        # injects to move its own measurement (Prop. 1). pad_delta is a per-tenant
        # per-slot offset [n].
        if pad_cost > 0.0 and pad_delta is not None:
            util -= pad_cost * L * np.abs(np.asarray(pad_delta, float))

        # --- learner: GATED INPUT ONLY ------------------------------------------
        if static:
            continue                                        # B6: weights never move
        if freeze_after is not None and k >= freeze_after:
            continue                                        # B4: stop updating
        itilde = gate.inputs(zk, m[k], mbar[k], sig, inst, L, k, K)
        if grad_noise > 0.0:
            # B2 (DP-gradient): protect the LEARNER, not the INPUT. Inputs are raw.
            lrn.step(itilde, qk)
            lrn.w = _noisy_project(lrn, rng, grad_noise)
        else:
            lrn.step(itilde, qk)

    return RunResult(
        w_path=w_path,
        utility=util,
        payments=pay,
        rho=bind_slots / float(K * L),
        welfare_path=np.zeros(K),
        K=K,
        L=L,
    )


def _noisy_project(lrn: OGD, rng: np.random.Generator, scale: float) -> np.ndarray:
    from .learner import project
    noise = rng.normal(0.0, scale, size=lrn.inst.n)
    noise -= noise.mean()            # keep it in the tangent space of the slice
    return project(lrn.w + lrn.eta * noise, lrn.inst)


# --------------------------------------------------------------------------- #
# Report strategies
# --------------------------------------------------------------------------- #
def truthful(inst: Instance) -> Callable[[int], np.ndarray]:
    th = inst.theta_arr
    return lambda k: th


def persistent_deviation(inst: Instance, tenant: int, d: float) -> Callable[[int], np.ndarray]:
    """Tenant `tenant` reports theta + d every epoch; everyone else truthful."""
    z = inst.theta_arr.copy()
    z[tenant] = float(np.clip(z[tenant] + d, inst.theta_lo[tenant], inst.theta_hi[tenant]))
    return lambda k: z


def change_point_deviation(inst: Instance, tenant: int, d: float, K: int) -> Callable[[int], np.ndarray]:
    """Truthful for the first half, then a persistent deviation (P1 curve)."""
    zt = inst.theta_arr
    zd = inst.theta_arr.copy()
    zd[tenant] = float(np.clip(zd[tenant] + d, inst.theta_lo[tenant], inst.theta_hi[tenant]))
    half = K // 2
    return lambda k: (zt if k < half else zd)
