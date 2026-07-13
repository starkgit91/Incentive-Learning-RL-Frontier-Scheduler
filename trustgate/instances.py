"""Problem instances for the trust-gated truthful PRB allocator.

Instance A -- the theory-matching instance (Appendix B of the paper). Two tenants,
one channel state, exponential-saturating valuations. Everything about it is
analytically certifiable, so it is the instance the validation suite checks
against and the one that carries the fragility/gate/scaling figures.

Instance B -- the RAN-flavored instance. Three slices (URLLC/eMBB/mMTC-like), an
i.i.d. exogenous channel, and *service floors* whose level is the load knob: the
floors are calibrated to hit a target binding frequency rho. Floors are what open
the second (resource) manipulation channel, so every rho-sweep must use a rule whose floors can actually bind:
RULE_PF (see allocators.py). RULE_W is NOT usable: at w=1 it already solves the welfare
program, so the weight is vacuous and there is nothing to learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Instance:
    name: str
    n: int                       # tenants / slices
    budget: float                # B, total PRB mass per slot
    theta: Tuple[float, ...]     # true types
    theta_lo: Tuple[float, ...]  # type-space lower bound (Myerson integration start)
    theta_hi: Tuple[float, ...]
    sat: Tuple[float, ...]       # saturation s_i in v = theta*q*(1-exp(-x/s))
    floors: Tuple[float, ...]    # f_i (0 = unfloored)
    channel_states: Tuple[float, ...]   # support of q_i(c); () => single state q=1
    weight_sum: float            # simplex slice: sum_i w_i = weight_sum
    weight_lo: float             # w_i >= weight_lo
    allocator: str               # "rule_p" (floor-free closed form) | "rule_w" (KKT)

    @property
    def theta_arr(self) -> np.ndarray:
        return np.asarray(self.theta, dtype=float)

    @property
    def sat_arr(self) -> np.ndarray:
        return np.asarray(self.sat, dtype=float)

    @property
    def floors_arr(self) -> np.ndarray:
        return np.asarray(self.floors, dtype=float)

    @property
    def has_floors(self) -> bool:
        return bool(np.any(self.floors_arr > 0))


def instance_a() -> Instance:
    """Theory-matching instance. N=2, B=4, q=1, theta=(1,1), Theta=[0.9,1.1],
    v_i(x)=theta_i(1-e^{-x}); weight slice w1+w2=2, w1 in [0.9,1.1].
    Per-slot first-best welfare W* = 2(1-e^{-2}) = 1.7293294..."""
    return Instance(
        name="A",
        n=2,
        budget=4.0,
        theta=(1.0, 1.0),
        theta_lo=(0.9, 0.9),
        theta_hi=(1.1, 1.1),
        sat=(1.0, 1.0),
        floors=(0.0, 0.0),
        channel_states=(),          # single state, q == 1
        weight_sum=2.0,
        weight_lo=0.9,              # with sum=2 this also caps w1 <= 1.1
        allocator="rule_p",
    )


def instance_b(f1: float = 0.0) -> Instance:
    """RAN-flavored instance. S1 URLLC-like, S2 eMBB-like, S3 mMTC-like.
    ``f1`` is the load knob: floors (f1, 0, f1/2), calibrated to a target rho."""
    return Instance(
        name="B",
        n=3,
        budget=12.0,
        theta=(1.0, 3.0, 0.5),
        theta_lo=(0.8, 2.4, 0.4),   # 0.8 * theta
        theta_hi=(1.2, 3.6, 0.6),   # 1.2 * theta
        sat=(1.0, 3.0, 0.5),
        floors=(float(f1), 0.0, float(f1) / 2.0),
        channel_states=(0.6, 1.0, 1.4),
        weight_sum=3.0,
        weight_lo=0.2,
        allocator="rule_pf",
    )


# --------------------------------------------------------------------------- #
# Valuation primitives
# --------------------------------------------------------------------------- #
def h(x: np.ndarray, q: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Type-free part of the valuation: h_i = q_i * (1 - exp(-x_i / s_i)).

    The valuation is multiplicative in the type, v_i = theta_i * h_i(x, c), which
    is exactly the single-parameter structure Myerson's threshold payment needs.
    """
    return q * (1.0 - np.exp(-x / sat))


def welfare(x: np.ndarray, types: np.ndarray, q: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """sum_i types_i * h_i(x). Broadcasts over a leading slot axis."""
    return np.sum(types * h(x, q, sat), axis=-1)


def draw_channel(inst: Instance, rng: np.random.Generator, T: int) -> np.ndarray:
    """Exogenous channel path q[T, n]. Independent of allocation, reports, weights --
    this exogeneity is what lets the channel enter the (report-invariant) learner."""
    if not inst.channel_states:
        return np.ones((T, inst.n), dtype=float)
    return rng.choice(np.asarray(inst.channel_states, dtype=float), size=(T, inst.n))
