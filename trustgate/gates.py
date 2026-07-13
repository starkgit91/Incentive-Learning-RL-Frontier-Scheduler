"""Trust gates -- the contribution of the paper.

The learner needs an estimate of the tenants' types. The naive choice is the
tenants' own REPORTS; that is exactly what makes a learning allocator fragile,
because a tenant's epoch-k report then steers every future weight (Thm 1: the
per-slot gain is Theta(1), so Slack = Theta(T)).

But in this problem the type is a *physical, measurable* quantity: theta_i is the
mean of the tenant's own traffic, and the operator measures the traffic anyway. So
we can feed the learner a MEASUREMENT instead of a report. The gate is the map

    itilde_{i,k} = clip( z_{i,k},  anchor_{i,k} - r,  anchor_{i,k} + r )

with an anchor built from measurements and a trust radius r:

* RINF      r = infinity, anchor irrelevant -> itilde = z.  The fragile baseline.
* EPOCH(r)  anchor = m_k (this epoch's traffic mean), r = r_L (a concentration
            radius): a report is trusted only insofar as it is corroborated.
* EPOCH0    r = 0, anchor = m_k. The learner sees ONLY the measurement.
* CUM(r_k)  anchor = mbar_k (running mean over all past epochs), r_k shrinking.
* CUM0      r = 0, anchor = mbar_k. The upgraded theorem's anchor.

The r = 0 members are the crux. Their itilde is a function of (traffic, channel)
ONLY -- it contains no report at all. Therefore the entire weight path w_1..w_K is
a deterministic function of quantities the tenant cannot influence, so within each
epoch the mechanism is a FIXED monotone rule with a Myerson payment. Truthfulness
is not "approximately" restored; the cross-epoch channel is *identically closed*:

    Slack = 0 exactly, at every horizon, load, sigma and L   (Theorem 3).

This is why r = 0 escapes the Omega(T^{2/3}) report-training barrier: that barrier
constrains what a learner fed *reports* can do, and this learner is not fed reports.

The price is that a measurement is noisy: itilde = m_k has sd sigma/sqrt(L), which
biases the learned weights (Thm 4). The running-mean anchor (CUM0) shrinks that sd
to sigma/sqrt(kL), which is what turns the price of invariance from O(sigma T/sqrt(L))
into O(sigma sqrt(T)) -- Theorem 4'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instances import Instance


def radius_epoch(sigma_i: np.ndarray, L: int, n: int, K: int, alpha: float | None = None) -> np.ndarray:
    """r_L = (sigma/sqrt(L)) * sqrt(2 ln(2 N K / alpha)). A union bound over all
    (tenant, epoch) pairs, so an honest report is inside the band w.h.p. and the
    gate never taxes honesty (that is the 'no tax on honesty' claim, E2)."""
    alpha = 1.0 / (K * L) if alpha is None else alpha
    return (sigma_i / np.sqrt(L)) * np.sqrt(2.0 * np.log(2.0 * n * K / alpha))


def radius_cum(sigma_i: np.ndarray, L: int, k: int, n: int, K: int, alpha: float | None = None) -> np.ndarray:
    """r_k = sigma * sqrt( 2 ln(2 N K / alpha) / ((k+1) L) ): the shrinking radius of the
    running-mean anchor, using the (k+1) epochs of traffic measured so far (k 0-based)."""
    alpha = 1.0 / (K * L) if alpha is None else alpha
    seen = (k + 1) * L
    return sigma_i * np.sqrt(2.0 * np.log(2.0 * n * K / alpha) / seen)


@dataclass
class Gate:
    """kind: 'rinf' | 'epoch' | 'cum'.  r_mult: multiples of the concentration radius
    (0.0 => the report is ignored entirely; inf => never gated)."""
    kind: str
    r_mult: float = 1.0

    @property
    def label(self) -> str:
        if self.kind == "rinf":
            return "RINF"
        if self.kind == "oracle":
            return "ORACLE"
        base = "EPOCH" if self.kind == "epoch" else "CUM"
        if self.r_mult == 0.0:
            return base + "0"
        return f"{base}(r={self.r_mult:g})"

    @property
    def report_invariant(self) -> bool:
        """True iff the learner's input cannot depend on any report. These are the
        mechanisms Theorem 3 covers, and the ones V3 checks bitwise."""
        if self.kind == "oracle":
            return True                 # trained on the true types: trivially invariant
        return self.kind != "rinf" and self.r_mult == 0.0

    def inputs(
        self,
        z: np.ndarray,          # [n] reports this epoch
        m_k: np.ndarray,        # [n] this epoch's measured traffic mean
        mbar_k: np.ndarray,     # [n] running mean of past epochs (nan at k=0)
        sigma_i: np.ndarray,    # [n]
        inst: Instance,
        L: int,
        k: int,
        K: int,
    ) -> np.ndarray:
        """The gated learner input i-tilde. NOTE: this NEVER touches the allocation or
        the payment -- those always consume the raw report z. Crossing those two wires
        is the single most likely bug in this codebase; V3 is the canary."""
        lo = np.asarray(inst.theta_lo, dtype=float)
        hi = np.asarray(inst.theta_hi, dtype=float)

        if self.kind == "oracle":
            # B5: the learner is handed the TRUE types. Not implementable (the whole
            # premise is that theta is private) -- it is the regret SKYLINE of Fig. 5,
            # and the reference E2 compares the gated learners against.
            return inst.theta_arr.copy()

        if self.kind == "rinf":
            return np.clip(z, lo, hi)

        if self.kind == "epoch":
            anchor = m_k
            r = self.r_mult * radius_epoch(sigma_i, L, inst.n, K)
        else:                                   # cum: inclusive running mean, so the
            anchor = mbar_k                     # anchor exists from epoch 0 onward and
            r = self.r_mult * radius_cum(sigma_i, L, k, inst.n, K)   # never falls back
            # to the report (that fallback would break the exact invariance of Thm 3).

        if self.r_mult == 0.0:
            return np.clip(anchor, lo, hi)      # report-invariant: z never enters
        return np.clip(np.clip(z, anchor - r, anchor + r), lo, hi)


# The mechanism family used across the experiments.
RINF = Gate("rinf")
EPOCH_R = Gate("epoch", 1.0)
EPOCH_2R = Gate("epoch", 2.0)
EPOCH0 = Gate("epoch", 0.0)
CUM_R = Gate("cum", 1.0)
CUM0 = Gate("cum", 0.0)
ORACLE = Gate("oracle")           # B5 skyline: trained on the true (private) types
