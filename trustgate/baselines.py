"""Baselines B1-B6 and the constant C1.

The point of Fig. 5 is that the trust family traces the lower-left (Regret, Slack)
frontier and Pareto-dominates the two ideas the literature would reach for first:

* B2 DP-gradient. Weakly-DP online learning (Huh & Kandasamy) protects the LEARNER by
  noising its gradient, while still feeding it raw reports. It is one-shot protection:
  the report still moves the weights in expectation, so slack barely falls until the
  noise is large enough to destroy the regret. Noise is added AFTER the gate is
  bypassed -- that is exactly the comparison being made.
* B4 Burn-in commit. Train on raw reports for T0, then freeze forever. Dominated,
  because the attacker simply front-loads the lie into the burn-in window.
* B3 Fines. The other honest use of a measurement: don't refuse to learn from an
  uncorroborated report, PRICE it. This does work -- for kappa >= C1 -- but it needs
  the operator to commit to a large penalty and it taxes honest tenants whose traffic
  fluctuates. The gate needs neither commitment nor a penalty.

A faithful Dai-Golrezaei-Jaillet baseline would need their long-term-cost setting; we
say so explicitly and use B4 (commitment) and B2 (DP) as the implementable
representatives of those two defense ideas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .allocators import rule_p
from .gates import CUM0, CUM_R, EPOCH0, ORACLE, RINF, Gate, radius_epoch
from .instances import Instance, h, instance_a
from .learner import plug_in_welfare, project


@dataclass
class Mechanism:
    """A row of Fig. 5: a gate plus whatever extra training pathway the baseline adds."""
    name: str
    gate: Gate
    family: str                      # "trust" | "baseline"
    kwargs: Dict = field(default_factory=dict)


def compute_C1(inst: Instance = None, verbose: bool = False) -> float:
    """C1 = 4 sqrt(N) C_w G_v G_x (1 + 2 theta_bar), computed numerically for the
    instance (Prop. 1 / App. C.5). It is the per-unit penalty above which pricing the
    report-measurement discrepancy makes manipulation unprofitable -- and the same
    constant sets the padding knee of E4. It is a conservative bound; the empirical
    knee sits at or below it, and we say so."""
    inst = instance_a() if inst is None else inst
    N = inst.n
    th_hi = float(max(inst.theta_hi))
    th_bar = float(np.mean(inst.theta))

    # G_v = max_i theta_i * phi'(0) = theta_hi  (phi' = e^{-x}/s <= 1/s, s = 1)
    G_v = th_hi / float(min(inst.sat))

    # G_x = max |dx_i / dz_i| for RULE_P over the (weight x report) box.
    ws = np.linspace(inst.weight_lo, inst.weight_sum - inst.weight_lo, 25)
    zs = np.linspace(inst.theta_lo[0], inst.theta_hi[0], 25)
    q1 = np.ones((1, N))
    G_x = 0.0
    eps = 1e-5
    for w1 in ws:
        w = project(np.array([w1] + [(inst.weight_sum - w1) / (N - 1)] * (N - 1)), inst)
        for z1 in zs:
            z = inst.theta_arr.copy(); z[0] = z1
            zp = z.copy(); zp[0] = z1 + eps
            x = rule_p(z[None, :], w, q1, inst)[0, 0]
            xp = rule_p(zp[None, :], w, q1, inst)[0, 0]
            G_x = max(G_x, abs((xp - x) / eps))

    # C_w = G_wtheta / m:  how far the learned weight moves per unit of input, where
    # G_wtheta = max |d/d(theta-hat) grad_w W| and m = min |d^2 W / dw^2| on the slice.
    def W_of(w, itilde):
        return plug_in_welfare(np.asarray(w, float), np.asarray(itilde, float), q1, inst)

    w0 = np.full(N, inst.weight_sum / N)
    dW = 1e-4
    m_curv, G_wt = np.inf, 0.0
    for z1 in zs:
        it = inst.theta_arr.copy(); it[0] = z1
        # d2W/dw1^2 along the slice direction (1,-1)/sqrt(2)
        v = np.zeros(N); v[0] = 1.0; v[1] = -1.0; v /= np.sqrt(2)
        f0 = W_of(w0, it); fp = W_of(w0 + dW * v, it); fm = W_of(w0 - dW * v, it)
        curv = abs((fp - 2 * f0 + fm) / dW ** 2)
        m_curv = min(m_curv, curv)
        # d/d(itilde) of grad_w W
        itp = it.copy(); itp[0] += eps
        g0 = (W_of(w0 + dW * v, it) - W_of(w0 - dW * v, it)) / (2 * dW)
        g1 = (W_of(w0 + dW * v, itp) - W_of(w0 - dW * v, itp)) / (2 * dW)
        G_wt = max(G_wt, abs((g1 - g0) / eps))
    C_w = G_wt / max(m_curv, 1e-9)

    C1 = 4.0 * np.sqrt(N) * C_w * G_v * G_x * (1.0 + 2.0 * th_bar)
    if verbose:
        print(f"  G_v={G_v:.4f}  G_x={G_x:.4f}  G_wtheta={G_wt:.4f}  m={m_curv:.4f}  "
              f"C_w={C_w:.4f}  ->  C1={C1:.4f}")
    return float(C1)


def e5_mechanisms(C1: float) -> List[Mechanism]:
    """Every marker in Fig. 5."""
    M: List[Mechanism] = []
    # trust family: the radius sweep, plus both r = 0 anchors
    for r in (0.0, 0.5, 1.0, 2.0, 4.0):
        g = Gate("epoch", r)
        M.append(Mechanism(g.label, g, "trust"))
    M.append(Mechanism("CUM0", CUM0, "trust"))
    M.append(Mechanism("CUM(r_k)", CUM_R, "trust"))
    # B1 report-trained
    M.append(Mechanism("B1 RINF", RINF, "baseline"))
    # B2 DP-gradient (raw reports + noisy gradient)
    for lam in (0.1, 0.3, 1.0, 3.0):
        M.append(Mechanism(f"B2 DP($\\lambda$={lam:g})", RINF, "baseline",
                           dict(grad_noise=lam)))
    # B3 fines (raw reports + a penalty on the report-measurement discrepancy)
    for kap in (0.5, 1.0, 2.0):
        M.append(Mechanism(f"B3 fine($\\kappa$={kap:g}$C_1$)", RINF, "baseline",
                           dict(fine_kappa=kap * C1)))
    # B4 burn-in commit (raw reports, then freeze)
    for frac in (0.1, 0.2, 0.4):
        M.append(Mechanism(f"B4 burn-in({frac:g}T)", RINF, "baseline",
                           dict(_freeze_frac=frac)))
    # B5 skyline, B6 floor
    M.append(Mechanism("B5 ORACLE", ORACLE, "anchor"))
    M.append(Mechanism("B6 static", ORACLE, "anchor", dict(static=True)))
    return M
