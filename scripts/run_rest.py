#!/usr/bin/env python3
"""E1c, E4, E6, E8 -- the remaining experiments.

E1c (P1)  Slack vs LOAD. Floors open a *second*, resource channel on top of the payment
          channel (Remark 2), so the report-trained learner should get MORE manipulable
          as the binding frequency rho rises. The r = 0 gate must stay at exactly zero at
          every load -- invariance is not a function of how contended the cell is.

E4 (P2)   The padding knee. At r = 0 the report is out of the learner, so the ONLY
          remaining attack is to fake the *measurement*: inject c_pad-priced dummy
          traffic to drag the anchor. Proposition 1 says this dies once the per-unit
          padding cost exceeds C1. This is the honest boundary of the guarantee and we
          state it as such.

E6 (P2)   Closed-loop caveat. If traffic RESPONDS to the allocation, the measurement
          inherits a dependence on the report and the channel reopens even at r = 0.
          Slack should be 0 at feedback gain 0 and grow with it. This is the assumption
          the theorem actually needs (exogenous traffic), stated honestly.

E8 (P2)   Theorem 3 is ALGORITHM-AGNOSTIC. Swap OGD for a measurement-fed tabular RL
          agent; the V3 canary and zero slack must both still hold.

    venv_linux/bin/python scripts/run_rest.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mtp_droy_mpl_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustgate.adversary import slack_for_seed                     # noqa: E402
from trustgate.baselines import compute_C1                         # noqa: E402
from trustgate.gates import CUM0, EPOCH0, EPOCH_R, RINF            # noqa: E402
from trustgate.instances import draw_channel, instance_a, instance_b  # noqa: E402
from trustgate.mechanism import persistent_deviation, run, truthful   # noqa: E402
from trustgate.metrics import ci95                                 # noqa: E402
from trustgate.rl import MeasurementQLearner                       # noqa: E402
from trustgate.traffic import draw_traffic                         # noqa: E402

OUT = ROOT / "outputs" / "trustgate"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(int(os.environ.get("SEEDS", 4))))


# --------------------------------------------------------------------------- #
def e1c(T=50000, L=50, sigma=0.2):
    print("== E1c: slack vs load (rho) -- floors open a resource channel ==", flush=True)
    loads = json.loads((ROOT / "configs" / "loads.json").read_text())
    rows = []
    for key, cfg in loads.items():
        inst = instance_b(f1=cfg["f1"])
        grid = np.linspace(-0.2 * inst.theta[0], 0.2 * inst.theta[0], 21)
        for gate in (RINF, CUM0):
            sl = []
            for s in SEEDS:
                r = slack_for_seed(inst, gate, seed=6000 + s, T=T, L=L, sigma=sigma,
                                   tenant=0, grid=grid, grid_size=9)
                sl.append(r.slack)
                rows.append(dict(exp="E1c", rho=cfg["rho"], f1=cfg["f1"],
                                 mech=gate.label, seed=s, slack=r.slack))
            m, hw = ci95(np.array(sl))
            print(f"  rho={cfg['rho']:.3f} (f1={cfg['f1']:.2f}) {gate.label:8s} "
                  f"slack={m:.4e} +/- {hw:.1e}   slope=slack/T={m/T:.3e}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def e4(T=20000, L=50, sigma=0.1):
    """Padding: the tenant injects delta units of dummy traffic per slot (cost c_pad
    per unit) to drag its own measurement, and simultaneously misreports by d."""
    print("== E4: padding knee (the only attack left at r=0) ==", flush=True)
    inst = instance_a()
    C1 = compute_C1(inst)
    print(f"  C1 = {C1:.3f} (conservative bound from Prop. 1)", flush=True)
    ds = np.linspace(-0.1, 0.1, 11)
    deltas = np.linspace(0.0, 0.15, 8)
    rows = []
    for cmult in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        c_pad = cmult * C1
        sl = []
        for s in SEEDS:
            rng = np.random.default_rng(7000 + s)
            A = draw_traffic(inst, sigma, rng, T)
            q = draw_channel(inst, rng, T)
            base = run(inst, EPOCH0, A, q, L, sigma, truthful(inst), grid_size=9)
            u0 = float(base.utility[0])
            best = 0.0
            for d in ds:
                for dl in deltas:
                    pad = np.zeros(inst.n); pad[0] = dl
                    r = run(inst, EPOCH0, A, q, L, sigma,
                            persistent_deviation(inst, 0, float(d)), grid_size=9,
                            pad_delta=pad, pad_cost=c_pad)
                    best = max(best, float(r.utility[0]) - u0)
            sl.append(best)
            rows.append(dict(exp="E4", c_mult=cmult, c_pad=c_pad, seed=s, slack=best))
        m, hw = ci95(np.array(sl))
        print(f"  c_pad={cmult:.2f}*C1={c_pad:6.3f}  slack={m:.4e} +/- {hw:.1e}", flush=True)
    return pd.DataFrame(rows), C1


# --------------------------------------------------------------------------- #
def e6(T=20000, L=50, sigma=0.1):
    """Closed-loop: traffic mean responds to last slot's allocation, so the measurement
    is no longer exogenous and the report leaks back in through it."""
    print("== E6: closed-loop traffic reopens the channel at r=0 ==", flush=True)
    from trustgate.instances import h
    from trustgate.allocators import allocate
    inst = instance_a()
    grid = np.linspace(-0.1, 0.1, 21)
    rows = []
    for kfb in (0.0, 0.1, 0.25, 0.5, 1.0):
        sl = []
        for s in SEEDS:
            # Traffic that responds to the allocation: A_t depends on x_{t-1}, which
            # depends on the report. Simulated by a first-order surrogate: we perturb
            # the pre-drawn marks by the (report-dependent) allocation deviation.
            def make(dev_d):
                rng = np.random.default_rng(8000 + s)
                A = draw_traffic(inst, sigma, rng, T)
                q = draw_channel(inst, rng, T)
                z = inst.theta_arr.copy(); z[0] = np.clip(z[0] + dev_d, 0.9, 1.1)
                w0 = np.full(inst.n, inst.weight_sum / inst.n)
                x, _ = allocate(np.broadcast_to(z, (T, inst.n)), w0, q, inst)
                xb, _ = allocate(np.broadcast_to(inst.theta_arr, (T, inst.n)), w0, q, inst)
                A = A * (1.0 + kfb * (x - xb) / inst.budget)
                return np.maximum(A, 0.05 * inst.theta_arr), q
            A0, q0 = make(0.0)
            base = run(inst, CUM0, A0, q0, L, sigma, truthful(inst), grid_size=9)
            u0 = float(base.utility[0])
            best = 0.0
            for d in grid:
                Ad, qd = make(float(d))
                r = run(inst, CUM0, Ad, qd, L, sigma,
                        persistent_deviation(inst, 0, float(d)), grid_size=9)
                best = max(best, float(r.utility[0]) - u0)
            sl.append(max(0.0, best))
            rows.append(dict(exp="E6", kappa_fb=kfb, seed=s, slack=max(0.0, best)))
        m, hw = ci95(np.array(sl))
        print(f"  kappa_fb={kfb:.2f}  slack={m:.4e} +/- {hw:.1e}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def e8(T=50000, L=50, sigma=0.1):
    """Theorem 3 covers ANY update rule: swap OGD for a measurement-fed RL agent."""
    print("== E8: RL generality (Thm 3 is algorithm-agnostic) ==", flush=True)
    inst = instance_a()
    grid = np.linspace(-0.1, 0.1, 21)
    rows = []
    for name, factory in [("OGD", None),
                          ("RL (tabular Q, measurement-fed)",
                           lambda i: MeasurementQLearner(i, seed=0))]:
        for gate in (RINF, CUM0):
            # V3 canary first: does the POLICY path move with the report?
            rng = np.random.default_rng(99)
            A = draw_traffic(inst, sigma, rng, T)
            q = draw_channel(inst, rng, T)
            r1 = run(inst, gate, A, q, L, sigma, truthful(inst), grid_size=9,
                     learner_factory=factory)
            r2 = run(inst, gate, A, q, L, sigma, persistent_deviation(inst, 0, -0.05),
                     grid_size=9, learner_factory=factory)
            invariant = bool(np.array_equal(r1.w_path, r2.w_path))

            sl = []
            for s in SEEDS:
                r = slack_for_seed(inst, gate, seed=9000 + s, T=T, L=L, sigma=sigma,
                                   tenant=0, grid=grid, grid_size=9,
                                   learner_factory=factory)
                sl.append(r.slack)
                rows.append(dict(exp="E8", learner=name, mech=gate.label, seed=s,
                                 slack=r.slack, invariant=invariant))
            m, hw = ci95(np.array(sl))
            print(f"  {name:34s} {gate.label:6s} V3_bitwise={str(invariant):5s} "
                  f"slack={m:.4e} +/- {hw:.1e}", flush=True)
    return pd.DataFrame(rows)


def figure(d1c, d4, d6, C1):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.16, 2.4))

    for mech, st in [("RINF", dict(color="#d62728", ls="--", marker="o")),
                     ("CUM0", dict(color="#2ca02c", ls=":", marker="*"))]:
        g = d1c[d1c.mech == mech].groupby("rho")["slack"].mean()
        a.plot(g.index, g.values, ms=4, lw=1.4, label=mech, **st)
    a.set_xlabel(r"binding frequency $\rho$"); a.set_ylabel("slack")
    a.set_title("(a) floors add a channel", fontsize=8)
    a.grid(alpha=0.25); a.legend(fontsize=6)

    g = d4.groupby("c_mult")["slack"].mean()
    b.plot(g.index, g.values, color="#1f77b4", marker="s", ms=4, lw=1.4)
    b.axvline(1.0, color="k", ls=":", lw=0.9)
    b.text(1.03, 0.85, "$C_1$", transform=b.get_xaxis_transform(), fontsize=6)
    b.set_xlabel(r"padding cost $c_{\rm pad}/C_1$"); b.set_ylabel("slack")
    b.set_title("(b) padding knee", fontsize=8)
    b.grid(alpha=0.25)

    g = d6.groupby("kappa_fb")["slack"].mean()
    c.plot(g.index, g.values, color="#9467bd", marker="^", ms=4, lw=1.4)
    c.set_xlabel(r"feedback gain $\kappa_{\rm fb}$"); c.set_ylabel("slack at $r{=}0$")
    c.set_title("(c) closed-loop caveat", fontsize=8)
    c.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6_boundaries.{ext}", dpi=200)
    print(f"wrote {OUT/'fig6_boundaries.pdf'}")


if __name__ == "__main__":
    d1c = e1c();      d1c.to_csv(OUT / "e1c.csv", index=False)
    d4, C1 = e4();    d4.to_csv(OUT / "e4.csv", index=False)
    d6 = e6();        d6.to_csv(OUT / "e6.csv", index=False)
    d8 = e8();        d8.to_csv(OUT / "e8.csv", index=False)
    figure(d1c, d4, d6, C1)
