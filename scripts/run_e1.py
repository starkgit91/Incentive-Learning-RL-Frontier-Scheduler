#!/usr/bin/env python3
"""E1 (P0) -> paper Fig. 2: fragility, the gate, and invariance.

E1a  Slack vs horizon T, for the whole trust family.
     Expected: RINF linear in T with slope ~ c0 = 1.479e-4 (the certified analytic
     line is overlaid); EPOCH(r) bounded; CUM(r_k) sublinear; EPOCH0 / CUM0 flat at
     exactly 0.
E1b  Slack vs epoch length L (RINF). Expected: FLAT in L -- the visual kill-shot for
     "just freeze the policy longer". Freezing does not close a channel that is open
     by construction; only removing the report from the learner does.

    venv_linux/bin/python scripts/run_e1.py
"""
from __future__ import annotations

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

from trustgate import analytic as AN                       # noqa: E402
from trustgate.adversary import slack_for_seed             # noqa: E402
from trustgate.gates import CUM0, CUM_R, EPOCH0, EPOCH_2R, EPOCH_R, RINF  # noqa: E402
from trustgate.instances import instance_a                 # noqa: E402
from trustgate.metrics import ci95                         # noqa: E402

OUT = ROOT / "outputs" / "trustgate"
OUT.mkdir(parents=True, exist_ok=True)

INST = instance_a()
SIGMA = 0.1
GRID = np.linspace(-0.1, 0.1, 41)
GQ = 11                       # payment quadrature points (11 is ample for Instance A)
MECHS = [RINF, EPOCH_R, EPOCH_2R, CUM_R, EPOCH0, CUM0]


def e1a(seeds, Ts, L=50) -> pd.DataFrame:
    rows = []
    for T in Ts:
        for gate in MECHS:
            for s in seeds:
                r = slack_for_seed(INST, gate, seed=1000 + s, T=int(T), L=L, sigma=SIGMA,
                                   tenant=0, grid=GRID, grid_size=GQ)
                rows.append(dict(exp="E1a", T=int(T), L=L, mech=gate.label, seed=s,
                                 slack=r.slack, best_d=r.best_d))
            sub = [x["slack"] for x in rows if x["T"] == T and x["mech"] == gate.label]
            m, hw = ci95(np.array(sub))
            print(f"  T={int(T):>7d} {gate.label:12s} slack={m:.4e} +/- {hw:.1e}", flush=True)
    return pd.DataFrame(rows)


def e1b(seeds, Ls, T=50000) -> pd.DataFrame:
    rows = []
    for L in Ls:
        for s in seeds:
            r = slack_for_seed(INST, RINF, seed=2000 + s, T=T, L=int(L), sigma=SIGMA,
                               tenant=0, grid=GRID, grid_size=GQ)
            rows.append(dict(exp="E1b", T=T, L=int(L), mech="RINF", seed=s,
                             slack=r.slack, best_d=r.best_d))
        sub = [x["slack"] for x in rows if x["L"] == L]
        m, hw = ci95(np.array(sub))
        print(f"  L={int(L):>4d} RINF slack={m:.4e} +/- {hw:.1e}", flush=True)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})

    style = {
        "RINF":        dict(color="#d62728", ls="--", marker="o", label="RINF (report-trained)"),
        "EPOCH(r=1)":  dict(color="#1f77b4", ls="-", marker="s", label=r"EPOCH($r_L$)"),
        "EPOCH(r=2)":  dict(color="#1f77b4", ls="-.", marker="^", label=r"EPOCH($2r_L$)"),
        "CUM(r=1)":    dict(color="#2ca02c", ls="-", marker="v", label=r"CUM($r_k$)"),
        "EPOCH0":      dict(color="#1f77b4", ls=":", marker="D", label="EPOCH0  ($r{=}0$)"),
        "CUM0":        dict(color="#2ca02c", ls=":", marker="*", label="CUM0  ($r{=}0$)"),
    }

    a = df[df.exp == "E1a"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.9))

    for mech, st in style.items():
        g = a[a.mech == mech].groupby("T")["slack"]
        if g.count().empty:
            continue
        Ts = np.array(sorted(g.groups.keys()), dtype=float)
        mu = np.array([g.get_group(t).mean() for t in Ts])
        hw = np.array([ci95(g.get_group(t).to_numpy())[1] for t in Ts])
        ax.plot(Ts, mu, lw=1.4, ms=3.5, **st)
        ax.fill_between(Ts, mu - hw, mu + hw, color=st["color"], alpha=0.15, lw=0)

    Ts = np.array(sorted(a["T"].unique()), dtype=float)
    c0 = AN.certified_constants()["c0"]
    ax.plot(Ts, c0 * Ts, color="k", lw=0.9, alpha=0.8,
            label="analytic $c_0 T$  ($c_0$=%.2e)" % c0)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("horizon $T$ (slots)")
    ax.set_ylabel("incentive slack (tenant utility gain)")
    ax.set_title("(a) fragility, the gate, and exact invariance", fontsize=8)
    ax.legend(fontsize=5.6, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)

    b = df[df.exp == "E1b"]
    if not b.empty:
        g = b.groupby("L")["slack"]
        Ls = np.array(sorted(g.groups.keys()), dtype=float)
        mu = np.array([g.get_group(l).mean() for l in Ls])
        hw = np.array([ci95(g.get_group(l).to_numpy())[1] for l in Ls])
        ax2.errorbar(Ls, mu, yerr=hw, color="#d62728", ls="--", marker="o", ms=4, lw=1.4,
                     capsize=2.5, label="RINF")
        ax2.axhline(float(mu.mean()), color="k", lw=0.8, alpha=0.6, ls=":",
                    label="flat in $L$")
        ax2.axhline(0.0, color="#1f77b4", lw=1.6, ls=":", label="EPOCH0 / CUM0 $\\equiv 0$")
        ax2.set_xscale("log", base=2)
        ax2.set_ylim(bottom=-0.05 * float(mu.max()))
        ax2.set_xlabel("epoch length $L$ (slots)")
        ax2.set_ylabel(f"slack at $T={int(b['T'].iloc[0]):,}$")
        ax2.set_title("(b) freezing longer does NOT help", fontsize=8)
        ax2.legend(fontsize=6)
        ax2.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_fragility.{ext}", dpi=200)
    print(f"wrote {OUT/'fig2_fragility.pdf'}")


if __name__ == "__main__":
    seeds = list(range(int(os.environ.get("SEEDS", 8))))
    print("== E1a: slack vs horizon ==", flush=True)
    dfa = e1a(seeds, Ts=[2500, 5000, 10000, 25000, 50000, 100000])
    print("== E1b: slack vs epoch length (RINF) ==", flush=True)
    dfb = e1b(seeds, Ls=[25, 50, 100, 200])
    df = pd.concat([dfa, dfb], ignore_index=True)
    df.to_csv(OUT / "e1.csv", index=False)
    figure(df)
