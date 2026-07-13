#!/usr/bin/env python3
"""E5 (P1) -> paper Fig. 5: the (Regret, Slack) frontier against every baseline.

Each marker is one mechanism. Lower-left is better: no manipulation, no learning loss.
The claim is that the trust family traces the lower-left frontier and Pareto-dominates
the defenses the literature would reach for first:

  B2 DP-gradient  -- noise the learner's gradient but still feed it raw reports.
                     One-shot protection: slack barely falls until the noise is large
                     enough that regret has exploded.
  B4 Burn-in      -- train on raw reports, then commit. Dominated: the attacker simply
                     front-loads the lie into the burn-in window.
  B3 Fines        -- price the report/measurement discrepancy. This DOES work at
                     kappa >= C1, but needs operator commitment and taxes honest
                     tenants whose traffic fluctuates.
  B5 ORACLE / B6 static -- the skyline and the floor of the plot.

The r = 0 members sit exactly on the Slack = 0 axis at a small regret. Nothing else can.

    venv_linux/bin/python scripts/run_e5.py
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

from trustgate.adversary import slack_for_seed             # noqa: E402
from trustgate.baselines import compute_C1, e5_mechanisms  # noqa: E402
from trustgate.instances import instance_b                 # noqa: E402
from trustgate.metrics import WelfareOracle, ci95          # noqa: E402

OUT = ROOT / "outputs" / "trustgate"
OUT.mkdir(parents=True, exist_ok=True)


def load_f1(target=0.05) -> float:
    p = ROOT / "configs" / "loads.json"
    if p.exists():
        return json.loads(p.read_text())[f"{target:.2f}"]["f1"]
    return 1.5


def main(seeds, T=50000, L=50, sigma=0.2):
    f1 = load_f1(0.05)
    inst = instance_b(f1=f1)
    oracle = WelfareOracle(inst)
    C1 = compute_C1(inst)
    K = T // L
    grid = np.linspace(-0.2 * inst.theta[0], 0.2 * inst.theta[0], 21)
    print(f"  Instance B f1={f1:.3f}  C1={C1:.3f}  W*={oracle.W_star:.4f}", flush=True)

    rows = []
    for M in e5_mechanisms(C1):
        kw = dict(M.kwargs)
        frac = kw.pop("_freeze_frac", None)
        if frac is not None:
            kw["freeze_after"] = int(frac * K)
        for s in seeds:
            r = slack_for_seed(inst, M.gate, seed=5000 + s, T=T, L=L, sigma=sigma,
                               tenant=0, grid=grid, grid_size=9, **kw)
            rows.append(dict(mech=M.name, family=M.family, seed=s,
                             slack=r.slack, regret=oracle.regret(r.w_path, L)))
        sub = pd.DataFrame([x for x in rows if x["mech"] == M.name])
        print(f"  {M.name:22s} slack/T={sub.slack.mean()/T:.3e}  "
              f"regret/T={sub.regret.mean()/T:.3e}", flush=True)
    return pd.DataFrame(rows), T


def figure(df, T):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    g = df.groupby(["mech", "family"], as_index=False).agg(
        slack=("slack", "mean"), regret=("regret", "mean"))
    g["slack"] /= T
    g["regret"] /= T

    fam_style = {
        "trust":    dict(color="#1f77b4", marker="o", s=42, zorder=5, label="trust family (ours)"),
        "baseline": dict(color="0.45", marker="s", s=26, zorder=3, label="baselines B1-B4"),
        "anchor":   dict(color="k", marker="*", s=70, zorder=4, label="B5 oracle / B6 static"),
    }
    seen = set()
    for _, r in g.iterrows():
        st = dict(fam_style[r.family])
        lbl = st.pop("label")
        ax.scatter(r.regret, r.slack, edgecolor="white", linewidth=0.4,
                   label=None if r.family in seen else lbl, **st)
        seen.add(r.family)
        if r.family == "trust" or r.mech.startswith(("B1", "B5", "B6")):
            ax.annotate(r.mech.replace("$", "").replace("\\lambda", "lam").replace("\\kappa", "k"),
                        (r.regret, r.slack), fontsize=4.4, xytext=(3, 2),
                        textcoords="offset points")

    ax.axhline(0, color="#2ca02c", lw=1.0, ls=":")
    ax.text(0.99, 0.02, "Slack $=0$: exact truthfulness", transform=ax.transAxes,
            fontsize=5.4, ha="right", va="bottom", color="#2ca02c")
    ax.set_xlabel("regret / $T$  $\\rightarrow$ worse learning")
    ax.set_ylabel("slack / $T$  $\\rightarrow$ more manipulable")
    ax.set_title("(Regret, Slack) frontier", fontsize=8)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=5.6, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_frontier.{ext}", dpi=200)
    print(f"wrote {OUT/'fig5_frontier.pdf'}")


if __name__ == "__main__":
    seeds = list(range(int(os.environ.get("SEEDS", 4))))
    print("== E5: regret-slack frontier vs baselines ==", flush=True)
    df, T = main(seeds)
    df.to_csv(OUT / "e5.csv", index=False)
    figure(df, T)
