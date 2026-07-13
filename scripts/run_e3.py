#!/usr/bin/env python3
"""E3 (P0) -> paper Fig. 4: scaling and dissolution; per-epoch vs running-mean anchor.

This figure carries the paper's upgraded headline. Plot (Regret + Slack)/sqrt(T)
against T on log-log axes:

  * EPOCH-anchored mechanisms must trade L against sigma. The per-epoch measurement
    m_k has sd sigma/sqrt(L), so the estimation bias forces L to grow with T; the
    combined cost then scales as T^{3/4} and the normalized curve has slope ~ +1/4
    at sigma = 0.3.
  * CUM-anchored mechanisms anchor on the RUNNING mean, whose sd is sigma/sqrt(kL)
    and shrinks with the epoch index. Summing that over the horizon gives an
    O(sigma sqrt(T)) bias, so the combined cost is O((1+sigma) sqrt(T)) and the
    normalized curve is FLAT (slope ~ 0) at every sigma.

That flatness is the point: with a running-mean anchor the incentive cost of learning
is O((1+sigma) sqrt(T)) for EVERY constant sigma -- not only in the vanishing-noise
regime sigma <= T^{-1/2}. It also sidesteps the Omega(T^{2/3}) report-training barrier
entirely, because this learner is never fed a report.

    venv_linux/bin/python scripts/run_e3.py
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

from trustgate.adversary import slack_for_seed          # noqa: E402
from trustgate.gates import CUM0, CUM_R, EPOCH0, EPOCH_R  # noqa: E402
from trustgate.instances import instance_a              # noqa: E402
from trustgate.metrics import WelfareOracle, ci95       # noqa: E402

OUT = ROOT / "outputs" / "trustgate"
OUT.mkdir(parents=True, exist_ok=True)

INST = instance_a()
ORACLE = WelfareOracle(INST)
GRID = np.linspace(-0.1, 0.1, 41)
GQ = 11
C_L = 1.0          # calibrated so L(T=1e4, sigma=0.3) ~ 30 = c * 0.3 * 100


def epoch_L(T: int, sigma: float) -> int:
    """Per-T optimized epoch length for the EPOCH-anchored family."""
    return int(np.clip(round(C_L * sigma * np.sqrt(T)), 10, 500))


def run(seeds, sigmas=(0.05, 0.3), Ts=(1000, 3000, 10000, 30000, 100000)) -> pd.DataFrame:
    rows = []
    for sigma in sigmas:
        for T in Ts:
            for gate, anchor in [(EPOCH0, "EPOCH"), (EPOCH_R, "EPOCH"),
                                 (CUM0, "CUM"), (CUM_R, "CUM")]:
                # EPOCH family tunes L with T; CUM family does not need to.
                L = epoch_L(T, sigma) if anchor == "EPOCH" else 50
                for s in seeds:
                    r = slack_for_seed(INST, gate, seed=3000 + s, T=int(T), L=L,
                                       sigma=sigma, tenant=0, grid=GRID, grid_size=GQ)
                    reg = ORACLE.regret(r.w_path, L)
                    rows.append(dict(sigma=sigma, T=int(T), L=L, anchor=anchor,
                                     mech=gate.label, seed=s, slack=r.slack,
                                     regret=reg, combined=(reg + r.slack)))
                sub = [x for x in rows if x["T"] == T and x["mech"] == gate.label
                       and x["sigma"] == sigma]
                c = np.array([x["combined"] for x in sub]) / np.sqrt(T)
                m, hw = ci95(c)
                print(f"  sigma={sigma:.2f} T={int(T):>7d} L={L:>3d} {gate.label:12s} "
                      f"(Reg+Slk)/sqrt(T)={m:.4f} +/- {hw:.3f}", flush=True)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})

    style = {
        "EPOCH0":     dict(color="#1f77b4", ls=":", marker="D", label="EPOCH0 ($r{=}0$)"),
        "EPOCH(r=1)": dict(color="#1f77b4", ls="-", marker="s", label=r"EPOCH($r_L$)"),
        "CUM0":       dict(color="#2ca02c", ls=":", marker="*", label="CUM0 ($r{=}0$)"),
        "CUM(r=1)":   dict(color="#2ca02c", ls="-", marker="v", label=r"CUM($r_k$)"),
    }
    sigmas = sorted(df.sigma.unique())
    fig, axes = plt.subplots(1, len(sigmas), figsize=(7.16, 3.0), sharey=False)

    for ax, sigma in zip(np.atleast_1d(axes), sigmas):
        d = df[df.sigma == sigma]
        txt = []
        for mech, st in style.items():
            g = d[d.mech == mech]
            if g.empty:
                continue
            Ts = np.array(sorted(g["T"].unique()), float)
            y = np.array([g[g["T"] == t]["combined"].mean() / np.sqrt(t) for t in Ts])
            hw = np.array([ci95(g[g["T"] == t]["combined"].to_numpy() / np.sqrt(t))[1]
                           for t in Ts])
            ax.plot(Ts, y, lw=1.4, ms=3.5, **st)
            ax.fill_between(Ts, y - hw, y + hw, color=st["color"], alpha=0.15, lw=0)
            # least-squares slope in log-log
            pos = y > 0
            slope = np.polyfit(np.log(Ts[pos]), np.log(y[pos]), 1)[0] if pos.sum() >= 2 else np.nan
            txt.append(f"{st['label']}: slope {slope:+.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("horizon $T$ (slots)")
        ax.set_ylabel(r"$(\mathrm{Regret}+\mathrm{Slack})/\sqrt{T}$")
        ax.set_title(rf"$\sigma={sigma}$", fontsize=8)
        ax.grid(alpha=0.25, which="both")
        ax.text(0.02, 0.02, "\n".join(txt), transform=ax.transAxes, fontsize=5.4,
                va="bottom", ha="left",
                bbox=dict(fc="white", ec="0.7", alpha=0.85, lw=0.4))
        ax.legend(fontsize=5.8, loc="upper right")

    fig.suptitle(r"flat $\Rightarrow$ combined cost $O(\sqrt{T})$;  slope $+1/4$ "
                 r"$\Rightarrow$ the $T^{3/4}$ law", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_scaling.{ext}", dpi=200)
    print(f"wrote {OUT/'fig4_scaling.pdf'}")


if __name__ == "__main__":
    seeds = list(range(int(os.environ.get("SEEDS", 5))))
    print("== E3: scaling, per-epoch vs running-mean anchor ==", flush=True)
    df = run(seeds)
    df.to_csv(OUT / "e3.csv", index=False)
    figure(df)
