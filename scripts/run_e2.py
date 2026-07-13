#!/usr/bin/env python3
"""E2 (P1) -> paper Fig. 3: the price of invariance, and no tax on honesty.

All tenants are TRUTHFUL here -- this experiment is about what the gate costs when
nobody is cheating. Two things must be true for the mechanism to be worth using:

1. NO TAX ON HONESTY. A gate with a real trust radius (EPOCH(r_L)) must not clip an
   honest report, so its regret must be statistically indistinguishable from the
   ORACLE that trains on the true (private) types. If it is not, the radius is too
   tight and the gate is punishing honest tenants whose traffic merely fluctuated.
2. THE PRICE IS BOUNDED. The r = 0 gates refuse the report entirely and learn from a
   noisy measurement, so they must pay something. Theorem 4 says the per-epoch anchor
   pays O(sigma T / sqrt(L)) -- a gap growing LINEARLY in sigma. Theorem 4' says the
   running-mean anchor pays only O(sigma sqrt(T)), so its gap should stay small even
   at large sigma. That gap is the entire price of exact truthfulness.

    venv_linux/bin/python scripts/run_e2.py
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

from trustgate.gates import CUM0, EPOCH0, EPOCH_R, ORACLE     # noqa: E402
from trustgate.instances import draw_channel, instance_b      # noqa: E402
from trustgate.mechanism import run, truthful                 # noqa: E402
from trustgate.metrics import WelfareOracle, ci95             # noqa: E402
from trustgate.traffic import draw_traffic                    # noqa: E402

OUT = ROOT / "outputs" / "trustgate"
OUT.mkdir(parents=True, exist_ok=True)

SIGMAS = [0.05, 0.1, 0.2, 0.4, 0.8]
MECHS = [ORACLE, EPOCH_R, EPOCH0, CUM0]


def load_f1(target=0.05) -> float:
    p = ROOT / "configs" / "loads.json"
    if p.exists():
        return json.loads(p.read_text())[f"{target:.2f}"]["f1"]
    return 1.5


def main(seeds, T=50000, L=50):
    f1 = load_f1(0.05)
    inst = instance_b(f1=f1)
    oracle = WelfareOracle(inst)
    print(f"  Instance B, f1={f1:.3f} (rho~0.05), W*={oracle.W_star:.5f}", flush=True)

    rows = []
    for sigma in SIGMAS:
        for gate in MECHS:
            for s in seeds:
                rng = np.random.default_rng(4000 + s)
                A = draw_traffic(inst, sigma, rng, T)
                q = draw_channel(inst, rng, T)
                r = run(inst, gate, A, q, L, sigma, truthful(inst), compute_payments=False)
                rows.append(dict(sigma=sigma, mech=gate.label, seed=s,
                                 regret=oracle.regret(r.w_path, L), rho=r.rho))
            sub = np.array([x["regret"] for x in rows
                            if x["sigma"] == sigma and x["mech"] == gate.label])
            m, hw = ci95(sub / T)
            print(f"  sigma={sigma:.2f} {gate.label:12s} regret/T={m:.3e} +/- {hw:.1e}",
                  flush=True)
    return pd.DataFrame(rows), T


def figure(df, T):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    style = {
        "ORACLE":     dict(color="k", ls="-", marker="o", lw=1.0, label="ORACLE (true types)"),
        "EPOCH(r=1)": dict(color="#1f77b4", ls="-", marker="s", label=r"EPOCH($r_L$)"),
        "EPOCH0":     dict(color="#1f77b4", ls=":", marker="D", label="EPOCH0 ($r{=}0$)"),
        "CUM0":       dict(color="#2ca02c", ls=":", marker="*", label="CUM0 ($r{=}0$)"),
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.9))
    sig = np.array(SIGMAS, float)

    for mech, st in style.items():
        g = df[df.mech == mech]
        y = np.array([g[g.sigma == s]["regret"].mean() / T for s in SIGMAS])
        hw = np.array([ci95(g[g.sigma == s]["regret"].to_numpy() / T)[1] for s in SIGMAS])
        a1.plot(sig, y, ms=3.5, **st)
        a1.fill_between(sig, y - hw, y + hw, color=st["color"], alpha=0.15, lw=0)

    a1.set_xlabel(r"traffic noise $\sigma_{\rm rel}$")
    a1.set_ylabel("regret / $T$ (per slot)")
    a1.set_title("(a) price of invariance", fontsize=8)
    a1.grid(alpha=0.25); a1.legend(fontsize=6)

    # gap to oracle -- the actual price
    orc = np.array([df[(df.mech == "ORACLE") & (df.sigma == s)]["regret"].mean()
                    for s in SIGMAS])
    for mech, st in style.items():
        if mech == "ORACLE":
            continue
        g = df[df.mech == mech]
        gap = np.array([g[g.sigma == s]["regret"].mean() for s in SIGMAS]) - orc
        a2.plot(sig, gap / T, ms=3.5, **st)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xlabel(r"traffic noise $\sigma_{\rm rel}$")
    a2.set_ylabel("(regret $-$ oracle) / $T$")
    a2.set_title("(b) no tax on honesty; $r{=}0$ gap is the price", fontsize=8)
    a2.grid(alpha=0.25); a2.legend(fontsize=6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_invariance_price.{ext}", dpi=200)
    print(f"wrote {OUT/'fig3_invariance_price.pdf'}")


if __name__ == "__main__":
    seeds = list(range(int(os.environ.get("SEEDS", 6))))
    print("== E2: price of invariance / no tax on honesty ==", flush=True)
    df, T = main(seeds)
    df.to_csv(OUT / "e2.csv", index=False)
    figure(df, T)
