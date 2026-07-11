#!/usr/bin/env python3
"""Corrected frontier evaluation (CRN-paired slack, hindsight regret, fixed rho).

Replaces the old run_gtmd_experiments.py measurement, which (a) compared unpaired
runs, (b) multiplied the gain by rho (circular), (c) never measured regret, and
(d) used an estimator whose report influence did not scale as O(1/L).

Outputs to --output-dir:
  frontier_v2_allruns.csv   one row per (load, L, seed)
  frontier_v2_summary.csv   seed-averaged
  frontier_v2_panels.png    4-panel figure (slack, rho, regret, combined cost)
  frontier_slack_vs_L.png / rho_invariance_vs_L.png  paper Fig. 2 panels
  summary.md
"""
from __future__ import annotations

import argparse
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

from gtmd_rl.config import default_config
from gtmd_rl.frontier_eval import DEFAULT_MULTIPLIERS, run_frontier_v2
from gtmd_rl.plotting import plot_frontier_v2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loads", default="0.55,0.85,1.15")
    p.add_argument("--epoch-lengths", default="30,60,120,240")
    p.add_argument("--total-slots", type=int, default=2400)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/gtmd_frontier_v2")
    args = p.parse_args()

    loads = [float(x) for x in args.loads.split(",")]
    Ls = [int(x) for x in args.epoch_lengths.split(",")]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for s in range(args.seed_offset, args.seed_offset + args.seeds):
        df = run_frontier_v2(
            config=default_config(),
            loads=loads,
            epoch_lengths=Ls,
            total_slots=args.total_slots,
            seeds=(s,),
            multipliers=DEFAULT_MULTIPLIERS,
        )
        frames.append(df)
        # checkpoint after every seed
        allruns = pd.concat(frames, ignore_index=True)
        allruns.to_csv(outdir / "frontier_v2_allruns.csv", index=False)
        summary = allruns.groupby(["load", "L"], as_index=False).mean(numeric_only=True)
        summary.to_csv(outdir / "frontier_v2_summary.csv", index=False)
        plot_frontier_v2(allruns, outdir)
        print(f"=== seed {s} complete, checkpointed ===", flush=True)

    allruns = pd.concat(frames, ignore_index=True)
    agg = allruns.groupby(["load", "L"], as_index=False).agg(
        rho_hat=("rho_hat", "mean"),
        ic_slack=("ic_slack", "mean"),
        regret=("regret", "mean"),
        best_multiplier=("best_multiplier", "mean"),
    )
    lines = ["# Frontier v2 (CRN-paired) summary", "", agg.round(3).to_markdown(index=False), ""]
    # simple H1 diagnostic: correlation between slack and rho*T/L
    pred = agg["rho_hat"] * args.total_slots / agg["L"]
    mask = pred > 0
    if mask.sum() >= 3:
        corr = float(np.corrcoef(agg.loc[mask, "ic_slack"], pred[mask])[0, 1])
        lines.append(f"H1 diagnostic: corr(slack, rho*T/L) over binding cells = **{corr:.3f}**")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
