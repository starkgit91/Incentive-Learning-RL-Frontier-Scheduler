#!/usr/bin/env python3
"""Multi-seed averaged frontier sweep for clean H1/H2 figures.

Averaging over independent seeds smooths the binding-frequency and slack estimates
(the single-seed curves are noisy because the RL/adversary trajectories vary), which
sharpens the visual signature of Lemma 1 (rho flat in L) and Theorem 2 (slack null
where rho -> 0). Outputs the same figure/CSV set as run_gtmd_experiments.py.
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
from gtmd_rl.experiments import run_frontier_sweep
from gtmd_rl.plotting import plot_frontier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loads", default="0.55,0.85,1.15")
    p.add_argument("--epoch-lengths", default="30,60,120,240")
    p.add_argument("--total-slots", type=int, default=2400)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--adversary-train-episodes", type=int, default=3)
    p.add_argument("--output-dir", default="outputs/gtmd_frontier_avg")
    args = p.parse_args()

    loads = [float(x) for x in args.loads.split(",")]
    Ls = [int(x) for x in args.epoch_lengths.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()

    frames = []
    for s in range(args.seeds):
        sweep, _, _ = run_frontier_sweep(
            config=config,
            loads=tuple(loads),
            epoch_lengths=tuple(Ls),
            total_slots=args.total_slots,
            seed=42 + 1000 * s,
            adversary_train_episodes=args.adversary_train_episodes,
            collect_sample_slots=False,
        )
        sweep["seed"] = s
        frames.append(sweep)
        print(f"seed {s} done", flush=True)

        # Checkpoint after every seed so a long run never loses progress and the
        # figure can be regenerated from however many seeds have completed.
        allrows = pd.concat(frames, ignore_index=True)
        allrows.to_csv(output_dir / "frontier_allseeds.csv", index=False)
        num = allrows.select_dtypes("number").columns
        avg = allrows.groupby(["load", "L"], as_index=False)[list(num)].mean()
        avg = avg.drop(columns=[c for c in ["seed"] if c in avg.columns])
        avg.to_csv(output_dir / "sweep_results.csv", index=False)
        plot_frontier(avg, output_dir)

    figs = plot_frontier(avg, output_dir)
    print("averaged over", args.seeds, "seeds")
    cols = ["load", "L", "rho_hat", "ic_slack", "theory_rho_T_over_L"]
    print(avg[cols].round(4).to_string(index=False))
    for f in figs:
        print("wrote", f)


if __name__ == "__main__":
    main()
