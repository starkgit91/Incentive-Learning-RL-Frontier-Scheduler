#!/usr/bin/env python3
"""Average the incentive-learning frontier sweep over several seeds.

The per-seed tabular adversary/planner estimates are noisy; averaging over seeds
gives smoother slack-vs-L and rho-vs-L curves for the paper figures while keeping
every run reproducible. Outputs to outputs/gtmd_frontier_avg/.
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

from gtmd_rl.config import default_config
from gtmd_rl.experiments import run_frontier_sweep
from gtmd_rl.plotting import plot_frontier


def main() -> None:
    seeds = [11, 23, 42, 71, 101]
    loads = (0.55, 0.85, 1.15)
    epoch_lengths = (30, 60, 120, 240)
    total_slots = 2400
    config = default_config()
    out = Path("outputs/gtmd_frontier_avg")
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for sd in seeds:
        sweep, _, _ = run_frontier_sweep(
            config=config, loads=loads, epoch_lengths=epoch_lengths,
            total_slots=total_slots, seed=sd, adversary_train_episodes=3,
            collect_sample_slots=False,
        )
        sweep["seed"] = sd
        frames.append(sweep)
        print(f"seed {sd} done")

    allruns = pd.concat(frames, ignore_index=True)
    agg = (
        allruns.groupby(["load", "L"], as_index=False)
        .agg({
            "rho_hat": "mean", "strategic_rho_hat": "mean", "ic_slack": "mean",
            "raw_strategic_gain": "mean", "theory_rho_T_over_L": "mean",
            "sla_violation_rate": "mean", "throughput_mbps": "mean", "mean_delay_ms": "mean",
        })
    )
    agg["ic_slack_std"] = (
        allruns.groupby(["load", "L"])["ic_slack"].std().reset_index(drop=True)
    )
    allruns.to_csv(out / "frontier_allseeds.csv", index=False)
    agg.to_csv(out / "sweep_results.csv", index=False)
    figs = plot_frontier(agg, out)
    print("wrote", out / "sweep_results.csv")
    for f in figs:
        print("wrote", f)
    print("\n=== averaged frontier ===")
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
