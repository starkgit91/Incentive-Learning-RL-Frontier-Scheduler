#!/usr/bin/env python3
"""Truthful vs. misreported robustness of the integrated DSIC+RL mechanism, against
classical schedulers and against our own allocator with the payment switched off.

Produces, in outputs/robustness/:
  robustness_dsic.png            -- U(m) dominant-strategy curve, gain vs load, gain/harm bars
  robustness_epochs.png          -- DSIC+RL over epochs, truthful vs misreport
  efficiency_vs_strategyproofness.png -- the synthesis quadrant
  robustness_summary.md, *.csv

    venv_linux/bin/python scripts/run_robustness.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mtp_droy_mpl_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gtmd_rl.config import default_config  # noqa: E402
from gtmd_rl.robustness import run_robustness, run_gain_vs_load  # noqa: E402
from gtmd_rl.plotting import (  # noqa: E402
    plot_robustness, plot_robustness_epochs, plot_responsiveness_strategyproof,
)


def efficiency_from_comparison(comp_csv: Path, config, load: float) -> dict:
    """Priority-weighted 'protected QoS' per scheduler at the nearest load in the
    deployed scheduler comparison: sum_i priority_i * (1 - SLA_i). Higher = better.
    GTMD-noPay shares GTMD-RL's deployed allocation, so it inherits its efficiency."""
    if not comp_csv.exists():
        return {}
    df = pd.read_csv(comp_csv)
    load_pick = df["load"].iloc[(df["load"] - load).abs().argsort().iloc[0]]
    df = df[df["load"] == load_pick]
    prio = np.asarray(config.priorities, dtype=float)
    names = [s.name for s in config.slices]
    eff = {}
    for _, row in df.iterrows():
        slas = np.array([row.get(f"sla_{nm}", np.nan) for nm in names], dtype=float)
        if np.any(np.isnan(slas)):
            continue
        eff[row["scheduler"]] = float(np.sum(prio * (1.0 - slas)))
    if "GTMD-RL" in eff:
        eff["GTMD-noPay"] = eff["GTMD-RL"]
    return eff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--load", type=float, default=1.1, help="headline (contended) load")
    p.add_argument("--epoch-length", type=int, default=60)
    p.add_argument("--total-slots", type=int, default=1800)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--gain-loads", default="0.7,0.85,1.0,1.15,1.3")
    p.add_argument("--payment-grid", type=int, default=13)
    p.add_argument("--comparison-csv", default="outputs/scheduler_comparison_v4/comparison_metrics.csv")
    p.add_argument("--output-dir", default="outputs/robustness")
    args = p.parse_args()

    cfg = default_config()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seeds))

    print(f"== robustness at load {args.load} ==", flush=True)
    result = run_robustness(cfg, load=args.load, epoch_length=args.epoch_length,
                            total_slots=args.total_slots, seeds=seeds,
                            payment_grid=args.payment_grid, verbose=True)

    print("== manipulation gain vs load ==", flush=True)
    gain_loads = [float(x) for x in args.gain_loads.split(",")]
    gl = run_gain_vs_load(cfg, loads=gain_loads, epoch_length=args.epoch_length,
                          total_slots=args.total_slots, seeds=seeds[:3],
                          payment_grid=args.payment_grid, verbose=False)

    p1 = plot_robustness(result, gl, outdir)
    p2 = plot_robustness_epochs(result, outdir)
    eff = efficiency_from_comparison(Path(args.comparison_csv), cfg, args.load)
    p3 = plot_responsiveness_strategyproof(result.summary, outdir, efficiency=eff)

    result.rows.to_csv(outdir / "robustness_rows.csv", index=False)
    result.summary.to_csv(outdir / "robustness_summary.csv", index=False)
    gl.to_csv(outdir / "gain_vs_load.csv", index=False)

    s = result.summary.set_index("scheduler")
    order = ["RoundRobin+Floors", "MaxCQI+Floors", "ProportionalFair+Floors", "GTMD-noPay", "GTMD-RL"]
    order = [n for n in order if n in s.index]
    lines = [
        "# Truthful vs. misreported robustness",
        "",
        f"Headline load {args.load}, {args.seeds} seeds, L={args.epoch_length}, "
        f"T={args.total_slots}. Strategic tenant = eMBB; best response over report "
        f"multipliers, common random numbers. GTMD-noPay is our allocator with the "
        f"Myerson payment switched off (isolates the payment as the truthfulness lever).",
        "",
        "| Scheduler | best-response $m^*$ | manipulation gain % | honest-slice Mbps lost |",
        "|---|---|---|---|",
    ]
    for n in order:
        lines.append(f"| {n} | {s.loc[n,'best_mult']:.2f} | {s.loc[n,'gain_pct']:.2f} | "
                     f"{-s.loc[n,'delta_honest_thr']:.2f} |")
    lines += [
        "",
        "Reading: GTMD-RL's best response is $m^*\\approx1$ (truthful) with ~0% gain and no "
        "harm to honest slices -- dominant-strategy truthfulness. The SAME allocator without "
        "the payment (GTMD-noPay) is gamed for a double-digit gain that starves the honest "
        "slices. Demand-blind RR/MaxCQI cannot be gamed but also cannot exploit demand "
        "(their efficiency loss shows in the deployed comparison).",
    ]
    if eff:
        lines += ["", "Deployed-path efficiency (priority-weighted protected QoS): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in sorted(eff.items(), key=lambda x: -x[1]))]
    (outdir / "robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {p1}\nwrote {p2}" + (f"\nwrote {p3}" if p3 else ""))
    print(result.summary[["scheduler", "best_mult", "gain_pct", "delta_honest_thr"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
