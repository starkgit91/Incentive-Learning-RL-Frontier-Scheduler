#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mtp_droy_mpl_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gtmd_rl.config import default_config
from gtmd_rl.experiments import run_frontier_sweep
from gtmd_rl.mechanism import check_monotonicity
from gtmd_rl.plotting import plot_epoch_learning, plot_frontier


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_summary(output_dir: Path, sweep: pd.DataFrame, figures: list[Path]) -> Path:
    best = sweep.sort_values("ic_slack").head(1).iloc[0]
    max_raw = sweep.sort_values("raw_strategic_gain", ascending=False).head(1).iloc[0]
    rho_spread = (
        sweep.groupby("load")["rho_hat"].agg(["min", "max"]).assign(spread=lambda d: d["max"] - d["min"])
    )
    summary = output_dir / "summary.md"
    lines = [
        "# DSIC-RL Frontier Experiment Summary",
        "",
        "This run implements the local simulation counterpart of the INFOCOM draft:",
        "DSIC epoch-frozen reports feed a Bayesian demand estimator, which feeds an",
        "epoch-frozen RL controller for PRB weights. A Q-learning tenant searches for",
        "profitable cross-epoch report multipliers.",
        "",
        "## Key outputs",
        "",
        f"- Best measured slack: `{best['ic_slack']:.4f}` at load `{best['load']:.2f}`, L `{int(best['L'])}`.",
        f"- Largest raw strategic gain before floor-localization: `{max_raw['raw_strategic_gain']:.4f}` at load `{max_raw['load']:.2f}`, L `{int(max_raw['L'])}`.",
        f"- Mean rho over all runs: `{sweep['rho_hat'].mean():.4f}`.",
        f"- Mean SLA violation rate: `{sweep['sla_violation_rate'].mean():.4f}`.",
        f"- Mean throughput: `{sweep['throughput_mbps'].mean():.4f}` Mbps.",
        "",
        "## Rho invariance by load",
        "",
        rho_spread.to_markdown(),
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig}`")
    lines.extend(
        [
            "",
            "## CSV files",
            "",
            "- `sweep_results.csv`: one row per `(load, L)` pair.",
            "- `epoch_traces.csv`: per-epoch truthful and strategic traces.",
            "- `network_trace_sample.csv`: per-slot sample with PRB, throughput, latency, CQI, SNR, BER, and binding indicators.",
        ]
    )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSIC-RL PRB allocation frontier experiments.")
    parser.add_argument("--loads", default="0.55,0.85,1.15", help="Comma-separated offered-load factors.")
    parser.add_argument("--epoch-lengths", default="30,60,120,240", help="Comma-separated epoch lengths.")
    parser.add_argument("--total-slots", type=int, default=2400, help="Slots per scenario.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adversary-train-episodes", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = default_config()
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, message = check_monotonicity(config)
    if not ok:
        raise RuntimeError(message)
    print(message)

    sweep, epochs, slots = run_frontier_sweep(
        config=config,
        loads=parse_csv_floats(args.loads),
        epoch_lengths=parse_csv_ints(args.epoch_lengths),
        total_slots=args.total_slots,
        seed=args.seed,
        adversary_train_episodes=args.adversary_train_episodes,
        collect_sample_slots=True,
    )

    sweep_path = output_dir / "sweep_results.csv"
    epoch_path = output_dir / "epoch_traces.csv"
    slots_path = output_dir / "network_trace_sample.csv"
    sweep.to_csv(sweep_path, index=False)
    epochs.to_csv(epoch_path, index=False)
    slots.to_csv(slots_path, index=False)

    figures = plot_frontier(sweep, output_dir)
    learning = plot_epoch_learning(epochs, output_dir)
    if learning is not None:
        figures.append(learning)
    summary = write_summary(output_dir, sweep, figures)

    print(f"wrote {sweep_path}")
    print(f"wrote {epoch_path}")
    print(f"wrote {slots_path}")
    print(f"wrote {summary}")
    for fig in figures:
        print(f"wrote {fig}")


if __name__ == "__main__":
    main()
