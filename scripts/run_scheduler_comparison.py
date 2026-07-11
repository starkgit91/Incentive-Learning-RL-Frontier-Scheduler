#!/usr/bin/env python3
"""Compare GTMD-RL against Round-Robin, Max-CQI and Proportional-Fair schedulers.

Outputs (default ``outputs/scheduler_comparison/``):
  - comparison_metrics.csv     one row per (scheduler, load)
  - comparison_slots.csv       per-slot sample trace at the reference load
  - scheduler_comparison_bars.png
  - throughput_fairness_tradeoff.png
  - comparison_summary.md
"""
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

from gtmd_rl.comparison import run_comparison
from gtmd_rl.config import default_config
from gtmd_rl.plotting import plot_scheduler_comparison


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_summary(output_dir: Path, metrics: pd.DataFrame) -> Path:
    cols = [
        "scheduler",
        "load",
        "sum_throughput_mbps",
        "p95_latency_ms",
        "sla_violation_rate",
        "jain_fairness",
        "floor_satisfaction",
        "wasted_prbs_per_slot",
    ]
    lines = [
        "# Scheduler Comparison Summary",
        "",
        "GTMD-RL (epoch-frozen RL weights + monotone DSIC allocator with hard floors)",
        "versus the classical 5G MAC schedulers, all driven on identical arrival and",
        "channel realisations. Baselines are shown with hard floors enforced so the",
        "SLA comparison is fair.",
        "",
        "## Full metric table",
        "",
        metrics[cols].round(4).to_markdown(index=False),
        "",
        "## Headline (highest load)",
        "",
    ]
    top = metrics[metrics["load"] == metrics["load"].max()].set_index("scheduler")
    if "GTMD-RL" in top.index:
        g = top.loc["GTMD-RL"]
        best_thr = top["sum_throughput_mbps"].idxmax()
        best_sla = top["sla_violation_rate"].idxmin()
        lines += [
            f"- GTMD-RL SLA violation rate: `{g['sla_violation_rate']:.4f}` "
            f"(best baseline: `{best_sla}` at `{top.loc[best_sla, 'sla_violation_rate']:.4f}`).",
            f"- GTMD-RL throughput: `{g['sum_throughput_mbps']:.3f}` Mbps "
            f"(max-throughput policy: `{best_thr}` at `{top.loc[best_thr, 'sum_throughput_mbps']:.3f}`).",
            f"- GTMD-RL Jain fairness: `{g['jain_fairness']:.4f}`; floor satisfaction: `{g['floor_satisfaction']:.4f}`.",
        ]
    path = output_dir / "comparison_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="GTMD-RL vs classical scheduler comparison.")
    parser.add_argument("--loads", default="0.7,1.0,1.3")
    parser.add_argument("--epoch-length", type=int, default=60)
    parser.add_argument("--total-slots", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/scheduler_comparison")
    args = parser.parse_args()

    config = default_config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loads = parse_csv_floats(args.loads)

    metrics, slots = run_comparison(
        config=config,
        loads=tuple(loads),
        epoch_length=args.epoch_length,
        total_slots=args.total_slots,
        seed=args.seed,
        collect_slots_for_load=loads[len(loads) // 2],
    )

    metrics_path = output_dir / "comparison_metrics.csv"
    slots_path = output_dir / "comparison_slots.csv"
    metrics.to_csv(metrics_path, index=False)
    slots.to_csv(slots_path, index=False)
    figures = plot_scheduler_comparison(metrics, output_dir)
    summary = write_summary(output_dir, metrics)

    print(f"wrote {metrics_path}")
    print(f"wrote {slots_path}")
    for fig in figures:
        print(f"wrote {fig}")
    print(f"wrote {summary}")
    print()
    print(metrics[["scheduler", "load", "sum_throughput_mbps", "p95_latency_ms",
                   "sla_violation_rate", "jain_fairness"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
