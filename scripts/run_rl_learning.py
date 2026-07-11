#!/usr/bin/env python3
"""Demonstrate that the epoch-frozen RL controller learns its weight policy.

Produces outputs/rl_learning/rl_learning_curve.png and a summary. See
gtmd_rl/learning_demo.py for the setup and the (contextual-bandit) learner.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mtp_droy_mpl_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gtmd_rl.learning_demo import run_learning_demo
from gtmd_rl.plotting import plot_learning


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--epoch-length", type=int, default=100)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=40)
    p.add_argument("--load", type=float, default=1.0)
    p.add_argument("--output-dir", default="outputs/rl_learning")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    demo = run_learning_demo(load=args.load, n_epochs=args.epochs,
                             epoch_length=args.epoch_length, seeds=args.seeds,
                             eval_every=args.eval_every)
    path = plot_learning(demo, outdir)

    ns, op = demo.norm_score_mean, demo.opt_rate_mean
    lines = [
        "# RL learning demonstration",
        "",
        f"Contextual-bandit controller (UCB, gamma=0), {args.seeds} seeds, "
        f"{args.epochs} epochs, L={args.epoch_length}.",
        "",
        f"- Normalized QoS reward (0=random policy, 1=per-state oracle): "
        f"**{ns[0]:.2f} -> {ns[-1]:.2f}**.",
        f"- Optimal-action selection rate (random baseline {1/demo.n_actions:.2f}): "
        f"**{op[0]:.2f} -> {op[-1]:.2f}**.",
        "",
        "The rising curves show the controller learning the state->weight-profile map "
        "(route the discretionary surplus to the stressed slice). The ceiling is below "
        "1 because several demand contexts have near-tied best actions.",
    ]
    (outdir / "learning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"norm score {ns[0]:.2f} -> {ns[-1]:.2f} | opt-rate {op[0]:.2f} -> {op[-1]:.2f} "
          f"(random {1/demo.n_actions:.2f})")


if __name__ == "__main__":
    main()
