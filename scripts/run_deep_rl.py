#!/usr/bin/env python3
"""Train and compare deep-RL controllers (DQN, PPO, A2C) against the tabular
contextual bandit on the epoch-frozen DSIC weight policy.

All learners share the same fine simplex action set and the same priced-mechanism
reward; each is scored against a Monte-Carlo oracle of the true per-context action
values. Produces outputs/deep_rl/deep_rl_comparison.png and a summary.

    venv_linux/bin/python scripts/run_deep_rl.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mtp_droy_mpl_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gtmd_rl.deep_learning_demo import run_deep_demo  # noqa: E402
from gtmd_rl.plotting import plot_deep_learning  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--epoch-length", type=int, default=80)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--eval-ctx", type=int, default=20)
    p.add_argument("--eval-mc", type=int, default=20)
    p.add_argument("--degree", type=int, default=6, help="simplex-lattice degree (#actions grows with it)")
    p.add_argument("--load", type=float, default=1.0)
    p.add_argument("--agents", nargs="+",
                   default=["Tabular bandit", "DQN", "PPO", "A2C"])
    p.add_argument("--output-dir", default="outputs/deep_rl")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    result = run_deep_demo(
        agents=tuple(args.agents), load=args.load, n_epochs=args.epochs,
        epoch_length=args.epoch_length, seeds=args.seeds, eval_every=args.eval_every,
        degree=args.degree, eval_ctx=args.eval_ctx, eval_mc=args.eval_mc, verbose=True,
    )
    path = plot_deep_learning(result, outdir)

    rand = 1.0 / result.n_actions
    lines = [
        "# Deep-RL controllers vs the tabular contextual bandit",
        "",
        f"Same {result.n_actions}-action simplex weight set and priced-mechanism reward "
        f"for every learner; {args.seeds} seeds, {args.epochs} epochs, L={args.epoch_length}. "
        f"Each learner is scored by its greedy policy against a Monte-Carlo oracle of the "
        f"true per-context action values (0 = random action, 1 = per-context oracle).",
        "",
        "| Controller | class | final normalized reward | final optimal-action rate |",
        "|---|---|---|---|",
    ]
    kind = {"Tabular bandit": "tabular bandit (gamma=0)", "DQN": "deep value (Double-DQN)",
            "PPO": "deep policy grad (clipped)", "A2C": "deep actor-critic"}
    for name in result.agents:
        nf, of = result.final_table[name]
        lines.append(f"| {name} | {kind.get(name,'')} | {nf:.2f} | {of:.2f} (random {rand:.2f}) |")
    lines += [
        "",
        "The deep controllers see the continuous demand belief (contention level, "
        "per-slice demand-to-floor stress, per-slice share) instead of the tabular "
        "learner's coarse (load-bin, stressed-slice) cell, and choose among the same "
        f"{result.n_actions} simplex weight profiles. Each epoch is a single-step episode "
        "(the demand type is i.i.d. across epochs, Assumption 4), so the DQN target reduces "
        "to E[reward|s,a] and the PPO/A2C advantage to reward - V(s): the deep agents are "
        "faithful contextual bandits, not misapplied sequential-MDP learners.",
    ]
    (outdir / "deep_rl_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    for name in result.agents:
        nf, of = result.final_table[name]
        print(f"  {name:15s} final norm={nf:.2f} opt={of:.2f}")


if __name__ == "__main__":
    main()
