#!/usr/bin/env python3
"""Quantify whether the RL action space has any leverage on reward, and whether
the reward-optimal action is state-dependent (the two things that must both hold
for there to be anything to learn)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtmd_rl.config import default_config
from gtmd_rl.network import NRTraceGenerator
from gtmd_rl.rl import BayesianDemandEstimator, EpochFrozenQLearner, network_reward
from gtmd_rl.mechanism import weighted_greedy_allocator
from gtmd_rl.frontier_eval import reference_theta_base, draw_epoch_type


def epoch_reward_for_action(cfg, load, seed, epoch, action_id, planner, est):
    """Total network reward of holding one fixed action over one epoch."""
    env = NRTraceGenerator(cfg, load=load, seed=seed)
    base = reference_theta_base(cfg, load)
    theta = draw_epoch_type(cfg, base, seed, epoch)
    weights = planner.weights_for_action(action_id, est)
    tot = 0.0
    for _ in range(60):
        env.theta = theta.copy()
        st = env.current_state()
        dec = weighted_greedy_allocator(st, theta, weights, cfg)
        _, res = env.step(dec.allocation_prbs)
        tot += network_reward(cfg, res.throughput_mbps, res.latency_ms,
                              res.sla_violation, res.wasted_prbs)
    return tot


def main():
    cfg = default_config()
    planner = EpochFrozenQLearner(cfg, seed=0)
    n_act = len(planner.action_templates)
    print(f"actions: {n_act}")
    for load in (0.9, 1.2, 1.6):
        print(f"\n===== load {load} =====")
        best_actions = []
        spreads = []
        for epoch in range(8):
            est = BayesianDemandEstimator(cfg)
            base = reference_theta_base(cfg, load)
            theta = draw_epoch_type(cfg, base, epoch * 7, epoch)
            est.update(theta, observed_mean=theta, n_obs=60)
            rewards = [epoch_reward_for_action(cfg, load, epoch * 7, epoch, a, planner, est)
                       for a in range(n_act)]
            rewards = np.array(rewards)
            best = int(np.argmax(rewards))
            spread = float(rewards.max() - rewards.min())
            rel = spread / (abs(rewards.mean()) + 1e-9)
            best_actions.append(best)
            spreads.append(rel)
            dom = int(np.argmax(theta))
            print(f"  epoch {epoch}: dom-slice={cfg.slices[dom].name:5s} "
                  f"best_action={best} reward_spread={spread:8.1f} ({100*rel:5.1f}% of mean)")
        print(f"  --> distinct best-actions across epochs: {sorted(set(best_actions))} "
              f"(if 1 value: nothing state-dependent to learn)")
        print(f"  --> mean relative reward spread: {100*np.mean(spreads):.1f}%")


if __name__ == "__main__":
    main()
