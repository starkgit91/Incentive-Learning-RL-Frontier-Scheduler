"""Demonstration that the epoch-frozen RL controller actually learns a policy.

Framing. Each epoch draws an i.i.d. demand type with a shifting hotspot (one slice
surges), the controller observes the demand belief, and picks one of the weight
strategies; the reward is a priority-weighted QoS score (fraction of each slice's
offered load served, minus an SLA-violation penalty). The reward-optimal strategy
depends on which slice is stressed that epoch, so a rising reward curve is genuine
learning of a state->action map, not drift.

This is a *controlled study of the learning mechanism*: hard floors are kept at a
realistic minimum-guarantee level (so the agent controls enough of the band for the
choice to matter), epochs are long enough to average out arrival noise, and demand
variance is moderate -- the regime in which the weight choice has a signal-to-noise
ratio the bandit can resolve. The incentive--learning frontier (Sec. IX-B) uses the
full, noisier network reward; here we isolate whether the controller learns at all.

Correct learner. Across epochs the type is i.i.d. (Assumption 4), so this is a
contextual bandit, not a sequential MDP: the learner uses gamma=0, a sample-average
value, and UCB exploration that tries every action before ranking. (The old
epsilon-greedy Q-learner with gamma=0.92 bootstrapped across unrelated epochs and
locked onto the first action it tried -- the reason the earlier curve was flat.)

Clean measurement. We build a held-out set of contexts and, by Monte-Carlo, the TRUE
mean reward of every action in every state. At training checkpoints we freeze the
greedy policy and score it against those true means: the normalized score (0 = a
random policy, 1 = the per-state oracle) and the optimal-action rate. Reporting the
greedy policy on true means removes the per-epoch difficulty noise that made the raw
reward look flat, so the learning is legible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

from .config import SimulationConfig, default_config
from .mechanism import weighted_greedy_allocator
from .network import NRTraceGenerator
from .rl import BayesianDemandEstimator, EpochFrozenQLearner


def demo_config() -> SimulationConfig:
    """Default scenario but with floors at a realistic minimum-guarantee level
    (32% of the band, vs 68% by default), so the discretionary surplus the RL
    controls is large enough for its weight choice to move the QoS reward."""
    base = default_config()
    floors = (5, 8, 3)
    return replace(base, slices=tuple(replace(s, floor_prbs=f) for s, f in zip(base.slices, floors)))


def qos_reward(config: SimulationConfig, result, state) -> float:
    """Priority-weighted fraction of offered load served, minus an SLA penalty.
    Smooth in the allocation (partial service still counts), so the split genuinely
    moves the reward -- unlike a capacity-saturated log-throughput term."""
    prio = np.asarray(config.priorities, dtype=float)
    sla_ms = np.asarray(config.sla_latency_ms, dtype=float)
    offered = state.queue_bits + result.arrivals_bits
    sat = np.clip(result.served_bits / np.maximum(offered, 1.0), 0.0, 1.0)
    viol = (result.latency_ms > sla_ms).astype(float)
    return float(np.sum(prio * (sat - 0.6 * viol)))


def _base_theta(cfg: SimulationConfig, load: float) -> np.ndarray:
    fl = np.asarray(cfg.floor_prbs, dtype=float)
    return np.clip(load * cfg.total_prbs * fl / fl.sum(), cfg.theta_min, cfg.theta_max)


def _draw_type(cfg, base, rng, sigma=0.10, smin=2.0, smax=2.6):
    theta = base * rng.lognormal(0.0, sigma, size=cfg.n_slices)
    hot = int(rng.integers(0, cfg.n_slices))
    theta[hot] *= float(rng.uniform(smin, smax))
    return np.clip(theta, cfg.theta_min, cfg.theta_max)


def _epoch_reward(cfg, env_seed, theta, weights, L) -> float:
    env = NRTraceGenerator(cfg, load=1.0, seed=env_seed)
    tot = 0.0
    for _ in range(L):
        env.theta = theta.copy()
        st = env.current_state()
        # Priced mechanism path (no static priority): the RL weight is the full
        # allocation lever, which is what makes the learning problem non-trivial.
        dec = weighted_greedy_allocator(st, theta, weights, cfg, serve_backlog=False)
        _, res = env.step(dec.allocation_prbs)
        tot += qos_reward(cfg, res, st)
    return tot / L


def _mc_state_action_means(cfg, base, L, n_mc=400, seed=77):
    """Monte-Carlo the true mean reward of every action in every visited state."""
    planner = EpochFrozenQLearner(cfg, seed=0)
    nA = len(planner.action_templates)
    acc: Dict[Tuple[int, int], np.ndarray] = {}
    cnt: Dict[Tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for it in range(n_mc):
        theta = _draw_type(cfg, base, rng)
        est = BayesianDemandEstimator(cfg)
        est.update(theta, observed_mean=theta, n_obs=L)
        state = planner.discretize(est, 0.0, 0.0)
        if state not in acc:
            acc[state] = np.zeros(nA)
            cnt[state] = np.zeros(nA)
        for a in range(nA):
            acc[state][a] += _epoch_reward(cfg, 991 * it + a, theta, planner.weights_for_action(a, est), L)
            cnt[state][a] += 1
    return {s: acc[s] / np.maximum(cnt[s], 1) for s in acc}, cnt


@dataclass
class DemoResult:
    checkpoints: np.ndarray
    norm_score_mean: np.ndarray
    norm_score_std: np.ndarray
    opt_rate_mean: np.ndarray
    opt_rate_std: np.ndarray
    n_actions: int
    load: float
    state_weight: Dict


def _score_policy(planner, state_means, state_freq, cfg, base, L) -> Tuple[float, float]:
    """Frequency-weighted normalized score and optimal-action rate of the greedy
    policy against the MC-true per-state action means."""
    est_probe = BayesianDemandEstimator(cfg)
    norm_num = w_sum = hit_num = 0.0
    for state, means in state_means.items():
        # Re-derive the greedy action for this state by probing the learner's Q.
        if state in planner.q and planner.counts[state].sum() > 0:
            aid = int(np.argmax(planner.q[state]))
        else:
            aid = int(np.argmax(planner.q.default_factory()))  # unseen -> action 0
        lo, hi = float(means.mean()), float(means.max())
        norm = (means[aid] - lo) / (hi - lo) if hi > lo else 1.0
        w = float(state_freq[state].sum())
        norm_num += w * norm
        hit_num += w * (1.0 if aid == int(means.argmax()) else 0.0)
        w_sum += w
    return norm_num / max(w_sum, 1e-9), hit_num / max(w_sum, 1e-9)


def run_learning_demo(
    config: SimulationConfig | None = None,
    load: float = 1.0,
    n_epochs: int = 1000,
    epoch_length: int = 100,
    seeds: int = 8,
    eval_every: int = 40,
) -> DemoResult:
    cfg = demo_config() if config is None else config
    base = _base_theta(cfg, load)
    state_means, state_freq = _mc_state_action_means(cfg, base, epoch_length)

    all_norm, all_opt, checkpoints = [], [], None
    for s in range(seeds):
        planner = EpochFrozenQLearner(cfg, seed=s + 3000)
        rng = np.random.default_rng(4242 + 17 * s)
        norm_curve, opt_curve, ck = [], [], []
        for epoch in range(n_epochs):
            theta = _draw_type(cfg, base, rng)
            est = BayesianDemandEstimator(cfg)
            est.update(theta, observed_mean=theta, n_obs=epoch_length)
            aid, _, _ = planner.select_action(est, 0.0, 0.0, train=True)
            r = _epoch_reward(cfg, 7919 * s + epoch, theta, planner.weights_for_action(aid, est), epoch_length)
            planner.update(planner.discretize(est, 0.0, 0.0), r)
            if epoch % eval_every == 0:
                ns, orate = _score_policy(planner, state_means, state_freq, cfg, base, epoch_length)
                norm_curve.append(ns)
                opt_curve.append(orate)
                ck.append(epoch)
        all_norm.append(norm_curve)
        all_opt.append(opt_curve)
        checkpoints = np.array(ck)

    norm = np.array(all_norm)
    opt = np.array(all_opt)
    return DemoResult(
        checkpoints=checkpoints,
        norm_score_mean=norm.mean(0), norm_score_std=norm.std(0),
        opt_rate_mean=opt.mean(0), opt_rate_std=opt.std(0),
        n_actions=len(EpochFrozenQLearner(cfg).action_templates),
        load=load, state_weight=state_freq,
    )
