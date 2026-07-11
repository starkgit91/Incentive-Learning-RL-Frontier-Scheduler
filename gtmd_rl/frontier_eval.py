"""Common-random-numbers evaluation of the incentive--learning frontier.

This module measures exactly the three quantities the paper's theorems are about,
in the way the theory defines them:

* **Incentive slack** (Theorems 2-3): the *best-response* utility gain of a
  strategic tenant. For each persistent misreport multiplier ``m`` we rerun the
  entire episode on **identical arrival and channel realizations** (common random
  numbers -- the environment's RNG stream does not depend on the allocation, so
  seeding it identically reproduces the same traffic sample path) and with the
  planner seeded identically (same exploration schedule). The only difference
  between runs is the report, so the utility difference *is* the manipulation
  effect, not learning-trajectory noise. Slack = max(0, max_m U(m) - U(truthful)).

* **Learning regret** (Theorem 3): total reward of the best *fixed* weight
  profile in hindsight, evaluated on the same realizations, minus the learner's
  total reward. This is regret against the comparator class the epoch-frozen
  learner actually optimizes over.

* **Binding frequency** rho (Lemma 1): fraction of slots on which some floor is
  allocation-limited, with the corrected indicator (the floor, not the tenant's
  own demand, is the active constraint).

The old ``experiments.py`` sweep compared unpaired runs and then multiplied the
gain by rho -- which injects the theory's conclusion into the metric. Nothing
here is gated by rho: if slack tracks rho*T/L, that must now emerge from data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import SimulationConfig, default_config
from .mechanism import critical_value_payment, weighted_greedy_allocator
from .network import NRTraceGenerator, NetworkState
from .rl import BayesianDemandEstimator, EpochFrozenQLearner, EpochMetrics, network_reward

# Small deviations isolate the cross-epoch channel: within-epoch DSIC makes the
# within-epoch loss second-order in (m-1), while the weight-steering gain is
# first-order, so the best response is typically a mild persistent misreport.
DEFAULT_MULTIPLIERS: tuple = (0.9, 0.95, 1.05, 1.1, 1.2)


def reference_theta_base(config: SimulationConfig, load: float) -> np.ndarray:
    """Mean demand intensity (PRBs/slot) per slice at admission-control load
    ``load`` = E[sum_i theta_i] / B, with per-slice means proportional to the
    service floors. This is the standard offered-load definition: load < 1 is
    the underloaded regime, load >= 1 is overload. Means proportional to floors
    put every slice's type astride its own floor, so floor-binding is a genuine
    possibility for all tenants rather than only for the largest slice."""
    floors = np.asarray(config.floor_prbs, dtype=float)
    share = floors / max(float(floors.sum()), 1e-9)
    base = load * float(config.total_prbs) * share
    return np.clip(base, config.theta_min, config.theta_max)


def draw_epoch_type(
    config: SimulationConfig,
    base_theta: np.ndarray,
    seed: int,
    epoch: int,
    sigma: float = 0.35,
) -> np.ndarray:
    """Assumption 4 verbatim: the private type is drawn i.i.d. per epoch from an
    epoch-stationary law -- lognormal fluctuation around the load-scaled mean with
    slice-specific burst mixtures. The law does not depend on L, which is what
    makes the Lemma-1 invariance test meaningful, and the draw is a deterministic
    function of (seed, epoch), so it is identical across CRN-paired variants."""
    rng = np.random.default_rng(100003 * int(seed) + int(epoch))
    # Lognormal fluctuation around the load-scaled mean, plus mild slice-specific
    # bursts. (Deliberately NOT the large shifting hotspot used by the learning
    # demo: here the goal is a stationary law whose binding frequency is a clean
    # function of load -- the Lemma-1 invariance test -- not a state-dependent
    # decision to learn.)
    theta = base_theta * rng.lognormal(0.0, sigma, size=config.n_slices)
    for i, spec in enumerate(config.slices):
        if rng.random() < spec.burst_probability * 1.5:
            theta[i] *= float(rng.uniform(1.3, min(spec.burst_multiplier, 3.0)))
    return np.clip(theta, config.theta_min, config.theta_max)


@dataclass
class EpisodeOutcome:
    total_reward: float
    epoch_rewards: List[float]
    tenant_utility: float
    rho: float
    binding_slots: int
    total_slots: int


def run_policy_episode(
    config: SimulationConfig,
    load: float,
    epoch_length: int,
    total_slots: int,
    seed: int,
    planner_seed: int,
    fixed_action: Optional[int] = None,
    misreport_mult: float = 1.0,
    tenant: int = 1,
    compute_payments: bool = False,
    payment_grid: int = 17,
) -> EpisodeOutcome:
    """One full episode under CRN.

    ``fixed_action=None`` runs the training Q-learner (the mechanism under test);
    an integer runs that weight template as a fixed policy (hindsight comparator).
    ``misreport_mult`` scales the strategic tenant's report every epoch.
    """
    env = NRTraceGenerator(config, load=load, seed=seed)
    estimator = BayesianDemandEstimator(config)
    planner = EpochFrozenQLearner(config, seed=planner_seed)
    n_epochs = int(total_slots // epoch_length)

    total_reward = 0.0
    epoch_rewards: List[float] = []
    tenant_utility = 0.0
    binding_slots = 0
    prev_delay, prev_rho = 0.0, 0.0

    base_theta = reference_theta_base(config, load)

    for epoch in range(n_epochs):
        # Assumption 4: type drawn i.i.d. per epoch from an exogenous stationary
        # law; fixed for the L slots of the epoch (arrival noise still fluctuates
        # slot-by-slot around it). The env's own theta drift is overridden so the
        # binding frequency cannot become an artifact of report staleness.
        epoch_theta = draw_epoch_type(config, base_theta, seed, epoch)
        true_theta = epoch_theta
        reports = true_theta.copy()
        if misreport_mult != 1.0:
            reports[tenant] = float(
                np.clip(reports[tenant] * misreport_mult, config.theta_min, config.theta_max)
            )

        if fixed_action is None:
            _, weights, _ = planner.select_action(
                estimator, mean_delay_ms=prev_delay, rho_recent=prev_rho, train=True
            )
        else:
            weights = planner.weights_for_action(int(fixed_action), estimator)

        states: List[NetworkState] = []
        value_sum = 0.0
        rewards: List[float] = []
        lat_acc: List[np.ndarray] = []
        bind_any = 0
        obs_samples = np.zeros(config.n_slices, dtype=float)

        for _ in range(epoch_length):
            env.theta = epoch_theta.copy()  # pin the type; RNG streams untouched (CRN-safe)
            slot_theta = env.theta.copy()
            state = env.current_state()
            cap = np.maximum(env.prb_capacity_bits(), 1.0)
            decision = weighted_greedy_allocator(state, reports, weights, config)
            _, result = env.step(decision.allocation_prbs)

            reward = network_reward(
                config,
                result.throughput_mbps,
                result.latency_ms,
                result.sla_violation,
                result.wasted_prbs,
            )
            rewards.append(reward)
            lat_acc.append(result.latency_ms)
            if np.any(decision.binding > 0):
                bind_any += 1
            # Tenant's per-slot value at its TRUE type (quasi-linear utility).
            value_sum += float(slot_theta[tenant]) * float(decision.allocation_prbs[tenant])
            # Controller-side traffic observation: realized arrival intensity in
            # PRBs/slot -- measurable at the gNB, not falsifiable by the report.
            obs_samples += result.arrivals_bits / cap
            if compute_payments:
                states.append(state.copy())

        payment = 0.0
        if compute_payments:
            payment = critical_value_payment(
                tenant, reports, weights, states, config, grid_size=payment_grid
            )
        tenant_utility += value_sum - payment

        epoch_reward = float(np.sum(rewards))
        epoch_rewards.append(epoch_reward)
        total_reward += epoch_reward
        binding_slots += bind_any

        rho_epoch = bind_any / float(epoch_length)
        mean_delay = float(np.mean(lat_acc))
        # Belief update: report + L per-slot observations (Assumption 3 dilution).
        estimator.update(reports, observed_mean=obs_samples / epoch_length, n_obs=epoch_length)
        if fixed_action is None:
            planner.remember(
                EpochMetrics(epoch_reward, rho_epoch, mean_delay, 0.0, 0.0, 0.0)
            )
            next_key = planner.discretize(estimator, mean_delay, planner.recent_rho())
            planner.update(next_key, epoch_reward)
        prev_delay, prev_rho = mean_delay, rho_epoch

    return EpisodeOutcome(
        total_reward=total_reward,
        epoch_rewards=epoch_rewards,
        tenant_utility=tenant_utility,
        rho=binding_slots / float(max(n_epochs * epoch_length, 1)),
        binding_slots=binding_slots,
        total_slots=n_epochs * epoch_length,
    )


def evaluate_config(
    config: SimulationConfig,
    load: float,
    epoch_length: int,
    total_slots: int,
    seed: int,
    tenant: int = 1,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    payment_grid: int = 17,
) -> Dict[str, float]:
    """Slack, regret and rho for one (load, L, seed) cell, all under CRN."""
    planner_seed = seed + 9999

    truthful = run_policy_episode(
        config, load, epoch_length, total_slots, seed, planner_seed,
        fixed_action=None, misreport_mult=1.0, tenant=tenant,
        compute_payments=True, payment_grid=payment_grid,
    )

    # Hindsight regret vs the best fixed weight profile on the same realizations.
    n_actions = len(EpochFrozenQLearner(config).action_templates)
    fixed_totals = []
    for a in range(n_actions):
        out = run_policy_episode(
            config, load, epoch_length, total_slots, seed, planner_seed,
            fixed_action=a, misreport_mult=1.0, tenant=tenant, compute_payments=False,
        )
        fixed_totals.append(out.total_reward)
    best_fixed = float(np.max(fixed_totals))
    regret = max(0.0, best_fixed - truthful.total_reward)

    # Best-response slack over persistent misreport multipliers, CRN-paired.
    best_gain, best_mult, strategic_rho = 0.0, 1.0, truthful.rho
    for m in multipliers:
        out = run_policy_episode(
            config, load, epoch_length, total_slots, seed, planner_seed,
            fixed_action=None, misreport_mult=float(m), tenant=tenant,
            compute_payments=True, payment_grid=payment_grid,
        )
        gain = out.tenant_utility - truthful.tenant_utility
        if gain > best_gain:
            best_gain, best_mult, strategic_rho = gain, float(m), out.rho

    # Normalized slack: the manipulation gain as a fraction of the tenant's truthful
    # utility. Reporting slack in raw utility units makes a ~0.2% best-response gain
    # look like a large, jumpy number; as a percentage it reads as what it is -- a
    # near-zero residual that is exactly 0 when floors never bind (H3) and grows only
    # mildly with rho*T/L (H1).
    slack_pct = 100.0 * best_gain / max(abs(truthful.tenant_utility), 1e-9)

    return {
        "load": float(load),
        "L": int(epoch_length),
        "seed": int(seed),
        "rho_hat": truthful.rho,
        "ic_slack": best_gain,
        "ic_slack_pct": slack_pct,
        "best_multiplier": best_mult,
        "strategic_rho": strategic_rho,
        "regret": regret,
        "learned_reward": truthful.total_reward,
        "best_fixed_reward": best_fixed,
        "truthful_utility": truthful.tenant_utility,
        "theory_rho_T_over_L": truthful.rho * total_slots / float(epoch_length),
        "theory_sqrt_LT": float(np.sqrt(epoch_length * total_slots)),
    }


def run_frontier_v2(
    config: Optional[SimulationConfig] = None,
    loads: Sequence[float] = (0.55, 0.85, 1.15),
    epoch_lengths: Sequence[int] = (30, 60, 120, 240),
    total_slots: int = 2400,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5),
    tenant: int = 1,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    verbose: bool = True,
) -> pd.DataFrame:
    config = default_config() if config is None else config
    rows: List[dict] = []
    for load in loads:
        for L in epoch_lengths:
            for s in seeds:
                row = evaluate_config(
                    config, float(load), int(L), int(total_slots),
                    seed=42 + 1000 * int(s) + int(load * 100) + int(L),
                    tenant=tenant, multipliers=multipliers,
                )
                rows.append(row)
                if verbose:
                    print(
                        f"load={load:.2f} L={L:4d} seed={s}: rho={row['rho_hat']:.4f} "
                        f"slack={row['ic_slack']:9.1f} (m*={row['best_multiplier']:.2f}) "
                        f"regret={row['regret']:8.1f}",
                        flush=True,
                    )
    return pd.DataFrame(rows)
