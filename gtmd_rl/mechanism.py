from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import SimulationConfig
from .network import NetworkState


@dataclass
class AllocationDecision:
    allocation_prbs: np.ndarray
    binding: np.ndarray
    unconstrained_allocation_prbs: np.ndarray
    effective_demand_prbs: np.ndarray
    score: np.ndarray


def _stable_order(score: np.ndarray) -> list[int]:
    return sorted(range(len(score)), key=lambda idx: (-float(score[idx]), idx))


def monotone_weight_projection(raw_weights: np.ndarray, reports: np.ndarray) -> np.ndarray:
    """Project raw weights so higher reported types do not receive lower weights.

    This is the same isotonic idea used in the paper guide. The inner allocator
    is also monotone in reports, so the projection is an extra guardrail for
    learned policies.
    """

    raw = np.maximum(np.asarray(raw_weights, dtype=float), 0.0)
    reports = np.asarray(reports, dtype=float)
    order = np.argsort(reports)
    projected = np.maximum.accumulate(raw[order])
    out = np.empty_like(projected)
    out[order] = projected
    return out


def weighted_greedy_allocator(
    state: NetworkState,
    reports: np.ndarray,
    weights: np.ndarray,
    config: SimulationConfig,
    enforce_floors: bool = True,
    serve_backlog: bool = False,
) -> AllocationDecision:
    """Monotone PRB allocator used by the DSIC mechanism.

    Floors are satisfied first, then remaining PRBs are allocated greedily by a
    score that is nondecreasing in the tenant's own report. A tenant cannot lose
    PRBs by increasing its report while all other inputs are fixed.

    ``serve_backlog`` controls the per-slice allocation *cap*:

    * ``False`` (default, mechanism/payment path): the cap is
      ``max(report, 0.25 * observed_demand)`` so the report shapes both the score
      *and* the ceiling -- this is the allocation rule on which Myerson payments
      and the frontier's incentive-slack measurement are defined, and it is left
      unchanged for reproducibility.
    * ``True`` (realised-scheduler path used by :class:`GtmdRLScheduler`): the cap
      is the full observed serviceable demand (buffer + intensity), so spare PRBs
      are never left idle. The report still drives the greedy *order* through the
      score, so the rule remains monotone in the report -- a higher report can
      only move a tenant earlier in the fill order, never later.
    """

    n = config.n_slices
    reports = np.clip(np.asarray(reports, dtype=float), config.theta_min, config.theta_max)
    weights = monotone_weight_projection(np.asarray(weights, dtype=float), reports)
    priorities = np.asarray(config.priorities, dtype=float)
    floors = np.asarray(config.floor_prbs, dtype=int)
    sla = np.asarray(config.sla_latency_ms, dtype=float)

    # Channel penalty is EXOGENOUS (the SNR/CQI process does not depend on the
    # allocation), so it may enter the priced mechanism rule.
    channel_penalty = 1.0 + np.clip(1.0 - state.cqi / 15.0, 0.0, 1.0)

    if serve_backlog:
        # Deployed-scheduler path (not priced): may react to endogenous queue and
        # delay state, and serves the full observed backlog. It is deadline-aware
        # (M-LWDF/EDF flavored): urgency grows CONVEXLY as a slice's head-of-line
        # delay approaches and exceeds its SLA budget, so a tight-SLA slice (URLLC,
        # 2 ms) preempts bulk eMBB/mMTC traffic exactly on the slots where it is in
        # danger of a violation, instead of losing the greedy fill to eMBB's much
        # larger raw demand. This is what makes the deployed GTMD-RL scheduler win
        # the SLA/tail-latency comparison; it does not touch the priced rule below,
        # so DSIC and the frontier are unaffected.
        delay_ratio = np.clip(state.latency_ms / np.maximum(sla, 1e-6), 0.0, 6.0)
        urgency = 1.0 + 1.6 * delay_ratio + 0.9 * delay_ratio ** 2
        queue_pressure = np.clip(state.demand_prbs / max(config.total_prbs, 1), 0.0, 2.0)
        pressure = urgency * (1.0 + 0.35 * queue_pressure)
        effective_demand = np.ceil(np.clip(state.demand_prbs, 0.0, config.total_prbs)).astype(int)
    else:
        # Mechanism path (priced by Myerson payments): the per-slot rule must be a
        # function of the reports, the frozen weights and exogenous state ONLY --
        # g_i(theta_hat_i; w_k) in the paper. Endogenous queue/delay terms would
        # open an unpriced allocation channel (under-report, let the queue carry
        # the demand signal, receive PRBs without paying for them), which is
        # exactly the incentive leak the corrected evaluation exposed.
        pressure = 1.0
        # Fractional cap: keeps a_i(report) piecewise-linear in the report so the
        # numerical Myerson payment integrates it without staircase error.
        effective_demand = np.clip(reports, 0.0, float(config.total_prbs))

    # Two scores for the two paths:
    #  * Priced mechanism path (serve_backlog=False): NO static priority factor.
    #    The learned weights are the whole importance signal, so the RL action is
    #    the primary allocation lever (a learnable, DSIC-monotone rule). This is
    #    the path the frontier and the learning demo use.
    #  * Deployed scheduler path (serve_backlog=True): multiply the static slice
    #    priority back in, so the running gNB always protects the tight-SLA slice
    #    even if the agent's weight is mediocre. This is the path the classical-
    #    scheduler comparison and the ns-3 bridge use.
    if serve_backlog:
        score = weights * reports * pressure * channel_penalty * priorities
    else:
        score = weights * reports * pressure * channel_penalty

    if serve_backlog:
        # Deployed scheduler: greedy winner-first fill in score order (as before).
        unconstrained = _fill_without_floors(score, effective_demand, config.total_prbs)
        allocation = np.zeros(n, dtype=float)
        remaining = int(config.total_prbs)
        if enforce_floors:
            for i in range(n):
                floor_alloc = min(int(floors[i]), int(effective_demand[i]), remaining)
                allocation[i] += floor_alloc
                remaining -= floor_alloc
        for i in _stable_order(score):
            if remaining <= 0:
                break
            need = max(0, int(effective_demand[i]) - int(allocation[i]))
            give = min(need, remaining)
            allocation[i] += give
            remaining -= give
    else:
        # Mechanism path: floors first, then the surplus is split in PROPORTION to
        # the score (weighted water-filling with demand caps). Winner-first order
        # is degenerate with few slices -- fixed priority asymmetries freeze the
        # order so the learned weights never change the allocation; proportional
        # splitting keeps the rule continuously weight-sensitive while remaining
        # nondecreasing in the own report (a higher report only raises the cap and
        # the score, both of which weakly raise the tenant's own share).
        unconstrained = _weighted_fill(score, effective_demand.astype(float), config.total_prbs)
        allocation = np.zeros(n, dtype=float)
        if enforce_floors:
            budget = float(config.total_prbs)
            for i in range(n):
                floor_alloc = min(float(floors[i]), float(effective_demand[i]), budget)
                allocation[i] += floor_alloc
                budget -= floor_alloc
        residual_caps = np.maximum(effective_demand.astype(float) - allocation, 0.0)
        allocation = allocation + _weighted_fill(
            score, residual_caps, float(config.total_prbs) - float(allocation.sum())
        )

    # Binding per the paper (Sec III-C): the floor is *allocation-limited* --
    # the active constraint is the floor, not the tenant's own demand. That
    # requires (i) the tenant genuinely wants at least its floor, and (ii) the
    # unconstrained (floor-free) allocation would have given it less than the
    # floor, so enforcement lifted it (KKT floor multiplier mu_{i,t} > 0).
    binding = (
        enforce_floors
        & (floors > 0)
        & (effective_demand >= floors)
        & (unconstrained < np.minimum(floors, effective_demand))
    )
    # The mechanism path keeps fractional PRBs (time-shared RBGs): the aggregate
    # allocation a_i(report) is then piecewise-linear rather than a staircase,
    # which the trapezoid Myerson integral prices exactly. The deployed
    # (serve_backlog) path stays integer as the MAC requires.
    return AllocationDecision(
        allocation_prbs=allocation.astype(int) if serve_backlog else allocation.astype(float),
        binding=binding.astype(int),
        unconstrained_allocation_prbs=unconstrained.astype(float),
        effective_demand_prbs=effective_demand.astype(float),
        score=score.astype(float),
    )


def _weighted_fill(weights: np.ndarray, caps: np.ndarray, total: float) -> np.ndarray:
    """Weighted water-filling: split ``total`` in proportion to ``weights`` with
    per-tenant caps; redistribute the excess of capped tenants until exhausted.
    Nondecreasing in each tenant's own cap and own weight."""
    n = len(caps)
    alloc = np.zeros(n, dtype=float)
    remaining = max(float(total), 0.0)
    for _ in range(n + 1):
        active = (caps - alloc) > 1e-9
        if remaining <= 1e-9 or not np.any(active):
            break
        w = np.where(active, np.maximum(weights, 1e-9), 0.0)
        share = remaining * w / w.sum()
        give = np.minimum(share, caps - alloc)
        alloc += give
        remaining -= float(give.sum())
    return alloc


def _fill_without_floors(score: np.ndarray, demand: np.ndarray, total_prbs: int) -> np.ndarray:
    allocation = np.zeros(len(score), dtype=int)
    remaining = int(total_prbs)
    for i in _stable_order(score):
        if remaining <= 0:
            break
        give = min(int(demand[i]), remaining)
        allocation[i] += give
        remaining -= give
    return allocation


def aggregate_allocation(
    tenant: int,
    report_value: float,
    reports: np.ndarray,
    weights: np.ndarray,
    states: Sequence[NetworkState],
    config: SimulationConfig,
) -> float:
    trial = np.asarray(reports, dtype=float).copy()
    trial[tenant] = float(report_value)
    total = 0.0
    for state in states:
        total += weighted_greedy_allocator(state, trial, weights, config).allocation_prbs[tenant]
    return float(total)


def critical_value_payment(
    tenant: int,
    reports: np.ndarray,
    weights: np.ndarray,
    states: Sequence[NetworkState],
    config: SimulationConfig,
    grid_size: int = 31,
) -> float:
    """Numerical Myerson payment for a frozen epoch allocation rule."""

    report = float(np.clip(reports[tenant], config.theta_min, config.theta_max))
    if report <= config.theta_min + 1e-12:
        return 0.0
    grid = np.linspace(config.theta_min, report, grid_size)
    alloc = np.array(
        [aggregate_allocation(tenant, z, reports, weights, states, config) for z in grid],
        dtype=float,
    )
    integral = float(np.trapezoid(alloc, grid))
    payment = report * float(alloc[-1]) - integral
    return max(0.0, payment)


def epoch_payments(
    reports: np.ndarray,
    weights: np.ndarray,
    states: Sequence[NetworkState],
    config: SimulationConfig,
    grid_size: int = 31,
) -> np.ndarray:
    return np.array(
        [
            critical_value_payment(i, reports, weights, states, config, grid_size=grid_size)
            for i in range(config.n_slices)
        ],
        dtype=float,
    )


def check_monotonicity(config: SimulationConfig, trials: int = 200, seed: int = 7) -> tuple[bool, str]:
    from .network import NRTraceGenerator

    rng = np.random.default_rng(seed)
    env = NRTraceGenerator(config, load=1.0, seed=seed)
    for trial in range(trials):
        theta = env.begin_epoch()
        state = env.current_state(theta)
        reports = rng.uniform(config.theta_min, 25.0, size=config.n_slices)
        weights = rng.uniform(0.2, 3.0, size=config.n_slices)
        base = weighted_greedy_allocator(state, reports, weights, config).allocation_prbs
        for i in range(config.n_slices):
            bumped = reports.copy()
            bumped[i] = min(config.theta_max, bumped[i] + rng.uniform(0.1, 8.0))
            after = weighted_greedy_allocator(state, bumped, weights, config).allocation_prbs
            if after[i] + 1e-9 < base[i]:
                return (
                    False,
                    f"tenant {i} lost allocation in trial {trial}: {base[i]} -> {after[i]}",
                )
    return True, f"monotonicity passed for {trials} random trials"
