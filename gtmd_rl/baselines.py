"""Classical PRB schedulers used as comparison baselines for GTMD-RL.

Every scheduler exposes the same interface: given the current :class:`NetworkState`
and the per-PRB capacity, it returns an integer PRB allocation vector that sums to
at most ``config.total_prbs``. This lets the comparison harness drive each policy on
an identical arrival/channel realisation and compare only the allocation logic.

Baselines implemented:

* ``RoundRobinScheduler``   -- channel-blind equal frequency sharing (3GPP RR).
* ``MaxCqiScheduler``       -- best-CQI / max-rate; throughput-greedy, starves cell edge.
* ``ProportionalFairScheduler`` -- PF metric = instantaneous_rate / avg_throughput.
* ``FloorFirstWrapper``     -- makes any baseline honour hard service floors first.
* ``GtmdRLScheduler``       -- our epoch-frozen RL weights + monotone weighted-greedy
                               DSIC allocator (the proposed policy).

All schedulers are deterministic given the environment seed, so a run is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import SimulationConfig
from .mechanism import weighted_greedy_allocator
from .network import NetworkState
from .rl import BayesianDemandEstimator, EpochFrozenQLearner


# --------------------------------------------------------------------------- #
# Common scheduler protocol
# --------------------------------------------------------------------------- #
class Scheduler:
    """Base class. ``allocate`` returns an int PRB vector summing to <= total_prbs."""

    name: str = "base"

    def reset(self) -> None:  # pragma: no cover - overridden when stateful
        pass

    def begin_epoch(self, reports: np.ndarray) -> None:  # pragma: no cover
        """Hook called once per epoch; only GTMD-RL uses it."""

    def allocate(
        self,
        state: NetworkState,
        prb_capacity_bits: np.ndarray,
        config: SimulationConfig,
    ) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


def _demand_prbs(state: NetworkState, config: SimulationConfig) -> np.ndarray:
    """Integer PRB demand a slice can actually use this slot (queue + intensity)."""
    return np.ceil(np.clip(state.demand_prbs, 0.0, config.total_prbs)).astype(int)


def _fill_by_priority(
    order: List[int],
    demand: np.ndarray,
    total_prbs: int,
    preassigned: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Greedily hand PRBs to slices in ``order`` up to their integer demand."""
    n = len(demand)
    alloc = np.zeros(n, dtype=int) if preassigned is None else preassigned.astype(int).copy()
    remaining = int(total_prbs) - int(alloc.sum())
    for i in order:
        if remaining <= 0:
            break
        need = max(0, int(demand[i]) - int(alloc[i]))
        give = min(need, remaining)
        alloc[i] += give
        remaining -= give
    return alloc


# --------------------------------------------------------------------------- #
# Round Robin
# --------------------------------------------------------------------------- #
class RoundRobinScheduler(Scheduler):
    """Equal frequency sharing across backlogged slices, channel-blind.

    Each active slice receives ``B / n_active`` PRBs (capped at its demand). Any
    residue is handed out in a rotating order so that no slice is favoured over time.
    """

    name = "RoundRobin"

    def __init__(self) -> None:
        self._rotation = 0

    def reset(self) -> None:
        self._rotation = 0

    def allocate(self, state, prb_capacity_bits, config) -> np.ndarray:
        demand = _demand_prbs(state, config)
        active = [i for i in range(config.n_slices) if demand[i] > 0]
        alloc = np.zeros(config.n_slices, dtype=int)
        if not active:
            return alloc
        share = config.total_prbs // len(active)
        for i in active:
            alloc[i] = min(int(demand[i]), share)
        # Distribute the remainder round-robin starting from the rotation pointer.
        order = active[self._rotation % len(active):] + active[: self._rotation % len(active)]
        alloc = _fill_by_priority(order, demand, config.total_prbs, preassigned=alloc)
        self._rotation += 1
        return alloc


# --------------------------------------------------------------------------- #
# Max-CQI / Best-Rate
# --------------------------------------------------------------------------- #
class MaxCqiScheduler(Scheduler):
    """Throughput-greedy: allocate to the slice with the best instantaneous rate.

    Maximises cell throughput but is unfair -- a persistently poor-channel slice
    (typically mMTC / cell edge) can be starved, which shows up as SLA violations.
    """

    name = "MaxCQI"

    def allocate(self, state, prb_capacity_bits, config) -> np.ndarray:
        demand = _demand_prbs(state, config)
        # Sort slices by per-PRB capacity (spectral efficiency), highest first.
        order = sorted(range(config.n_slices), key=lambda i: (-float(prb_capacity_bits[i]), i))
        return _fill_by_priority(order, demand, config.total_prbs)


# --------------------------------------------------------------------------- #
# Proportional Fair
# --------------------------------------------------------------------------- #
class ProportionalFairScheduler(Scheduler):
    """PF metric r_inst / R_avg, with R_avg an EWMA of delivered throughput.

    Allocates one PRB at a time to the slice with the current maximum PF metric,
    which trades cell throughput against long-run fairness -- the standard 5G MAC
    reference scheduler.
    """

    name = "ProportionalFair"

    def __init__(self, beta: float = 0.1, eps: float = 1e-3):
        self.beta = float(beta)
        self.eps = float(eps)
        self._avg_rate: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._avg_rate = None

    def allocate(self, state, prb_capacity_bits, config) -> np.ndarray:
        n = config.n_slices
        if self._avg_rate is None:
            self._avg_rate = np.full(n, self.eps, dtype=float)
        demand = _demand_prbs(state, config)
        inst_rate = np.asarray(prb_capacity_bits, dtype=float)  # bits per PRB this slot
        alloc = np.zeros(n, dtype=int)
        remaining = int(config.total_prbs)
        need = demand.copy()
        while remaining > 0 and np.any(need > 0):
            metric = np.where(need > 0, inst_rate / np.maximum(self._avg_rate, self.eps), -np.inf)
            i = int(np.argmax(metric))
            if not np.isfinite(metric[i]):
                break
            alloc[i] += 1
            need[i] -= 1
            remaining -= 1
        # Update the throughput EWMA from what was actually served.
        served_bits = alloc * inst_rate
        self._avg_rate = (1.0 - self.beta) * self._avg_rate + self.beta * served_bits
        return alloc


# --------------------------------------------------------------------------- #
# Floor-aware wrapper
# --------------------------------------------------------------------------- #
class FloorFirstWrapper(Scheduler):
    """Satisfy hard per-slice floors first, then run the wrapped baseline on the rest.

    This isolates the effect of the *floor constraint* from the effect of the
    *allocation policy*: with floors on, every baseline is guaranteed the minimum,
    so SLA differences reflect how the surplus is distributed.
    """

    def __init__(self, inner: Scheduler):
        self.inner = inner
        self.name = f"{inner.name}+Floors"

    def reset(self) -> None:
        self.inner.reset()

    def begin_epoch(self, reports: np.ndarray) -> None:
        self.inner.begin_epoch(reports)

    def allocate(self, state, prb_capacity_bits, config) -> np.ndarray:
        demand = _demand_prbs(state, config)
        floors = np.asarray(config.floor_prbs, dtype=int)
        alloc = np.minimum(floors, np.maximum(demand, floors))  # reserve the floor
        alloc = np.minimum(alloc, config.total_prbs)
        # Cap the total reserved floors at the budget (extreme scarcity guard).
        if alloc.sum() > config.total_prbs:
            order = sorted(range(config.n_slices), key=lambda i: -float(config.priorities[i]))
            alloc = _fill_by_priority(order, floors, config.total_prbs)
            return alloc
        # Remaining budget -> inner policy, but do not double-count the floor.
        remaining = int(config.total_prbs - alloc.sum())
        if remaining <= 0:
            return alloc
        surplus_state = state
        inner_alloc = self.inner.allocate(surplus_state, prb_capacity_bits, config)
        residual_demand = np.maximum(_demand_prbs(state, config) - alloc, 0)
        # Re-run the inner ordering but only over the residual budget/demand.
        order = sorted(
            range(config.n_slices),
            key=lambda i: (-int(inner_alloc[i]), i),
        )
        alloc = _fill_by_priority(order, alloc + residual_demand, config.total_prbs, preassigned=alloc)
        return alloc


# --------------------------------------------------------------------------- #
# GTMD-RL (our policy)
# --------------------------------------------------------------------------- #
class GtmdRLScheduler(Scheduler):
    """Epoch-frozen RL weights + Bayesian demand belief + monotone DSIC allocator.

    The RL controller picks a weight profile once per epoch (frozen for the epoch,
    which is what preserves within-epoch DSIC). The Bayesian estimator tracks the
    demand belief from reports. The monotone weighted-greedy allocator satisfies
    hard floors first and then distributes the surplus in weight order.
    """

    name = "GTMD-RL"

    def __init__(
        self,
        config: SimulationConfig,
        seed: int = 0,
        train: bool = True,
        enforce_floors: bool = True,
    ):
        self.config = config
        self.train = bool(train)
        self.enforce_floors = bool(enforce_floors)
        self.estimator = BayesianDemandEstimator(config)
        self.planner = EpochFrozenQLearner(config, seed=seed + 4242)
        self._weights = np.asarray(config.priorities, dtype=float)
        self._reports = np.array([s.base_mbps for s in config.slices], dtype=float)
        self._prev_delay = 0.0
        self._prev_rho = 0.0

    def reset(self) -> None:
        self.estimator = BayesianDemandEstimator(self.config)
        self._weights = np.asarray(self.config.priorities, dtype=float)
        self._prev_delay = 0.0
        self._prev_rho = 0.0

    def begin_epoch(self, reports: np.ndarray) -> None:
        self._reports = np.asarray(reports, dtype=float).copy()
        self.estimator.update(self._reports)
        _, weights, _ = self.planner.select_action(
            self.estimator,
            mean_delay_ms=self._prev_delay,
            rho_recent=self._prev_rho,
            train=self.train,
        )
        self._weights = weights

    def observe_epoch(self, mean_delay_ms: float, rho: float, reward: float) -> None:
        """Feed epoch outcome back into the RL controller (used during training)."""
        next_key = self.planner.discretize(self.estimator, mean_delay_ms, rho)
        if self.train:
            self.planner.update(next_key, reward)
        self._prev_delay = mean_delay_ms
        self._prev_rho = rho

    def allocate(self, state, prb_capacity_bits, config) -> np.ndarray:
        decision = weighted_greedy_allocator(
            state,
            self._reports,
            self._weights,
            config,
            enforce_floors=self.enforce_floors,
            serve_backlog=True,
        )
        return decision.allocation_prbs.astype(int)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_baseline_schedulers(with_floors: bool = True) -> Dict[str, Scheduler]:
    """Return the classical baselines; floor-aware if requested."""
    base: List[Scheduler] = [
        RoundRobinScheduler(),
        MaxCqiScheduler(),
        ProportionalFairScheduler(),
    ]
    if with_floors:
        return {s.name: s for s in base} | {
            FloorFirstWrapper(RoundRobinScheduler()).name: FloorFirstWrapper(RoundRobinScheduler()),
            FloorFirstWrapper(MaxCqiScheduler()).name: FloorFirstWrapper(MaxCqiScheduler()),
            FloorFirstWrapper(ProportionalFairScheduler()).name: FloorFirstWrapper(
                ProportionalFairScheduler()
            ),
        }
    return {s.name: s for s in base}
