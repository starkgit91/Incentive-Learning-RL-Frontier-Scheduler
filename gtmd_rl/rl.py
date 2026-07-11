from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np

from .config import SimulationConfig


def simplex_action_templates(
    n_slices: int, degree: int = 6, w_min: float = 0.3, w_max: float = 9.0
) -> np.ndarray:
    """Simplex-lattice weight profiles: every composition of ``degree`` units over
    the ``n_slices`` slices, mapped affinely from the unit simplex to weights in
    ``[w_min, w_max]``. Yields ``C(degree + n_slices - 1, n_slices - 1)`` actions
    (28 for the default 3 slices, degree 6), a fine, uniform sweep of the weight
    simplex -- the "more actions" the deep controllers exploit.

    The lattice includes the corners (favor one slice), all edges (favor a pair),
    and the balanced centre, so the coarse hand-picked 8-action set is (up to the
    grid resolution) a subset. Using the *same* generated set for every learner
    keeps the tabular/DQN/PPO comparison apples-to-apples.
    """

    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in compositions(total - i, parts - 1):
                yield (i,) + rest

    grid = np.array(list(compositions(degree, n_slices)), dtype=float) / float(degree)
    return w_min + grid * (w_max - w_min)


def templates_to_weights(templates: np.ndarray, action_id: int, mean: np.ndarray) -> np.ndarray:
    """Map an action id to a per-slice weight vector, lightly modulated by the
    demand belief. Shared by EVERY learner (tabular, DQN, PPO) and the MC-true
    evaluator, so a given (action, context) pair always yields the same allocation
    weights -- the only thing that differs between learners is *which* action they
    pick. Keeping this single-sourced is what makes the reward of action ``a`` in
    context ``theta`` identical across agents (a fair comparison)."""
    mean = np.asarray(mean, dtype=float)
    demand = mean / max(float(np.mean(mean)), 1e-6)
    tmpl = np.asarray(templates, dtype=float)[action_id]
    return np.maximum(0.05, tmpl * (0.75 + 0.25 * demand))


class BayesianDemandEstimator:
    """Discounted Gaussian demand belief over reports *and* per-slot observations.

    The between-epoch update realizes Assumption 3 of the paper (bounded marginal
    influence) by construction: each epoch the belief conjugately combines

    * the tenant's report (one sample, variance ``report_var``), and
    * the ``n_obs = L`` per-slot traffic observations the controller measures
      anyway (BSR/arrival intensity, per-sample variance ``obs_var``).

    The report's weight in the posterior mean is
    ``(1/report_var) / (prior + 1/report_var + L/obs_var) = O(1/L)``,
    so a single epoch's misreport perturbs the next epoch's weights by O(1/L),
    which is exactly the dilution the frontier theorem needs. Calling ``update``
    without observations falls back to the report-only path (used by baselines).
    """

    def __init__(
        self,
        config: SimulationConfig,
        discount: float = 0.96,
        report_var: float = 4.0,
        obs_var: float = 25.0,
    ):
        self.config = config
        self.discount = float(discount)
        self.report_var = float(report_var)
        self.obs_var = float(obs_var)
        self.mean = np.array([s.base_mbps for s in config.slices], dtype=float)
        self.mean = np.clip(self.mean, config.theta_min, config.theta_max)
        self.precision = np.ones(config.n_slices, dtype=float) / 25.0

    def update(
        self,
        reports: np.ndarray,
        observed_mean: np.ndarray | None = None,
        n_obs: int = 0,
    ) -> None:
        reports = np.clip(np.asarray(reports, dtype=float), self.config.theta_min, self.config.theta_max)
        prior_precision = self.discount * self.precision
        report_precision = np.ones_like(prior_precision) / self.report_var
        numer = prior_precision * self.mean + report_precision * reports
        denom = prior_precision + report_precision
        if observed_mean is not None and n_obs > 0:
            obs_precision = float(n_obs) / self.obs_var
            observed_mean = np.clip(
                np.asarray(observed_mean, dtype=float), self.config.theta_min, self.config.theta_max
            )
            numer = numer + obs_precision * observed_mean
            denom = denom + obs_precision
        self.mean = numer / denom
        self.precision = np.clip(denom, 1e-6, 1e6)

    def report_influence(self, n_obs: int = 0) -> np.ndarray:
        """d(posterior mean)/d(report): should scale as O(1/n_obs). For tests."""
        prior_precision = self.discount * self.precision
        report_precision = np.ones_like(prior_precision) / self.report_var
        denom = prior_precision + report_precision + float(n_obs) / self.obs_var
        return report_precision / denom

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(1.0 / np.maximum(self.precision, 1e-9))


@dataclass
class EpochMetrics:
    reward: float
    rho: float
    mean_delay_ms: float
    sla_violation_rate: float
    throughput_mbps: float
    wasted_prbs: float


class EpochFrozenQLearner:
    """Small tabular RL controller whose action is frozen for one epoch.

    The state uses the belief over tenant demand plus recent radio/SLA pressure.
    Actions are weight profiles consumed by the monotone DSIC allocator.
    """

    def __init__(
        self,
        config: SimulationConfig,
        seed: int = 0,
        alpha: float = 0.18,
        gamma: float = 0.0,
        epsilon: float = 0.18,
        epsilon_decay: float = 0.997,
        ucb_c: float = 1.5,
        action_templates: np.ndarray | None = None,
    ):
        self.config = config
        self.rng = np.random.default_rng(seed)
        # Optional custom action set (e.g. the fine simplex lattice shared with the
        # deep controllers); falls back to the hand-picked 8-action default below.
        self._action_templates = (
            None
            if action_templates is None
            else np.asarray(action_templates, dtype=float)[:, : config.n_slices]
        )
        self.alpha = float(alpha)
        # Across epochs the type is drawn i.i.d. (Assumption 4), so the weight
        # choice this epoch does not affect the next epoch's state: this is a
        # contextual bandit, not a sequential MDP. gamma defaults to 0 (no
        # bootstrap) and the value is a sample average -- the paper's "no-regret
        # learner over epochs". A nonzero gamma is available for experiments that
        # let queues carry across epochs.
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.ucb_c = float(ucb_c)
        n_act = len(self.action_templates)
        self.q: Dict[Tuple[int, int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(n_act, dtype=float)
        )
        # Per-(state,action) visit counts: drive UCB exploration and the
        # sample-average step size (each action's value is the mean reward seen).
        self.counts: Dict[Tuple[int, int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(n_act, dtype=float)
        )
        self.state_visits: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
        self.last_state: Tuple[int, int, int, int] | None = None
        self.last_action: int | None = None
        self.history: Deque[EpochMetrics] = deque(maxlen=12)

    @property
    def action_templates(self) -> np.ndarray:
        """Per-slice weight strategies spanning the 3-slice weight simplex.

        The RL weight is now the primary allocation lever (priority is not applied
        again in the allocator score), so these must be genuinely different splits:
        the corners (favor one slice), edges (favor a pair), the balanced centre,
        and the priority-proportional profile. Which strategy is best depends on
        the epoch's demand mix, so the agent has a real state->action map to learn.
        """
        if self._action_templates is not None:
            return self._action_templates
        return np.array(
            [
                [5.0, 1.3, 0.7],   # 0: priority-proportional (sensible default)
                [9.0, 1.0, 1.0],   # 1: protect URLLC hard
                [1.0, 9.0, 1.0],   # 2: favor eMBB (throughput)
                [1.0, 1.0, 9.0],   # 3: favor mMTC
                [5.0, 5.0, 1.0],   # 4: URLLC + eMBB
                [1.0, 5.0, 5.0],   # 5: eMBB + mMTC
                [1.0, 1.0, 1.0],   # 6: equal share
                [7.0, 3.0, 1.0],   # 7: strong priority tilt
            ],
            dtype=float,
        )[:, : self.config.n_slices]

    def discretize(
        self,
        estimator: BayesianDemandEstimator,
        mean_delay_ms: float,
        rho_recent: float,
    ) -> Tuple[int, int, int, int]:
        # Coarse, well-sampled context: (contention level, stressed slice). Finer
        # bins (delay, rho, 6-way load) fragment the same context across many keys
        # so each is seen too rarely to learn from; two coarse features keep the
        # bandit's per-state sample count high while capturing what the optimal
        # weight profile actually depends on.
        load_ratio = float(np.sum(estimator.mean) / max(self.config.total_prbs, 1))
        load_bin = int(np.digitize(load_ratio, [0.75, 1.15]))  # under / near / over
        # Which slice is most stressed relative to its floor -- the observable
        # signature of the demand hotspot. (argmax of the raw belief is nearly
        # constant because one slice has the largest baseline; normalizing by the
        # floor exposes the slice that is actually surging.)
        floors = np.maximum(np.asarray(self.config.floor_prbs, dtype=float), 1.0)
        stressed = int(np.argmax(estimator.mean / floors))
        return load_bin, stressed

    def weights_for_action(self, action_id: int, estimator: BayesianDemandEstimator) -> np.ndarray:
        """Weight profile for one action; shared by the learner and the fixed-policy
        hindsight baselines so regret compares identical policy classes.

        The action template is the strategy and dominates; it is only lightly
        modulated by the demand belief (so a template can react to which slice is
        currently heavy without erasing the strategy). This keeps the weight the
        primary lever, which is what gives the agent a learnable reward signal.
        """
        return templates_to_weights(self.action_templates, action_id, estimator.mean)

    def select_action(
        self,
        estimator: BayesianDemandEstimator,
        mean_delay_ms: float,
        rho_recent: float,
        train: bool = True,
    ) -> tuple[int, np.ndarray, Tuple[int, int, int, int]]:
        state = self.discretize(estimator, mean_delay_ms, rho_recent)
        n = self.counts[state]
        if not train:
            action_id = int(np.argmax(self.q[state]))
        else:
            untried = np.where(n == 0)[0]
            if len(untried) > 0:
                # Try every action once before ranking -- the original epsilon-greedy
                # learner skipped this, leaving 7/8 actions at a stale zero value and
                # locking greedy onto the first one tried.
                action_id = int(untried[0])
            elif self.rng.random() < self.epsilon:
                action_id = int(self.rng.integers(0, len(self.action_templates)))
            else:
                action_id = int(np.argmax(self.q[state]))

        weights = self.weights_for_action(action_id, estimator)
        self.last_state = state
        self.last_action = action_id
        return action_id, weights, state

    def update(self, next_state: Tuple[int, int, int, int], reward: float) -> None:
        if self.last_state is None or self.last_action is None:
            return
        s, a = self.last_state, self.last_action
        self.counts[s][a] += 1.0
        self.state_visits[s] += 1
        # Sample-average toward the (bootstrapped, if gamma>0) return. With the
        # default gamma=0 this is the incremental mean of the observed rewards.
        target = float(reward) + self.gamma * float(np.max(self.q[next_state]))
        step = 1.0 / self.counts[s][a]
        self.q[s][a] += step * (target - self.q[s][a])
        self.epsilon = max(0.02, self.epsilon * self.epsilon_decay)

    def remember(self, metrics: EpochMetrics) -> None:
        self.history.append(metrics)

    def recent_rho(self) -> float:
        if not self.history:
            return 0.0
        return float(np.mean([m.rho for m in self.history]))

    def recent_delay(self) -> float:
        if not self.history:
            return 0.0
        return float(np.mean([m.mean_delay_ms for m in self.history]))


def network_reward(config: SimulationConfig, throughput_mbps: np.ndarray, latency_ms: np.ndarray, sla: np.ndarray, wasted_prbs: np.ndarray) -> float:
    priorities = np.asarray(config.priorities, dtype=float)
    sla_ms = np.asarray(config.sla_latency_ms, dtype=float)
    throughput_term = float(np.sum(np.log1p(throughput_mbps) * (0.7 + 0.15 * priorities)))
    delay_term = float(np.sum(priorities * np.clip(latency_ms / np.maximum(sla_ms, 1e-6), 0.0, 10.0)))
    violation_term = float(np.sum(priorities * sla))
    waste_term = float(np.sum(wasted_prbs) / max(config.total_prbs, 1))
    return throughput_term - 0.7 * delay_term - 3.0 * violation_term - 0.35 * waste_term
