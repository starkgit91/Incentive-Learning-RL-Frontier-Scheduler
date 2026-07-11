from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import numpy as np


class TabularReportAdversary:
    """Q-learning tenant that learns cross-epoch report multipliers."""

    def __init__(
        self,
        tenant_id: int = 1,
        multipliers: tuple[float, ...] = (0.65, 0.85, 1.0, 1.2, 1.5, 1.9),
        seed: int = 0,
        alpha: float = 0.22,
        gamma: float = 0.88,
        epsilon: float = 0.25,
    ):
        self.tenant_id = int(tenant_id)
        self.multipliers = np.asarray(multipliers, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.q = defaultdict(lambda: np.zeros(len(self.multipliers), dtype=float))
        self.last_state = None
        self.last_action = None

    def state_key(self, planner_action: int, own_theta: float, rho_recent: float) -> Tuple[int, int, int]:
        theta_bin = int(np.digitize(float(own_theta), [2.0, 5.0, 10.0, 18.0, 30.0]))
        rho_bin = int(np.digitize(float(rho_recent), [0.05, 0.15, 0.35, 0.6]))
        return int(planner_action), theta_bin, rho_bin

    def choose_multiplier(
        self,
        planner_action: int,
        own_theta: float,
        rho_recent: float,
        train: bool = True,
    ) -> float:
        key = self.state_key(planner_action, own_theta, rho_recent)
        if train and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, len(self.multipliers)))
        else:
            action = int(np.argmax(self.q[key]))
        self.last_state = key
        self.last_action = action
        return float(self.multipliers[action])

    def update(self, reward: float, planner_action: int, own_theta: float, rho_recent: float) -> None:
        if self.last_state is None or self.last_action is None:
            return
        next_key = self.state_key(planner_action, own_theta, rho_recent)
        target = float(reward) + self.gamma * float(np.max(self.q[next_key]))
        values = self.q[self.last_state]
        values[self.last_action] += self.alpha * (target - values[self.last_action])

    def greedy(self) -> "TabularReportAdversary":
        clone = TabularReportAdversary(
            tenant_id=self.tenant_id,
            multipliers=tuple(float(x) for x in self.multipliers),
            seed=0,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=0.0,
        )
        clone.q = self.q
        return clone
