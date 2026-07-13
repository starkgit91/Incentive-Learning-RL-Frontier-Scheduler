"""E8: a measurement-fed RL learner, to show Theorem 3 is ALGORITHM-AGNOSTIC.

Theorem 3 says nothing about *how* the weights are updated. It says only that if the
learner's inputs carry no report, the weight path is a deterministic function of
quantities the tenant cannot influence -- and therefore every epoch is a fixed monotone
rule priced by Myerson, so truthfulness is exact. Any update rule inherits this: OGD,
FTRL, a bandit, a deep policy gradient, anything.

We demonstrate it by swapping OGD for tabular Q-learning over a discretized weight grid
and re-running the same V3 canary and the same adversary grid. The RL agent's STATE and
REWARD are built from measurements and the channel only -- never from a report, and never
from an endogenous quantity (no queues, no realized payments). If a report leaked into
the state or the reward, V3 would fail immediately.

This is deliberately NOT presented as the paper's main mechanism. The mechanism is the
gate; the RL agent is evidence that the gate's guarantee does not depend on the learner.
"""

from __future__ import annotations

import numpy as np

from .instances import Instance
from .learner import plug_in_welfare, project


class MeasurementQLearner:
    """Tabular Q-learning over a grid of weight vectors.

    State  : (binned running-mean measurement per tenant, epoch bucket)  -- measurable
    Action : an index into a discretized grid of weight vectors
    Reward : the plug-in welfare of the epoch computed from the GATED input and the
             realized channel -- i.e. exactly the same measurable objective OGD ascends.

    Same interface as learner.OGD (`.w` and `.step(itilde, q)`), so mechanism.py does not
    care which learner it is driving.
    """

    def __init__(self, inst: Instance, n_grid: int = 21, eta: float = 0.5,
                 gamma: float = 0.0, epsilon: float = 0.1, n_bins: int = 5,
                 seed: int = 0):
        self.inst = inst
        self.rng = np.random.default_rng(seed)
        self.alpha = float(eta)
        self.gamma = float(gamma)     # epochs are i.i.d. in the measurement -> bandit
        self.epsilon = float(epsilon)
        self.n_bins = int(n_bins)

        # Action set: a grid over the weight slice (w1 free, the rest split evenly).
        lo, hi = inst.weight_lo, inst.weight_sum - (inst.n - 1) * inst.weight_lo
        self.actions = []
        for w1 in np.linspace(lo, hi, n_grid):
            w = np.full(inst.n, (inst.weight_sum - w1) / (inst.n - 1))
            w[0] = w1
            self.actions.append(project(w, inst))
        self.actions = np.array(self.actions)

        self.Q: dict[tuple, np.ndarray] = {}
        self.C: dict[tuple, np.ndarray] = {}      # per-(state,action) visit counts
        self.w = project(np.full(inst.n, inst.weight_sum / inst.n), inst)

    # -- state is a function of the MEASUREMENT only -------------------------------
    def _state(self, itilde: np.ndarray) -> tuple:
        lo = np.asarray(self.inst.theta_lo, float)
        hi = np.asarray(self.inst.theta_hi, float)
        b = np.clip(((itilde - lo) / np.maximum(hi - lo, 1e-12) * self.n_bins).astype(int),
                    0, self.n_bins - 1)
        return tuple(int(v) for v in b)

    def _qc(self, s: tuple):
        if s not in self.Q:
            self.Q[s] = np.zeros(len(self.actions))
            self.C[s] = np.zeros(len(self.actions))
        return self.Q[s], self.C[s]

    def step(self, itilde: np.ndarray, q: np.ndarray) -> np.ndarray:
        s = self._state(itilde)
        qs, cs = self._qc(s)

        # Across epochs the measurement is i.i.d., so this is a CONTEXTUAL BANDIT, not
        # a sequential MDP (gamma = 0). Try every action once before ranking: with Q
        # initialised at zero and rewards strictly positive, a plain argmax would lock
        # onto action 0 forever, leaving the weight path CONSTANT -- which would pass
        # the V3 canary vacuously instead of testing it.
        untried = np.flatnonzero(cs == 0)
        if untried.size:
            a = int(untried[0])
        elif self.rng.random() < self.epsilon:
            a = int(self.rng.integers(len(self.actions)))
        else:
            a = int(np.argmax(qs))

        w_next = self.actions[a]
        # Reward: the plug-in welfare of this epoch under the chosen weight, built from
        # itilde (the gated input) and q (the channel) ONLY -- never from a report.
        r = plug_in_welfare(w_next, itilde, q, self.inst)
        cs[a] += 1.0
        qs[a] += (r - qs[a]) / cs[a]          # sample mean (gamma = 0)
        self.w = w_next
        return self.w
