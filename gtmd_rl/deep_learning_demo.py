"""Deep-RL controllers (DQN, PPO, A2C) for the epoch-frozen weight policy, and a
head-to-head comparison against the tabular contextual bandit.

Why deep RL here, and why it is still a *bandit*. Across epochs the demand type is
drawn i.i.d. (Assumption 4), so the weight choice this epoch does not shape the
next epoch's state: the learning problem is a **contextual bandit**, not a
sequential MDP. We therefore run every neural agent with single-step episodes
(``done=True`` every epoch), which makes the discount irrelevant -- the DQN target
collapses to ``E[reward | s, a]`` (a neural Q-regression / neural bandit) and the
PPO/A2C advantage collapses to ``reward - V(s)`` (a contextual policy gradient).
This keeps the deep agents faithful to the same no-regret-over-epochs object the
paper's theory is about, while replacing the coarse 2-bin tabular state with a
continuous demand-belief feature vector and the hand-picked 8 actions with a fine
28-point simplex lattice.

The point of the study. The tabular bandit is exact but *coarse*: it can only pick
one weight profile per (load-bin, stressed-slice) cell, so within a cell it cannot
adapt to how heavy the surge is. A function approximator over the continuous belief
can. We give all learners the SAME 28-action simplex set and the SAME priced
mechanism reward, then score each one's greedy policy against a Monte-Carlo oracle
of the true per-context action values. The question is whether deep approximation
buys a higher normalized score than the coarse table -- and how DQN/PPO/A2C rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from .config import SimulationConfig
from .learning_demo import _base_theta, _draw_type, _epoch_reward, demo_config
from .rl import (
    BayesianDemandEstimator,
    EpochFrozenQLearner,
    simplex_action_templates,
    templates_to_weights,
)
from .rl_sota_models import A2CAgent, DQNAgent, PPOAgent, PPOBuffer

# Tiny networks on a laptop CPU: cap the thread pool so we do not oversubscribe.
torch.set_num_threads(min(4, torch.get_num_threads()))


# --------------------------------------------------------------------------- #
# Continuous state features (shared by all neural agents)
# --------------------------------------------------------------------------- #
def encode_features(cfg: SimulationConfig, est: BayesianDemandEstimator) -> np.ndarray:
    """Continuous context for the deep controllers: the contention level, each
    slice's demand-to-floor stress, and each slice's demand share. This is a
    strict refinement of the tabular (load-bin, stressed-slice) discretization --
    the deep agent sees *how much* a slice is surging, not just which one is."""
    mean = np.asarray(est.mean, dtype=float)
    floors = np.maximum(np.asarray(cfg.floor_prbs, dtype=float), 1.0)
    load = float(np.sum(mean) / max(cfg.total_prbs, 1))
    stress = mean / floors / 4.0                       # ~O(1)
    share = mean / max(float(np.sum(mean)), 1e-6)      # in [0, 1]
    return np.concatenate([[load], stress, share]).astype(np.float32)


def feature_dim(cfg: SimulationConfig) -> int:
    return 1 + 2 * cfg.n_slices


def _fresh_estimator(cfg: SimulationConfig, theta: np.ndarray, L: int) -> BayesianDemandEstimator:
    est = BayesianDemandEstimator(cfg)
    est.update(theta, observed_mean=theta, n_obs=L)
    return est


# --------------------------------------------------------------------------- #
# Uniform adapter interface: select -> observe -> greedy
# --------------------------------------------------------------------------- #
class _Adapter:
    name = "base"

    def select(self, feats, est, train=True) -> int:
        raise NotImplementedError

    def observe(self, feats, action_id, reward) -> None:
        raise NotImplementedError

    def greedy(self, feats, est) -> int:
        raise NotImplementedError


class TabularAdapter(_Adapter):
    """The existing epoch-frozen contextual-bandit table, on the expanded action
    set. Reference point: exact values, coarse state."""

    name = "Tabular bandit"

    def __init__(self, cfg, templates, seed):
        self.learner = EpochFrozenQLearner(cfg, seed=seed, action_templates=templates)

    def select(self, feats, est, train=True):
        aid, _, _ = self.learner.select_action(est, 0.0, 0.0, train=train)
        return aid

    def observe(self, feats, action_id, reward):
        # select_action already set last_state/last_action; gamma=0 so the
        # next_state argument (bootstrap target) is unused -- pass last_state.
        self.learner.update(self.learner.last_state, reward)

    def greedy(self, feats, est):
        aid, _, _ = self.learner.select_action(est, 0.0, 0.0, train=False)
        return aid


class DQNAdapter(_Adapter):
    """Double-DQN over the continuous belief. Single-step episodes -> the TD target
    is just the reward, so this is a neural contextual bandit (Q-regression)."""

    name = "DQN"

    def __init__(self, cfg, templates, seed):
        self.agent = DQNAgent(
            state_dim=feature_dim(cfg), action_dim=len(templates),
            learning_rate=1e-3, gamma=0.0, epsilon=0.35, epsilon_decay=0.999,
            epsilon_min=0.03, buffer_size=4000, batch_size=64, target_update_freq=20,
            seed=seed,
        )

    def _t(self, feats):
        return torch.as_tensor(feats, dtype=torch.float32, device=self.agent.device).unsqueeze(0)

    def select(self, feats, est, train=True):
        return int(self.agent.select_action(self._t(feats), train=train))

    def observe(self, feats, action_id, reward):
        self.agent.store_transition(np.asarray(feats, np.float32), int(action_id),
                                    float(reward), np.asarray(feats, np.float32), True)
        self.agent.train_step()

    def greedy(self, feats, est):
        return int(self.agent.select_action(self._t(feats), train=False))


class _OnPolicyAdapter(_Adapter):
    """Common rollout plumbing for PPO/A2C: buffer (feats, action, value, logp,
    reward) for a fixed number of epochs, then flush one policy update."""

    def __init__(self, rollout_len):
        self.rollout_len = int(rollout_len)
        self._s: List[np.ndarray] = []
        self._a: List[int] = []
        self._v: List[float] = []
        self._lp: List[float] = []
        self._r: List[float] = []
        self._pending = None

    def _flush(self):
        raise NotImplementedError

    def observe(self, feats, action_id, reward):
        self._r.append(float(reward))
        if len(self._r) >= self.rollout_len:
            self._flush()
            self._s, self._a, self._v, self._lp, self._r = [], [], [], [], []


class PPOAdapter(_OnPolicyAdapter):
    """PPO with a clipped objective and a value baseline. Single-step episodes make
    the GAE advantage reduce to ``reward - V(s)`` -- a contextual policy gradient."""

    name = "PPO"

    def __init__(self, cfg, templates, seed, rollout_len=32):
        super().__init__(rollout_len)
        self.agent = PPOAgent(
            state_dim=feature_dim(cfg), action_dim=len(templates),
            learning_rate=5e-4, gamma=0.0, lam=0.95, clip_ratio=0.2,
            entropy_coef=0.02, n_epochs=8, batch_size=32, seed=seed,
        )

    def _t(self, feats):
        return torch.as_tensor(feats, dtype=torch.float32, device=self.agent.device).unsqueeze(0)

    def select(self, feats, est, train=True):
        a, logp, v = self.agent.select_action(self._t(feats))
        self._s.append(np.asarray(feats, np.float32)); self._a.append(int(a))
        self._v.append(float(v)); self._lp.append(float(logp))
        return int(a)

    def _flush(self):
        buf = PPOBuffer(states=self._s, actions=self._a, rewards=self._r,
                        values=self._v, dones=[True] * len(self._r), log_probs=self._lp)
        self.agent.update(buf)

    def greedy(self, feats, est):
        with torch.no_grad():
            logits = self.agent.actor(self._t(feats))
            return int(logits.argmax(dim=1).item())


class A2CAdapter(_OnPolicyAdapter):
    """Synchronous advantage actor-critic; single-step returns = reward."""

    name = "A2C"

    def __init__(self, cfg, templates, seed, rollout_len=16):
        super().__init__(rollout_len)
        self.agent = A2CAgent(
            state_dim=feature_dim(cfg), action_dim=len(templates),
            learning_rate=7e-4, gamma=0.0, entropy_coef=0.02, seed=seed,
        )

    def _t(self, feats):
        return torch.as_tensor(feats, dtype=torch.float32, device=self.agent.device).unsqueeze(0)

    def select(self, feats, est, train=True):
        a, v = self.agent.select_action(self._t(feats))
        self._s.append(np.asarray(feats, np.float32)); self._a.append(int(a))
        return int(a)

    def _flush(self):
        self.agent.update(self._s, self._a, self._r, next_value=0.0)

    def greedy(self, feats, est):
        with torch.no_grad():
            logits = self.agent.actor(self._t(feats))
            return int(logits.argmax(dim=1).item())


ADAPTERS: Dict[str, Callable] = {
    "Tabular bandit": TabularAdapter,
    "DQN": DQNAdapter,
    "PPO": PPOAdapter,
    "A2C": A2CAdapter,
}


# --------------------------------------------------------------------------- #
# Shared MC-true evaluation set
# --------------------------------------------------------------------------- #
@dataclass
class _TestCtx:
    theta: np.ndarray
    est: BayesianDemandEstimator
    feats: np.ndarray
    action_means: np.ndarray
    best: int
    lo: float
    hi: float


def build_eval_set(cfg, base, templates, L, n_ctx=24, n_mc=6, seed=77) -> List[_TestCtx]:
    """Held-out contexts with the Monte-Carlo TRUE mean reward of every action.
    The greedy policy of each learner is later scored against these means, so the
    comparison is on the exact same yardstick (oracle=1, action-average=0)."""
    rng = np.random.default_rng(seed)
    nA = len(templates)
    out: List[_TestCtx] = []
    for c in range(n_ctx):
        theta = _draw_type(cfg, base, rng)
        est = _fresh_estimator(cfg, theta, L)
        means = np.zeros(nA)
        for a in range(nA):
            w = templates_to_weights(templates, a, est.mean)
            means[a] = np.mean([_epoch_reward(cfg, 4099 * c + 7 * a + m, theta, w, L)
                                for m in range(n_mc)])
        out.append(_TestCtx(theta, est, encode_features(cfg, est), means,
                            int(means.argmax()), float(means.mean()), float(means.max())))
    return out


def score_greedy(adapter: _Adapter, test: List[_TestCtx]) -> Tuple[float, float]:
    """Frequency-flat normalized score and optimal-action rate of the greedy policy."""
    norm = hit = 0.0
    for ctx in test:
        aid = adapter.greedy(ctx.feats, ctx.est)
        if ctx.hi > ctx.lo:
            norm += (ctx.action_means[aid] - ctx.lo) / (ctx.hi - ctx.lo)
        else:
            norm += 1.0
        hit += 1.0 if aid == ctx.best else 0.0
    n = max(len(test), 1)
    return norm / n, hit / n


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class DeepDemoResult:
    checkpoints: np.ndarray
    curves: Dict[str, Dict[str, np.ndarray]]  # name -> {norm_mean,norm_std,opt_mean,opt_std}
    n_actions: int
    load: float
    agents: List[str]
    final_table: Dict[str, Tuple[float, float]]  # name -> (norm_final, opt_final)


def run_deep_demo(
    config: SimulationConfig | None = None,
    agents: Tuple[str, ...] = ("Tabular bandit", "DQN", "PPO", "A2C"),
    load: float = 1.0,
    n_epochs: int = 2500,
    epoch_length: int = 60,
    seeds: int = 3,
    eval_every: int = 100,
    degree: int = 6,
    eval_ctx: int = 24,
    eval_mc: int = 6,
    verbose: bool = True,
) -> DeepDemoResult:
    cfg = demo_config() if config is None else config
    templates = simplex_action_templates(cfg.n_slices, degree=degree)
    base = _base_theta(cfg, load)
    test = build_eval_set(cfg, base, templates, epoch_length, n_ctx=eval_ctx, n_mc=eval_mc)
    if verbose:
        print(f"[eval] {len(test)} contexts x {len(templates)} actions; "
              f"mean oracle-gap {np.mean([t.hi - t.lo for t in test]):.3f}")

    curves = {name: {"norm": [], "opt": []} for name in agents}
    checkpoints = None
    for name in agents:
        norm_seeds, opt_seeds = [], []
        for s in range(seeds):
            adapter = ADAPTERS[name](cfg, templates, seed=1000 * s + 41)
            rng = np.random.default_rng(9001 + 31 * s)
            ns_curve, op_curve, ck = [], [], []
            for epoch in range(n_epochs):
                theta = _draw_type(cfg, base, rng)
                est = _fresh_estimator(cfg, theta, epoch_length)
                feats = encode_features(cfg, est)
                aid = adapter.select(feats, est, train=True)
                w = templates_to_weights(templates, aid, est.mean)
                r = _epoch_reward(cfg, 7919 * s + epoch, theta, w, epoch_length)
                adapter.observe(feats, aid, r)
                if epoch % eval_every == 0:
                    nrm, opt = score_greedy(adapter, test)
                    ns_curve.append(nrm); op_curve.append(opt); ck.append(epoch)
            norm_seeds.append(ns_curve); opt_seeds.append(op_curve)
            checkpoints = np.array(ck)
        nrm = np.array(norm_seeds); opt = np.array(opt_seeds)
        curves[name] = {
            "norm_mean": nrm.mean(0), "norm_std": nrm.std(0),
            "opt_mean": opt.mean(0), "opt_std": opt.std(0),
        }
        if verbose:
            print(f"[{name:15s}] norm {nrm.mean(0)[0]:.2f} -> {nrm.mean(0)[-1]:.2f} | "
                  f"opt {opt.mean(0)[0]:.2f} -> {opt.mean(0)[-1]:.2f}")

    final_table = {name: (float(curves[name]["norm_mean"][-1]),
                          float(curves[name]["opt_mean"][-1])) for name in agents}
    return DeepDemoResult(
        checkpoints=checkpoints, curves=curves, n_actions=len(templates),
        load=load, agents=list(agents), final_table=final_table,
    )
