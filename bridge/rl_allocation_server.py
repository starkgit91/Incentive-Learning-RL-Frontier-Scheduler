#!/usr/bin/env python3
"""RL+DSIC PRB-allocation server for the ns-3 5G-LENA bridge.

This is the near-RT-RIC controller. It owns the game-theoretic mechanism (DSIC
reports + Myerson payments), the Bayesian demand belief, and the epoch-frozen RL
policy that emits per-slice PRB weights. Once per epoch it computes a per-slice
PRB budget and ships it over a TCP socket to whichever client is connected --
the ns-3 gNB scenario (``bridge/ns3/gtmd-nr-bridge.cc``) in a real run, or the
Python mock client (``bridge/mock_ns3_client.py``) for protocol testing.

The client applies the budget in its 3GPP PHY/MAC and returns measured per-slice
KPIs, which the server folds back into the RL reward -- a genuine closed loop.

Run:
    python3 bridge/rl_allocation_server.py --epochs 40 --epoch-length 60 --load 1.1
then, in another terminal:
    python3 bridge/mock_ns3_client.py           # or the ns-3 scenario
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge.protocol import (
    ALLOC,
    CONFIG,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DONE,
    HELLO,
    KPI,
    AllocMessage,
    KpiMessage,
    NDJSONChannel,
)
from gtmd_rl.adversary import TabularReportAdversary
from gtmd_rl.config import SimulationConfig, default_config
from gtmd_rl.mechanism import epoch_payments, weighted_greedy_allocator
from gtmd_rl.network import NRTraceGenerator
from gtmd_rl.rl import BayesianDemandEstimator, EpochFrozenQLearner


def epoch_prb_budget(weights: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Per-slice PRB entitlement for one epoch: floors first, surplus by weight.

    Sums to exactly ``config.total_prbs`` (largest-remainder rounding) and is
    monotone in the weight, so a higher-weight slice never gets fewer PRBs. This
    is the hard split the gNB enforces for the whole (frozen) epoch.
    """
    n = config.n_slices
    floors = np.asarray(config.floor_prbs, dtype=int)
    budget = np.minimum(floors, config.total_prbs).astype(int)
    if budget.sum() > config.total_prbs:  # scarcity guard: prioritise by weight
        order = sorted(range(n), key=lambda i: -float(weights[i]))
        budget = np.zeros(n, dtype=int)
        remaining = int(config.total_prbs)
        for i in order:
            give = min(int(floors[i]), remaining)
            budget[i] += give
            remaining -= give
        return budget
    remaining = int(config.total_prbs - budget.sum())
    w = np.maximum(np.asarray(weights, dtype=float), 1e-9)
    share = remaining * w / w.sum()
    extra = np.floor(share).astype(int)
    budget += extra
    # Largest-remainder to place the last few PRBs deterministically.
    leftover = remaining - int(extra.sum())
    frac_order = sorted(range(n), key=lambda i: -(share[i] - np.floor(share[i])))
    for k in range(leftover):
        budget[frac_order[k % n]] += 1
    return budget.astype(int)


def kpi_reward(config: SimulationConfig, kpi: KpiMessage) -> float:
    """Turn measured KPIs into the epoch reward that trains the RL policy."""
    priorities = np.asarray(config.priorities, dtype=float)
    thr = np.asarray(kpi.throughput_mbps, dtype=float)
    sla = np.asarray(kpi.sla_violation, dtype=float)
    delay = np.asarray(kpi.mean_delay_ms, dtype=float)
    sla_ms = np.asarray(config.sla_latency_ms, dtype=float)
    thr_term = float(np.sum(np.log1p(np.maximum(thr, 0.0)) * (0.7 + 0.15 * priorities)))
    delay_term = float(np.sum(priorities * np.clip(delay / np.maximum(sla_ms, 1e-6), 0.0, 10.0)))
    viol_term = float(np.sum(priorities * sla))
    return thr_term - 0.7 * delay_term - 3.0 * viol_term


class RLAllocationServer:
    def __init__(
        self,
        config: SimulationConfig,
        load: float,
        epochs: int,
        epoch_length: int,
        seed: int = 42,
        adversary_tenant: Optional[int] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        trace_path: Optional[Path] = None,
    ):
        self.config = config
        self.load = float(load)
        self.epochs = int(epochs)
        self.epoch_length = int(epoch_length)
        self.seed = int(seed)
        self.host = host
        self.port = port
        self.trace_path = trace_path
        self.env = NRTraceGenerator(config, load=load, seed=seed)
        self.estimator = BayesianDemandEstimator(config)
        self.planner = EpochFrozenQLearner(config, seed=seed + 100)
        self.adversary_tenant = adversary_tenant
        self.adversary = (
            TabularReportAdversary(tenant_id=adversary_tenant, seed=seed + 7)
            if adversary_tenant is not None
            else None
        )
        self._records: List[dict] = []

    def _config_message(self) -> dict:
        c = self.config
        return {
            "type": CONFIG,
            "slices": [s.name for s in c.slices],
            "priorities": list(c.priorities),
            "floor_prbs": list(c.floor_prbs),
            "sla_latency_ms": list(c.sla_latency_ms),
            "total_prbs": c.total_prbs,
            "prb_bandwidth_hz": c.prb_bandwidth_hz,
            "slot_ms": c.slot_ms,
            "epoch_length": self.epoch_length,
            "epochs": self.epochs,
            "load": self.load,
        }

    def _next_alloc(self, epoch: int) -> AllocMessage:
        true_theta = self.env.begin_epoch()
        reports = true_theta.copy()
        if self.adversary is not None:
            mult = self.adversary.choose_multiplier(0, true_theta[self.adversary_tenant],
                                                    self.planner.recent_rho(), train=False)
            reports[self.adversary_tenant] = float(np.clip(
                reports[self.adversary_tenant] * mult, self.config.theta_min, self.config.theta_max))
        self.estimator.update(reports)
        _, weights, _ = self.planner.select_action(
            self.estimator,
            mean_delay_ms=self.planner.recent_delay(),
            rho_recent=self.planner.recent_rho(),
            train=True,
        )
        budget = epoch_prb_budget(weights, self.config)
        cap = self.env.prb_capacity_bits()
        # advance the environment channel/traffic over the epoch so the next epoch
        # sees fresh conditions (KPIs come from the client, not from here)
        for _ in range(self.epoch_length):
            self.env.step(np.zeros(self.config.n_slices, dtype=int))
        return AllocMessage(
            epoch=epoch,
            prbs=budget.tolist(),
            weights=[float(x) for x in weights],
            reports=[float(x) for x in reports],
            cqi=[float(x) for x in self.env.cqi],
            prb_capacity_bits=[float(x) for x in cap],
        )

    def _handle_kpi(self, alloc: AllocMessage, kpi: KpiMessage) -> None:
        reward = kpi_reward(self.config, kpi)
        rho = float(np.mean([1.0 if u >= b - 0.5 and b > 0 else 0.0
                             for u, b in zip(kpi.prb_used or alloc.prbs, alloc.prbs)]))
        mean_delay = float(np.mean(kpi.mean_delay_ms))
        next_key = self.planner.discretize(self.estimator, mean_delay, rho)
        self.planner.update(next_key, reward)
        from gtmd_rl.rl import EpochMetrics
        self.planner.remember(EpochMetrics(reward, rho, mean_delay,
                                           float(np.mean(kpi.sla_violation)),
                                           float(np.sum(kpi.throughput_mbps)), 0.0))
        for i, sname in enumerate(s.name for s in self.config.slices):
            self._records.append({
                "epoch": alloc.epoch, "slice": sname,
                "prb_budget": alloc.prbs[i], "weight": alloc.weights[i],
                "report": alloc.reports[i], "cqi": alloc.cqi[i],
                "throughput_mbps": kpi.throughput_mbps[i],
                "mean_delay_ms": kpi.mean_delay_ms[i],
                "sla_violation": kpi.sla_violation[i],
                "prb_used": (kpi.prb_used or alloc.prbs)[i],
                "reward": reward,
            })

    def serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[server] listening on {self.host}:{self.port} "
              f"(load={self.load}, epochs={self.epochs}, L={self.epoch_length})")
        conn, addr = srv.accept()
        print(f"[server] client connected from {addr}")
        ch = NDJSONChannel(conn)
        try:
            hello = ch.recv()
            if not hello or hello.get("type") != HELLO:
                print(f"[server] expected HELLO, got {hello}")
                return
            print(f"[server] HELLO from role={hello.get('role')}")
            ch.send(self._config_message())

            for epoch in range(self.epochs):
                alloc = self._next_alloc(epoch)
                ch.send(alloc.to_dict())
                reply = ch.recv()
                if reply is None:
                    print("[server] client closed early")
                    break
                if reply.get("type") != KPI:
                    print(f"[server] expected KPI, got {reply.get('type')}")
                    break
                kpi = KpiMessage.from_dict(reply)
                self._handle_kpi(alloc, kpi)
                agg_thr = sum(kpi.throughput_mbps)
                agg_sla = float(np.mean(kpi.sla_violation))
                print(f"[server] epoch {epoch:3d}  budget={alloc.prbs}  "
                      f"thr={agg_thr:6.2f}Mbps  sla={agg_sla:.3f}")
            ch.send({"type": DONE, "epochs": self.epochs})
        finally:
            ch.close()
            srv.close()
            if self.trace_path is not None and self._records:
                import pandas as pd
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(self._records).to_csv(self.trace_path, index=False)
                print(f"[server] wrote bridge trace -> {self.trace_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="RL+DSIC PRB allocation server (ns-3 bridge).")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--load", type=float, default=1.1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--epoch-length", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adversary-tenant", type=int, default=None,
                   help="Slice index that misreports (default none).")
    p.add_argument("--trace", default="outputs/bridge/bridge_trace.csv")
    args = p.parse_args()

    server = RLAllocationServer(
        config=default_config(),
        load=args.load,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        seed=args.seed,
        adversary_tenant=args.adversary_tenant,
        host=args.host,
        port=args.port,
        trace_path=Path(args.trace) if args.trace else None,
    )
    server.serve()


if __name__ == "__main__":
    main()
