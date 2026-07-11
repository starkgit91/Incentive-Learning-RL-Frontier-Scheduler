#!/usr/bin/env python3
"""Python stand-in for the ns-3 5G-LENA client, for testing the socket protocol.

It speaks the exact same NDJSON wire protocol the ns-3 scenario speaks, so you can
validate the whole RL+DSIC control loop end-to-end on one machine before building
ns-3. It receives a per-slice PRB budget each epoch, runs a small slot-level
queueing model in which each slice is *hard-capped* to its allocated PRBs (which is
what the gNB does to a slice), and returns measured per-slice throughput, delay and
SLA violation -- the same fields the ns-3 FlowMonitor reports.

Run the server first, then:
    python3 bridge/mock_ns3_client.py
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

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
    AllocMessage,
    KpiMessage,
    NDJSONChannel,
)
from gtmd_rl.config import default_config
from gtmd_rl.network import NRTraceGenerator


def simulate_epoch(env: NRTraceGenerator, alloc: AllocMessage, cfg, epoch_length: int) -> KpiMessage:
    """Run ``epoch_length`` slots with each slice hard-capped to its PRB budget."""
    budget = np.asarray(alloc.prbs, dtype=int)
    thr, delay, sla, used = [], [], [], []
    for _ in range(epoch_length):
        state = env.current_state()
        # The gNB serves each slice up to its PRB entitlement (or its backlog).
        demand = np.ceil(np.clip(state.demand_prbs, 0, cfg.total_prbs)).astype(int)
        applied = np.minimum(budget, demand)
        _, result = env.step(applied)
        thr.append(result.throughput_mbps)
        delay.append(result.latency_ms)
        sla.append(result.sla_violation)
        used.append(applied.astype(float))
    thr = np.vstack(thr).mean(axis=0)
    delay = np.vstack(delay).mean(axis=0)
    sla = np.vstack(sla).mean(axis=0)
    used = np.vstack(used).mean(axis=0)
    return KpiMessage(
        epoch=alloc.epoch,
        throughput_mbps=[float(x) for x in thr],
        mean_delay_ms=[float(x) for x in delay],
        sla_violation=[float(x) for x in sla],
        prb_used=[float(x) for x in used],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Mock ns-3 client for the RL+DSIC bridge.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--load", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    cfg = default_config()
    sock = socket.create_connection((args.host, args.port))
    ch = NDJSONChannel(sock)
    print(f"[mock-ns3] connected to {args.host}:{args.port}")
    ch.send({"type": HELLO, "role": "mock-ns3", "version": 1})

    cfgmsg = ch.recv()
    assert cfgmsg and cfgmsg["type"] == CONFIG, cfgmsg
    epoch_length = int(cfgmsg["epoch_length"])
    print(f"[mock-ns3] CONFIG: slices={cfgmsg['slices']} floors={cfgmsg['floor_prbs']} "
          f"B={cfgmsg['total_prbs']} L={epoch_length}")

    env = NRTraceGenerator(cfg, load=args.load, seed=args.seed)
    n = 0
    while True:
        msg = ch.recv()
        if msg is None or msg.get("type") == DONE:
            print(f"[mock-ns3] DONE after {n} epochs")
            break
        assert msg["type"] == ALLOC, msg
        alloc = AllocMessage(
            epoch=int(msg["epoch"]), prbs=msg["prbs"], weights=msg["weights"],
            reports=msg["reports"], cqi=msg["cqi"], prb_capacity_bits=msg["prb_capacity_bits"],
        )
        kpi = simulate_epoch(env, alloc, cfg, epoch_length)
        ch.send(kpi.to_dict())
        n += 1
    ch.close()


if __name__ == "__main__":
    main()
