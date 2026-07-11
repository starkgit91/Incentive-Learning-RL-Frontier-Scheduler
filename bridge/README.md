# GTMD-RL ⇄ ns-3 5G-LENA Socket Bridge

This directory contains the **Tx/Rx bridge** that connects the Python RL+DSIC
resource controller to the ns-3 **5G-LENA** NR simulator over a plain TCP socket.
No `ns3-ai` / `ns3-gym` middleware is required — it is raw POSIX sockets on the
C++ side and the `socket` module on the Python side, exactly as requested.

```
        ┌──────────────────────────────────────────┐        ┌──────────────────────────────┐
        │  Python  (near-RT RIC controller)        │        │  ns-3 5G-LENA (gNB + UEs)    │
        │                                          │        │                              │
        │  DSIC reports ─► Bayesian demand belief  │  TCP   │  1 gNB, 3 slices (URLLC/     │
        │        │                                 │ socket │  eMBB/mMTC), OFDMA-QoS sched │
        │        ▼                                 │ NDJSON │                              │
        │  Epoch-frozen RL policy ─► PRB weights   │ ─────► │  apply per-slice PRB budget  │
        │        │                                 │ ALLOC  │  (rate-cap = budget×capacity)│
        │        ▼                                 │        │            │                 │
        │  epoch_prb_budget()  ── per-slice PRBs ──┼───────►│  run L slots on 3GPP PHY/MAC │
        │                                          │        │            ▼                 │
        │  RL reward  ◄── throughput/delay/SLA ────┼◄────── │  FlowMonitor per-slice KPIs  │
        │                                          │  KPI   │                              │
        └──────────────────────────────────────────┘        └──────────────────────────────┘
```

The controller sends a per-slice PRB budget once per epoch; the gNB enforces it,
runs the epoch on a 3GPP-compliant PHY/MAC, and returns measured per-slice KPIs
that become the RL reward. This is the closed near-RT-RIC loop from the paper.

## Files

| File | Role |
|------|------|
| `protocol.py`             | NDJSON wire protocol shared by both sides |
| `rl_allocation_server.py` | Python server: DSIC + RL + PRB-budget streaming (Tx), KPI ingest (Rx) |
| `mock_ns3_client.py`      | Python stand-in for ns-3 to test the protocol with no build |
| `ns3/gtmd-nr-bridge.cc`   | ns-3 5G-LENA scenario: socket client, PRB enforcement, FlowMonitor KPIs |

## 1. Protocol-only test (no ns-3 build needed)

Terminal A:
```bash
python3 bridge/rl_allocation_server.py --epochs 20 --epoch-length 60 --load 1.1 --port 5005
```
Terminal B:
```bash
python3 bridge/mock_ns3_client.py --port 5005 --load 1.1
```
The server prints the per-epoch PRB budget and the KPIs it gets back, and writes
`outputs/bridge/bridge_trace.csv`. This validates the entire control loop end-to-end.

## 2. Real 5G-LENA run

Assumes ns-3-dev with the `nr` (5G-LENA) module built (`./ns3 build`).

```bash
# one-time: drop the scenario into ns-3's scratch/ and build it
cp bridge/ns3/gtmd-nr-bridge.cc  $NS3_DIR/scratch/
cd $NS3_DIR && ./ns3 build gtmd-nr-bridge
```

Then, from the project root:
```bash
# Terminal A — start the controller first (it listens):
python3 bridge/rl_allocation_server.py --epochs 8 --epoch-length 40 --load 1.1 --port 5005 \
        --trace outputs/bridge/ns3_bridge_trace.csv

# Terminal B — start the ns-3 gNB (it connects):
cd $NS3_DIR && ./ns3 run "gtmd-nr-bridge --serverHost=127.0.0.1 --serverPort=5005"
```
The server writes `outputs/bridge/ns3_bridge_trace.csv` with the per-slice PRB
budget, weight, report, CQI and the **measured** ns-3 throughput / delay / SLA.

### ns-3 scenario command-line options
| Flag | Default | Meaning |
|------|---------|---------|
| `--serverHost` | `127.0.0.1` | controller host |
| `--serverPort` | `5005`      | controller TCP port |
| `--numerology` | `0`         | NR numerology (0 → 1 ms slot, matches the model) |
| `--bandwidth`  | `20e6`      | channel bandwidth (Hz) |
| `--centralFrequency` | `3.5e9` | carrier frequency (Hz) |

## How the PRB budget is enforced

**Mode A (implemented, default).** Each slice's aggregate offered rate is capped to
`budget_i × capacity_per_PRB_i` (bits/slot → bits/s) by reprogramming the slice's
`OnOffApplication` `DataRate` at every epoch boundary. A hard PRB allocation caps a
slice's deliverable rate to exactly this, so throughput and delay reflect the
controller's decision. The OFDMA-QoS scheduler resolves any residual contention by
slice priority. This compiles and runs on **stock** ns-3.46 + 5G-LENA.

**Mode B (optional, scheduler-subclass).** For hard MAC-level PRB enforcement you can
subclass `NrMacSchedulerOfdmaRR` and override `GetUeCompareDlFn()` to order UEs by the
per-slice RL weight, plus cap each slice's assigned RBGs in `AssignedDlResources()`.
Register it with `NS_OBJECT_ENSURE_REGISTERED` and pass its `TypeId` to
`nr->SetSchedulerTypeId(...)`. This is the realization the paper's Evaluation section
refers to; it requires rebuilding against your exact `nr` version and is left as a
documented extension because Mode A already yields faithful per-slice KPIs.

## Wire protocol (NDJSON, one JSON object per line)

```
ns3    → {"type":"hello","role":"ns3","version":1}
server → {"type":"config","slices":["URLLC","eMBB","mMTC"],"floor_prbs":[10,18,6],
          "sla_latency_ms":[2,8,20],"total_prbs":50,"epoch_length":60,"epochs":8,...}
loop:
  server → {"type":"alloc","epoch":k,"prbs":[..],"weights":[..],"reports":[..],
            "cqi":[..],"prb_capacity_bits":[..]}
  ns3    → {"type":"kpi","epoch":k,"throughput_mbps":[..],"mean_delay_ms":[..],
            "sla_violation":[..],"prb_used":[..]}
server → {"type":"done","epochs":8}
```

## Troubleshooting
- *`Could not connect to RL server`*: start `rl_allocation_server.py` **before** the ns-3 run.
- *ns-3 hangs at epoch 0*: the server sends `alloc` only after receiving `hello`; check the port matches.
- *Throughput looks saturated*: raise `--bandwidth` or lower `--load`; the cell is congested (expected at load > 1).
