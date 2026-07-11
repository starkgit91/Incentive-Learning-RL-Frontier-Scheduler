"""Wire protocol shared by the Python RL+DSIC server and the ns-3 5G-LENA client.

Transport: a single TCP connection. Messages are newline-delimited JSON (NDJSON),
i.e. one compact JSON object per line terminated by ``\\n``. This is trivial to
parse from both Python (``socket`` / ``json``) and C++ (read until ``\\n``,
``nlohmann/json`` or a tiny hand parser), which is what makes it a clean Tx/Rx
bridge without any middleware such as ns3-ai.

Message flow (server = RL+DSIC controller, client = ns-3 gNB/RIC scenario):

    client -> HELLO       announce role + capabilities
    server -> CONFIG      slice specs (floors, SLA, priorities, PRB bandwidth)
    loop over epochs:
        server -> ALLOC   per-slice PRB budget + weights for the next epoch
        client -> KPI     measured per-slice throughput/delay/SLA for that epoch
    server -> DONE        end of episode

The server consumes the returned KPIs as the RL reward, closing the loop: this is
exactly the near-RT-RIC control loop, with the learned policy in Python and the
3GPP-compliant PHY/MAC in ns-3.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Message type tags.
HELLO = "hello"
CONFIG = "config"
ALLOC = "alloc"
KPI = "kpi"
DONE = "done"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005


def encode(msg: Dict[str, Any]) -> bytes:
    """Serialise one message to a single NDJSON line (compact, newline-terminated)."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


class NDJSONChannel:
    """Buffered newline-delimited-JSON reader/writer over a connected socket."""

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self._buf = b""

    def send(self, msg: Dict[str, Any]) -> None:
        self.conn.sendall(encode(msg))

    def recv(self) -> Optional[Dict[str, Any]]:
        """Return the next message, or ``None`` if the peer closed the connection."""
        while b"\n" not in self._buf:
            chunk = self.conn.recv(65536)
            if not chunk:
                if self._buf.strip():
                    line, self._buf = self._buf, b""
                    return json.loads(line.decode("utf-8"))
                return None
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        line = line.strip()
        if not line:
            return self.recv()
        return json.loads(line.decode("utf-8"))

    def close(self) -> None:
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()


@dataclass
class AllocMessage:
    epoch: int
    prbs: List[int]           # per-slice integer PRB budget, sums to <= total_prbs
    weights: List[float]      # per-slice RL weights (allocation priority)
    reports: List[float]      # per-slice DSIC demand reports (PRB-intensity)
    cqi: List[float]          # per-slice CQI used to size PRB capacity
    prb_capacity_bits: List[float]  # bits carried by one PRB this epoch, per slice

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": ALLOC,
            "epoch": self.epoch,
            "prbs": [int(x) for x in self.prbs],
            "weights": [float(x) for x in self.weights],
            "reports": [float(x) for x in self.reports],
            "cqi": [float(x) for x in self.cqi],
            "prb_capacity_bits": [float(x) for x in self.prb_capacity_bits],
        }


@dataclass
class KpiMessage:
    epoch: int
    throughput_mbps: List[float]
    mean_delay_ms: List[float]
    sla_violation: List[float]
    prb_used: List[float]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KpiMessage":
        return KpiMessage(
            epoch=int(d["epoch"]),
            throughput_mbps=[float(x) for x in d["throughput_mbps"]],
            mean_delay_ms=[float(x) for x in d["mean_delay_ms"]],
            sla_violation=[float(x) for x in d["sla_violation"]],
            prb_used=[float(x) for x in d.get("prb_used", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": KPI,
            "epoch": self.epoch,
            "throughput_mbps": self.throughput_mbps,
            "mean_delay_ms": self.mean_delay_ms,
            "sla_violation": self.sla_violation,
            "prb_used": self.prb_used,
        }
