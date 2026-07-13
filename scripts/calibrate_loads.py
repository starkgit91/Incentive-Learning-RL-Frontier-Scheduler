#!/usr/bin/env python3
"""Instance B load calibration: bisect the floor f1 to hit a target binding
frequency rho, and cache the (f1, rho) pairs.

rho is the LOAD KNOB of the paper: it is the fraction of slots on which some floor is
allocation-limited, i.e. the fraction of slots on which the floors open a *resource*
channel on top of the payment channel (Remark 2). It must be measured on a TRUTHFUL
run with RULE_W -- RULE_P can never bind a floor at all, which is why every rho sweep
uses the waterfilling rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustgate.gates import CUM0                       # noqa: E402
from trustgate.instances import draw_channel, instance_b  # noqa: E402
from trustgate.mechanism import run, truthful          # noqa: E402
from trustgate.traffic import draw_traffic             # noqa: E402

OUT = ROOT / "configs"
OUT.mkdir(exist_ok=True)


def measure_rho(f1: float, T: int = 20000, L: int = 50, sigma: float = 0.2,
                seeds=(0, 1, 2)) -> float:
    inst = instance_b(f1=f1)
    rhos = []
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        A = draw_traffic(inst, sigma, rng, T)
        q = draw_channel(inst, rng, T)
        r = run(inst, CUM0, A, q, L, sigma, truthful(inst), compute_payments=False)
        rhos.append(r.rho)
    return float(np.mean(rhos))


def bisect_f1(target: float, lo: float = 0.0, hi: float = 4.0, iters: int = 14) -> float:
    if target <= 0.0:
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if measure_rho(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("== Instance B load calibration (RULE_W, truthful) ==")
    print("  f1 grid probe:")
    for f1 in (0.0, 1.0, 1.5, 2.0, 2.5):
        print(f"    f1={f1:.2f} -> rho={measure_rho(f1):.4f}", flush=True)

    loads = {}
    for target in (0.0, 0.05, 0.10, 0.20):
        f1 = bisect_f1(target)
        rho = measure_rho(f1)
        loads[f"{target:.2f}"] = dict(target_rho=target, f1=round(f1, 5), rho=round(rho, 5))
        print(f"  target rho={target:.2f} -> f1={f1:.4f}  (measured rho={rho:.4f})", flush=True)

    (OUT / "loads.json").write_text(json.dumps(loads, indent=2))
    print(f"wrote {OUT/'loads.json'}")
