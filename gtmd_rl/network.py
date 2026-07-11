from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Dict, Tuple

import numpy as np

from .config import SimulationConfig


@dataclass
class NetworkState:
    queue_bits: np.ndarray
    demand_prbs: np.ndarray
    cqi: np.ndarray
    snr_db: np.ndarray
    ber: np.ndarray
    latency_ms: np.ndarray
    throughput_mbps: np.ndarray

    def copy(self) -> "NetworkState":
        return NetworkState(
            queue_bits=self.queue_bits.copy(),
            demand_prbs=self.demand_prbs.copy(),
            cqi=self.cqi.copy(),
            snr_db=self.snr_db.copy(),
            ber=self.ber.copy(),
            latency_ms=self.latency_ms.copy(),
            throughput_mbps=self.throughput_mbps.copy(),
        )


@dataclass
class SlotResult:
    allocation_prbs: np.ndarray
    arrivals_bits: np.ndarray
    served_bits: np.ndarray
    throughput_mbps: np.ndarray
    latency_ms: np.ndarray
    sla_violation: np.ndarray
    floor_violation: np.ndarray
    wasted_prbs: np.ndarray


class NRTraceGenerator:
    """Synthetic 5G NR slice environment.

    It models the variables needed by the paper experiments: PRBs, throughput,
    latency, CQI, SNR, BER, buffer demand, service floors, and per-slice traffic.
    The simulator is not a replacement for 5G-LENA; it is a fast local
    implementation of the same control problem for mechanism/RL development.
    """

    def __init__(self, config: SimulationConfig, load: float, seed: int | None = None):
        self.config = config
        self.load = float(load)
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.reset()

    def reset(self) -> NetworkState:
        n = self.config.n_slices
        self.queue_bits = np.zeros(n, dtype=float)
        self.throughput_ewma = np.zeros(n, dtype=float)
        self.latency_ms = np.zeros(n, dtype=float)
        self.snr_db = np.array(
            [
                self.rng.normal(s.snr_mean_db, s.snr_std_db)
                for s in self.config.slices
            ],
            dtype=float,
        )
        self._update_link_from_snr()
        self.theta = self._target_theta()
        return self.current_state()

    def _update_link_from_snr(self) -> None:
        req = np.asarray(self.config.cqi_required_snr_db, dtype=float)
        eff = np.asarray(self.config.cqi_efficiency, dtype=float)
        cqi = np.searchsorted(req, self.snr_db, side="right")
        self.cqi = np.clip(cqi, 1, 15).astype(float)
        self.spectral_efficiency = eff[self.cqi.astype(int) - 1]
        snr_linear = np.power(10.0, self.snr_db / 10.0)
        self.ber = np.array([0.5 * erfc(sqrt(max(x, 1e-9))) for x in snr_linear])

    def _advance_channel(self) -> None:
        means = np.array([s.snr_mean_db for s in self.config.slices], dtype=float)
        stds = np.array([s.snr_std_db for s in self.config.slices], dtype=float)
        innovation = self.rng.normal(0.0, stds * 0.18)
        self.snr_db = 0.94 * self.snr_db + 0.06 * means + innovation
        self.snr_db = np.clip(self.snr_db, -8.0, 28.0)
        self._update_link_from_snr()

    def prb_capacity_bits(self) -> np.ndarray:
        return (
            self.config.prb_bandwidth_hz
            * self.config.slot_s
            * self.config.phy_overhead
            * self.spectral_efficiency
        )

    def _target_theta(self) -> np.ndarray:
        cap = np.maximum(self.prb_capacity_bits(), 1.0)
        theta = np.zeros(self.config.n_slices, dtype=float)
        for i, spec in enumerate(self.config.slices):
            mean_bits = spec.base_mbps * 1e6 * self.config.slot_s * self.load
            theta[i] = mean_bits / cap[i]
        return np.clip(theta, self.config.theta_min, self.config.theta_max)

    def _advance_traffic(self) -> None:
        target = self._target_theta()
        noise = self.rng.normal(0.0, 0.05 * np.maximum(target, 1.0))
        self.theta = 0.965 * self.theta + 0.035 * target + noise
        for i, spec in enumerate(self.config.slices):
            if self.rng.random() < spec.burst_probability * 0.08:
                self.theta[i] += self.rng.uniform(0.3, spec.burst_multiplier) * target[i]
        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)

    def begin_epoch(self) -> np.ndarray:
        """Return private demand-intensity types reported for one epoch.

        A type is measured in PRBs per slot. Tenants know these intensities;
        the controller only sees reported versions of them.
        """

        return self.theta.copy()

    def current_state(self, theta: np.ndarray | None = None) -> NetworkState:
        if theta is None:
            theta = self.theta
        cap = np.maximum(self.prb_capacity_bits(), 1.0)
        expected_bits = np.asarray(theta, dtype=float) * cap
        demand = (self.queue_bits + expected_bits) / cap
        return NetworkState(
            queue_bits=self.queue_bits.copy(),
            demand_prbs=np.clip(demand, 0.0, self.config.total_prbs).astype(float),
            cqi=self.cqi.copy(),
            snr_db=self.snr_db.copy(),
            ber=self.ber.copy(),
            latency_ms=self.latency_ms.copy(),
            throughput_mbps=self.throughput_ewma.copy(),
        )

    def sample_arrivals(self, theta: np.ndarray | None = None) -> np.ndarray:
        if theta is None:
            theta = self.theta
        cap = np.maximum(self.prb_capacity_bits(), 1.0)
        arrivals = np.zeros(self.config.n_slices, dtype=float)
        for i, spec in enumerate(self.config.slices):
            mean_bits = max(theta[i] * cap[i], 0.0)
            if spec.name == "URLLC":
                burst = 0.0
                if self.rng.random() < spec.burst_probability:
                    burst = self.rng.uniform(1.5, spec.burst_multiplier) * mean_bits
                arrivals[i] = self.rng.gamma(shape=2.0, scale=max(mean_bits, 1.0) / 2.0) + burst
            elif spec.name == "eMBB":
                arrivals[i] = max(0.0, self.rng.normal(mean_bits, 0.18 * mean_bits + 1.0))
            else:
                spike = 0.0
                if self.rng.random() < spec.burst_probability:
                    spike = self.rng.uniform(3.0, spec.burst_multiplier) * mean_bits
                arrivals[i] = self.rng.exponential(max(mean_bits, 1.0) * 0.65) + spike
        return arrivals

    def evaluate_allocation(
        self,
        state: NetworkState,
        allocation_prbs: np.ndarray,
        arrivals_bits: np.ndarray,
    ) -> SlotResult:
        allocation = np.asarray(allocation_prbs, dtype=float)
        cap_bits = allocation * self.prb_capacity_bits()
        available = state.queue_bits + arrivals_bits
        served = np.minimum(available, cap_bits)
        next_queue = np.maximum(available - served, 0.0)

        throughput = served / self.config.slot_s / 1e6
        service_rate_bps = np.maximum(cap_bits / self.config.slot_s, 1.0)
        latency = np.where(next_queue > 0.0, next_queue / service_rate_bps * 1000.0, 0.0)
        latency = np.clip(latency, 0.0, 500.0)
        sla = latency > np.asarray(self.config.sla_latency_ms, dtype=float)
        floors = np.asarray(self.config.floor_prbs, dtype=float)
        floor_violation = (state.demand_prbs >= floors) & (allocation + 1e-9 < floors)
        wasted = np.maximum(allocation - state.demand_prbs, 0.0)
        return SlotResult(
            allocation_prbs=allocation.astype(int),
            arrivals_bits=arrivals_bits.copy(),
            served_bits=served,
            throughput_mbps=throughput,
            latency_ms=latency,
            sla_violation=sla.astype(int),
            floor_violation=floor_violation.astype(int),
            wasted_prbs=wasted,
        )

    def step(
        self,
        allocation_prbs: np.ndarray,
        theta: np.ndarray | None = None,
    ) -> Tuple[NetworkState, SlotResult]:
        state_before = self.current_state(theta)
        arrivals = self.sample_arrivals(theta)
        result = self.evaluate_allocation(state_before, allocation_prbs, arrivals)

        self.queue_bits = np.maximum(
            state_before.queue_bits + arrivals - result.served_bits,
            0.0,
        )
        self.throughput_ewma = 0.85 * self.throughput_ewma + 0.15 * result.throughput_mbps
        self.latency_ms = result.latency_ms.copy()
        self._advance_traffic()
        self._advance_channel()
        return self.current_state(), result


def summarize_slot_results(results: list[SlotResult]) -> Dict[str, float]:
    if not results:
        return {}
    alloc = np.vstack([r.allocation_prbs for r in results])
    thr = np.vstack([r.throughput_mbps for r in results])
    lat = np.vstack([r.latency_ms for r in results])
    sla = np.vstack([r.sla_violation for r in results])
    waste = np.vstack([r.wasted_prbs for r in results])
    return {
        "mean_alloc_prbs": float(alloc.mean()),
        "sum_throughput_mbps": float(thr.sum(axis=1).mean()),
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "sla_violation_rate": float(sla.mean()),
        "wasted_prb_rate": float(waste.sum(axis=1).mean()),
    }
