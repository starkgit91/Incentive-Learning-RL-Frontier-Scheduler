from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class SliceSpec:
    name: str
    priority: float
    sla_latency_ms: float
    floor_prbs: int
    base_mbps: float
    burst_probability: float
    burst_multiplier: float
    snr_mean_db: float
    snr_std_db: float


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for the local 5G NR slicing simulation.

    The numerical values are intentionally modest so the full rho-vs-L sweep can
    run on a laptop while preserving the mechanisms used in the paper draft.
    """

    total_prbs: int = 50
    slot_ms: float = 1.0
    prb_bandwidth_hz: float = 180_000.0
    phy_overhead: float = 0.82
    theta_min: float = 0.05
    theta_max: float = 60.0
    seed: int = 42
    output_dir: Path = Path("outputs/gtmd_frontier")

    # CQI 1..15 spectral efficiencies, close to common LTE/NR MCS tables.
    cqi_efficiency: Tuple[float, ...] = (
        0.1523,
        0.2344,
        0.3770,
        0.6016,
        0.8770,
        1.1758,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,
    )
    cqi_required_snr_db: Tuple[float, ...] = (
        -6.7,
        -4.7,
        -2.3,
        0.2,
        2.4,
        4.3,
        5.9,
        8.1,
        10.3,
        11.7,
        14.1,
        16.3,
        18.7,
        21.0,
        22.7,
    )

    slices: Tuple[SliceSpec, ...] = field(
        default_factory=lambda: (
            SliceSpec(
                name="URLLC",
                priority=5.0,
                sla_latency_ms=2.0,
                floor_prbs=10,
                base_mbps=1.8,
                burst_probability=0.08,
                burst_multiplier=5.0,
                snr_mean_db=17.0,
                snr_std_db=2.5,
            ),
            SliceSpec(
                name="eMBB",
                priority=1.3,
                sla_latency_ms=8.0,
                floor_prbs=18,
                base_mbps=11.0,
                burst_probability=0.03,
                burst_multiplier=2.2,
                snr_mean_db=19.0,
                snr_std_db=3.0,
            ),
            SliceSpec(
                name="mMTC",
                priority=0.7,
                sla_latency_ms=20.0,
                floor_prbs=6,
                base_mbps=1.2,
                burst_probability=0.04,
                burst_multiplier=12.0,
                snr_mean_db=12.0,
                snr_std_db=4.0,
            ),
        )
    )

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def slot_s(self) -> float:
        return self.slot_ms / 1000.0

    @property
    def floor_prbs(self) -> Tuple[int, ...]:
        return tuple(s.floor_prbs for s in self.slices)

    @property
    def priorities(self) -> Tuple[float, ...]:
        return tuple(s.priority for s in self.slices)

    @property
    def sla_latency_ms(self) -> Tuple[float, ...]:
        return tuple(s.sla_latency_ms for s in self.slices)


def default_config() -> SimulationConfig:
    return SimulationConfig()
