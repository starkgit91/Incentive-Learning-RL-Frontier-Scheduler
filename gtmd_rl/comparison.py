"""Apples-to-apples comparison of GTMD-RL against classical PRB schedulers.

Every scheduler is driven on its *own* copy of the environment seeded identically,
so arrivals and channel realisations are the same for all of them and the only
difference is the allocation logic. We report the standard slice-KPI panel plus
fairness and floor-satisfaction, which are the axes on which max-CQI, round-robin,
proportional-fair and our floor-aware RL policy actually differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .baselines import GtmdRLScheduler, Scheduler, build_baseline_schedulers
from .config import SimulationConfig, default_config
from .network import NRTraceGenerator
from .rl import network_reward


def jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    denom = len(x) * float(np.sum(x**2))
    if denom <= 0:
        return 1.0
    return float(np.sum(x) ** 2) / denom


@dataclass
class SchedulerRun:
    name: str
    slot_frame: pd.DataFrame
    metrics: Dict[str, float]


def _warmup_gtmd(
    scheduler: GtmdRLScheduler,
    config: SimulationConfig,
    load: float,
    epoch_length: int,
    total_slots: int,
    seed: int,
    episodes: int = 6,
) -> None:
    """Train the epoch-frozen RL controller before the measured comparison run."""
    for ep in range(episodes):
        env = NRTraceGenerator(config, load=load, seed=seed + 17 * ep)
        scheduler.reset()
        n_epochs = int(total_slots // epoch_length)
        prev_delay, prev_rho = 0.0, 0.0
        for epoch in range(n_epochs):
            reports = env.begin_epoch()
            scheduler.begin_epoch(reports)
            lat_acc, bind_acc, rew_acc = [], [], []
            for _ in range(epoch_length):
                state = env.current_state()
                cap = env.prb_capacity_bits()
                alloc = scheduler.allocate(state, cap, config)
                _, result = env.step(alloc)
                lat_acc.append(result.latency_ms)
                bind_acc.append(result.floor_violation)
                rew_acc.append(
                    network_reward(
                        config,
                        result.throughput_mbps,
                        result.latency_ms,
                        result.sla_violation,
                        result.wasted_prbs,
                    )
                )
            mean_delay = float(np.mean(lat_acc))
            rho = float(np.mean([np.any(b > 0) for b in bind_acc]))
            scheduler.observe_epoch(mean_delay, rho, float(np.sum(rew_acc)))
            prev_delay, prev_rho = mean_delay, rho


def run_single_scheduler(
    scheduler: Scheduler,
    config: SimulationConfig,
    load: float,
    epoch_length: int,
    total_slots: int,
    seed: int,
    collect_slots: bool = False,
) -> SchedulerRun:
    env = NRTraceGenerator(config, load=load, seed=seed)
    scheduler.reset()
    n_epochs = int(total_slots // epoch_length)
    slice_names = [s.name for s in config.slices]
    floors = np.asarray(config.floor_prbs, dtype=float)

    thr_rows: List[np.ndarray] = []
    lat_rows: List[np.ndarray] = []
    sla_rows: List[np.ndarray] = []
    alloc_rows: List[np.ndarray] = []
    waste_rows: List[np.ndarray] = []
    floor_ok_rows: List[np.ndarray] = []
    demand_rows: List[np.ndarray] = []
    slot_records: List[dict] = []

    is_gtmd = isinstance(scheduler, GtmdRLScheduler)

    for epoch in range(n_epochs):
        reports = env.begin_epoch()
        scheduler.begin_epoch(reports)
        for offset in range(epoch_length):
            slot = epoch * epoch_length + offset
            state = env.current_state()
            cap = env.prb_capacity_bits()
            demand = np.ceil(np.clip(state.demand_prbs, 0.0, config.total_prbs))
            alloc = scheduler.allocate(state, cap, config).astype(int)
            _, result = env.step(alloc)

            thr_rows.append(result.throughput_mbps)
            lat_rows.append(result.latency_ms)
            sla_rows.append(result.sla_violation)
            alloc_rows.append(result.allocation_prbs.astype(float))
            waste_rows.append(result.wasted_prbs)
            demand_rows.append(demand)
            need_floor = demand >= floors
            floor_ok = np.where(need_floor, alloc >= np.minimum(floors, demand), 1.0)
            floor_ok_rows.append(floor_ok.astype(float))

            if collect_slots:
                for i, sname in enumerate(slice_names):
                    slot_records.append(
                        {
                            "scheduler": scheduler.name,
                            "slot": slot,
                            "epoch": epoch,
                            "slice": sname,
                            "load": load,
                            "alloc_prbs": int(result.allocation_prbs[i]),
                            "throughput_mbps": float(result.throughput_mbps[i]),
                            "latency_ms": float(result.latency_ms[i]),
                            "cqi": float(state.cqi[i]),
                            "snr_db": float(state.snr_db[i]),
                            "ber": float(state.ber[i]),
                            "demand_prbs": float(state.demand_prbs[i]),
                            "sla_violation": int(result.sla_violation[i]),
                            "floor_ok": float(floor_ok[i]),
                        }
                    )
        if is_gtmd:
            mean_delay = float(np.mean(lat_rows[-epoch_length:]))
            rho = float(np.mean([np.any(f < 1) for f in floor_ok_rows[-epoch_length:]]))
            scheduler.observe_epoch(mean_delay, rho, 0.0)

    thr = np.vstack(thr_rows)
    lat = np.vstack(lat_rows)
    sla = np.vstack(sla_rows)
    alloc = np.vstack(alloc_rows)
    waste = np.vstack(waste_rows)
    floor_ok = np.vstack(floor_ok_rows)

    per_slice_thr = thr.mean(axis=0)
    metrics = {
        "scheduler": scheduler.name,
        "load": float(load),
        "sum_throughput_mbps": float(thr.sum(axis=1).mean()),
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "sla_violation_rate": float(sla.mean()),
        "jain_fairness": jain_index(per_slice_thr),
        "floor_satisfaction": float(floor_ok.mean()),
        "wasted_prbs_per_slot": float(waste.sum(axis=1).mean()),
        "prb_utilization": float(alloc.sum(axis=1).mean() / max(config.total_prbs, 1)),
    }
    for i, sname in enumerate(slice_names):
        metrics[f"thr_{sname}"] = float(per_slice_thr[i])
        metrics[f"p95lat_{sname}"] = float(np.percentile(lat[:, i], 95))
        metrics[f"sla_{sname}"] = float(sla[:, i].mean())

    frame = pd.DataFrame(slot_records) if collect_slots else pd.DataFrame()
    return SchedulerRun(name=scheduler.name, slot_frame=frame, metrics=metrics)


def run_comparison(
    config: Optional[SimulationConfig] = None,
    loads: tuple = (0.7, 1.0, 1.3),
    epoch_length: int = 60,
    total_slots: int = 2400,
    seed: int = 42,
    collect_slots_for_load: Optional[float] = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep several loads; return (metrics_table, sample_slot_trace)."""
    config = default_config() if config is None else config
    rows: List[dict] = []
    slot_frames: List[pd.DataFrame] = []

    for load in loads:
        base_seed = seed + int(load * 1000)
        schedulers: Dict[str, Scheduler] = build_baseline_schedulers(with_floors=True)
        gtmd = GtmdRLScheduler(config, seed=base_seed, train=True, enforce_floors=True)
        _warmup_gtmd(gtmd, config, load, epoch_length, total_slots, base_seed)
        gtmd.train = False  # freeze the learned policy for the measured run
        schedulers[gtmd.name] = gtmd

        for name, sched in schedulers.items():
            collect = collect_slots_for_load is not None and abs(load - collect_slots_for_load) < 1e-9
            run = run_single_scheduler(
                sched, config, load, epoch_length, total_slots, base_seed, collect_slots=collect
            )
            rows.append(run.metrics)
            if collect and not run.slot_frame.empty:
                slot_frames.append(run.slot_frame)

    metrics = pd.DataFrame(rows)
    slots = pd.concat(slot_frames, ignore_index=True) if slot_frames else pd.DataFrame()
    return metrics, slots
