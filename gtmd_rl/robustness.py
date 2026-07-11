"""Truthful vs. misreported robustness: the DSIC+RL mechanism against classical
schedulers, under honest and strategically-inflated demand reports.

The point of the mechanism, stated operationally. DSIC means truthful reporting is
a dominant strategy: inflating your demand report cannot raise your utility,
because the Myerson critical-value payment prices the extra PRBs away. Classical
schedulers (round-robin, max-CQI, proportional-fair) charge nothing for demand, so
a tenant that inflates its report simply captures more PRBs for free -- its own
service rises (a positive *manipulation gain*) while the honest slices lose PRBs
and take SLA hits. We quantify both sides for every scheduler on common random
numbers, so truthful and misreported runs differ ONLY in the strategic tenant's
report, never in the traffic/channel sample path.

Two experiments:

* **Over epochs (one scheduler, ours):** truthful vs.\ a fixed aggressive misreport,
  showing (i) the RL system reward converging, (ii) the honest-slice SLA staying put
  under misreporting (robustness), and (iii) the strategic tenant's cumulative gain
  hugging zero (DSIC).

* **Across schedulers:** for each scheduler, the best-response manipulation gain and
  the harm it inflicts on the honest slices, truthful vs.\ misreported. The headline:
  our gain is ~0 where every non-priced scheduler -- including our OWN allocator with
  the payment switched off (``GTMD-noPay``) -- is gameable, which isolates the payment
  (not the allocation rule) as the truthfulness lever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .baselines import build_baseline_schedulers
from .config import SimulationConfig, default_config
from .frontier_eval import draw_epoch_type, reference_theta_base
from .mechanism import critical_value_payment, weighted_greedy_allocator
from .network import NRTraceGenerator, NetworkState
from .rl import (
    BayesianDemandEstimator,
    EpochFrozenQLearner,
    EpochMetrics,
    network_reward,
)

# The strategic tenant (eMBB): large, elastic demand -- the natural manipulator.
STRATEGIC_TENANT = 1
# Best-response search grid. Under-report (0.85) and a range of over-reports; the
# priced mechanism should prefer ~1.0, the non-priced schedulers a >1 inflation.
DEFAULT_MULTIPLIERS: Tuple[float, ...] = (0.85, 1.0, 1.25, 1.5, 2.0, 3.0)
# Scheduler ids. GTMD-noPay and GTMD-RL come from ONE run (same allocation; the only
# difference is whether the Myerson payment is subtracted), which is the ablation.
BASELINES = ("RoundRobin+Floors", "MaxCQI+Floors", "ProportionalFair+Floors")
GTMD_KINDS = ("GTMD-noPay", "GTMD-RL")


def _report_state(state: NetworkState, mult_vec: np.ndarray, config: SimulationConfig) -> NetworkState:
    """The scheduler's *view* of demand: the true per-slice PRB demand scaled by each
    slice's report multiplier (1 for honest slices, ``m`` for the manipulator). The
    true demand is untouched and is what actually gets served -- only the declared
    request the scheduler allocates against is inflated."""
    rep = state.copy()
    rep.demand_prbs = np.clip(state.demand_prbs * mult_vec, 0.0, float(config.total_prbs))
    return rep


@dataclass
class EpisodeKPIs:
    # System (from true serving)
    sla_rate: float
    sla_honest: float          # mean SLA over slices != strategic tenant
    p95_latency_ms: float
    p95_urllc_ms: float        # URLLC (slice 0) tail -- the priority-weighted win
    throughput_mbps: float
    throughput_honest: float    # mean served throughput over slices != strategic tenant
    floor_satisfaction: float
    jain_fairness: float
    wasted_prbs: float
    # Strategic tenant
    tenant_value: float        # sum_t theta_true[t]*min(alloc, true_demand)
    tenant_payment: float      # Myerson payment (0 for non-priced schedulers)
    rho: float
    # Optional per-epoch series (for the over-epoch figure)
    epoch_reward: Optional[np.ndarray] = None
    epoch_sla: Optional[np.ndarray] = None
    epoch_tenant_util: Optional[np.ndarray] = None


def _jain(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = len(x) * float(np.sum(x**2))
    return float(np.sum(x) ** 2) / d if d > 0 else 1.0


def run_episode(
    kind: str,
    config: SimulationConfig,
    load: float,
    epoch_length: int,
    total_slots: int,
    seed: int,
    planner_seed: int,
    misreport_mult: float = 1.0,
    tenant: int = STRATEGIC_TENANT,
    payment_grid: int = 13,
    collect_series: bool = False,
) -> EpisodeKPIs:
    """One CRN episode. ``kind`` is a baseline id, or 'GTMD' for the priced DSIC+RL
    mechanism (which fills both GTMD-noPay and GTMD-RL). The strategic tenant scales
    its report by ``misreport_mult`` every epoch; the type law and all RNG streams
    are a deterministic function of ``seed`` so runs are perfectly paired."""
    env = NRTraceGenerator(config, load=load, seed=seed)
    n_epochs = int(total_slots // epoch_length)
    base_theta = reference_theta_base(config, load)
    floors = np.asarray(config.floor_prbs, dtype=float)
    sla_ms = np.asarray(config.sla_latency_ms, dtype=float)

    is_gtmd = kind == "GTMD"
    if is_gtmd:
        estimator = BayesianDemandEstimator(config)
        planner = EpochFrozenQLearner(config, seed=planner_seed)
    else:
        scheduler = build_baseline_schedulers(with_floors=True)[kind]
        scheduler.reset()

    mult_vec = np.ones(config.n_slices, dtype=float)
    mult_vec[tenant] = float(misreport_mult)

    lat_rows: List[np.ndarray] = []
    sla_rows: List[np.ndarray] = []
    thr_rows: List[np.ndarray] = []
    alloc_rows: List[np.ndarray] = []
    waste_rows: List[np.ndarray] = []
    floor_ok_rows: List[np.ndarray] = []
    tenant_value = 0.0
    tenant_payment = 0.0
    binding_slots = 0
    prev_delay, prev_rho = 0.0, 0.0
    ep_reward, ep_sla, ep_tenant_util = [], [], []
    tenant_util_cum = 0.0

    for epoch in range(n_epochs):
        epoch_theta = draw_epoch_type(config, base_theta, seed, epoch)
        reports = epoch_theta.copy()
        reports[tenant] = float(np.clip(reports[tenant] * misreport_mult,
                                        config.theta_min, config.theta_max))
        if is_gtmd:
            _, weights, _ = planner.select_action(
                estimator, mean_delay_ms=prev_delay, rho_recent=prev_rho, train=True
            )

        states: List[NetworkState] = []
        lat_acc, sla_acc, thr_acc = [], [], []
        obs_samples = np.zeros(config.n_slices, dtype=float)
        ep_rew, bind_any, ep_alloc, ep_waste, ep_floor_ok = 0.0, 0, [], [], []
        ep_value = 0.0

        for _ in range(epoch_length):
            env.theta = epoch_theta.copy()          # pin the type (CRN-safe)
            true_theta = env.theta.copy()
            state = env.current_state()
            cap = np.maximum(env.prb_capacity_bits(), 1.0)
            true_demand = np.ceil(np.clip(state.demand_prbs, 0.0, config.total_prbs))

            if is_gtmd:
                dec = weighted_greedy_allocator(state, reports, weights, config)
                alloc = dec.allocation_prbs
                if np.any(dec.binding > 0):
                    bind_any += 1
                if collect_series or True:
                    states.append(state.copy())
            else:
                rep_state = _report_state(state, mult_vec, config)
                alloc = scheduler.allocate(rep_state, cap, config).astype(float)

            _, result = env.step(alloc)
            lat_acc.append(result.latency_ms)
            sla_acc.append(result.sla_violation)
            thr_acc.append(result.throughput_mbps)
            ep_alloc.append(result.allocation_prbs.astype(float))
            ep_waste.append(result.wasted_prbs)
            need_floor = true_demand >= floors
            floor_ok = np.where(need_floor, result.allocation_prbs >= np.minimum(floors, true_demand), 1.0)
            ep_floor_ok.append(floor_ok.astype(float))
            obs_samples += result.arrivals_bits / cap
            # Valuation: willingness-to-pay for USEFUL PRBs (capped at true demand),
            # so inflating beyond what a slice can use yields no extra value.
            ep_value += float(true_theta[tenant]) * float(min(alloc[tenant], true_demand[tenant]))
            ep_rew += network_reward(config, result.throughput_mbps, result.latency_ms,
                                     result.sla_violation, result.wasted_prbs)

        # Myerson payment (priced mechanism only). Non-priced schedulers pay nothing.
        payment = 0.0
        if is_gtmd:
            payment = critical_value_payment(tenant, reports, weights, states, config, grid_size=payment_grid)
        tenant_value += ep_value
        tenant_payment += payment
        binding_slots += bind_any

        lat_rows.extend(lat_acc); sla_rows.extend(sla_acc); thr_rows.extend(thr_acc)
        alloc_rows.extend(ep_alloc); waste_rows.extend(ep_waste); floor_ok_rows.extend(ep_floor_ok)

        mean_delay = float(np.mean(lat_acc))
        rho_epoch = bind_any / float(epoch_length)
        if is_gtmd:
            estimator.update(reports, observed_mean=obs_samples / epoch_length, n_obs=epoch_length)
            planner.remember(EpochMetrics(ep_rew, rho_epoch, mean_delay, 0.0, 0.0, 0.0))
            planner.update(planner.discretize(estimator, mean_delay, planner.recent_rho()), ep_rew)
        prev_delay, prev_rho = mean_delay, rho_epoch

        if collect_series:
            tenant_util_cum += ep_value - payment
            ep_reward.append(ep_rew)
            ep_sla.append(float(np.mean(sla_acc)))
            ep_tenant_util.append(tenant_util_cum)

    lat = np.vstack(lat_rows); sla = np.vstack(sla_rows); thr = np.vstack(thr_rows)
    alloc = np.vstack(alloc_rows); waste = np.vstack(waste_rows); floor_ok = np.vstack(floor_ok_rows)
    honest = [i for i in range(config.n_slices) if i != tenant]

    return EpisodeKPIs(
        sla_rate=float(sla.mean()),
        sla_honest=float(sla[:, honest].mean()),
        p95_latency_ms=float(np.percentile(lat, 95)),
        p95_urllc_ms=float(np.percentile(lat[:, 0], 95)),
        throughput_mbps=float(thr.sum(axis=1).mean()),
        throughput_honest=float(thr[:, honest].sum(axis=1).mean()),
        floor_satisfaction=float(floor_ok.mean()),
        jain_fairness=_jain(thr.mean(axis=0)),
        wasted_prbs=float(waste.sum(axis=1).mean()),
        tenant_value=tenant_value,
        tenant_payment=tenant_payment,
        rho=binding_slots / float(max(n_epochs * epoch_length, 1)),
        epoch_reward=np.array(ep_reward) if collect_series else None,
        epoch_sla=np.array(ep_sla) if collect_series else None,
        epoch_tenant_util=np.array(ep_tenant_util) if collect_series else None,
    )


# --------------------------------------------------------------------------- #
# Cross-scheduler best-response study
# --------------------------------------------------------------------------- #
@dataclass
class SchedulerRobustness:
    name: str
    truthful: EpisodeKPIs
    best_misreport: EpisodeKPIs
    best_mult: float
    gain_abs: float
    gain_pct: float            # manipulation gain as % of truthful utility
    mults: np.ndarray = field(default_factory=lambda: np.array([]))
    util_curve: np.ndarray = field(default_factory=lambda: np.array([]))  # U(m), normalized to U(1)=1


def _utility(k: EpisodeKPIs) -> float:
    return k.tenant_value - k.tenant_payment


def evaluate_scheduler(
    name: str, config, load, L, T, seed, multipliers=DEFAULT_MULTIPLIERS, payment_grid=13,
    gtmd_cache: Optional[Dict[float, EpisodeKPIs]] = None,
) -> SchedulerRobustness:
    """Truthful KPIs and the strategic tenant's BEST-RESPONSE misreport for one
    scheduler. 'GTMD-noPay' and 'GTMD-RL' are both derived from the single 'GTMD'
    priced run: noPay counts value only, RL counts value minus payment. When both
    are evaluated, pass a shared ``gtmd_cache`` so the expensive priced+payment
    episodes are computed once."""
    planner_seed = seed + 9999
    is_no_pay = name == "GTMD-noPay"
    kind = "GTMD" if name in GTMD_KINDS else name

    def util(k: EpisodeKPIs) -> float:
        return k.tenant_value if is_no_pay else _utility(k)

    # Sweep the full report-multiplier grid once; the truthful point is m=1.
    grid = sorted(set([1.0]) | {float(m) for m in multipliers})
    kpis: Dict[float, EpisodeKPIs] = {}
    for m in grid:
        if kind == "GTMD" and gtmd_cache is not None and m in gtmd_cache:
            kpis[m] = gtmd_cache[m]
            continue
        kpis[m] = run_episode(kind, config, load, L, T, seed, planner_seed,
                              misreport_mult=float(m), payment_grid=payment_grid)
        if kind == "GTMD" and gtmd_cache is not None:
            gtmd_cache[m] = kpis[m]
    truthful = kpis[1.0]
    base_u = util(truthful)
    utils = np.array([util(kpis[m]) for m in grid])
    # Normalize the utility curve to U(truthful)=1 so schedulers are comparable.
    util_norm = utils / max(abs(base_u), 1e-9)
    best_i = int(np.argmax(utils))
    best_m = float(grid[best_i]); best = kpis[grid[best_i]]
    best_gain = float(utils[best_i] - base_u)
    gain_pct = 100.0 * best_gain / max(abs(base_u), 1e-9)
    return SchedulerRobustness(name, truthful, best, best_m, best_gain, gain_pct,
                               mults=np.array(grid), util_curve=util_norm)


@dataclass
class RobustnessResult:
    load: float
    schedulers: List[str]
    rows: pd.DataFrame                 # one row per (scheduler, seed)
    summary: pd.DataFrame              # aggregated per scheduler
    epoch_series: Dict[str, Dict[str, np.ndarray]]  # over-epoch (GTMD) truthful/misreport
    util_curves: Dict[str, np.ndarray]              # scheduler -> normalized U(m) (seed-averaged)
    mults: np.ndarray = field(default_factory=lambda: np.array([]))


def run_robustness(
    config: Optional[SimulationConfig] = None,
    load: float = 1.15,
    epoch_length: int = 60,
    total_slots: int = 1800,
    seeds: Sequence[int] = (0, 1, 2, 3),
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    payment_grid: int = 13,
    over_epoch_mult: float = 1.5,
    verbose: bool = True,
) -> RobustnessResult:
    config = default_config() if config is None else config
    schedulers = list(BASELINES) + list(GTMD_KINDS)
    rows: List[dict] = []
    curve_acc: Dict[str, List[np.ndarray]] = {n: [] for n in schedulers}
    grid_ref: np.ndarray = np.array([])
    for s in seeds:
        seed = 42 + 1000 * int(s) + int(load * 100)
        gtmd_cache: Dict[float, EpisodeKPIs] = {}   # share GTMD episodes across noPay/RL
        for name in schedulers:
            r = evaluate_scheduler(name, config, load, epoch_length, total_slots, seed,
                                   multipliers, payment_grid, gtmd_cache=gtmd_cache)
            # Manipulation harm to honest slices, self-paired (misreport - truthful):
            # served-throughput drop is a non-saturating measure of who gets starved.
            d_honest_thr = r.best_misreport.throughput_honest - r.truthful.throughput_honest
            rows.append({
                "scheduler": name, "seed": int(s), "best_mult": r.best_mult,
                "gain_abs": r.gain_abs, "gain_pct": r.gain_pct,
                "sla_honest_truthful": r.truthful.sla_honest,
                "sla_honest_misreport": r.best_misreport.sla_honest,
                "thr_honest_truthful": r.truthful.throughput_honest,
                "thr_honest_misreport": r.best_misreport.throughput_honest,
                "delta_honest_thr": d_honest_thr,
                "throughput_truthful": r.truthful.throughput_mbps,
                "floor_truthful": r.truthful.floor_satisfaction,
                "jain_truthful": r.truthful.jain_fairness,
            })
            curve_acc[name].append(r.util_curve)
            grid_ref = r.mults
            if verbose:
                print(f"load={load:.2f} seed={s} {name:22s} m*={r.best_mult:.2f} "
                      f"gain={r.gain_pct:6.2f}%  honest_thr "
                      f"{r.truthful.throughput_honest:.2f}->{r.best_misreport.throughput_honest:.2f}",
                      flush=True)

    rows_df = pd.DataFrame(rows)
    summary = rows_df.groupby("scheduler", as_index=False).mean(numeric_only=True)
    util_curves = {n: (np.mean(np.vstack(curve_acc[n]), axis=0) if curve_acc[n] else np.array([]))
                   for n in schedulers}

    # Over-epoch series for the DSIC+RL mechanism (truthful vs one aggressive misreport).
    seed0 = 42 + int(load * 100)
    truth = run_episode("GTMD", config, load, epoch_length, total_slots, seed0, seed0 + 9999,
                        misreport_mult=1.0, payment_grid=payment_grid, collect_series=True)
    mis = run_episode("GTMD", config, load, epoch_length, total_slots, seed0, seed0 + 9999,
                      misreport_mult=over_epoch_mult, payment_grid=payment_grid, collect_series=True)
    epoch_series = {
        "truthful": {"reward": truth.epoch_reward, "sla": truth.epoch_sla, "util": truth.epoch_tenant_util},
        "misreport": {"reward": mis.epoch_reward, "sla": mis.epoch_sla, "util": mis.epoch_tenant_util},
        "mult": np.array([over_epoch_mult]),
    }
    return RobustnessResult(load, schedulers, rows_df, summary, epoch_series,
                            util_curves=util_curves, mults=grid_ref)


def run_gain_vs_load(
    config: Optional[SimulationConfig] = None,
    loads: Sequence[float] = (0.7, 0.85, 1.0, 1.15, 1.3),
    epoch_length: int = 60,
    total_slots: int = 1800,
    seeds: Sequence[int] = (0, 1, 2),
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    payment_grid: int = 13,
    verbose: bool = True,
) -> pd.DataFrame:
    """Best-response manipulation gain (%) vs offered load, per scheduler. The DSIC
    mechanism should stay flat at ~0 across loads; the unpriced allocator's gain
    should climb with contention."""
    config = default_config() if config is None else config
    schedulers = list(BASELINES) + list(GTMD_KINDS)
    rows: List[dict] = []
    for load in loads:
        for s in seeds:
            seed = 42 + 1000 * int(s) + int(load * 100)
            gtmd_cache: Dict[float, EpisodeKPIs] = {}
            for name in schedulers:
                r = evaluate_scheduler(name, config, float(load), epoch_length, total_slots,
                                       seed, multipliers, payment_grid, gtmd_cache=gtmd_cache)
                rows.append({"scheduler": name, "load": float(load), "seed": int(s),
                             "gain_pct": r.gain_pct, "best_mult": r.best_mult})
                if verbose:
                    print(f"load={load:.2f} seed={s} {name:22s} gain={r.gain_pct:6.2f}% "
                          f"m*={r.best_mult:.2f}", flush=True)
    return pd.DataFrame(rows)
