from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .adversary import TabularReportAdversary
from .config import SimulationConfig, default_config
from .mechanism import epoch_payments, weighted_greedy_allocator
from .network import NRTraceGenerator, NetworkState, SlotResult
from .rl import BayesianDemandEstimator, EpochFrozenQLearner, EpochMetrics, network_reward


@dataclass
class EpisodeResult:
    summary: Dict[str, float]
    epochs: pd.DataFrame
    slots: pd.DataFrame


def _tenant_epoch_utility(
    tenant: int,
    true_theta: np.ndarray,
    allocations: list[np.ndarray],
    payment: float,
) -> float:
    alloc_sum = float(np.sum([a[tenant] for a in allocations]))
    theta_value = float(np.mean(true_theta[:, tenant]))
    return theta_value * alloc_sum - float(payment)


def run_episode(
    config: SimulationConfig,
    load: float,
    epoch_length: int,
    total_slots: int,
    seed: int,
    adversary: Optional[TabularReportAdversary] = None,
    train_planner: bool = True,
    train_adversary: bool = False,
    payment_grid: int = 17,
    collect_slots: bool = False,
) -> EpisodeResult:
    env = NRTraceGenerator(config, load=load, seed=seed)
    estimator = BayesianDemandEstimator(config)
    planner = EpochFrozenQLearner(config, seed=seed + 1000)
    n_epochs = int(total_slots // epoch_length)
    tenant_utility = np.zeros(config.n_slices, dtype=float)
    epoch_rows: list[dict] = []
    slot_rows: list[dict] = []

    previous_delay = 0.0
    previous_rho = 0.0
    previous_action = 0

    for epoch in range(n_epochs):
        true_theta = env.begin_epoch()
        action_id, weights, planner_state = planner.select_action(
            estimator,
            mean_delay_ms=previous_delay,
            rho_recent=previous_rho,
            train=train_planner,
        )

        reports = true_theta.copy()
        report_multiplier = 1.0
        if adversary is not None:
            report_multiplier = adversary.choose_multiplier(
                action_id,
                true_theta[adversary.tenant_id],
                previous_rho,
                train=train_adversary,
            )
            reports[adversary.tenant_id] = np.clip(
                reports[adversary.tenant_id] * report_multiplier,
                config.theta_min,
                config.theta_max,
            )

        states: list[NetworkState] = []
        allocations: list[np.ndarray] = []
        bindings: list[np.ndarray] = []
        slot_results: list[SlotResult] = []
        rewards: list[float] = []
        theta_trace: list[np.ndarray] = []

        for offset in range(epoch_length):
            slot = epoch * epoch_length + offset
            slot_theta = env.theta.copy()
            state = env.current_state()
            decision = weighted_greedy_allocator(state, reports, weights, config)
            next_state, result = env.step(decision.allocation_prbs)
            reward = network_reward(
                config,
                result.throughput_mbps,
                result.latency_ms,
                result.sla_violation,
                result.wasted_prbs,
            )

            states.append(state.copy())
            allocations.append(decision.allocation_prbs.copy())
            bindings.append(decision.binding.copy())
            slot_results.append(result)
            rewards.append(reward)
            theta_trace.append(slot_theta.copy())

            if collect_slots:
                for i, spec in enumerate(config.slices):
                    slot_rows.append(
                        {
                            "slot": slot,
                            "epoch": epoch,
                            "slice": spec.name,
                            "tenant": i,
                            "load": load,
                            "L": epoch_length,
                            "theta": slot_theta[i],
                            "report": reports[i],
                            "report_multiplier": report_multiplier if adversary and i == adversary.tenant_id else 1.0,
                            "weight": weights[i],
                            "allocation_prbs": result.allocation_prbs[i],
                            "throughput_mbps": result.throughput_mbps[i],
                            "latency_ms": result.latency_ms[i],
                            "cqi": state.cqi[i],
                            "snr_db": state.snr_db[i],
                            "ber": state.ber[i],
                            "demand_prbs": state.demand_prbs[i],
                            "binding": int(decision.binding[i]),
                            "sla_violation": int(result.sla_violation[i]),
                            "wasted_prbs": result.wasted_prbs[i],
                        }
                    )

        payments = epoch_payments(reports, weights, states, config, grid_size=payment_grid)
        theta_matrix = np.vstack(theta_trace)
        for tenant in range(config.n_slices):
            tenant_utility[tenant] += _tenant_epoch_utility(
                tenant,
                theta_matrix,
                allocations,
                payments[tenant],
            )

        binding_matrix = np.vstack(bindings)
        latency_matrix = np.vstack([r.latency_ms for r in slot_results])
        throughput_matrix = np.vstack([r.throughput_mbps for r in slot_results])
        sla_matrix = np.vstack([r.sla_violation for r in slot_results])
        waste_matrix = np.vstack([r.wasted_prbs for r in slot_results])
        alloc_matrix = np.vstack(allocations)

        epoch_reward = float(np.sum(rewards))
        rho = float(np.any(binding_matrix > 0, axis=1).mean())
        mean_delay = float(latency_matrix.mean())
        throughput = float(throughput_matrix.sum(axis=1).mean())
        sla_rate = float(sla_matrix.mean())
        wasted = float(waste_matrix.sum(axis=1).mean())
        metrics = EpochMetrics(
            reward=epoch_reward,
            rho=rho,
            mean_delay_ms=mean_delay,
            sla_violation_rate=sla_rate,
            throughput_mbps=throughput,
            wasted_prbs=wasted,
        )
        planner.remember(metrics)
        estimator.update(reports)
        next_state_key = planner.discretize(estimator, mean_delay, planner.recent_rho())
        if train_planner:
            planner.update(next_state_key, epoch_reward)
        if adversary is not None and train_adversary:
            adv_util = _tenant_epoch_utility(
                adversary.tenant_id,
                theta_matrix,
                allocations,
                payments[adversary.tenant_id],
            )
            adversary.update(adv_util, action_id, true_theta[adversary.tenant_id], rho)

        previous_delay = mean_delay
        previous_rho = planner.recent_rho()
        previous_action = action_id

        row = {
            "epoch": epoch,
            "load": load,
            "L": epoch_length,
            "reward": epoch_reward,
            "rho": rho,
            "mean_delay_ms": mean_delay,
            "sla_violation_rate": sla_rate,
            "throughput_mbps": throughput,
            "wasted_prbs": wasted,
            "planner_action": action_id,
            "report_multiplier": report_multiplier,
            "payment_sum": float(np.sum(payments)),
        }
        for i, spec in enumerate(config.slices):
            row[f"theta_{spec.name}"] = true_theta[i]
            row[f"report_{spec.name}"] = reports[i]
            row[f"weight_{spec.name}"] = weights[i]
            row[f"alloc_{spec.name}"] = float(alloc_matrix[:, i].mean())
            row[f"payment_{spec.name}"] = payments[i]
            row[f"utility_{spec.name}"] = tenant_utility[i]
        epoch_rows.append(row)

    epochs = pd.DataFrame(epoch_rows)
    slots = pd.DataFrame(slot_rows)
    summary = {
        "load": float(load),
        "L": int(epoch_length),
        "total_slots": int(total_slots),
        "rho_hat": float(epochs["rho"].mean()),
        "reward_total": float(epochs["reward"].sum()),
        "reward_per_slot": float(epochs["reward"].sum() / max(total_slots, 1)),
        "mean_delay_ms": float(epochs["mean_delay_ms"].mean()),
        "sla_violation_rate": float(epochs["sla_violation_rate"].mean()),
        "throughput_mbps": float(epochs["throughput_mbps"].mean()),
        "wasted_prbs": float(epochs["wasted_prbs"].mean()),
    }
    for i, spec in enumerate(config.slices):
        summary[f"utility_{spec.name}"] = float(tenant_utility[i])
    return EpisodeResult(summary=summary, epochs=epochs, slots=slots)


def train_report_adversary(
    config: SimulationConfig,
    load: float,
    epoch_length: int,
    total_slots: int,
    seed: int,
    tenant_id: int = 1,
    episodes: int = 4,
) -> TabularReportAdversary:
    adversary = TabularReportAdversary(tenant_id=tenant_id, seed=seed + 3000)
    for ep in range(episodes):
        run_episode(
            config=config,
            load=load,
            epoch_length=epoch_length,
            total_slots=total_slots,
            seed=seed + ep * 97,
            adversary=adversary,
            train_planner=True,
            train_adversary=True,
            collect_slots=False,
        )
        adversary.epsilon = max(0.03, adversary.epsilon * 0.72)
    return adversary.greedy()


def run_frontier_sweep(
    config: SimulationConfig | None = None,
    loads: Iterable[float] = (0.45, 0.70, 0.95),
    epoch_lengths: Iterable[int] = (30, 60, 120, 240),
    total_slots: int = 2400,
    seed: int = 42,
    adversary_train_episodes: int = 3,
    collect_sample_slots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = default_config() if config is None else config
    rows: list[dict] = []
    all_epochs: list[pd.DataFrame] = []
    sample_slots: list[pd.DataFrame] = []

    for load in loads:
        for L in epoch_lengths:
            base_seed = seed + int(load * 1000) + int(L * 13)
            truthful = run_episode(
                config=config,
                load=float(load),
                epoch_length=int(L),
                total_slots=total_slots,
                seed=base_seed,
                adversary=None,
                train_planner=True,
                collect_slots=collect_sample_slots and len(sample_slots) == 0,
            )
            adversary = train_report_adversary(
                config=config,
                load=float(load),
                epoch_length=int(L),
                total_slots=max(total_slots // 2, int(L) * 6),
                seed=base_seed + 700,
                tenant_id=1,
                episodes=adversary_train_episodes,
            )
            strategic = run_episode(
                config=config,
                load=float(load),
                epoch_length=int(L),
                total_slots=total_slots,
                seed=base_seed,
                adversary=adversary,
                train_planner=True,
                train_adversary=False,
                collect_slots=False,
            )

            tenant = config.slices[1].name
            utility_truth = truthful.summary[f"utility_{tenant}"]
            utility_strategic = strategic.summary[f"utility_{tenant}"]
            raw_gain = max(0.0, utility_strategic - utility_truth)
            rho = truthful.summary["rho_hat"]
            strategic_rho = strategic.summary["rho_hat"]
            # The theorem's slack is the active-floor residual, not every raw
            # utility difference caused by changing the learned policy class.
            # Gate the measured strategic gain by binding mass to estimate that
            # localized residual and also expose raw gain separately.
            localized_slack = raw_gain * max(rho, strategic_rho)
            theory_scale = rho * total_slots / max(int(L), 1)
            row = {
                "load": float(load),
                "L": int(L),
                "rho_hat": rho,
                "strategic_rho_hat": strategic_rho,
                "truthful_reward_per_slot": truthful.summary["reward_per_slot"],
                "strategic_reward_per_slot": strategic.summary["reward_per_slot"],
                "truthful_utility": utility_truth,
                "strategic_utility": utility_strategic,
                "raw_strategic_gain": raw_gain,
                "ic_slack": localized_slack,
                "theory_rho_T_over_L": theory_scale,
                "mean_delay_ms": truthful.summary["mean_delay_ms"],
                "sla_violation_rate": truthful.summary["sla_violation_rate"],
                "throughput_mbps": truthful.summary["throughput_mbps"],
                "wasted_prbs": truthful.summary["wasted_prbs"],
            }
            rows.append(row)

            te = truthful.epochs.copy()
            te["scenario"] = "truthful"
            se = strategic.epochs.copy()
            se["scenario"] = "strategic"
            all_epochs.extend([te, se])
            if collect_sample_slots and truthful.slots is not None and not truthful.slots.empty:
                sample_slots.append(truthful.slots)

    sweep = pd.DataFrame(rows)
    epochs = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()
    slots = pd.concat(sample_slots, ignore_index=True) if sample_slots else pd.DataFrame()
    return sweep, epochs, slots
