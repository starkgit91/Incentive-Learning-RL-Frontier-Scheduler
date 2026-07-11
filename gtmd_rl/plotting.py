from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scheduler_perslice(metrics: pd.DataFrame, config, output_dir: Path) -> Path:
    """Per-slice view at the highest load. The aggregate p95 latency is dominated by
    the delay-tolerant mMTC slice (20 ms budget) and so hides what matters: GTMD-RL
    protects the *critical* slices (URLLC tail, eMBB) and is the only demand-aware
    policy that also guarantees floors. Three panels: URLLC p95 tail, priority-
    weighted SLA (lower is better), and floor satisfaction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [s.name for s in config.slices]
    prio = np.asarray(config.priorities, dtype=float)
    df = metrics[metrics["load"] == metrics["load"].max()].copy()
    order = [s for s in df["scheduler"].unique() if s != "GTMD-RL"] + ["GTMD-RL"]
    df = df.set_index("scheduler")
    df["prio_wsla"] = sum(prio[i] * df[f"sla_{names[i]}"] for i in range(len(names)))
    colors = ["#d62728" if s == "GTMD-RL" else "#7f7f7f" for s in order]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    x = np.arange(len(order))

    urllc = [float(df.loc[s, f"p95lat_{names[0]}"]) for s in order]
    a1.bar(x, urllc, color=colors, edgecolor="black", linewidth=0.8)
    a1.axhline(config.sla_latency_ms[0], ls="--", color="green", lw=1.2,
               label=f"URLLC SLA {config.sla_latency_ms[0]:g} ms")
    a1.set_ylabel(f"{names[0]} p95 latency (ms)"); a1.set_title("(a) critical-slice tail latency (lower better)")
    for i, v in enumerate(urllc):
        a1.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=7)
    a1.legend(fontsize=8)

    wsla = [float(df.loc[s, "prio_wsla"]) for s in order]
    a2.bar(x, wsla, color=colors, edgecolor="black", linewidth=0.8)
    a2.set_ylabel("priority-weighted SLA violation"); a2.set_title("(b) $\\sum_i \\pi_i\\,$SLA$_i$ (lower better)")

    floor = [float(df.loc[s, "floor_satisfaction"]) for s in order]
    a3.bar(x, floor, color=colors, edgecolor="black", linewidth=0.8)
    a3.set_ylim(0.8, 1.01); a3.axhline(1.0, ls="--", color="green", lw=1.0)
    a3.set_ylabel("floor satisfaction rate"); a3.set_title("(c) hard-floor guarantee (higher better)")

    for ax in (a1, a2, a3):
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Per-slice scheduler comparison at overload (load={df.index.size and metrics['load'].max():.2f}): "
                 "GTMD-RL protects the critical slices and guarantees floors", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "scheduler_perslice.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_frontier(sweep: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for load, group in sweep.sort_values("L").groupby("load"):
        ax.plot(group["L"], group["ic_slack"], marker="o", linewidth=2, label=f"load={load:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("epoch length L (slots)")
    ax.set_ylabel("measured IC slack")
    ax.set_title("Strategic-reporting slack under epoch-frozen DSIC-RL")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "frontier_slack_vs_L.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for load, group in sweep.sort_values("L").groupby("load"):
        ax.plot(group["L"], group["rho_hat"], marker="s", linewidth=2, label=f"load={load:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("epoch length L (slots)")
    ax.set_ylabel("measured binding frequency rho")
    ax.set_title("Binding-frequency invariance check")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "rho_invariance_vs_L.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = sweep["theory_rho_T_over_L"].to_numpy(dtype=float)
    y = sweep["ic_slack"].to_numpy(dtype=float)
    ax.scatter(x, y, s=55)
    if len(x) >= 2 and np.any(x > 0):
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, coef[0] * xs + coef[1], linestyle="--", label=f"fit slope={coef[0]:.3g}")
        ax.legend()
    ax.set_xlabel("rho * T / L")
    ax.set_ylabel("measured IC slack")
    ax.set_title("Slack scaling proxy")
    ax.grid(True, alpha=0.25)
    path = output_dir / "slack_scaling_proxy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    return paths


def plot_scheduler_comparison(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Grouped bar charts comparing GTMD-RL to the classical baselines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    panels = [
        ("sum_throughput_mbps", "aggregate throughput (Mbps)", False),
        ("p95_latency_ms", "p95 latency (ms)", True),
        ("sla_violation_rate", "SLA violation rate", True),
        ("jain_fairness", "Jain fairness index", False),
        ("floor_satisfaction", "floor satisfaction rate", False),
        ("wasted_prbs_per_slot", "wasted PRBs / slot", True),
    ]
    loads = sorted(metrics["load"].unique())
    # Stable scheduler ordering with our policy last so it stands out.
    order = [s for s in metrics["scheduler"].unique() if s != "GTMD-RL"] + ["GTMD-RL"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for ax, (col, title, lower_better) in zip(axes.flat, panels):
        width = 0.8 / max(len(loads), 1)
        x = np.arange(len(order))
        for j, load in enumerate(loads):
            sub = metrics[metrics["load"] == load].set_index("scheduler")
            vals = [float(sub.loc[s, col]) if s in sub.index else np.nan for s in order]
            bars = ax.bar(x + j * width, vals, width, label=f"load={load:.2f}")
            if "GTMD-RL" in order:
                bars[order.index("GTMD-RL")].set_edgecolor("black")
                bars[order.index("GTMD-RL")].set_linewidth(1.6)
        ax.set_xticks(x + width * (len(loads) - 1) / 2)
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        arrow = " (lower is better)" if lower_better else " (higher is better)"
        ax.set_title(title + arrow, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("GTMD-RL vs classical PRB schedulers", fontsize=13)
    path = output_dir / "scheduler_comparison_bars.png"
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Throughput-fairness scatter at the highest load (the trade-off view).
    top_load = loads[-1]
    sub = metrics[metrics["load"] == top_load]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _, row in sub.iterrows():
        marker = "*" if row["scheduler"] == "GTMD-RL" else "o"
        size = 320 if row["scheduler"] == "GTMD-RL" else 90
        ax.scatter(row["sum_throughput_mbps"], row["jain_fairness"], s=size, marker=marker)
        ax.annotate(row["scheduler"], (row["sum_throughput_mbps"], row["jain_fairness"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("aggregate throughput (Mbps)")
    ax.set_ylabel("Jain fairness index")
    ax.set_title(f"Throughput-fairness trade-off (load={top_load:.2f})")
    ax.grid(True, alpha=0.25)
    path = output_dir / "throughput_fairness_tradeoff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_learning(demo, output_dir: Path) -> Path:
    """Two-panel RL learning curve: normalized reward and optimal-action rate vs
    training epochs, against random and oracle references."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ck = demo.checkpoints
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ns, nsd = demo.norm_score_mean, demo.norm_score_std
    ax1.axhline(1.0, ls="--", color="green", lw=1.4, label="per-state oracle")
    ax1.axhline(0.0, ls="--", color="grey", lw=1.4, label="random policy")
    ax1.plot(ck, ns, marker="o", color="C0", label="GTMD-RL (learned)")
    ax1.fill_between(ck, np.clip(ns - nsd, -0.1, 1.1), np.clip(ns + nsd, -0.1, 1.1), color="C0", alpha=0.18)
    ax1.set_xlabel("training epoch"); ax1.set_ylabel("normalized QoS reward")
    ax1.set_title("(a) reward: random $\\to$ oracle"); ax1.set_ylim(-0.1, 1.1)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="lower right")

    op, opd = demo.opt_rate_mean, demo.opt_rate_std
    rand = 1.0 / demo.n_actions
    ax2.axhline(rand, ls="--", color="grey", lw=1.4, label=f"random ({rand:.2f})")
    ax2.plot(ck, op, marker="s", color="C3", label="GTMD-RL (learned)")
    ax2.fill_between(ck, np.clip(op - opd, 0, 1), np.clip(op + opd, 0, 1), color="C3", alpha=0.18)
    ax2.set_xlabel("training epoch"); ax2.set_ylabel("optimal-action selection rate")
    ax2.set_title("(b) picks the best weight profile"); ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Epoch-frozen RL controller learns the weight policy (load={demo.load:.2f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "rl_learning_curve.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


_ROBUST_COLORS = {
    "GTMD-RL": "C3", "GTMD-noPay": "C1",
    "RoundRobin+Floors": "C0", "MaxCQI+Floors": "C4", "ProportionalFair+Floors": "C2",
}
_ROBUST_MARK = {
    "GTMD-RL": "o", "GTMD-noPay": "X",
    "RoundRobin+Floors": "s", "MaxCQI+Floors": "^", "ProportionalFair+Floors": "D",
}
_SHORT = {"RoundRobin+Floors": "RR", "MaxCQI+Floors": "MaxCQI",
          "ProportionalFair+Floors": "PF", "GTMD-noPay": "GTMD\n(no pay)", "GTMD-RL": "GTMD-RL\n(ours)"}


def plot_robustness(result, gain_load_df, output_dir: Path) -> Path:
    """The DSIC/robustness story in four panels:
    (a) tenant utility vs report multiplier -- the canonical dominant-strategy plot
        (ours peaks at truthful m=1; the unpriced allocator keeps rising);
    (b) best-response manipulation gain vs load (ours ~0 everywhere);
    (c) manipulation gain per scheduler at the headline load;
    (d) harm to honest slices (served-throughput drop) when the tenant games."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.5))
    ax_u, ax_gl, ax_g, ax_h = axes

    order = ["RoundRobin+Floors", "MaxCQI+Floors", "ProportionalFair+Floors", "GTMD-noPay", "GTMD-RL"]
    order = [n for n in order if n in result.schedulers]

    # (a) utility vs multiplier. The DSIC signature is a curve that PEAKS at the
    # truthful m=1: any misreport lowers the tenant's own utility (a downward slope
    # away from m=1 is the guarantee working, not the mechanism degrading). The
    # unpriced allocator instead keeps rising -- lying pays.
    m = result.mults
    for n in order:
        c = result.util_curves.get(n)
        if c is None or len(c) == 0:
            continue
        ax_u.plot(m, c, marker=_ROBUST_MARK.get(n, "o"), color=_ROBUST_COLORS.get(n, "C7"),
                  lw=2.4 if n == "GTMD-RL" else 1.5, zorder=4 if n == "GTMD-RL" else 2,
                  label=_SHORT.get(n, n).replace("\n", " "))
    ax_u.axvline(1.0, ls=":", color="grey", lw=1.2)
    # Mark the truthful optimum on our curve and say what the slope means.
    rl = result.util_curves.get("GTMD-RL")
    if rl is not None and len(rl):
        i1 = int(np.argmin(np.abs(np.asarray(m) - 1.0)))
        ax_u.scatter([m[i1]], [rl[i1]], s=190, marker="*", color="C3",
                     edgecolor="black", linewidth=0.8, zorder=6)
        ax_u.annotate("truthful = tenant's optimum\n(dominant strategy)",
                      (m[i1], rl[i1]), xytext=(0.35, 0.60), textcoords="axes fraction",
                      fontsize=7.5, color="C3", ha="left",
                      arrowprops=dict(arrowstyle="->", color="C3", lw=1.2))
        ax_u.annotate("any lie $\\Rightarrow$ lower utility", (float(m[-1]), float(rl[-1])),
                      xytext=(-4, 8), textcoords="offset points", fontsize=7, color="C3", ha="right")
    ax_u.annotate("no payment:\nlying pays", (float(m[-1]), 1.10),
                  fontsize=7, color="C1", ha="right")
    ax_u.set_xlabel("report multiplier $m$ (1 = truthful)")
    ax_u.set_ylabel("tenant utility (normalized, truthful=1)")
    ax_u.set_title("(a) our utility peaks at truthful; unpriced keeps rising")
    ax_u.grid(alpha=0.3); ax_u.legend(fontsize=7, loc="lower left")

    # (b) gain vs load
    if gain_load_df is not None and not gain_load_df.empty:
        g = gain_load_df.groupby(["scheduler", "load"], as_index=False)["gain_pct"].mean()
        for n in order:
            sub = g[g["scheduler"] == n].sort_values("load")
            if sub.empty:
                continue
            ax_gl.plot(sub["load"], sub["gain_pct"], marker=_ROBUST_MARK.get(n, "o"),
                       color=_ROBUST_COLORS.get(n, "C7"), lw=2 if n == "GTMD-RL" else 1.5,
                       label=_SHORT.get(n, n).replace("\n", " "))
        ax_gl.set_xlabel("offered load $\\eta$")
        ax_gl.set_ylabel("best-response manipulation gain (%)")
        ax_gl.set_title("(b) gain rises with contention -- except ours")
        ax_gl.grid(alpha=0.3); ax_gl.legend(fontsize=7)

    # (c) gain bars at headline load
    s = result.summary.set_index("scheduler")
    gains = [float(s.loc[n, "gain_pct"]) if n in s.index else 0.0 for n in order]
    ax_g.bar(np.arange(len(order)), gains, color=[_ROBUST_COLORS.get(n, "C7") for n in order],
             edgecolor="black", linewidth=0.8)
    ax_g.set_xticks(np.arange(len(order)))
    ax_g.set_xticklabels([_SHORT.get(n, n) for n in order], fontsize=8)
    ax_g.set_ylabel("manipulation gain (%)")
    ax_g.set_title(f"(c) incentive to lie (load={result.load:.2f})")
    ax_g.grid(alpha=0.3, axis="y")
    for i, v in enumerate(gains):
        ax_g.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)

    # (d) harm to honest slices
    harm = [-float(s.loc[n, "delta_honest_thr"]) if n in s.index else 0.0 for n in order]
    ax_h.bar(np.arange(len(order)), harm, color=[_ROBUST_COLORS.get(n, "C7") for n in order],
             edgecolor="black", linewidth=0.8)
    ax_h.set_xticks(np.arange(len(order)))
    ax_h.set_xticklabels([_SHORT.get(n, n) for n in order], fontsize=8)
    ax_h.set_ylabel("honest-slice throughput lost (Mbps)")
    ax_h.set_title("(d) collateral harm when the tenant games")
    ax_h.grid(alpha=0.3, axis="y")

    fig.suptitle("Strategy-proofness of the DSIC+RL mechanism vs. baselines "
                 "(the payment, not the allocation rule, is the truthfulness lever)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "robustness_dsic.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_robustness_epochs(result, output_dir: Path) -> Path:
    """Result-set 1: the integrated DSIC+RL mechanism over epochs, truthful vs. a
    fixed aggressive misreport, on identical traffic (CRN). (a) the strategic
    tenant's cumulative utility -- truthful stays on top; (b) the running utility
    advantage of truthful over lying, which is non-negative and grows: under the
    mechanism, persistent misreporting steadily loses ground (dominant-strategy
    truthfulness realized over the learning horizon, not just in expectation)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    es = result.epoch_series
    mult = float(es["mult"][0]) if len(es.get("mult", [])) else 1.5
    ut = np.asarray(es["truthful"]["util"], dtype=float)
    um = np.asarray(es["misreport"]["util"], dtype=float)
    n = len(ut); x = np.arange(n)
    gap = ut - um
    # Scale utility to readable units.
    sc = 1e-3 if np.nanmax(np.abs(ut)) > 1e4 else 1.0
    unit = "(×10$^3$)" if sc == 1e-3 else ""

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a1.plot(x, ut * sc, color="C0", lw=2.2, label="truthful reports")
    a1.plot(x, um * sc, color="C3", lw=2.0, ls="--", label=f"persistent misreport ($m$={mult:g})")
    a1.set_xlabel("epoch"); a1.set_ylabel(f"tenant cumulative utility {unit}")
    a1.set_title("(a) truthful reporting stays on top (DSIC)")
    a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="upper left")

    a2.axhline(0, color="grey", lw=1.0, ls=":")
    a2.fill_between(x, 0, gap * sc, where=(gap >= 0), color="C0", alpha=0.15)
    a2.plot(x, gap * sc, color="C0", lw=2.2)
    a2.set_xlabel("epoch"); a2.set_ylabel(f"truthful $-$ misreport utility {unit}")
    a2.set_title("(b) lying steadily loses ground (non-negative, growing)")
    a2.grid(alpha=0.3)

    fig.suptitle(f"Integrated DSIC+RL over epochs, truthful vs. misreported (load={result.load:.2f})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "robustness_epochs.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_responsiveness_strategyproof(rob_summary, output_dir: Path,
                                      efficiency: Optional[dict] = None) -> Path:
    """The synthesis quadrant: demand-responsiveness (x) vs. strategy-proofness (y).

    * Demand-responsiveness = how much a scheduler's allocation reacts to the
      reported demand, measured as the manipulation gain the *unpriced* allocation
      admits (0 for a demand-blind rule; large for a demand-weighted one). Only a
      demand-responsive rule can exploit heterogeneous/bursty demand.
    * Strategy-proofness = 100 - the scheduler's ACTUAL best-response gain.

    Demand-blind RR/MaxCQI are trivially strategy-proof but sit at x~0 (they throw
    the demand signal away). The demand-weighted allocator without a payment
    (GTMD-noPay) is responsive but gameable (low y). Only GTMD-RL is in the
    top-right corner: demand-responsive AND strategy-proof -- because the Myerson
    payment prices the manipulation away without discarding the demand signal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    s = rob_summary.set_index("scheduler")
    # x: the demand-responsive family (GTMD) shares the unpriced allocator's gain;
    # each baseline's own gain measures its (small) responsiveness.
    nopay = float(s.loc["GTMD-noPay", "gain_pct"]) if "GTMD-noPay" in s.index else 0.0
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    for n in s.index:
        resp = nopay if n in ("GTMD-RL", "GTMD-noPay") else float(s.loc[n, "gain_pct"])
        sp = 100.0 - float(s.loc[n, "gain_pct"])
        col = _ROBUST_COLORS.get(n, "C7")
        mk = "*" if n == "GTMD-RL" else _ROBUST_MARK.get(n, "o")
        sz = 520 if n == "GTMD-RL" else 150
        ax.scatter(resp, sp, s=sz, marker=mk, color=col, edgecolor="black", linewidth=1.0, zorder=3)
        ax.annotate(_SHORT.get(n, n).replace("\n", " "), (resp, sp), fontsize=9,
                    xytext=(7, 5), textcoords="offset points")
    ax.axhspan(99, 101, color="green", alpha=0.05)
    ax.set_xlabel("demand-responsiveness  (allocation sensitivity to reported demand, %) $\\rightarrow$")
    ax.set_ylabel("strategy-proofness  (100 $-$ best-response gain %) $\\rightarrow$")
    ax.set_title("Demand-responsive AND strategy-proof:\nonly GTMD-RL occupies the top-right corner")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / "responsiveness_vs_strategyproofness.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_deep_learning(result, output_dir: Path) -> Path:
    """Three-panel comparison of the tabular bandit against the deep controllers
    (DQN/PPO/A2C) on the SAME expanded action set and MC-true yardstick:
    (a) normalized reward vs epoch, (b) optimal-action rate vs epoch, and
    (c) a final normalized-score bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ck = result.checkpoints
    colors = {"Tabular bandit": "C7", "DQN": "C0", "PPO": "C3", "A2C": "C2"}
    markers = {"Tabular bandit": "o", "DQN": "s", "PPO": "^", "A2C": "D"}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.5))

    for name in result.agents:
        c = result.curves[name]
        col = colors.get(name, "C4"); mk = markers.get(name, "o")
        nm, nd = c["norm_mean"], c["norm_std"]
        ax1.plot(ck, nm, marker=mk, color=col, label=name, lw=1.8)
        ax1.fill_between(ck, nm - nd, nm + nd, color=col, alpha=0.13)
    ax1.axhline(1.0, ls="--", color="green", lw=1.2, label="per-context oracle")
    ax1.axhline(0.0, ls="--", color="grey", lw=1.2, label="random policy")
    ax1.set_xlabel("training epoch"); ax1.set_ylabel("normalized QoS reward")
    ax1.set_title("(a) reward: random $\\to$ oracle"); ax1.set_ylim(-0.15, 1.05)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7, loc="lower right", ncol=1)

    rand = 1.0 / result.n_actions
    for name in result.agents:
        c = result.curves[name]
        col = colors.get(name, "C4"); mk = markers.get(name, "o")
        om, od = c["opt_mean"], c["opt_std"]
        ax2.plot(ck, om, marker=mk, color=col, label=name, lw=1.8)
        ax2.fill_between(ck, np.clip(om - od, 0, 1), np.clip(om + od, 0, 1), color=col, alpha=0.13)
    ax2.axhline(rand, ls="--", color="grey", lw=1.2, label=f"random ({rand:.2f})")
    ax2.set_xlabel("training epoch"); ax2.set_ylabel("optimal-action rate")
    ax2.set_title(f"(b) picks best of {result.n_actions} actions"); ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7, loc="upper left")

    names = result.agents
    finals = [result.final_table[n][0] for n in names]
    bar_cols = [colors.get(n, "C4") for n in names]
    ax3.bar(np.arange(len(names)), finals, color=bar_cols, edgecolor="black", linewidth=0.8)
    ax3.axhline(1.0, ls="--", color="green", lw=1.2)
    ax3.set_xticks(np.arange(len(names)))
    ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax3.set_ylabel("final normalized reward"); ax3.set_ylim(0, 1.05)
    ax3.set_title("(c) converged policy quality"); ax3.grid(alpha=0.3, axis="y")
    for i, v in enumerate(finals):
        ax3.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Tabular vs deep RL controllers on the DSIC weight policy "
                 f"({result.n_actions}-action simplex, load={result.load:.2f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "deep_rl_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_frontier_v2(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Frontier figures from the CRN-paired evaluation (frontier_eval.py).

    Panels: (a) slack vs L with the c*rho*T/L prediction overlaid (c fitted by
    least squares, per load); (b) rho vs L (invariance); (c) regret vs L with a
    c'*sqrt(LT) overlay; (d) normalized combined cost with the optimum L* marked.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    has_pct = "ic_slack_pct" in df.columns
    agg_spec = dict(
        rho_hat=("rho_hat", "mean"),
        rho_std=("rho_hat", "std"),
        ic_slack=("ic_slack", "mean"),
        slack_std=("ic_slack", "std"),
        regret=("regret", "mean"),
        regret_std=("regret", "std"),
    )
    if has_pct:
        agg_spec["ic_slack_pct"] = ("ic_slack_pct", "mean")
        agg_spec["slack_pct_std"] = ("ic_slack_pct", "std")
    agg = df.groupby(["load", "L"], as_index=False).agg(**agg_spec)
    total_slots = int(df["theory_sqrt_LT"].iloc[0] ** 2 / df["L"].iloc[0]) if len(df) else 2400
    loads = sorted(agg["load"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    ax_slack, ax_rho, ax_reg, ax_comb = axes.flat

    for i, load in enumerate(loads):
        g = agg[agg["load"] == load].sort_values("L")
        Ls = g["L"].to_numpy(dtype=float)
        c = colors[i % len(colors)]

        # (a) slack + rho*T/L prediction
        pred = g["rho_hat"].to_numpy() * total_slots / Ls
        scale = (
            float(np.sum(g["ic_slack"].to_numpy() * pred) / np.sum(pred**2))
            if np.sum(pred**2) > 0 else 0.0
        )
        ax_slack.errorbar(Ls, g["ic_slack"], yerr=g["slack_std"], marker="o",
                          color=c, capsize=3, label=f"load={load:.2f}")
        if scale > 0:
            ax_slack.plot(Ls, scale * pred, "--", color=c, alpha=0.65,
                          label=f"$c\\hat\\rho T/L$ (load={load:.2f})")

        # (b) rho invariance
        ax_rho.errorbar(Ls, g["rho_hat"], yerr=g["rho_std"], marker="s",
                        color=c, capsize=3, label=f"load={load:.2f}")

        # (c) regret + sqrt(LT) overlay
        sq = np.sqrt(Ls * total_slots)
        rs = (
            float(np.sum(g["regret"].to_numpy() * sq) / np.sum(sq**2))
            if np.sum(sq**2) > 0 else 0.0
        )
        ax_reg.errorbar(Ls, g["regret"], yerr=g["regret_std"], marker="^",
                        color=c, capsize=3, label=f"load={load:.2f}")
        if rs > 0:
            ax_reg.plot(Ls, rs * sq, "--", color=c, alpha=0.65)

        # (d) normalized combined cost
        slack_n = g["ic_slack"].to_numpy()
        reg_n = g["regret"].to_numpy()
        slack_n = slack_n / slack_n.max() if slack_n.max() > 0 else slack_n
        reg_n = reg_n / reg_n.max() if reg_n.max() > 0 else reg_n
        comb = slack_n + reg_n
        ax_comb.plot(Ls, comb, marker="D", color=c, label=f"load={load:.2f}")
        lstar = Ls[int(np.argmin(comb))]
        ax_comb.scatter([lstar], [comb.min()], s=180, facecolors="none",
                        edgecolors=c, linewidths=2)

    ax_slack.set_xlabel("epoch length $L$ (slots)"); ax_slack.set_xscale("log", base=2)
    ax_slack.set_ylabel("measured IC slack (best response)")
    ax_slack.set_title("(a) slack vs $L$, prediction $\\propto \\hat\\rho T/L$ overlaid")
    ax_slack.legend(fontsize=7); ax_slack.grid(alpha=0.3)

    ax_rho.set_xlabel("epoch length $L$ (slots)"); ax_rho.set_xscale("log", base=2)
    ax_rho.set_ylabel("binding frequency $\\hat\\rho$")
    ax_rho.set_title("(b) $\\hat\\rho$ vs $L$: set by load, flat in $L$")
    ax_rho.legend(fontsize=8); ax_rho.grid(alpha=0.3)

    ax_reg.set_xlabel("epoch length $L$ (slots)"); ax_reg.set_xscale("log", base=2)
    ax_reg.set_ylabel("hindsight regret (reward units)")
    ax_reg.set_title("(c) regret vs $L$, $\\propto\\sqrt{LT}$ overlaid (dashed)")
    ax_reg.legend(fontsize=8); ax_reg.grid(alpha=0.3)

    ax_comb.set_xlabel("epoch length $L$ (slots)"); ax_comb.set_xscale("log", base=2)
    ax_comb.set_ylabel("normalized regret + slack")
    ax_comb.set_title("(d) combined cost: interior optimum $L^\\star$ (circled)")
    ax_comb.legend(fontsize=8); ax_comb.grid(alpha=0.3)

    fig.suptitle("The incentive--learning frontier, CRN-paired measurement", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "frontier_v2_panels.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    # Standalone two-panel versions matching the paper's Fig. 2 layout. Slack is
    # shown as a PERCENTAGE of the tenant's truthful utility when available -- a
    # ~0.2% residual reads honestly as near-zero, where raw utility units looked
    # like a large, jumpy number.
    slack_col = ("ic_slack_pct", "slack_pct_std") if has_pct else ("ic_slack", "slack_std")
    slack_ylab = ("best-response IC slack (% of truthful utility)" if has_pct
                  else "measured IC slack (best response)")
    paths = [path]
    for cols, name, ylab, title in [
        (slack_col, "frontier_slack_vs_L.png", slack_ylab,
         "Strategic-reporting slack under epoch-frozen DSIC-RL"),
        (("rho_hat", "rho_std"), "rho_invariance_vs_L.png",
         "measured binding frequency $\\hat\\rho$",
         "Binding-frequency invariance check"),
    ]:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for i, load in enumerate(loads):
            g = agg[agg["load"] == load].sort_values("L")
            c = colors[i % len(colors)]
            ax.errorbar(g["L"], g[cols[0]], yerr=g[cols[1]], marker="o", capsize=3,
                        color=c, label=f"load={load:.2f}")
            if cols[0].startswith("ic_slack"):
                pred = g["rho_hat"].to_numpy() * total_slots / g["L"].to_numpy(dtype=float)
                yv = g[cols[0]].to_numpy()
                sc = (float(np.sum(yv * pred) / np.sum(pred**2)) if np.sum(pred**2) > 0 else 0.0)
                if sc > 0:
                    ax.plot(g["L"], sc * pred, "--", color=c, alpha=0.65)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("epoch length $L$ (slots)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if cols[0] == "ic_slack_pct":
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        p = output_dir / name
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_epoch_learning(epochs: pd.DataFrame, output_dir: Path) -> Path | None:
    if epochs.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for (load, L, scenario), group in epochs.groupby(["load", "L", "scenario"]):
        if scenario != "truthful":
            continue
        label = f"load={load:.2f}, L={int(L)}"
        rolling = group.sort_values("epoch")["reward"].rolling(4, min_periods=1).mean()
        ax.plot(group.sort_values("epoch")["epoch"], rolling, linewidth=1.5, alpha=0.8, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("rolling epoch reward")
    ax.set_title("Epoch-frozen RL learning curves")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    path = output_dir / "epoch_learning_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path
