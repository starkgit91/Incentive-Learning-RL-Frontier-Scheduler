# GTMD-RL: Deliverables & Run Guide

Truthful (DSIC) reinforcement-learning PRB allocation for 5G NR network slices,
with a game-theoretic demand-reporting mechanism, a scheduler comparison, and a
verified Python ⇄ ns-3 5G-LENA socket bridge.

Everything below runs from the project root with the Linux virtualenv `venv_linux`
(created this session; the old macOS `.venv` is broken on Linux).

```bash
python3 -m venv venv_linux && venv_linux/bin/pip install numpy pandas matplotlib scipy tabulate
# deep-RL controllers (DQN/PPO/A2C) additionally need CPU torch:
venv_linux/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

---

## 1. Mechanism + RL core (`gtmd_rl/`)

| File | What it does |
|------|--------------|
| `config.py` | 3-slice (URLLC/eMBB/mMTC) 5G NR config: floors, SLAs, priorities, CQI table |
| `network.py` | Synthetic NR trace generator: PRBs, throughput, latency, CQI, SNR, BER, queues, floors |
| `mechanism.py` | Monotone weighted-greedy allocator + numerical Myerson (DSIC) payments + monotonicity check |
| `rl.py` | Bayesian demand estimator + epoch-frozen tabular Q-learner + reward; `simplex_action_templates` (fine 28-action weight lattice) + `templates_to_weights` shared by all learners |
| `rl_sota_models.py` | **NEW** — deep-RL controllers: Double-DQN, PPO, A2C (torch) |
| `deep_learning_demo.py` | **NEW** — adapts DQN/PPO/A2C to the contextual-bandit (single-step) setting over the continuous demand belief; head-to-head vs the tabular bandit on a shared MC-true oracle |
| `robustness.py` | **NEW** — truthful vs misreported study: best-response manipulation gain + harm to honest slices, DSIC+RL vs RR/MaxCQI/PF and vs our own allocator with the payment OFF (`GTMD-noPay`), all CRN-paired |
| `adversary.py` | Q-learning tenant that searches for profitable cross-epoch misreports |
| `baselines.py` | **NEW** — Round-Robin, Max-CQI, Proportional-Fair, floor-aware wrappers, GTMD-RL scheduler |
| `comparison.py` | **NEW** — apples-to-apples harness driving all schedulers on identical traffic |
| `experiments.py` | Frontier sweep (rho vs L, incentive slack) |
| `plotting.py` | Frontier + scheduler-comparison figures |

## 2. Experiment scripts (`scripts/`)

```bash
# CORRECTED frontier evaluation (CRN-paired best-response slack, hindsight regret,
# allocation-limited binding indicator). THIS is the paper's measurement.
venv_linux/bin/python scripts/run_frontier_v2.py --loads 0.6,0.9,1.2 --seeds 6
# (older run_gtmd_experiments.py / run_frontier_averaged.py are DEPRECATED:
#  their slack metric multiplied gain by rho — circular — and compared unpaired runs)

# Demonstrate the RL controller learns (norm reward 0.32->0.84, opt-action 0.29->0.63)
venv_linux/bin/python scripts/run_rl_learning.py   # -> outputs/rl_learning/rl_learning_curve.png
venv_linux/bin/python scripts/diagnose_rl.py       # action-leverage / state-dependence check

# Deep-RL controllers (DQN/PPO/A2C) vs the tabular bandit on the SAME 28-action
# simplex + priced reward, scored against an MC-true per-context oracle (3 seeds).
# Final normalized reward: PPO 0.62, tabular 0.59, DQN 0.58, A2C 0.21.
venv_linux/bin/python scripts/run_deep_rl.py       # -> outputs/deep_rl/deep_rl_comparison.png

# TRUTHFUL vs MISREPORTED robustness (the DSIC guarantee, operationalized).
# GTMD-RL best-response m*=1 (0% gain); the SAME allocator with the payment OFF
# (GTMD-noPay) is gamed for ~13.5% gain, stealing 1.7 Mbps from honest slices.
# -> robustness_dsic.png (U(m) curve, gain-vs-load, gain/harm bars),
#    robustness_epochs.png, responsiveness_vs_strategyproofness.png, scheduler_perslice.png
venv_linux/bin/python scripts/run_robustness.py    # -> outputs/robustness/

# GTMD-RL vs Round-Robin / Max-CQI / Proportional-Fair
venv_linux/bin/python scripts/run_scheduler_comparison.py --output-dir outputs/scheduler_comparison_final
```

Key result: under overload GTMD-RL gives the **lowest URLLC tail latency (3.5 ms vs
500 ms for max-CQI/PF)** and **lowest SLA-violation rate**, guarantees floors 100%,
and is the **only DSIC (truthful)** policy. The floors-always-on (ρ=1) baseline is
measurably worse — the single-lever barrier showing up empirically.

## 3. Python ⇄ ns-3 5G-LENA socket bridge (`bridge/`)

Raw TCP sockets (no ns3-ai). Python is the near-RT-RIC controller; ns-3 is the gNB.

```bash
# Protocol-only test (no ns-3 build):
venv_linux/bin/python bridge/rl_allocation_server.py --epochs 20 --epoch-length 60 --port 5005 &
venv_linux/bin/python bridge/mock_ns3_client.py --port 5005

# Real 5G-LENA (nr module already built at ~/ns-3-dev):
cp bridge/ns3/gtmd-nr-bridge.cc ~/ns-3-dev/scratch/ && (cd ~/ns-3-dev && ./ns3 build gtmd-nr-bridge)
venv_linux/bin/python bridge/rl_allocation_server.py --epochs 5 --epoch-length 40 --port 5577 &
cd ~/ns-3-dev && ./ns3 run "gtmd-nr-bridge --serverPort=5577"
```

Verified end-to-end this session: the ns-3 gNB receives per-epoch PRB budgets, runs
the NR OFDMA PHY/MAC, and returns FlowMonitor throughput/delay that trains the RL
policy. See `bridge/README.md` for the wire protocol and Mode-A/Mode-B enforcement.

## 4. INFOCOM paper (`paper/`)

- `main.tex` — 8-page IEEE two-column draft (compiles with `tectonic` or `pdflatex`).
- `figures/` — architecture, frontier, ρ-invariance, scheduler comparison.
- Build: `tectonic -X compile main.tex` (or upload the folder to Overleaf/Prism).

Adds to the original theory draft: Algorithm 1, the 5G-LENA bridge section, and a
full Evaluation with real numbers (Tables I–III, Figs 2–3).
