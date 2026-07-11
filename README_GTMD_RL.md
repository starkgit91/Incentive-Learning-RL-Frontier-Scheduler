# DSIC-RL PRB Allocation Implementation

This folder now contains a runnable local implementation of the paper pipeline:

- synthetic 5G NR slice trace generation with PRBs, throughput, latency, CQI, SNR, BER, queue demand, and service-floor binding;
- epoch-frozen monotone weighted-greedy allocation;
- numerical Myerson critical-value payments for within-epoch DSIC;
- Bayesian demand estimation from tenant reports;
- epoch-frozen tabular RL over PRB weight profiles;
- Q-learning strategic tenant that searches for profitable cross-epoch misreports;
- rho-vs-L frontier sweep and figures for the current INFOCOM draft.

Run the default experiment:

```bash
python3 scripts/run_gtmd_experiments.py
```

Fast smoke test:

```bash
python3 scripts/run_gtmd_experiments.py --loads 0.55 --epoch-lengths 30,60 --total-slots 600 --adversary-train-episodes 1
```

Outputs are written to `outputs/gtmd_frontier/`:

- `sweep_results.csv`
- `epoch_traces.csv`
- `network_trace_sample.csv`
- `frontier_slack_vs_L.png`
- `rho_invariance_vs_L.png`
- `slack_scaling_proxy.png`
- `epoch_learning_curves.png`
- `summary.md`

The local simulator is not a 5G-LENA replacement. It is the reproducible mechanism and RL development layer that mirrors the paper's experiment before porting the scheduler hook to ns-3/ns3-ai.
