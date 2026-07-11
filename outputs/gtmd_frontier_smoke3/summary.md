# DSIC-RL Frontier Experiment Summary

This run implements the local simulation counterpart of the INFOCOM draft:
DSIC epoch-frozen reports feed a Bayesian demand estimator, which feeds an
epoch-frozen RL controller for PRB weights. A Q-learning tenant searches for
profitable cross-epoch report multipliers.

## Key outputs

- Best measured slack: `0.0000` at load `0.55`, L `30`.
- Mean rho over all runs: `0.0253`.
- Mean SLA violation rate: `0.1749`.
- Mean throughput: `13.0357` Mbps.

## Rho invariance by load

|   load |        min |   max |    spread |
|-------:|-----------:|------:|----------:|
|   0.55 | 0          |  0    | 0         |
|   0.85 | 0          |  0.05 | 0.05      |
|   1.15 | 0.00166667 |  0.1  | 0.0983333 |

## Figures

- `outputs/gtmd_frontier_smoke3/frontier_slack_vs_L.png`
- `outputs/gtmd_frontier_smoke3/rho_invariance_vs_L.png`
- `outputs/gtmd_frontier_smoke3/slack_scaling_proxy.png`
- `outputs/gtmd_frontier_smoke3/epoch_learning_curves.png`

## CSV files

- `sweep_results.csv`: one row per `(load, L)` pair.
- `epoch_traces.csv`: per-epoch truthful and strategic traces.
- `network_trace_sample.csv`: per-slot sample with PRB, throughput, latency, CQI, SNR, BER, and binding indicators.
