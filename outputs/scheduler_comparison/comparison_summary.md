# Scheduler Comparison Summary

GTMD-RL (epoch-frozen RL weights + monotone DSIC allocator with hard floors)
versus the classical 5G MAC schedulers, all driven on identical arrival and
channel realisations. Baselines are shown with hard floors enforced so the
SLA comparison is fair.

## Full metric table

| scheduler               |   load |   sum_throughput_mbps |   p95_latency_ms |   sla_violation_rate |   jain_fairness |   floor_satisfaction |   wasted_prbs_per_slot |
|:------------------------|-------:|----------------------:|-----------------:|---------------------:|----------------:|---------------------:|-----------------------:|
| RoundRobin              |    0.7 |               12.4022 |           1.4341 |               0.0181 |          0.5876 |               0.9982 |                 1.4874 |
| MaxCQI                  |    0.7 |               12.4022 |           1.6722 |               0.0265 |          0.5876 |               0.9931 |                 1.4768 |
| ProportionalFair        |    0.7 |               12.4022 |           1.4953 |               0.0201 |          0.5876 |               0.9969 |                 1.4913 |
| RoundRobin+Floors       |    0.7 |               12.4022 |           0.3582 |               0.0042 |          0.5876 |               1      |                14.453  |
| MaxCQI+Floors           |    0.7 |               12.4022 |           0.3551 |               0.0054 |          0.5876 |               1      |                14.4457 |
| ProportionalFair+Floors |    0.7 |               12.4022 |           0.3562 |               0.0043 |          0.5876 |               1      |                14.4498 |
| GTMD-RL                 |    0.7 |               12.4022 |           1.4586 |               0.0188 |          0.5876 |               1      |                 1.4848 |
| RoundRobin              |    1   |               16.7305 |           2.6572 |               0.0193 |          0.6367 |               0.9874 |                 1.347  |
| MaxCQI                  |    1   |               16.733  |           3.7463 |               0.0325 |          0.6366 |               0.9894 |                 1.365  |
| ProportionalFair        |    1   |               16.7302 |           3.9141 |               0.0321 |          0.6367 |               0.9806 |                 1.3589 |
| RoundRobin+Floors       |    1   |               16.731  |           4.0192 |               0.0246 |          0.6367 |               1      |                 8.9211 |
| MaxCQI+Floors           |    1   |               16.7328 |           4.2635 |               0.0239 |          0.6366 |               1      |                 8.9226 |
| ProportionalFair+Floors |    1   |               16.731  |           4.8205 |               0.0257 |          0.6367 |               1      |                 8.917  |
| GTMD-RL                 |    1   |               16.733  |           2.5659 |               0.0196 |          0.6366 |               1      |                 1.3551 |
| RoundRobin              |    1.3 |               23.1722 |          75.3735 |               0.2525 |          0.6661 |               0.9381 |                 0.9089 |
| MaxCQI                  |    1.3 |               23.1722 |         500      |               0.2342 |          0.6661 |               0.9144 |                 1.0536 |
| ProportionalFair        |    1.3 |               23.1722 |         500      |               0.2761 |          0.6661 |               0.8621 |                 0.9701 |
| RoundRobin+Floors       |    1.3 |               23.1722 |         183.343  |               0.3508 |          0.6661 |               1      |                 4.51   |
| MaxCQI+Floors           |    1.3 |               23.1722 |         373.404  |               0.2661 |          0.6661 |               1      |                 4.709  |
| ProportionalFair+Floors |    1.3 |               23.1722 |         121.896  |               0.3401 |          0.6661 |               1      |                 4.6272 |
| GTMD-RL                 |    1.3 |               23.1722 |         221.992  |               0.2319 |          0.6661 |               1      |                 0.8894 |

## Headline (highest load)

- GTMD-RL SLA violation rate: `0.2319` (best baseline: `GTMD-RL` at `0.2319`).
- GTMD-RL throughput: `23.172` Mbps (max-throughput policy: `RoundRobin` at `23.172`).
- GTMD-RL Jain fairness: `0.6661`; floor satisfaction: `1.0000`.
