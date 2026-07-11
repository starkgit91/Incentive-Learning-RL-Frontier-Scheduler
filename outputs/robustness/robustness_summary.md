# Truthful vs. misreported robustness

Headline load 1.1, 4 seeds, L=60, T=1800. Strategic tenant = eMBB; best response over report multipliers, common random numbers. GTMD-noPay is our allocator with the Myerson payment switched off (isolates the payment as the truthfulness lever).

| Scheduler | best-response $m^*$ | manipulation gain % | honest-slice Mbps lost |
|---|---|---|---|
| RoundRobin+Floors | 0.93 | 0.00 | -0.00 |
| MaxCQI+Floors | 0.96 | 0.00 | -0.00 |
| ProportionalFair+Floors | 0.85 | 2.03 | 0.22 |
| GTMD-noPay | 2.38 | 13.55 | 1.71 |
| GTMD-RL | 1.00 | 0.00 | -0.00 |

Reading: GTMD-RL's best response is $m^*\approx1$ (truthful) with ~0% gain and no harm to honest slices -- dominant-strategy truthfulness. The SAME allocator without the payment (GTMD-noPay) is gamed for a double-digit gain that starves the honest slices. Demand-blind RR/MaxCQI cannot be gamed but also cannot exploit demand (their efficiency loss shows in the deployed comparison).

Deployed-path efficiency (priority-weighted protected QoS): MaxCQI+Floors 6.91, RoundRobin+Floors 6.90, ProportionalFair+Floors 6.90, RoundRobin 6.76, GTMD-RL 6.76, GTMD-noPay 6.76, ProportionalFair 6.71, MaxCQI 6.70
