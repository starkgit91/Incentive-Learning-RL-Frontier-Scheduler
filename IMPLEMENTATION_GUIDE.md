# Complete Implementation Guide: DSIC + RL for 5G PRB Resource Allocation

## Thesis Overview
**Title:** Dynamic and Efficient Resource Allocation in 5G NR for Different Network Slices  
**Approach:** Game Theory (DSIC Mechanism) + Reinforcement Learning  
**Goal:** Implement a truthful auction mechanism combined with RL-based optimization for PRB allocation across 5G network slices

## Project Structure

```
MTP-Droy/
├── gtmd_rl/                    # Core implementation package
│   ├── __init__.py
│   ├── config.py              # Simulation configuration & slice definitions
│   ├── network.py             # 5G NR trace generator with realistic metrics
│   ├── mechanism.py           # DSIC mechanism with Myerson payments
│   ├── rl.py                  # RL controller (Q-learning) & Bayesian demand estimator
│   ├── adversary.py           # Strategic tenant adversary model
│   ├── experiments.py         # Frontier sweep & experiment runner
│   └── plotting.py            # Visualization utilities
├── scripts/
│   └── run_gtmd_experiments.py # Main entry point
├── IMPLEMENTATION_GUIDE.md    # This file
├── TECHNICAL_DETAILS.md       # Deep technical documentation
├── DATA_GENERATION.md         # 5G metrics generation documentation
└── resources/                 # Generated outputs
    └── outputs/
        └── gtmd_frontier/     # Results directory
```

## Key Components

### 1. Network Simulator (network.py)
Generates realistic 5G NR network traces with:
- **PRBs**: Physical Resource Blocks allocation (50 total)
- **CQI**: Channel Quality Indicator (1-15 scale)
- **SNR**: Signal-to-Noise Ratio in dB
- **BER**: Bit Error Rate
- **Throughput**: Data transmission rate (Mbps)
- **Latency**: Queuing and transmission delay (ms)
- **Queue State**: Buffer occupancy per slice

**Three Network Slices:**
- **URLLC** (Ultra-Reliable Low-Latency): SLA=2ms, Priority=5.0
- **eMBB** (enhanced Mobile Broadband): SLA=8ms, Priority=1.3
- **mMTC** (massive Machine-Type Communication): SLA=20ms, Priority=0.7

### 2. DSIC Mechanism (mechanism.py)
Implements truthful auction with:
- **Monotone Greedy Allocator**: Cannot receive fewer PRBs by reporting higher demand
- **Myerson Critical Value Payments**: Ensures truthful reporting is optimal
- **Floor Constraints**: Minimum PRB guarantees per slice
- **Binding Indicators**: Tracks when constraints are active

### 3. RL Controller (rl.py)
- **Bayesian Demand Estimator**: Updates belief over tenant demand types
- **EpochFrozenQLearner**: Tabular Q-learning with 8 weight profiles
- **Reward Function**: Balances throughput, latency, SLA violations, and PRB waste

### 4. Adversary Model (adversary.py)
- **TabularReportAdversary**: Learns cross-epoch report multipliers
- Tests mechanism robustness against strategic misreporting
- Uses Q-learning to maximize tenant utility

## Running the Complete Pipeline

### Quick Start (Smoke Test)
```bash
cd /home/darpan/Desktop/MTP/MTP-Droy
python3 scripts/run_gtmd_experiments.py --loads 0.55 --epoch-lengths 30,60 --total-slots 600 --adversary-train-episodes 1
```

### Full Experiment
```bash
python3 scripts/run_gtmd_experiments.py
```

### Custom Configuration
```bash
python3 scripts/run_gtmd_experiments.py \
  --loads 0.5,0.8,1.0,1.2,1.5 \
  --epoch-lengths 30,60,120,240 \
  --total-slots 4000 \
  --adversary-train-episodes 3 \
  --output-dir outputs/custom_run
```

## Output Files

### CSV Outputs
1. **sweep_results.csv**: Frontier points (load, L, metrics)
   - `rho_hat`: Incentive compatibility measure
   - `ic_slack`: Surplus gain from truthful reporting
   - `raw_strategic_gain`: Maximum deviation profit
   - `throughput_mbps`, `sla_violation_rate`, `mean_latency_ms`

2. **epoch_traces.csv**: Per-epoch traces
   - Truthful vs. strategic allocation outcomes
   - Payment calculations
   - Demand estimation updates

3. **network_trace_sample.csv**: Per-slot detailed metrics
   - PRB allocation, CQI, SNR, BER
   - Throughput, latency, SLA violations
   - Queue state and demand

### Visualizations
1. **frontier_slack_vs_L.png**: IC slack across epoch lengths
2. **rho_invariance_vs_L.png**: Rho stability
3. **slack_scaling_proxy.png**: Scaling behavior
4. **epoch_learning_curves.png**: RL convergence

## 5G Metrics Generation Details

### Channel Model (Rayleigh Fading)
- SNR autocorrelated AR(1) process: `SNR_t = 0.94*SNR_{t-1} + 0.06*mean + noise`
- CQI determined by SNR-to-efficiency mapping (3GPP NR MCS tables)
- BER computed from SNR: `BER ≈ 0.5 * erfc(sqrt(SNR_linear))`

### Traffic Model (Per-Slice)
- **URLLC**: Gamma-distributed arrivals + occasional bursts (8% prob, 5x multiplier)
- **eMBB**: Normal arrivals ± 18% std dev + 3% burst probability
- **mMTC**: Exponential arrivals + frequent large spikes (4% prob, 12x multiplier)

### PRB Capacity
```
capacity_bits = PRB_bandwidth * slot_duration * PHY_overhead * spectral_efficiency
capacity_bits = 180kHz * 1ms * 0.82 * efficiency[CQI]
```

### SLA Calculation
- Latency = queue_bits / service_rate_bps * 1000ms
- SLA_violation = latency > SLA_threshold

## Experimental Workflow

### Phase 1: Mechanism Validation
- Verify monotonicity of allocator
- Check payment correctness (critical values)
- Validate floor constraints

### Phase 2: Demand Estimation
- Bootstrap 20 trajectories
- Update Bayesian beliefs from tenant reports
- Compare estimated vs. actual demand

### Phase 3: RL Optimization
- Train 12 episodes on each (load, L) pair
- Use 8 learned weight profiles
- Track convergence of Q-values

### Phase 4: Adversary Training
- Train strategic tenant for K episodes
- Measure IC slack reduction
- Compute raw strategic gains

## Key Metrics Explained

### ρ (Rho) - Incentive Compatibility
- Measures expected surplus gain from misreporting
- ρ = E[utility_strategic - utility_truthful]
- Lower ρ → stronger IC guarantee
- Target: ρ << 0.1

### IC Slack
- Maximum advantage from worst-case deviation
- Computed by testing multiple report multipliers
- Used for IC guarantee certificates

### Strategic Gain
- Raw benefit before accounting for floor violations
- `gain = theta * (alloc_strategic - alloc_truthful) - payment_diff`
- Floor-localized gains indicate conservative mechanism

## Integration with 5G-LENA (Future)
1. Export learned weight profiles as xApp configuration
2. Hook into ns-3 scheduler for real-time allocation
3. Use CQI/SNR feedback from PHY layer
4. Port Bayesian estimator to ns-3-ai middleware

## Reproducibility
- Fixed seed (42) for all experiments
- Deterministic slot-level simulation
- Stateless epoch transitions
- CSV outputs include all parameters

## Troubleshooting

### ImportError: gtmd_rl
```bash
export PYTHONPATH=/home/darpan/Desktop/MTP/MTP-Droy:$PYTHONPATH
```

### Out of Memory
- Reduce `--total-slots` (default: 2400)
- Reduce epoch lengths (e.g., `--epoch-lengths 30,60`)

### Slow Execution
- Run smoke test first: `--total-slots 600 --adversary-train-episodes 1`
- Disable sample collection: `--collect-sample-slots False` (in code)

## Citation
```bibtex
@inproceedings{droy2024dsic,
  title={Dynamic Resource Allocation in 5G NR using DSIC Mechanisms and Reinforcement Learning},
  author={Droy, [Name]},
  booktitle={IEEE INFOCOM 2024},
  year={2024}
}
```

## References
- 3GPP NR Resource Allocation (TS 38.214)
- Myerson, R. B. "Optimal Auction Design" (Mathematics of Operations Research, 1981)
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
- Paper: See ../main.pdf for detailed problem formulation
