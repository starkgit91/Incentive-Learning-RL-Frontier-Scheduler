# 5G NR Data Generation: Technical Details

## 1. Channel State Information (CSI)

### SNR (Signal-to-Noise Ratio) Generation
The SNR evolves per-slice with realistic fading characteristics:

```python
SNR_t = α * SNR_{t-1} + (1-α) * SNR_mean + ε_t

where:
  α = 0.94            # Autocorrelation coefficient
  SNR_mean            # Slice-specific mean (URLLC: 17dB, eMBB: 19dB, mMTC: 12dB)
  ε_t ~ N(0, σ²)      # Innovation with slice-specific std dev
  σ_URLLC = 2.5dB, σ_eMBB = 3.0dB, σ_mMTC = 4.0dB
```

**Properties:**
- Captures channel correlation (mobile users experience gradual changes)
- Slice-specific characteristics (URLLC in better coverage, mMTC in challenging conditions)
- Bounded range: [-8dB, 28dB] to reflect practical 5G deployments

### CQI (Channel Quality Indicator) Mapping
CQI is a 4-bit index (1-15) based on SNR thresholds from 3GPP NR MCS Table:

| CQI | SNR Threshold (dB) | Spectral Efficiency (bps/Hz) | Use Case |
|-----|-------------------|------------------------------|----------|
| 1   | -6.7              | 0.1523                       | Poor coverage |
| 7   | 5.9               | 1.4766                       | Typical |
| 15  | 22.7              | 5.5547                       | Excellent |

```python
cqi = searchsorted(snr_to_efficiency_lut, snr_db)
cqi = clip(cqi, 1, 15)
```

### BER (Bit Error Rate) Calculation
Derived from SNR using error function:

```python
BER ≈ 0.5 * erfc(sqrt(SNR_linear))

where:
  SNR_linear = 10^(SNR_dB / 10)
  erfc = complementary error function
```

**Interpretation:**
- At SNR=10dB: BER ≈ 10^-5 (acceptable for eMBB)
- At SNR=20dB: BER ≈ 10^-9 (good for all slices)

## 2. Physical Resource Blocks (PRBs)

### PRB Capacity Calculation
Each PRB can carry data at rate determined by CQI:

```python
capacity_per_prb_bits = PRB_bandwidth * slot_duration * PHY_overhead * spectral_efficiency[CQI]

Typical values:
  PRB_bandwidth = 180 kHz (one RB in NR)
  slot_duration = 1 ms (one slot)
  PHY_overhead = 0.82 (DMRS, PDCCH, guard bands)
  spectral_efficiency = 0.15-5.5 bps/Hz (from CQI)

Result: ~22-1000 bytes per PRB per slot
```

### PRB Allocation Constraints
1. **Total**: 50 PRBs available per slot
2. **Floors**: Minimum allocation per slice
   - URLLC: 10 PRBs (20% reserved)
   - eMBB: 18 PRBs (36% reserved)
   - mMTC: 6 PRBs (12% reserved)
3. **Monotonicity**: Cannot decrease allocation with higher demand report

## 3. Traffic Model

### Arrival Process
Slice-specific stochastic processes for incoming data:

#### URLLC (Gamma + Bursts)
```python
mean_bits_per_slot = theta * capacity
arrivals = Gamma(shape=2, scale=mean_bits_per_slot/2)
if random() < 0.08:  # 8% burst probability
    arrivals += Uniform(1.5*mean, 5*mean)  # 5x burst multiplier
```
**Rationale:** Mission-critical traffic with occasional surges (e.g., emergency alerts)

#### eMBB (Normal Distribution)
```python
arrivals = max(0, N(mean_bits_per_slot, 0.18*mean_bits_per_slot + 1.0))
if random() < 0.03:  # 3% burst probability
    # Video streaming surge
```
**Rationale:** Continuous high-rate traffic with occasional video/download spikes

#### mMTC (Exponential + Massive Spikes)
```python
arrivals = Exponential(0.65 * mean_bits_per_slot)
if random() < 0.04:  # 4% spike probability
    arrivals += Uniform(3*mean, 12*mean)  # 12x spike multiplier
```
**Rationale:** Sporadic sensor data with rare synchronized events

### Demand Intensity (θ)
Represents average PRBs needed per slot for slice i:

```python
θ_i = (traffic_load * base_rate) / capacity_per_prb

Example (load=1.0):
  URLLC: 1.8 Mbps / 180 PRB-bits/ms ≈ 2.4 PRBs
  eMBB: 11.0 Mbps / 180 ≈ 15.2 PRBs
  mMTC: 1.2 Mbps / 180 ≈ 1.8 PRBs
```

## 4. Queue Dynamics

### Queue State Evolution
```python
Q_t+1 = max(0, Q_t + arrivals_t - served_t)

served_t = min(Q_t + arrivals_t, allocation_PRBs * capacity_bits)
```

### Latency Calculation
```python
latency_ms = (queue_bits_after / service_rate_bps) * 1000

where:
  service_rate_bps = allocation_PRBs * capacity_bits / slot_duration
```

### SLA Violation
```python
violation = latency_ms > SLA_threshold_ms

SLA Thresholds:
  URLLC: 2 ms
  eMBB: 8 ms
  mMTC: 20 ms
```

## 5. Allocation Metrics

### Throughput
```python
throughput_mbps[i] = served_bits[i] / slot_duration / 1e6
```

### Floor Binding
```python
binding[i] = (demand >= floor) AND 
             (constrained_alloc >= floor) AND
             (unconstrained_alloc < floor)
```
Indicates when floor constraint is actively reducing allocation efficiency.

### Wasted PRBs
```python
wasted[i] = max(0, allocation_PRBs[i] - demand_PRBs[i])
```
Represents overallocation beyond estimated demand.

## 6. 5G-Specific Considerations

### Physical Overhead (0.82)
- DMRS (DeModulation Reference Signal): ~8%
- PDCCH (Physical Downlink Control Channel): ~5%
- Guard bands and sync signals: ~5%
- Leaves 82% for data

### CQI Filtering in Real 5G
In actual 5G:
1. UE measures reference signal power
2. Estimates CQI based on target BLER (e.g., 10%)
3. Reports CQI (quantized) to gNB
4. Aperiodic/periodic reporting

Our simulator uses instantaneous SNR → CQI (ideal observer model).

### MCS (Modulation and Coding Scheme)
CQI directly maps to MCS:
- CQI 1: QPSK 1/8 → 0.15 bps/Hz
- CQI 7: 64-QAM 8/15 → 1.48 bps/Hz
- CQI 15: 256-QAM 948/1024 → 5.55 bps/Hz

## 7. Simulation Parameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Total PRBs | 50 | Modest for fast local simulation |
| Slot duration | 1 ms | 3GPP NR std (can be 0.5ms in FR2) |
| Simulation slots | 2400 | ~2.4 seconds per scenario |
| Epoch length | 30-240 | DSIC auction epoch duration |
| Slices | 3 (URLLC/eMBB/mMTC) | 3GPP network slicing taxonomy |
| Load range | 0.5-1.5 | Explores underutilization → congestion |

## 8. Validation

### Sanity Checks
1. **Capacity vs. Demand**: Expected throughput ≈ load * base_rate (when no congestion)
2. **Latency Growth**: Latency increases with load and queue depth
3. **Floor Satisfaction**: Allocation >= floor (except during extreme scarcity)
4. **CQI Monotonicity**: Higher SNR should not decrease CQI

### Example Trace (Load=1.0, URLLC slice)
```
Slot: 1
  SNR: 16.8 dB
  CQI: 6
  Spectral Eff: 1.18 bps/Hz
  Capacity: ~211 bytes/PRB
  Demand: 2.4 PRBs
  Allocation: 10 PRBs (floor)
  Arrivals: 1450 bits (Gamma)
  Queue After: 2190 bits
  Latency: 2.3 ms ⚠️ SLA VIOLATION (threshold: 2.0 ms)
```

## 9. Integration with Real 5G-LENA

### Required Modifications
1. **CQI Source**: Read from PHY layer CQI reporting
2. **Allocation Implementation**: Call gNB scheduler
3. **Payment Abstraction**: Compute via callback (not transmitted)
4. **Demand Reporting**: DSIC reports via network protocol extension
5. **RL State**: Share via ns3-ai middleware

### Data Flow
```
gNB PHY → CQI/SNR ─┐
                   ├─→ [DSIC Mechanism] ──→ Allocation
Tenant → Report ──┘       ↓
                      [RL Weights]
                          ↓
                  [Update & Output]
```

## 10. References

- 3GPP TS 38.214: "NR; Physical layer procedures for data"
- 3GPP TS 38.321: "NR; Medium Access Control (MAC)"
- https://www.3gpp.org/DynaReport/38-series.htm
- Myerson, R. (1981) "Optimal Auction Design" Math. Oper. Res.
- Fading Channel Models: Proakis & Salehi (2007)
