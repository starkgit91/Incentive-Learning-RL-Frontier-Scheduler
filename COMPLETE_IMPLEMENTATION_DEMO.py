#!/usr/bin/env python3
"""
Complete DSIC + RL Implementation for 5G PRB Resource Allocation
Thesis: Dynamic and Efficient Resource Allocation in 5G NR for Different Network Slices

This notebook integrates:
1. Realistic 5G network data generation (PRBs, CQI, SNR, BER, throughput, latency)
2. DSIC mechanism for truthful demand reporting
3. RL-based controller for weight optimization
4. Adversary model for robustness testing
5. Comprehensive experiments and analysis
"""

import sys
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mtp_droy_mpl")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from gtmd_rl.config import default_config
from gtmd_rl.network import NRTraceGenerator, NetworkState
from gtmd_rl.mechanism import weighted_greedy_allocator, epoch_payments
from gtmd_rl.rl import BayesianDemandEstimator, EpochFrozenQLearner, network_reward
from gtmd_rl.adversary import TabularReportAdversary


class Complete5GImplementation:
    """End-to-end 5G resource allocation with DSIC + RL"""

    def __init__(self, load=1.0, seed=42):
        self.config = default_config()
        self.load = load
        self.seed = seed
        self.env = NRTraceGenerator(self.config, load=load, seed=seed)
        self.estimator = BayesianDemandEstimator(self.config)
        self.planner = EpochFrozenQLearner(self.config, seed=seed + 1000)
        
    def demonstrate_data_generation(self, num_slots=10):
        """Show realistic 5G data being generated"""
        print("\n" + "="*80)
        print("5G NETWORK DATA GENERATION DEMONSTRATION")
        print("="*80)
        
        self.env.reset()
        records = []
        
        for slot in range(num_slots):
            # Get network state
            state = self.env.current_state()
            
            # Generate arrivals
            arrivals = self.env.sample_arrivals()
            
            # Simple round-robin allocation for demo
            alloc = np.array([8, 15, 5])
            
            # Step environment
            next_state, result = self.env.step(alloc)
            
            # Record metrics
            for i, slice_name in enumerate(["URLLC", "eMBB", "mMTC"]):
                records.append({
                    'Slot': slot,
                    'Slice': slice_name,
                    'CQI': int(state.cqi[i]),
                    'SNR_dB': f"{state.snr_db[i]:.1f}",
                    'BER': f"{state.ber[i]:.2e}",
                    'Demand_PRBs': f"{state.demand_prbs[i]:.1f}",
                    'Allocation_PRBs': result.allocation_prbs[i],
                    'Throughput_Mbps': f"{result.throughput_mbps[i]:.2f}",
                    'Latency_ms': f"{result.latency_ms[i]:.2f}",
                    'Queue_bits': f"{state.queue_bits[i]:.0f}",
                    'SLA_Violation': '✓' if result.sla_violation[i] else '✗'
                })
        
        df = pd.DataFrame(records)
        print("\nSample Network Trace (first 10 slots, 3 slices):")
        print(df.to_string(index=False))
        print(f"\nTotal records: {len(df)}")
        
    def demonstrate_dsic_mechanism(self):
        """Show DSIC mechanism with truthfulness properties"""
        print("\n" + "="*80)
        print("DSIC MECHANISM DEMONSTRATION")
        print("="*80)
        
        self.env.reset()
        theta = self.env.begin_epoch()
        state = self.env.current_state()
        
        print(f"\nTrue Demand (θ): {theta}")
        print(f"Slice Configuration:")
        for i, spec in enumerate(self.config.slices):
            print(f"  {spec.name:8} | SLA: {spec.sla_latency_ms:5.1f}ms | Priority: {spec.priority:4.2f} | Floor: {spec.floor_prbs:2} PRBs")
        
        # Test different weight profiles
        weights_profiles = [
            ("Balanced", np.array([1.0, 1.0, 1.0])),
            ("URLLC Priority", np.array([2.0, 1.0, 0.5])),
            ("eMBB Priority", np.array([0.8, 2.0, 0.8])),
        ]
        
        print("\nAllocation Results with Different Weight Profiles:")
        print("-" * 80)
        
        for name, weights in weights_profiles:
            decision = weighted_greedy_allocator(state, theta, weights, self.config)
            print(f"\n{name}: weights = {weights}")
            print(f"  Allocation PRBs: {decision.allocation_prbs}")
            print(f"  Binding: {decision.binding}")
            
            # Test monotonicity: increase one report
            test_theta = theta.copy()
            test_theta[1] += 2.0  # Increase eMBB demand
            decision2 = weighted_greedy_allocator(state, test_theta, weights, self.config)
            print(f"  After +2 on eMBB report: {decision2.allocation_prbs} (monotone? {all(decision2.allocation_prbs >= decision.allocation_prbs)})")
        
    def demonstrate_rl_controller(self):
        """Show RL controller learning weight profiles"""
        print("\n" + "="*80)
        print("RL CONTROLLER DEMONSTRATION")
        print("="*80)
        
        print(f"\nRL Action Templates (weight profiles):")
        for i, profile in enumerate(self.planner.action_templates):
            print(f"  Action {i}: {profile}")
        
        # Simulate one epoch
        self.env.reset()
        self.estimator = BayesianDemandEstimator(self.config)
        
        print(f"\nInitial Demand Belief:")
        print(f"  Mean: {self.estimator.mean}")
        print(f"  Std:  {self.estimator.std}")
        
        # Collect demand reports
        reports_list = []
        for _ in range(5):
            theta = self.env.begin_epoch()
            reports_list.append(theta)
            self.estimator.update(theta)
        
        print(f"\nUpdated Demand Belief (after 5 reports):")
        print(f"  Mean: {self.estimator.mean}")
        print(f"  Std:  {self.estimator.std}")
        
    def demonstrate_adversary(self):
        """Show strategic tenant learning to misreport"""
        print("\n" + "="*80)
        print("ADVERSARY DEMONSTRATION")
        print("="*80)
        
        adversary = TabularReportAdversary(
            tenant_id=1,  # eMBB tenant
            multipliers=(0.7, 0.9, 1.0, 1.1, 1.3, 1.6),
            seed=self.seed
        )
        
        print(f"\nAdversary Configuration:")
        print(f"  Tenant ID: {adversary.tenant_id} (eMBB)")
        print(f"  Report Multipliers: {adversary.multipliers}")
        print(f"  Learning Rate (α): {adversary.alpha}")
        print(f"  Discount (γ): {adversary.gamma}")
        
        print(f"\nTraining for 100 steps...")
        
        # Quick training
        self.env.reset()
        for step in range(100):
            theta = self.env.begin_epoch()
            planner_action = step % len(self.planner.action_templates)
            mult = adversary.choose_multiplier(planner_action, theta[1], 0.1, train=True)
            
            # Simulate utility gain from misreporting
            reward = np.random.normal(0.02, 0.05)  # Mean small gain
            adversary.update(reward, planner_action, theta[1], 0.1)
        
        print(f"\nAdversary Q-function sample (state=(action=0, theta_bin=1, rho_bin=1)):")
        state_key = (0, 1, 1)
        print(f"  Q-values: {adversary.q[state_key]}")
        best_action = np.argmax(adversary.q[state_key])
        print(f"  Best learned multiplier: {adversary.multipliers[best_action]:.2f}x")
        
    def run_complete_experiment(self, loads=[0.8, 1.0], epoch_lengths=[60, 120], total_slots=1200):
        """Run complete end-to-end experiment"""
        print("\n" + "="*80)
        print("COMPLETE EXPERIMENT: Frontier Sweep")
        print("="*80)
        
        results = []
        
        for load in loads:
            for L in epoch_lengths:
                print(f"\nRunning: load={load:.2f}, epoch_length={L}")
                
                env = NRTraceGenerator(self.config, load=load, seed=self.seed)
                estimator = BayesianDemandEstimator(self.config)
                planner = EpochFrozenQLearner(self.config, seed=self.seed + 1000)
                
                n_epochs = total_slots // L
                metrics = {
                    'load': load,
                    'L': L,
                    'mean_alloc': [],
                    'mean_throughput': [],
                    'mean_latency': [],
                    'sla_violation_rate': 0,
                    'wasted_prbs': []
                }
                
                for epoch in range(min(n_epochs, 3)):  # Limit to 3 epochs for demo
                    theta = env.begin_epoch()
                    _, weights, _ = planner.select_action(estimator, 0.0, 0.0, train=False)
                    estimator.update(theta)
                    
                    for slot in range(L):
                        state = env.current_state()
                        decision = weighted_greedy_allocator(state, theta, weights, self.config)
                        next_state, result = env.step(decision.allocation_prbs)
                        
                        metrics['mean_alloc'].append(result.allocation_prbs.mean())
                        metrics['mean_throughput'].append(result.throughput_mbps.mean())
                        metrics['mean_latency'].append(result.latency_ms.mean())
                        metrics['sla_violation_rate'] += result.sla_violation.sum()
                        metrics['wasted_prbs'].append(result.wasted_prbs.sum())
                
                total_slots_run = min(n_epochs, 3) * L * 3  # 3 slices
                results.append({
                    'load': load,
                    'L': L,
                    'mean_alloc_prbs': np.mean(metrics['mean_alloc']),
                    'mean_throughput_mbps': np.mean(metrics['mean_throughput']),
                    'mean_latency_ms': np.mean(metrics['mean_latency']),
                    'sla_violation_rate': metrics['sla_violation_rate'] / total_slots_run,
                    'wasted_prbs': np.mean(metrics['wasted_prbs'])
                })
        
        df_results = pd.DataFrame(results)
        print("\n" + "-" * 80)
        print("Experiment Results:")
        print(df_results.to_string(index=False))
        return df_results


def main():
    """Run complete demonstration"""
    print("\n\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " COMPLETE 5G RESOURCE ALLOCATION IMPLEMENTATION ".center(78) + "║")
    print("║" + " DSIC Mechanism + Reinforcement Learning ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    impl = Complete5GImplementation(load=1.0, seed=42)
    
    # Run demonstrations
    impl.demonstrate_data_generation(num_slots=10)
    impl.demonstrate_dsic_mechanism()
    impl.demonstrate_rl_controller()
    impl.demonstrate_adversary()
    
    # Run complete experiment
    df_results = impl.run_complete_experiment(
        loads=[0.8, 1.0],
        epoch_lengths=[60, 120],
        total_slots=720
    )
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nFor full experiments, run:")
    print("  python3 scripts/run_gtmd_experiments.py")
    print("\nFor custom configuration:")
    print("  python3 scripts/run_gtmd_experiments.py --loads 0.5,1.0,1.5 --epoch-lengths 30,120")
    print("\n")


if __name__ == "__main__":
    main()
