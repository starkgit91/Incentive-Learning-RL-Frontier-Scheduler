# RL Model Comparison: DQN vs PPO vs A2C for 5G Resource Allocation

## Quick Summary

| Model | When to Use | Pros | Cons |
|-------|-----------|------|------|
| **DQN** | Discrete action, budget constraints | Stable, efficient, proven | Needs replay buffer, offline |
| **PPO** | ⭐ **RECOMMENDED** | Easiest to tune, robust, best overall | Slower convergence |
| **A2C** | Very small state space | Fast convergence, simple | Unstable, high variance |
| **Tabular Q-Learning** | ❌ Not recommended | Very simple | No generalization, limited |

---

## 1. DQN (Deep Q-Network) - Conservative Approach

### Algorithm Overview

```
Initialize: Q-network, Target network
Replay Buffer: [S, A, R, S', done]

For each episode:
  For each step:
    state ← current state
    action ← ε-greedy(Q-network(state))
    reward, state' ← environment.step(action)
    
    Store (state, action, reward, state', done) in replay buffer
    
    If |replay buffer| > batch_size:
      Sample batch from replay buffer
      target_Q ← reward + γ * max_a Q-target(state')
      loss ← (Q-network(state, action) - target_Q)²
      Update Q-network via gradient descent
      
      Every τ steps:
        Q-target ← Q-network  (or soft update)
        
    Decay ε
```

### Why DQN for 5G

**Advantages:**
1. **Experience Replay**: Breaks temporal correlation (ideal for network traces)
2. **Target Network**: Stabilizes training (prevents overestimation)
3. **Double DQN**: Reduces overestimation bias (we implement this)
4. **Efficiency**: Offline training possible

**Disadvantages:**
1. Higher memory usage (replay buffer)
2. More hyperparameters to tune
3. Off-policy bias (can be suboptimal)

### Implementation for 5G

```python
from gtmd_rl.rl_sota_models import DQNAgent

# State dimension:
#   - load_ratio (1)
#   - mean_delay_ms (1)
#   - rho_recent (1)
#   - dominant_slice (1)
#   - demand_estimates (3 slices)
#   = 7 dimensions

agent = DQNAgent(
    state_dim=7,
    action_dim=8,  # 8 weight profiles
    learning_rate=1e-3,
    gamma=0.99,
    epsilon=0.1,
    target_update_freq=10,  # Update target every 10 gradient steps
    buffer_size=256,  # Small buffer for fast experiments
    batch_size=32,
)

# Training loop
for epoch in range(n_epochs):
    state = env.reset()
    
    for slot in range(epoch_length):
        state_tensor = agent.encode_state(load, delay, rho, dominant, estimates)
        action = agent.select_action(state_tensor, train=True)
        
        reward = allocate_and_execute(action)
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()  # Gradient update if buffer full
        
        state = next_state

# Evaluation (ε=0 for greedy)
for epoch in range(n_eval_epochs):
    state = env.reset()
    for slot in range(epoch_length):
        state_tensor = agent.encode_state(...)
        action = agent.select_action(state_tensor, train=False)  # ε=0
        reward = allocate_and_execute(action)
```

### Key Hyperparameters

```
Learning rate (lr): 1e-3 to 1e-4
  - Start with 1e-3, reduce if diverging
  
Gamma (γ): 0.99
  - Standard for 5G (long horizon)
  
Epsilon (ε): 0.1 → 0.01 (decay: 0.995)
  - Start with 0.1 for exploration
  - Decay to 0.01 for exploitation
  
Buffer size: 256-1024
  - Tradeoff: larger → more memory, less correlation
  - 256 works for epoch-frozen (no off-policy)
  
Batch size: 32-64
  - Standard; 32 good for small problems
  
Target update: every 10 gradient steps
  - Can be per episode if too slow
```

---

## 2. PPO (Proximal Policy Optimization) - Recommended ⭐

### Algorithm Overview

```
Initialize: Actor π(a|s), Critic V(s)
Buffer: [S, A, R, V(S), log π(A|S)]

For each epoch:
  Collect trajectories using current policy
  Compute advantages via GAE
  
  For n_epochs (e.g., 4):
    For each batch:
      Compute π_new(a|s) and V_new(s)
      
      ratio ← π_new(a|s) / π_old(a|s)
      
      # Clipped objective (core of PPO)
      actor_loss = -min(
        ratio × advantage,
        clip(ratio, 1-ε, 1+ε) × advantage
      )
      
      critic_loss = (V_new(s) - target)²
      
      entropy_bonus = -α × entropy(π_new)
      
      total_loss = actor_loss + β × critic_loss + entropy_bonus
      
      Update actor, critic
```

### Why PPO for 5G (BEST CHOICE)

**Advantages:**
1. ✅ **Easiest to tune**: Clipped objective is forgiving
2. ✅ **Robust**: Works with various learning rates
3. ✅ **Sample efficient**: Uses collected data multiple times (epochs)
4. ✅ **Stable**: Less prone to divergence than A3C
5. ✅ **Trust region**: Prevents huge policy updates
6. ✅ **Actor-Critic**: Lower variance than pure PG

**Disadvantages:**
1. Slower convergence (vs Q-learning)
2. Requires collecting full trajectories first

### Implementation for 5G

```python
from gtmd_rl.rl_sota_models import PPOAgent, PPOBuffer

agent = PPOAgent(
    state_dim=7,
    action_dim=8,
    learning_rate=3e-4,  # Common for PPO
    gamma=0.99,
    lam=0.95,  # GAE smoothing
    clip_ratio=0.2,  # PPO clipping range
    entropy_coef=0.01,  # Encourage exploration
    value_coef=0.5,  # Value network weight
    n_epochs=4,  # Reuse data 4 times
)

# Training loop
for episode in range(n_episodes):
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    
    state = env.reset()
    for step in range(epoch_length):
        state_tensor = agent.encode_state(...)
        action, log_prob, value = agent.select_action(state_tensor)
        
        reward = allocate_and_execute(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        dones.append(done)
        
        state = next_state
    
    # Update using collected trajectory
    buffer = PPOBuffer(
        states=states, actions=actions, rewards=rewards,
        values=values, dones=dones, log_probs=log_probs
    )
    agent.update(buffer)

# After training: use learned policy (greedy)
for epoch in range(n_eval):
    state = env.reset()
    for step in range(epoch_length):
        state_tensor = agent.encode_state(...)
        action, _, _ = agent.select_action(state_tensor)  # π is deterministic now
        reward = allocate_and_execute(action)
```

### Key Hyperparameters (PPO)

```
Learning rate: 3e-4
  - PPO is less sensitive; 1e-4 to 1e-3 all work
  
Clipping range (ε): 0.2
  - Standard; 0.1-0.3 are all reasonable
  
Lambda (λ): 0.95
  - Bias-variance tradeoff in GAE
  - 0.95 = heavy weighting on advantage (low bias)
  - 0.99 = smooth targets
  
Entropy coefficient: 0.01
  - Encourages exploration in early training
  - Reduce after convergence
  
Value coefficient: 0.5
  - Relative weight of critic vs actor loss
  
N epochs: 4
  - Number of passes over collected data
  - More epochs = more computation but better use of data
  
Gamma: 0.99
  - Long-horizon problem (5G)
```

### PPO Convergence Characteristics

```
Iteration    Loss        Entropy    Value Error
1            High        High       High
2-4          ↓↓↓         ↓↓         ↓
5-10         Stable      Low        Low  ← Converged!
11+          ~Stable     ~Low       ~Low
```

Expected convergence: **4-10 episodes** with proper tuning.

---

## 3. A2C (Advantage Actor-Critic) - Simple Alternative

### Algorithm Overview

```
Initialize: Actor π(a|s), Critic V(s)

For each episode:
  trajectories = []
  for step in episode:
    action ← π(s)
    reward, next_state ← environment.step(action)
    trajectories.append((s, a, r, s'))
  
  # Compute returns and advantages
  for t in reversed(trajectories):
    advantage = reward + γ×V(s') - V(s)
    
    # Actor update (policy gradient)
    actor_loss = -log π(a|s) × advantage - entropy_coef × entropy
    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()
    
    # Critic update (value function)
    critic_loss = (V(s) - return)²
    optimizer.zero_grad()
    critic_loss.backward()
    optimizer.step()
```

### When to Use A2C

**Good for:**
1. Very small state spaces (< 64 dimensions)
2. Fast prototyping (minimal code)
3. GPU not available (low memory)

**Bad for:**
1. High variance (can diverge)
2. Requires careful learning rate tuning
3. Slower convergence than PPO

### Our Recommendation

**For 5G resource allocation:**
- ❌ **Don't use A2C**: Our state is already 7-dimensional; PPO is better
- ✅ **Use PPO**: Most robust, easiest to tune, best results

---

## 4. Current Implementation vs SOTA

### Current (Tabular Q-Learning)

```python
class EpochFrozenQLearner:
    def __init__(self, ...):
        self.q = defaultdict(lambda: np.zeros(8))  # 8 action profiles
        
    def select_action(self, state):
        state_key = (load_bin, delay_bin, rho_bin, dominant)
        action = argmax(self.q[state_key])
        return action
    
    def update(self, state, action, reward, next_state):
        q_target = reward + γ × max(q[next_state])
        q[state][action] += α × (q_target - q[state][action])
```

**Limitations:**
- ❌ Only 8 actions (coarse weight profiles)
- ❌ No generalization across states
- ❌ Limited exploration
- ❌ State must be discretized

### Upgraded (DQN)

```python
agent = DQNAgent(state_dim=7, action_dim=8)

# Handles continuous state naturally
state = [load_ratio, mean_delay, rho, dominant, demand1, demand2, demand3]

# Neural network generalizes
q_values = agent.q_network(state)  # Can interpolate between states

# Efficient training
agent.store_transition(s, a, r, s')
loss = agent.train_step()  # Gradient descent with replay buffer
```

**Improvements:**
- ✅ Continuous state space (no discretization needed)
- ✅ Generalization via neural net
- ✅ Better exploration (ε-greedy over learned values)
- ✅ Proven in Atari, robotics, etc.

### Upgraded (PPO)

```python
actor = PPOActor(state_dim=7, action_dim=8)
critic = PPOCritic(state_dim=7)

# Collect trajectory
for step in epoch:
    action ~ π(·|state)  # Stochastic policy
    reward, next_state = allocate(action)
    trajectory.append((s, a, r, s'))

# Update with advantages
advantages = compute_gae(trajectory)
actor_loss = -log(π(a|s)) × advantages - entropy
critic_loss = (V(s) - return)²

# Multiple passes over data
for epoch in range(n_epochs):
    actor_loss.backward()
    critic_loss.backward()
    optimizer.step()
```

**Improvements:**
- ✅ Stochastic policy (better exploration)
- ✅ GAE advantage estimation (lower variance)
- ✅ Multiple passes on data (sample efficient)
- ✅ Clipping objective (stable)

---

## 5. Recommendation Matrix

```
Problem               | Best       | OK         | Avoid
─────────────────────┼──────────┼──────────┼──────────
Discrete actions      | DQN/PPO  | A2C      | Policy Grad
Large state space     | PPO      | DQN      | A2C
Sample efficiency     | PPO      | DQN      | A2C
Stability             | PPO      | DQN      | A2C
Convergence speed     | DQN      | PPO      | A2C
Implementation ease   | PPO      | A2C      | DQN
Memory usage          | PPO      | A2C      | DQN
─────────────────────┴──────────┴──────────┴──────────

FOR 5G RESOURCE ALLOCATION:
⭐ BEST: PPO
✅ GOOD: DQN
⚠️ NOT RECOMMENDED: A2C, Tabular Q-Learning
```

---

## 6. Hyperparameter Tuning Guide

### PPO (Recommended)

| Parameter | Value | Range | Sensitivity |
|-----------|-------|-------|-------------|
| lr | 3e-4 | 1e-4 to 1e-3 | Low |
| clip_ratio | 0.2 | 0.1 to 0.3 | Low |
| entropy_coef | 0.01 | 0.001 to 0.1 | Medium |
| gamma | 0.99 | 0.95 to 0.999 | Low |
| lambda | 0.95 | 0.9 to 0.99 | Low |
| n_epochs | 4 | 2 to 10 | Low |
| batch_size | 32 | 16 to 128 | Medium |

**Tuning recipe:**
1. Start with defaults above
2. If diverging: reduce lr or entropy_coef
3. If too slow: increase n_epochs or batch_size
4. If unstable: increase clip_ratio

### DQN

| Parameter | Value | Range | Sensitivity |
|-----------|-------|-------|-------------|
| lr | 1e-3 | 1e-4 to 1e-2 | High |
| gamma | 0.99 | 0.95 to 0.999 | Low |
| epsilon | 0.1 | 0.05 to 0.2 | Medium |
| epsilon_decay | 0.995 | 0.99 to 0.999 | Medium |
| buffer_size | 256 | 128 to 1024 | Low |
| batch_size | 32 | 16 to 64 | Medium |
| target_freq | 10 | 5 to 20 | Medium |

**Tuning recipe:**
1. lr is critical for DQN (start low, increase if needed)
2. Use double DQN (reduces overestimation)
3. Replay buffer should be large enough to mix experience
4. Target update frequency: too low → instability, too high → slow learning

---

## 7. Convergence Comparison

```
Method          | Convergence | Stability | Generalization
────────────────┼─────────────┼──────────┼────────────────
Tabular QL      | Fast        | Low      | None
A2C             | Very fast   | Low      | Medium
DQN             | Medium      | Medium   | Good
PPO             | Slow        | High     | Excellent
────────────────┴─────────────┴──────────┴────────────────

For 5G (epoch-frozen, 30-120 slots per epoch):
- Convergence time: 4-10 epochs (2-20 minutes)
- Stability critical (allocation affects SLA)
- Generalization important (load/channel changes)

→ PPO wins overall (good balance)
```

---

## 8. Implementation Status

### Current Code
- ✅ Tabular Q-Learning (working, not SOTA)
- ✅ Basic mechanism (working)
- ❌ DQN, PPO, A2C (created, need integration)

### Next Steps
1. Integrate DQN/PPO into experiments.py
2. Run comparison: Tabular vs DQN vs PPO
3. Measure convergence speed, stability, performance
4. Document results

---

## 9. Summary

| Aspect | Recommendation |
|--------|-----------------|
| **RL Model** | PPO (most robust) or DQN (most efficient) |
| **DSIC Mechanism** | ✅ Correctly implemented (Myerson payments) |
| **State Representation** | 7-D continuous (load, delay, rho, dominant, demand) |
| **Action Space** | 8 weight profiles (discrete, finite) |
| **Learning Loop** | Epoch-frozen (weights fixed 30-120 slots) |
| **Convergence** | 4-10 episodes with proper tuning |

**Publication-ready upgrade path:**
1. Keep current mechanism (excellent DSIC properties)
2. Replace Tabular QL with PPO or DQN
3. Show convergence curves, stability analysis
4. Claim: "First RL + DSIC combination with SOTA deep RL algorithms"

---

**Recommendation:** Start with **PPO**. It's the most forgiving and will give best results with minimal tuning.

