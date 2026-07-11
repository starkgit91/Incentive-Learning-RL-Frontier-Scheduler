# Architecture Comparison: Current vs SOTA

## Visual Architecture Diagrams

### Current Implementation (Tabular Q-Learning)

```
┌─────────────────────────────────────────────────────────┐
│                   5G Network State                       │
│   (θ: demands, channel, load, latency, QoS metrics)    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Bayesian Demand Estimator                     │
│   Input: θ → Output: Belief b = [μ₁, μ₂, μ₃]           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  State Discretization         │
         │  load_bin = discretize(load)  │
         │  delay_bin = discretize(delay)│
         │  rho_bin = discretize(rho)    │
         │  dominant = argmax(belief)    │
         │                               │
         │  State Key = (l,d,r,dom)      │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Tabular Q-Function          │
         │   Dict[state_key]             │
         │   = [Q(s,a₁), Q(s,a₂), ...]  │
         │                               │
         │   8 learned weight profiles   │
         └───────────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Select Action       │
              │  a* = argmax Q(s, a) │
              │  (ε-greedy)          │
              └──────────┬───────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   DSIC Mechanism                   │
        │   weighted_greedy_allocator()      │
        │   Allocation: [PRB₁, PRB₂, PRB₃]  │
        │   Payment: [P₁, P₂, P₃]           │
        └────────────┬───────────────────────┘
                     │
                     ▼
              ┌──────────────────┐
              │  Network Result  │
              │  reward = f(KPIs)│
              └──────────┬───────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  Tabular Q-Learning Update       │
        │  Q[s][a] += α(r + γmax Q[s'][a'])│
        │                                   │
        │  Update single (s, a) pair       │
        │  No generalization               │
        └──────────────────────────────────┘
```

**Limitations:**
- State space limited to ~64 bins (4 discretization levels)
- No generalization to new states
- Slow convergence
- Unstable training

---

## Recommended: PPO (Proximal Policy Optimization)

```
┌─────────────────────────────────────────────────────────┐
│                   5G Network State                       │
│   (θ: demands, channel, load, latency, QoS metrics)    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Bayesian Demand Estimator                     │
│   Input: θ → Output: Belief b = [μ₁, μ₂, μ₃]           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────┐
         │  Continuous State Encoding            │
         │  state = [                            │
         │    load_ratio ∈ [0, 2],              │
         │    mean_delay_ms ∈ [0, 100],         │
         │    rho_recent ∈ [0, 1],              │
         │    dominant_slice ∈ [0, 3],          │
         │    demand_est[0] ∈ [0, 100],         │
         │    demand_est[1] ∈ [0, 100],         │
         │    demand_est[2] ∈ [0, 100]          │
         │  ]                                    │
         │  ∈ ℝ⁷                                │
         └───────────────┬───────────────────────┘
                         │
         ┌───────────────┴─────────────────────────┬──────────┐
         │                                         │          │
         ▼                                         ▼          ▼
    ┌──────────────┐                        ┌──────────────┐
    │ Actor π(a|s) │                        │ Critic V(s)  │
    │              │                        │              │
    │ Input: s ∈ℝ⁷│                        │ Input: s ∈ℝ⁷ │
    │ Hidden: 256  │                        │ Hidden: 128  │
    │ Output: 8    │                        │ Output: 1    │
    │ logits for a │                        │ V-value      │
    └──────┬───────┘                        └──────┬───────┘
           │                                       │
           ▼                                       ▼
      [π(a₁|s), ..., π(a₈|s)]                    V(s)
           │                                       │
           └───────────────────┬───────────────────┘
                               │
                         ┌─────▼──────┐
                         │   Sample   │
                         │ a ~ π(·|s) │
                         │ (stochastic)│
                         └─────┬──────┘
                               │
                               ▼
        ┌────────────────────────────────────┐
        │   DSIC Mechanism                   │
        │   weighted_greedy_allocator()      │
        │   Allocation: [PRB₁, PRB₂, PRB₃]  │
        │   Payment: [P₁, P₂, P₃]           │
        └────────────┬───────────────────────┘
                     │
                     ▼
              ┌──────────────────┐
              │  Network Result  │
              │  reward = f(KPIs)│
              └──────────┬───────┘
                         │
    ┌────────────────────┴────────────────────┐
    │                                         │
    │  Trajectory Collection (per epoch)      │
    │  states, actions, rewards, dones        │
    │  values, log_probs                      │
    │                                         │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────────┐
    │   Advantage Estimation (GAE)                     │
    │   advantages = compute_gae(trajectory)           │
    │   returns = advantages + values                  │
    └────────────────┬───────────────────────────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌──────────────┐          ┌──────────────┐
    │   Actor      │          │   Critic     │
    │   Clipped    │          │   MSE        │
    │   Loss       │          │   Loss       │
    │              │          │              │
    │ L_π = -min(  │          │ L_V = (V(s)  │
    │  r·A,        │          │  -returns)² │
    │  clip(r,     │          │              │
    │  1-ε,1+ε)·A  │          │              │
    │ ) - H        │          │              │
    └──────┬───────┘          └──────┬───────┘
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  Update Networks (4x)   │
            │  for n_epochs:          │
            │    Loss.backward()       │
            │    Optimizer.step()      │
            │    Clip gradients        │
            └─────────────────────────┘
```

**Advantages:**
- ✅ Continuous state space (7-dim)
- ✅ Generalizes to new states
- ✅ Fast convergence
- ✅ Stable training
- ✅ Low variance (actor-critic)
- ✅ Proven SOTA (OpenAI, DeepMind)

---

## Alternative: DQN (Deep Q-Network)

```
┌─────────────────────────────────────────────────────────┐
│                   5G Network State                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────┐
         │  Continuous State Encoding (ℝ⁷)       │
         └───────────────┬───────────────────────┘
                         │
         ┌───────────────┴─────────────────────────┐
         │                                         │
         ▼                                         ▼
    ┌──────────────────┐              ┌──────────────────┐
    │ Q-Network        │              │ Target Network   │
    │ (Active)         │              │ (Delayed)        │
    │                  │              │                  │
    │ Input: s ∈ ℝ⁷    │              │ Input: s ∈ ℝ⁷    │
    │ Hidden: 128-64   │              │ Hidden: 128-64   │
    │ Output: 8        │              │ Output: 8        │
    │ Q(s, a₁)...Q(...) │              │ Q(s, a₁)...Q(...)│
    │                  │              │                  │
    │ π(s) = argmax(Q) │              │ (copies weights  │
    │ (greedy)         │              │  periodically)   │
    └──────┬───────────┘              └──────┬───────────┘
           │                                  │
           ├─────────────────┬────────────────┤
           │                 │                │
      ε-Greedy (explore)     │          Target Q-values
           │                 │                │
      Action a               │          (stable targets)
           │                 │                │
           └─────────────────┼────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │   DSIC Mechanism                   │
        │   weighted_greedy_allocator()      │
        │   Allocation, Payment             │
        └────────────┬───────────────────────┘
                     │
                     ▼
              ┌──────────────────┐
              │  Network Result  │
              │  reward = f(KPIs)│
              └──────────┬───────┘
                         │
                         ▼
    ┌──────────────────────────────────────┐
    │   Store in Replay Buffer             │
    │   (s, a, r, s', done)                │
    │   Buffer size: 256-1024              │
    └────────────────┬──────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │   Sample Batch from Buffer           │
    │   batch_size = 32 (random order)     │
    └────────────────┬──────────────────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Current Q-values │   │ Target Q-values  │
    │                  │   │                  │
    │ Q(s, a)          │   │ r + γ * max Q'   │
    │ from Q-Network   │   │ (s', a')         │
    │                  │   │ from Target Net  │
    └──────┬───────────┘   └──────┬───────────┘
           │                      │
           └──────────────┬───────┘
                          │
                          ▼
        ┌─────────────────────────────┐
        │  Temporal Difference (TD)   │
        │  Error                      │
        │                             │
        │  loss = (Q(s,a) - target)²  │
        │                             │
        │  target = r + γ*max Q'(s',a)│
        └────────────┬────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Update Q-Network          │
        │  loss.backward()            │
        │  optimizer.step()           │
        │  (every step)               │
        └────────────┬────────────────┘
                     │
         ┌───────────┴────────────────┐
         │                            │
         ▼                            ▼
    (every 10K steps)         Decay epsilon
    Copy weights to             (reduce explore)
    Target Network
```

**Characteristics:**
- ✅ Good generalization
- ✅ Experience replay (data reuse)
- ✅ Target network (stable)
- ⚠️ More hyperparameters
- ⚠️ Harder to tune
- ⚠️ Higher memory (replay buffer)

---

## Side-by-Side: Implementation Details

### State Encoding

**Tabular QL:**
```python
# Discretized (categorical)
state = (load_bin, delay_bin, rho_bin, dominant)
# Example: (2, 1, 0, 1) ∈ [0,3]⁴ = 256 possible states
# Missing: demand estimates
```

**PPO:**
```python
# Continuous (numerical)
state = np.array([
    load_ratio,      # ∈ [0.0, 2.0]
    mean_delay_ms,   # ∈ [0.0, 100.0]
    rho_recent,      # ∈ [0.0, 1.0]
    float(dominant), # ∈ [0.0, 3.0]
    demand_est[0],   # ∈ [0.0, 100.0]
    demand_est[1],   # ∈ [0.0, 100.0]
    demand_est[2],   # ∈ [0.0, 100.0]
], dtype=np.float32)
# ∈ ℝ⁷ = infinite possible states (continuous)
```

**DQN:**
```python
# Same as PPO
state = np.array([...], dtype=np.float32)  # ∈ ℝ⁷
```

---

### Action Selection

**Tabular QL:**
```python
def select_action(self, state_key, train=True):
    if train and random() < epsilon:
        return randint(0, 8)  # Random action
    else:
        q_values = self.q_table.get(state_key, zeros(8))
        return argmax(q_values)  # Greedy
```

**PPO:**
```python
def select_action(self, state_tensor, train=True):
    logits = self.actor(state_tensor)        # [batch=1, actions=8]
    probs = softmax(logits)
    action = sample_categorical(probs)       # Stochastic
    log_prob = log(probs[action])
    value = self.critic(state_tensor)        # [batch=1, values=1]
    return action, log_prob, value
```

**DQN:**
```python
def select_action(self, state_tensor, train=True):
    if train and random() < epsilon:
        return randint(0, 8)                 # ε-Greedy
    else:
        q_values = self.q_net(state_tensor)  # [batch=1, actions=8]
        return argmax(q_values)              # Greedy
```

---

### Learning Update

**Tabular QL:**
```python
def update(self, state, action, reward, next_state, done):
    # Single update
    q_current = self.q_table[state][action]
    q_next = max(self.q_table.get(next_state, zeros(8)))
    self.q_table[state][action] += alpha * (
        reward + gamma * q_next * (1 - done) - q_current
    )
```

**PPO:**
```python
def update(self, trajectory_buffer):
    # Batch update (multiple passes)
    advantages = self._compute_gae(trajectory_buffer)
    returns = advantages + trajectory_buffer.values
    
    for epoch in range(n_epochs):
        for batch in shuffle(trajectory_buffer):
            # Actor loss (clipped)
            ratio = exp(log_probs_new - log_probs_old)
            actor_loss = -min(ratio*advantages, clip(ratio, 1-clip_ratio, 1+clip_ratio)*advantages)
            
            # Critic loss
            critic_loss = 0.5 * (values_new - returns)**2
            
            # Total loss with entropy bonus
            loss = actor_loss + critic_coef*critic_loss - entropy_coef*entropy
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
```

**DQN:**
```python
def train_step(self):
    # Replay buffer sample
    batch = self.replay_buffer.sample(batch_size=32)
    
    # Current Q-values
    q_current = self.q_net(batch.states).gather(1, batch.actions)
    
    # Target Q-values (Double DQN)
    next_actions = self.q_net(batch.next_states).argmax(dim=1, keepdim=True)
    q_target = self.target_net(batch.next_states).gather(1, next_actions)
    target = batch.rewards + gamma * q_target * (1 - batch.dones)
    
    # TD loss
    loss = 0.5 * (q_current - target.detach())**2
    loss = loss.mean()
    
    # Update
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # Update target network (every N steps)
    if self.steps % target_update_freq == 0:
        self.target_net.load_state_dict(self.q_net.state_dict())
```

---

## Convergence Comparison

```
Performance vs Training Progress
│
│  PPO:    ════════════════════════════════════════════════→ 90%
│          ════╱╱╱╱╱╱╱╱════════════════════════════════════════
│
│  DQN:    ═══════════════╱╱╱╱════════════════════════════════→ 82%
│          ═════════╱╱╱╱╱═════════════════════════════════════
│
│  TabQL:  ╱╱╱╱╱╱╱╱════════════════════╱╱╱╱════════════════════→ 70%
│          ════════════════════════════════════════════════════
│
└──────────────────────────────────────────────────────────────────
  0        50        100       150       200       250      Epochs
```

---

## Summary

| Aspect | Tabular QL | DQN | PPO |
|--------|-----------|-----|-----|
| **State** | Discrete (256) | Continuous (ℝ⁷) | Continuous (ℝ⁷) |
| **Action** | Discrete (8) | Discrete (8) | Discrete (8) |
| **Network** | None | 2 MLPs | 2 MLPs |
| **Learning** | Per-step | Batch replay | Batch trajectory |
| **Update** | Value-based | Value-based | Policy-based |
| **Stability** | Unstable | Good | Excellent |
| **Tuning** | Easy | Hard | Easy |
| **Convergence** | Fast early | Slow early, fast late | Patient but stable |
| **Generalization** | None | Good | Excellent |
| **For 5G** | Not good | Good | ⭐ Best |

**RECOMMENDATION: Use PPO for publication**
