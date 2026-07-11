# RL learning demonstration

Contextual-bandit controller (UCB, gamma=0), 8 seeds, 1000 epochs, L=100.

- Normalized QoS reward (0=random policy, 1=per-state oracle): **0.32 -> 0.84**.
- Optimal-action selection rate (random baseline 0.12): **0.29 -> 0.63**.

The rising curves show the controller learning the state->weight-profile map (route the discretionary surplus to the stressed slice). The ceiling is below 1 because several demand contexts have near-tied best actions.
