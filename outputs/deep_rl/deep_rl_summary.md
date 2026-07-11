# Deep-RL controllers vs the tabular contextual bandit

Same 28-action simplex weight set and priced-mechanism reward for every learner; 3 seeds, 3000 epochs, L=80. Each learner is scored by its greedy policy against a Monte-Carlo oracle of the true per-context action values (0 = random action, 1 = per-context oracle).

| Controller | class | final normalized reward | final optimal-action rate |
|---|---|---|---|
| Tabular bandit | tabular bandit (gamma=0) | 0.59 | 0.20 (random 0.04) |
| DQN | deep value (Double-DQN) | 0.57 | 0.20 (random 0.04) |
| PPO | deep policy grad (clipped) | 0.63 | 0.18 (random 0.04) |
| A2C | deep actor-critic | 0.22 | 0.07 (random 0.04) |

The deep controllers see the continuous demand belief (contention level, per-slice demand-to-floor stress, per-slice share) instead of the tabular learner's coarse (load-bin, stressed-slice) cell, and choose among the same 28 simplex weight profiles. Each epoch is a single-step episode (the demand type is i.i.d. across epochs, Assumption 4), so the DQN target reduces to E[reward|s,a] and the PPO/A2C advantage to reward - V(s): the deep agents are faithful contextual bandits, not misapplied sequential-MDP learners.
