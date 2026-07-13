# Simulation Plan: Trust-Gated Truthful Learning PRB Allocator
## Target: IEEE INFOCOM 2027 (Honolulu). Abstract due 24 Jul 2026, full paper due 31 Jul 2026 (AoE).

Owner: Prashant Mishra. Reviewer: Dibbendu Roy.
Paper: "Learning From What You Measure: Verification-Gated and Report-Invariant Truthful PRB Allocation in O-RAN Slicing" (working draft v1).

---

## 0. Read this first

**Deadline reality.** The full paper is due in ~19 days. Every experiment below is tagged:

- **P0**: the paper cannot be submitted without it (Figs. 2 and 4, plus the validation suite).
- **P1**: strongly expected by reviewers (Figs. 3 and 5).
- **P2**: stretch; becomes a table or one sentence if time runs out (padding knee, closed-loop).

Do P0 end to end before touching P1. A submission with 2 clean, validated figures beats one with 5 rushed ones.

**Ground rules (non-negotiable).**

1. **Common random numbers (CRN).** Every slack measurement compares a deviating run and a truthful run on *identical* pre-generated channel and traffic arrays (same seed, same draw order). This mirrors the coupling in the proofs and cuts variance by orders of magnitude. Pre-generate `A[N,T]` and `q[N,T]` per seed; both runs consume the same arrays.
2. **No tuning against slack.** Learner hyperparameters (step size, regularizer) may be tuned only on truthful-run regret. If a slack result looks wrong, use the debug ladder in Section 13; never adjust parameters to make slack match theory.
3. **Reproducibility.** Every figure regenerates from `make figures` on cached CSVs. Every run logs: git commit, config hash, seed. INFOCOM encourages an anonymous open-source repo; we will provide one (Section 14).
4. **Report failures.** If an experiment contradicts a theorem after the validation suite passes, that is a *finding*, not a bug to hide. Escalate immediately; it changes the paper.

---

## 1. What we are demonstrating (theory to falsifiable predictions)

Notation: horizon `T` slots, epochs of length `L`, `K = T/L`, `N` tenants, budget `B`, true types `theta`, reports `z`, per-slot traffic noise scale `sigma`, measurement `m_k` (epoch mean of traffic), trust radius `r`.

| # | Claim (theorem) | Falsifiable prediction | Experiment |
|---|---|---|---|
| T1 | Fragility (Thm 1): report-trained learner, persistent misreport | Slack grows linearly in T with slope near c0 = 1.48e-4 (Instance A); slope insensitive to L; optimal deviation near d* = 0.042 | E1a, E1b, V2 |
| T1b | Floors add a resource channel (Rem. 2) | Slack slope increases with binding frequency rho | E1c |
| T2 | Gate (Thm 2): r = r_L caps per-slot gain at O(r_L); honest reports untouched | Gated slack bounded by a line of slope prop. to r_L; gated regret matches oracle | E1a, E2 |
| T3 | Invariance (Thm 3): r = 0 gives Slack = 0 exactly | Measured slack statistically indistinguishable from 0 at every load, every L, every sigma; weight paths bitwise identical across reports | E1a, V3 |
| T4 | Price of invariance (Thm 4): regret bias prop. to sigma*T/sqrt(L) with per-epoch anchor | Regret gap between r=0 and oracle scales linearly in sigma | E2 |
| T4' | **Upgrade**: running-mean anchor gives bias O(sigma*sqrt(T)); combined cost O((1+sigma)*sqrt(T)) | (Reg+Slk)/sqrt(T) flat in T for CUM anchors at all sigma; slope ~ +1/4 for EPOCH anchors at high sigma | E3 |
| P1 | Padding (Prop. 1): per-unit padding cost above threshold kills measurement gaming | Slack vs c_pad shows a knee at/below computed C1 | E4 |
| -- | Frontier: trust family Pareto-dominates DP-noise, fines, burn-in baselines in (Regret, Slack) | Scatter plot | E5 |
| -- | Closed-loop caveat: allocation-dependent traffic reopens the channel at r=0 | Slack at r=0 grows with feedback gain kappa_fb | E6 |

---

## 2. Environment and repo layout

- Python 3.11+, numpy, scipy, matplotlib, pandas. Optional: numba (Instance B waterfilling), joblib/multiprocessing for seed parallelism. No GPU needed.
- Fix `numpy` RNG via `np.random.default_rng(seed)`; one generator per run; pre-draw traffic/channel arrays.

```
trustgate-sim/
  configs/            # yaml or python dataclasses, one file per experiment
  src/
    instances.py      # Instance A, Instance B primitives
    traffic.py        # generators + sigma calibration
    allocators.py     # RULE_P, RULE_W (waterfilling)
    payments.py       # Myerson threshold via grid quadrature
    learner.py        # projected OGD on the simplex slice
    gates.py          # EPOCH(r), CUM(r_k), r=0 variants
    mechanism.py      # the full loop (Algorithm 1 of the paper)
    adversary.py      # deviation grids, CRN slack protocol
    metrics.py        # W*, Regret, Slack, rho
    baselines.py      # B1..B6
  tests/              # V1..V5 as pytest
  runs/               # CSV outputs, one row per (config, seed, deviation)
  figures/            # fig2_fragility.pdf, ...
  Makefile            # make validate / make e1 / ... / make figures
```

---

## 3. Model components (implementation spec)

### 3.1 Instances

**Instance A (theory-matching; validation + payment-channel experiments).** Exactly Appendix B of the paper.

| Item | Value |
|---|---|
| N, B, floors | 2, 4.0, none |
| Channel | single state, q = (1, 1) |
| Types | theta = (1.0, 1.0), Theta = [0.9, 1.1] |
| Valuation | v_i(x) = theta_i * (1 - exp(-x)) |
| Weight space | w1 + w2 = 2, w1 in [0.9, 1.1] (write omega = w1/w2) |
| W* per slot | 2*(1 - e^-2) = 1.729329... (allocation (2,2)) |

Analytic ground truth (numerically certified; use for V2 overlays):

```
x1*(a)      = 2 + 0.5*ln(a)
omega*(a)   = x1*(a) / (a*(4 - x1*(a)));   omega*'(1) = -0.5 exactly
kappa_R     = 0.014245     (rent slope in omega at omega=1)
u'(1)       = -0.0071227   (first-order deviation gain)
z*          = 0.94936      (numerically optimal persistent report)
peak gain   = 1.8430e-4    per slot
kappa_b     = 0.085755     (= max|u''|/2 on [0.9,1])
d*          = 0.041529,  c0 = 1.4790e-4   (certified lower-bound optimum)
```

Reference snippet for the analytic curve `u(z) - u(1)` (overlay in V2/Fig. 2 inset):

```python
import numpy as np
from scipy import integrate
B = 4.0
phi  = lambda x: 1 - np.exp(-x)
x1s  = lambda a: 2 + 0.5*np.log(a)
om   = lambda a: x1s(a) / (a*(B - x1s(a)))
x1   = lambda s, w: B*w*s/(w*s + 1)
def u(z):
    w = om(z)
    return (1-z)*phi(x1s(z)) + integrate.quad(lambda s: phi(x1(s, w)), 0.9, z)[0]
gain = lambda z: u(z) - u(1.0)   # analytic per-slot gain of persistent report z
```

**Instance B (RAN-flavored; floors, rho sweep, frontier).**

| Item | S1 (URLLC-like) | S2 (eMBB-like) | S3 (mMTC-like) |
|---|---|---|---|
| theta_i (nominal) | 1.0 | 3.0 | 0.5 |
| Theta_i | [0.8, 1.2]*theta_i | same | same |
| Saturation s_i | 1.0 | 3.0 | 0.5 |
| Floor f_i | swept (load knob) | 0 | f1/2 |

- Budget B = 12. Valuation `v_i = theta_i * q_i(c) * (1 - exp(-x/s_i))`.
- Channel: `q_i(c_t)` i.i.d. per slot per slice, uniform on {0.6, 1.0, 1.4}. Exogenous, independent of everything.
- Weight space: sum w = 3, w_i >= 0.2.
- **Load knob:** floors scaled to hit target binding frequency rho in {0, 0.05, 0.1, 0.2}. Procedure: run a truthful sim with candidate (f1, f3 = f1/2), measure rho, bisect on f1. Starting grid f1 in {0, 1.0, 1.5, 2.0, 2.5}. Record the calibrated (f1, rho) pairs in `configs/loads.yaml`; reuse everywhere.

### 3.2 Traffic generators and sigma calibration

- Default marks: `A_{i,t} ~ Gamma(shape k_i, scale theta_i/k_i)` with `k_i = (theta_i/sigma_i)^2`, giving mean theta_i and sd sigma_i, support >= 0. (Gamma is sub-exponential rather than sub-Gaussian; fine empirically, and note it in the paper's eval text.)
- Instance A: single sigma knob, sigma in {0.05, 0.1, 0.2, 0.4}. Instance B: relative knob, sigma_i = sigma_rel * theta_i, sigma_rel in {0.05, ..., 0.8}.
- **V4 check:** empirical sd of A within 2% of config; empirical sd of epoch means within 3% of sigma/sqrt(L).
- Closed-loop variant (E6 only): `mean_{i,t} = theta_i * (1 + kappa_fb * (x_{i,t-1} - xbar_i)/B)`, clipped at 0.05*theta_i, where xbar_i is the truthful long-run mean allocation (calibrate once).

### 3.3 Allocators

**RULE_P (paper eq. (6); Instance A and any floor-free config).**

```
x_i = f_i + (B - sum_F f) * (w_i * z_i * q_i) / sum_j (w_j * z_j * q_j)
```

Closed form, vectorize over slots. **Important:** RULE_P never has a binding floor (every floored slice gets f_i plus a strictly positive share). All rho-sweep experiments MUST use RULE_W. This also fixes an internal inconsistency in draft v1 (Section 12).

**RULE_W (weighted welfare program; Instance B and all floored experiments).**

Solves `max_x sum_i w_i * theta-hat_i * q_i * (1 - exp(-x_i/s_i))` s.t. `sum x_i = B`, `x_i >= f_i` (f_i = 0 if unfloored). KKT gives waterfilling:

```
a_i(c)   = s_i * ln( w_i * z_i * q_i(c) / s_i )       # unconstrained level at nu=0
x_i(nu)  = max( f_i, a_i - s_i * nu )                  # nu = ln(mu), water level
```

`sum_i x_i(nu)` is continuous and nonincreasing in nu: **bisection** on nu (60 iters, tol 1e-9) to hit B. Vectorize the bisection across slots (arrays over t). Binding indicator per slot: `b_t = 1` iff exists floored i with `x_i = f_i` and `a_i - s_i*nu < f_i`. Own-report monotonicity holds (raising z_i raises a_i).

Unit-check both rules: feasibility (sum = B to 1e-6, x >= f), monotonicity in own report on a random grid.

### 3.4 Payments (Myerson threshold, per epoch)

For multiplicative valuations `v_i = z * h_i(x, c)` with `h_i = q_i*(1 - exp(-x/s_i))`:

```
abar_{i,k}(s) = sum_{t in E_k} h_i( g_i((s, z_{-i,k}), w_k, c_t), c_t )
p_{i,k}       = z_i * abar(z_i) - integral_{theta_lo_i}^{z_i} abar(s) ds
```

- Quadrature: trapezoid on a fixed grid of G = 41 points spanning Theta_i; reuse allocator calls across the grid (vectorize grid x slots). epsilon-DSIC error <= vbar*L/G; compute and report this number in the paper.
- Cost: G re-allocations of L slots per tenant per epoch. Cheap for RULE_P; for RULE_W, batch the bisection over (grid x slots).
- Realized utility per epoch: `U_{i,k} = sum_t theta_i * h_i(g_i(z_k, w_k, c_t), c_t) - p_{i,k}` (true type in the value, reported type in allocation and payment).

### 3.5 Learner (projected OGD on the simplex slice)

- Empirical plug-in objective (paper eq. (7)-(8)), using the **gated inputs** i-tilde and the epoch's realized channel path:

```
What_k(w) = (1/L) * sum_{t in E_k} sum_j  itilde_{j,k} * h_j( g_j(itilde_k, w, c_t), c_t )
```

- Gradient: central differences along an orthonormal basis of the tangent {sum dw = 0}: 1 direction for N=2, 2 for N=3; delta = 1e-3; reuse the same channel path for +/- evaluations.
- Update: `w <- Proj_W( w + eta * ghat )`. Projection onto {sum w = N, w >= w_lo}: shift to the hyperplane, then iteratively clamp coordinates below w_lo and redistribute the deficit equally over unclamped coordinates (converges in <= N passes).
- Step size: default eta = 0.1. Sensitivity {0.05, 0.1, 0.2} on TRUTHFUL runs only; fix one value per instance and freeze.
- Optional regularizer `- (lambda_reg/2)*||w - w0||^2` inside What_k, default lambda_reg = 0. Only enable via the debug ladder (Section 13), and document it in the paper if used.
- **Critical wiring rule:** allocation and payments always consume the RAW reports z_k; only What_k consumes the gated inputs itilde_k. The single most likely implementation bug is crossing these. V3 catches it.

### 3.6 Trust gates

Let m_{i,k} = epoch-k mean of A_{i,t}; mbar_{i,k} = cumulative mean over slots 1..(k-1)L (all past epochs). alpha = 1/T. sigma known from config (a plug-in sigma-hat variant is a P2 robustness check).

- **EPOCH(r):** `itilde = clip(z, m_k - r, m_k + r)` then clip to Theta_i. Radius r_L = (sigma/sqrt(L)) * sqrt(2*ln(2*N*K/alpha)). Sweep as multiples of r_L.
- **EPOCH0:** itilde = clip(m_k, Theta_i). (r = 0, per-epoch anchor; the draft's Thm 4.)
- **CUM(r_k):** anchor mbar_{i,k}, radius `r_k = sigma * sqrt(2*ln(2*N*K/alpha) / ((k-1)*L))`. Epoch 1: itilde = clip(z, Theta) (one epoch of full trust; contributes O(L), note in paper).
- **CUM0:** itilde = clip(mbar_k, Theta). (r = 0, running-mean anchor; the upgraded theorem.)
- **RINF:** itilde = clip(z, Theta). (r = infinity, the fragile baseline.)

---

## 4. Metrics

- **W\*.** Instance A: exact, 1.729329 per slot. Instance B: offline, once per (theta, load) config: draw a fixed pool of M = 2e5 channel vectors; `W(w; theta)` = pool average of `sum_j theta_j h_j(g_j(theta, w, c))` (truthful inputs, allocation at true types); maximize by coarse grid then Nelder-Mead; cache w* and W*. Use the SAME pool to evaluate `W(w_k; theta)` during regret computation (common randomness makes the regret curve smooth).
- **Regret(T)** = `sum_k L * ( W* - W(w_k; theta) )`, evaluated on the cached pool (or exactly for Instance A).
- **Slack(T)** (CRN protocol): per seed s, simulate TRUTH(s), then DEV(s, d) for each grid deviation d on identical arrays; `slack(s) = max_d [ U_1^dev(d) - U_1^truth ]`; report mean and 95% t-CI over seeds. Save the full gain-vs-d curve per config (needed for V2 and for the max-bias caveat: max of noisy estimates is upward biased, CRN keeps this negligible; state seeds and CI method in the paper).
- **rho:** fraction of slots with binding indicator, truthful run, RULE_W only.

## 5. Adversary protocol

- **Persistent grid:** tenant 1 (also repeat with S2 in Instance B, P1) reports z = theta_1 + d every epoch, d on a 41-point grid over [-0.1, +0.1] (Instance A) or [-0.2, +0.2]*theta_1 (Instance B). Others truthful.
- **Refinement rule:** if the empirical argmax sits at a grid edge or the peak is under-resolved, refine with 0.002 spacing around it (Instance A: expect the peak near d = -0.05).
- **Change-point (P1):** same grid applied from t = T/2, truthful before. Reported as a second curve where used.
- **Padding (E4 only):** joint grid d x delta, delta in linspace(0, 0.15, 16); padding shifts the traffic array mean by +delta and subtracts `c_pad * delta` per slot from utility.
- Never implement a learned/RL adversary for the paper; the grid is the protocol (state in the paper that grid best response is exhaustive within the modeled strategy family).

## 6. Validation suite (all must pass before ANY experiment; keep as pytest)

- **V1 Within-epoch DSIC.** Fix w and a channel path; sweep own report on a 201-grid; assert `u_i(z)` is maximized at z = theta_i within the epsilon-DSIC quadrature bound. Run for both allocators, 5 random configs.
- **V2 Appendix-B replication (Instance A, sigma = 0.05, L = 50, T = 1e5, RINF, 10 seeds).**
  (a) under persistent z, learned omega converges: |omega_K - omega*(z)| <= 0.01;
  (b) measured per-slot gain vs d overlays the analytic `gain(z)` curve within CIs across the grid;
  (c) empirical argmax z in 0.949 +/- 0.005 and peak gain within 20% of 1.843e-4.
- **V3 Invariance / no-leakage.** At CUM0 and EPOCH0, two runs with different persistent reports on the same seed must produce **bitwise identical weight paths**. Then the paired CRN slack over the grid must be <= the epsilon-DSIC bound. This is the canary for the wiring rule in 3.5.
- **V4 Calibration.** Traffic sd and measurement sd as in 3.2.
- **V5 Learner sanity.** Truthful-run per-epoch plug-in suboptimality decays like ~1/sqrt(k) (log-log slope in [-0.7, -0.3]); no oscillation at the projection boundary; regret curve concave in T.

Milestone gate: do not start Section 7 until `make validate` is green and the V2 overlay plot is checked by DR.

---

## 7. Experiments

Common defaults unless stated: 20 seeds, 95% t-CIs, CRN, eta frozen per instance after 3.5.

### E1 (P0) Fragility and the gate  ->  paper Fig. 2 (replaces H1 [TO RUN])

- **E1a slack vs horizon.** Instance A, sigma = 0.1, L = 50. T in {2.5e3, 5e3, 1e4, 2.5e4, 5e4, 1e5}. Mechanisms: RINF, EPOCH(r_L), EPOCH(2 r_L), EPOCH0, CUM(r_k), CUM0.
  - Expected: RINF linear with slope approaching c0 ~ 1.5e-4 (overlay the analytic line); EPOCH(r_L) bounded by a line of slope C*r_L; CUM variants ~ sqrt(T); EPOCH0/CUM0 flat at 0 (CI contains 0).
  - Falsified if: RINF sublinear (first refine the d-grid near -0.05, then check V2); r = 0 slack CI excludes 0 (V3 leak).
- **E1b slack vs epoch length (inset of Fig. 2).** Instance A, RINF, T = 5e4, L in {25, 50, 100, 200}. Expected: flat in L. This is the visual kill-shot for "just freeze longer".
- **E1c slack slope vs load (P1; table in the paper if figure space is tight).** Instance B + RULE_W, calibrated rho in {0, 0.05, 0.1, 0.2}, T = 5e4, RINF vs CUM0. Expected: RINF slope increasing in rho; CUM0 flat at 0 at every load.

### E2 (P1) Price of invariance and no-tax-on-honesty  ->  paper Fig. 3 (replaces H2 [TO RUN])

- Instance B, truthful tenants only, T = 5e4, L = 50, rho ~ 0.05 config. sigma_rel in {0.05, 0.1, 0.2, 0.4, 0.8}.
- Mechanisms: ORACLE (train on true theta), EPOCH(r_L), EPOCH0, CUM0.
- Expected: EPOCH(r_L) regret statistically equal to ORACLE (no tax on honesty); EPOCH0 gap grows linearly in sigma; CUM0 gap small and ~ sigma*sqrt(T)/T per slot.
- Falsified if: gated regret exceeds oracle materially -> the gate is clipping honest inputs; check r_L formula, alpha, sigma calibration.

### E3 (P0) Scaling and dissolution, per-epoch vs running-mean anchor  ->  paper Fig. 4 (replaces H3 [TO RUN]; demonstrates the upgraded theorem)

- Instance A. sigma in {0.05, 0.3}. T in {1e3, 3e3, 1e4, 3e4, 1e5}.
- Mechanisms: EPOCH0 and EPOCH(r_L) with per-T optimized L = clip(round(c*sigma*sqrt(T)), 10, 500), c calibrated once so L(T=1e4, sigma=0.3) ~ 30; CUM0 and CUM(r_k) with fixed L = 50.
- Plot (Regret + Slack)/sqrt(T) vs T, log-log, with least-squares slope fits printed on the figure.
- Expected: EPOCH-anchored curves at sigma = 0.3 have slope ~ +1/4 (the T^{3/4} law); CUM-anchored flat (slope ~ 0) at both sigmas. At sigma = 0.05 all flat.
- This figure carries the paper's upgraded headline: with a running-mean anchor the incentive cost of learning is O((1+sigma)sqrt(T)) for every constant sigma, not only below sigma <= T^{-1/2}.

### E4 (P2) Padding knee  ->  small figure or Table II

- Instance A, EPOCH(r_L), T = 2e4, L = 50, sigma = 0.1. Joint (d, delta) adversary grid.
- Compute C1 = 4*sqrt(N)*C_w*G_v*G_x*(1 + 2*theta_bar) numerically for the instance: G_v = max theta*phi' = 1.1; G_x from the quotient form of RULE_P on the weight/report box; C_w = G_wtheta/m with m = min over inputs of |d2W/dw1^2| on the slice (numerically ~= 1.0 for Instance A) and G_wtheta by finite differences of grad_w W in the input. Script this; report the number.
- Sweep c_pad in {0, 0.25, 0.5, 0.75, 1, 1.5, 2} * C1. Expected: slack elevated at small c_pad, knee at or below C1 (C1 is conservative; say so in the paper).

### E5 (P1) Regret-slack frontier vs baselines  ->  paper Fig. 5

- Instance B, rho ~ 0.05, sigma_rel = 0.2, T = 5e4, L = 50. Plot Slack/T vs Regret/T, one marker per mechanism/parameter:
  - Trust family: EPOCH(r), r in {0, 0.5, 1, 2, 4} * r_L; plus CUM0 and CUM(r_k).
  - B1 RINF (report-trained).
  - B2 DP-gradient: add Gaussian noise of scale lam*G_g to each epoch gradient, lam in {0.1, 0.3, 1, 3}. Stands in for weakly-DP online learning; expected: slack barely drops until lam is huge, by which point regret has exploded (one-shot protection only).
  - B3 Fines: raw-report training plus transfer kappa*L*max(|z - m_k| - r_L, 0), kappa in {0.5, 1, 2} * C1. Expected: comparable to the gate at kappa >= C1 (consistent with the draft's App. C.5).
  - B4 Burn-in commit: train on raw reports for the first T0 in {0.1, 0.2, 0.4}*T, then freeze weights. Expected: dominated; the attacker simply front-loads the lie.
  - B5 ORACLE and B6 static uniform weights (the two skyline/floor anchors of the plot).
- Expected: the trust family traces the lower-left frontier and Pareto-dominates B1-B4.
- Note for the paper: a faithful DGJ exploration baseline needs their long-term-cost setting; we map it conceptually in Remark 3 and use B4 (commitment) and B2 (DP) as the implementable representatives of those two defense ideas. Say this explicitly in the eval text.

### E6 (P2) Closed-loop traffic robustness (supports the new small-gain remark)

- Instance A, CUM0, T = 2e4, L = 50, sigma = 0.1. Feedback gain kappa_fb in {0, 0.1, 0.25, 0.5, 1.0} per 3.2.
- Expected: slack ~ 0 at kappa_fb = 0 and increasing in kappa_fb. One curve, 0.5-column figure or two sentences with numbers.

---

## 8. Baseline implementation notes

- B2 noise is added to the gradient AFTER the gate is bypassed (inputs = raw reports); it protects the learner, not the inputs. That is the point being tested.
- B4 freeze means no OGD updates after T0; inputs during burn-in are raw reports.
- All baselines share the identical allocator, payments, adversary protocol, seeds, and CRN arrays. Only the training pathway differs.

## 9. Figure production spec (IEEE, double-blind)

- matplotlib, PDF output, embedded fonts (`pdf.fonttype = 42`), width 3.5 in (1-col) or 7.16 in (2-col), 8 pt fonts, greyscale-legible (line styles + markers, not color alone; INFOCOM reviewers may print B/W).
- Shaded 95% CI bands. Consistent mechanism styling everywhere:
  RINF = red dashed; EPOCH(r_L) = blue solid; EPOCH0 = blue dotted; CUM(r_k) = green solid; CUM0 = green dotted; ORACLE = black thin; baselines = grey markers.
- Filenames: `fig2_fragility.pdf`, `fig3_invariance_price.pdf`, `fig4_scaling.pdf`, `fig5_frontier.pdf`, `fig6_padding.pdf` (optional). Each produced by one script reading `runs/*.csv` only.

## 10. Compute budget

- Instance A run (T = 1e5, RULE_P, payments G = 41): vectorized numpy, ~1-3 s. E1a total ~ 6 T-values x 6 mechanisms x 20 seeds x 42 runs ~ 30k runs: a few hours single-core; parallelize over seeds (16 workers -> under 1 hour).
- Instance B run (T = 5e4, RULE_W bisection + payment grid): ~5-20 s with vectorized bisection (numba if needed). E5 ~ a few hundred runs: an evening.
- Whole plan < ~300 core-hours. One decent workstation suffices. Cache aggressively (W*, load calibration, channel pools).

## 11. Day-by-day schedule (paper due 31 Jul 2026 AoE; abstract 24 Jul)

| Dates | Deliverable | Gate |
|---|---|---|
| Jul 12-14 | Repo scaffold; Instance A; RULE_P; payments; learner; V1-V5 green | DR checks V2 overlay before proceeding |
| Jul 15-17 | E1a + E1b complete; Fig. 2 draft | Fig. 2 to DR by Jul 17 evening |
| Jul 18-19 | E3 complete; Fig. 4 draft; Instance B + RULE_W built; load calibration done | Fig. 4 to DR by Jul 19 |
| Jul 20-22 | E5 frontier + E2; Figs. 3 and 5 drafts | All four figures to DR by Jul 22 |
| Jul 23-24 | Abstract text frozen; EDAS abstract registered (24 Jul AoE) | |
| Jul 24-27 | E1c (rho table), change-point curves, seeds to 20 everywhere; P2 (E4/E6) only if P0/P1 frozen | |
| Jul 28-29 | Figure freeze; every number into LaTeX per Section 12; anonymized repo up | |
| Jul 30 | Full read; page fit; PDF checks (fonts, B/W print) | |
| Jul 31 | Submit with buffer; do not touch runs after freeze | |

If any P0 slips past Jul 20, drop E2 and E5 to a single combined half-column figure and say so; do not compress validation.

## 12. Numbers and edits the paper must carry (the [TO RUN] replacement map)

- Sec. IX H1 -> Fig. 2 (+ E1b inset, E1c table). Fix the prediction sentence: the optimal deviation is d*(rho) = d0* + Theta(rho) with d0* ~ 0.042 at rho = 0 (the current "d* = Theta(rho)" contradicts the floor-free instance).
- Sec. IX H2 -> Fig. 3. Sec. IX H3 -> Fig. 4 (now also demonstrating the running-mean upgrade). Padding probe -> Fig. 6 or Table II.
- Report in the eval text: seeds and CI method; G = 41 and the epsilon-DSIC bound vbar*L/G; eta and how it was tuned (truthful runs only); r_L and r_k values used; the computed C1; empirical c0 vs the certified 1.48e-4; calibrated (f1, rho) pairs.
- State which allocator each experiment uses (RULE_P floor-free, RULE_W floored); the draft currently implies eq. (6) throughout, but eq. (6) never binds floors.
- Traffic marks are Gamma (sub-exponential); one sentence noting the theory's sub-Gaussian assumption and that results are insensitive (P2: rerun one config with truncated-normal marks as a robustness line).

## 13. If results disagree with theory: debug ladder (in order, never skip)

1. V1 fails -> payment quadrature: grid coverage of Theta_i, trapezoid orientation, allocator reuse across grid.
2. V2(a) fails (no convergence to omega*(z)) -> eta too large, projection bug, or gradient sign; halve eta on truthful runs and retest.
3. V2(b) shape mismatch -> allocator or rent formula; check x1(z; omega*(z)) = x1*(z) numerically.
4. V3 fails -> gated input leaking into allocation/payments or raw report leaking into training (the wiring rule, 3.5).
5. RINF slack sublinear -> d-grid too coarse near the peak (refine +/-[0.02, 0.07] at 0.002); then horizon too short relative to transient (transient ~ sqrt(L*T)); then V2.
6. Slack CIs too wide -> verify CRN (paired differences should have sd << unpaired); raise seeds.
7. Learner instability on Instance B -> set lambda_reg = 0.5, document, and flag to DR (this interacts with a known assumption caveat in the paper; the comparator statement must then be adjusted).
8. Anything still inconsistent -> stop, write a one-page note with plots, escalate. Do not tune.

## 14. Anonymous artifact for submission

- Fresh repository (no git history), anonymous host (e.g., anonymous.4open.science or a scrubbed GitHub org). No names, emails, institution strings, or local paths in code, configs, notebooks, or PDF metadata.
- `README.md`: environment, `make validate`, `make e1 ... e6`, `make figures`, expected runtimes, seed lists.
- INFOCOM 2027 is strictly double-blind and the submission must be self-contained (no pointers to a technical report, even anonymous). The repo link is optional and reviewers are not obliged to open it; the paper must stand without it.

## 15. Definition of done

- `make validate` green; Figs. 2-5 regenerate from cached CSVs; every [TO RUN] in Sec. IX replaced; every number in the text traceable to a CSV row; DR sign-off on each figure against its "expected" clause above; anonymized repo builds from a clean clone.

---

## 16. E8 (P2, added 12 Jul): RL generality demonstration ("any learner, including DRL, is covered by Thm 3")

- **Goal:** empirically demonstrate that Theorem 3 is algorithm-agnostic: replace OGD with a small RL agent whose inputs are measurement-measurable, and show slack remains statistically zero under the full adversary grid.
- **Setup:** Instance A, T = 5e4, L = 50, sigma = 0.1. Agent: tabular Q-learning (or tiny actor-critic) over a discretized weight grid w1 in linspace(0.9, 1.1, 21). State: (binned running-mean measurement mbar_k for each tenant, epoch index bucket). Reward: empirical measured-welfare proxy of epoch k computed from measurements and channel only (never from reports). Action: pick w1 for the next epoch. Allocation and payments unchanged (raw reports, RULE_P, Myerson).
- **Wiring rule applies verbatim:** the agent's state/reward must contain no report-derived quantity and no endogenous quantity (no queues). V3 (bitwise-identical policy paths across different report sequences on the same seed) must pass for the RL agent too before measuring slack.
- **Expected:** slack CI contains 0 at every deviation (Thm 3 covers any update rule); regret within ~2x of OGD is acceptable (RL is not the point; invariance is).
- **Paper use:** one short remark + one sentence in the eval: "replacing OGD with a measurement-fed DRL agent leaves slack at zero, as Theorem 3 predicts for any update rule." Do NOT present the main mechanism as RL.
- **Budget:** ~1 day including the V3 rerun. Only start after P0 and P1 are frozen.
