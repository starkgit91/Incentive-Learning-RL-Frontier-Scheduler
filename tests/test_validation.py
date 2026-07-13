"""Validation suite V1-V5. `make validate` must be green before any experiment runs.

These are not unit tests of convenience -- each one is a falsifier for a specific way
the simulator could silently disagree with the paper.
"""
from __future__ import annotations

import numpy as np
import pytest

from trustgate import analytic as AN
from trustgate.adversary import slack_for_seed
from trustgate.allocators import allocate
from trustgate.gates import CUM0, EPOCH0, RINF
from trustgate.instances import draw_channel, h, instance_a, instance_b
from trustgate.learner import project
from trustgate.mechanism import persistent_deviation, run, truthful
from trustgate.payments import eps_dsic_bound, epoch_payment
from trustgate.traffic import draw_traffic, epoch_means, sigma_vector

INSTANCES = [instance_a(), instance_b(f1=1.5)]


# --------------------------------------------------------------------------- #
# V0 -- the model in code IS the model in the paper
# --------------------------------------------------------------------------- #
def test_v0_certified_constants():
    got = AN.certified_constants()
    expect = dict(W_star_per_slot=1.729329, omega_star_prime_1=-0.5, kappa_R=0.014245,
                  u_prime_1=-0.0071227, z_star=0.94936, peak_gain=1.8430e-4,
                  d_star=0.041529, c0=1.4790e-4)
    for k, want in expect.items():
        assert abs(got[k] - want) / abs(want) < 0.02, f"{k}: {got[k]} != {want}"


# --------------------------------------------------------------------------- #
# V1 -- within-epoch DSIC: with w frozen, truth maximizes own utility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", INSTANCES, ids=lambda i: i.name)
def test_v1_within_epoch_dsic(inst):
    rng = np.random.default_rng(3)
    L, G = 40, 41
    eps = eps_dsic_bound(inst, L, G)
    for trial in range(5):
        q = draw_channel(inst, rng, L)
        w = project(rng.uniform(inst.weight_lo, 1.5, size=inst.n), inst)
        z = np.array(inst.theta, float)
        for i in range(inst.n):
            grid = np.linspace(inst.theta_lo[i], inst.theta_hi[i], 61)
            us = []
            for s in grid:
                zz = z.copy(); zz[i] = s
                x, _ = allocate(np.broadcast_to(zz, (L, inst.n)), w, q, inst)
                p, _ = epoch_payment(i, zz, w, q, inst, grid_size=G)
                # value at the TRUE type, allocation/payment at the REPORT
                us.append(float(np.sum(inst.theta[i] * h(x, q, inst.sat_arr)[:, i]) - p))
            us = np.array(us)
            u_truth = us[int(np.argmin(np.abs(grid - inst.theta[i])))]
            assert us.max() - u_truth <= eps + 1e-9, (
                f"{inst.name} tenant {i}: misreport beats truth by "
                f"{us.max()-u_truth:.3e} > eps_DSIC {eps:.3e}")


# --------------------------------------------------------------------------- #
# V2 -- Appendix-B replication (Instance A)
# --------------------------------------------------------------------------- #
def test_v2a_learner_converges_to_omega_star():
    inst = instance_a()
    for d in (-0.05, 0.0, 0.05):
        z = 1.0 + d
        rng = np.random.default_rng(11)
        A = draw_traffic(inst, 0.05, rng, 30000)
        q = draw_channel(inst, rng, 30000)
        r = run(inst, RINF, A, q, 50, 0.05, persistent_deviation(inst, 0, d), grid_size=11)
        om = r.w_path[-1][0] / r.w_path[-1][1]
        assert abs(om - AN.omega_star(z)) < 0.01


def test_v2bc_gain_curve_matches_analytic():
    inst = instance_a()
    grid = np.linspace(-0.1, 0.1, 41)
    T = 30000
    gains = np.mean([slack_for_seed(inst, RINF, seed=s, T=T, L=50, sigma=0.05,
                                    grid=grid, grid_size=11).gains
                     for s in (1, 2, 3)], axis=0) / T
    analytic = np.array([AN.gain(1.0 + d) for d in grid])
    # shape agrees everywhere, and the peak is where theory says it is
    assert np.max(np.abs(gains - analytic)) < 3e-5
    assert abs(grid[int(np.argmax(gains))] - (AN.certified_constants()["z_star"] - 1)) < 0.01
    peak = gains.max()
    assert abs(peak - AN.certified_constants()["peak_gain"]) / AN.certified_constants()["peak_gain"] < 0.20


# --------------------------------------------------------------------------- #
# V3 -- invariance / no-leakage. THE canary for the wiring rule.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gate", [EPOCH0, CUM0], ids=lambda g: g.label)
@pytest.mark.parametrize("inst", INSTANCES, ids=lambda i: i.name)
def test_v3_bitwise_weight_path_invariance(inst, gate):
    """At r = 0 the learner's input contains no report, so two runs whose reports
    differ MUST produce bitwise identical weight paths -- and hence zero slack."""
    rng = np.random.default_rng(5)
    T, L, sigma = 8000, 50, 0.1
    A = draw_traffic(inst, sigma, rng, T)
    q = draw_channel(inst, rng, T)
    base = run(inst, gate, A, q, L, sigma, truthful(inst), grid_size=11)
    for d in (-0.08, -0.03, 0.05):
        dev = run(inst, gate, A, q, L, sigma, persistent_deviation(inst, 0, d), grid_size=11)
        assert np.array_equal(base.w_path, dev.w_path), (
            f"{inst.name}/{gate.label}: weight path moved with the report "
            f"(max |dw| = {np.abs(base.w_path - dev.w_path).max():.2e}) -- a report is "
            f"leaking into the learner")


@pytest.mark.parametrize("gate", [EPOCH0, CUM0], ids=lambda g: g.label)
def test_v3_slack_is_exactly_zero(gate):
    inst = instance_a()
    for s in (1, 2, 3):
        r = slack_for_seed(inst, gate, seed=s, T=8000, L=50, sigma=0.1,
                           grid=np.linspace(-0.1, 0.1, 21), grid_size=11)
        assert r.slack == 0.0, f"{gate.label}: slack {r.slack:.3e} != 0"


def test_v3_rinf_is_fragile():
    """The control: without the gate the channel really is open (else V3 proves nothing)."""
    inst = instance_a()
    T = 10000
    sl = np.mean([slack_for_seed(inst, RINF, seed=s, T=T, L=50, sigma=0.05,
                                 grid=np.linspace(-0.1, 0.1, 21), grid_size=11).slack
                  for s in (1, 2, 3)])
    c0T = AN.certified_constants()["c0"] * T
    assert sl > 0.5 * c0T, "RINF should be fragile with slope ~ c0"


# --------------------------------------------------------------------------- #
# V4 -- traffic / measurement calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", INSTANCES, ids=lambda i: i.name)
def test_v4_calibration(inst):
    rng = np.random.default_rng(9)
    sigma, T, L = 0.2, 200_000, 50
    A = draw_traffic(inst, sigma, rng, T)
    sig = sigma_vector(inst, sigma)
    assert np.allclose(A.mean(0), inst.theta_arr, rtol=0.02)
    assert np.allclose(A.std(0), sig, rtol=0.02)
    m = epoch_means(A, L)
    assert np.allclose(m.std(0), sig / np.sqrt(L), rtol=0.05)


# --------------------------------------------------------------------------- #
# V5 -- learner sanity on truthful runs
# --------------------------------------------------------------------------- #
def test_v5_learner_converges_and_regret_is_sane():
    from trustgate.metrics import WelfareOracle
    inst = instance_a()
    oracle = WelfareOracle(inst)
    rng = np.random.default_rng(21)
    T, L = 40000, 50
    A = draw_traffic(inst, 0.1, rng, T)
    q = draw_channel(inst, rng, T)
    r = run(inst, EPOCH0, A, q, L, 0.1, truthful(inst), grid_size=11)
    # the truthful, gated learner should find the first-best weight
    assert np.allclose(r.w_path[-1], oracle.w_star, atol=0.02)
    # and regret must be sublinear (concave in T)
    reg = oracle.regret(r.w_path, L)
    assert 0 <= reg < 0.05 * T * oracle.W_star
