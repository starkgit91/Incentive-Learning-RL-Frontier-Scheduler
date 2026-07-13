"""Closed-form ground truth for Instance A (paper Appendix B).

This module is the yardstick the whole simulator is checked against. Everything
here is derived, not fitted, so if a simulation disagrees with it the simulation
is wrong (see the debug ladder in the plan, Sec. 13).

The setup. Tenant 1 persistently reports z (tenant 2 truthful, z2 = 1). A learner
that trains on *reports* converges to the weight that maximizes the reported
welfare, and that weight is a function of z -- which is precisely the cross-epoch
channel a strategic tenant can steer. We can compute the whole thing in closed
form:

* the reported-welfare-optimal split of the budget:
      x1*(a) = B/2 + (s/2) ln a          (B=4, s=1  =>  2 + 0.5 ln a)
  from d/dx1 [ a(1-e^{-x1}) + (1-e^{-(B-x1)}) ] = 0.
* the weight RULE_P needs to realize it (omega = w1/w2):
      omega*(a) = x1*(a) / ( a (B - x1*(a)) )
* tenant 1's realized (Myerson) utility at true type 1 while reporting z, with the
  learner's weight frozen at omega*(z):
      u(z) = (1 - z) phi(x1*(z)) + int_{theta_lo}^{z} phi(x1(s; omega*(z))) ds
  where phi(x) = 1 - e^{-x} and x1(s; w) = B w s / (w s + 1) is RULE_P.

The per-slot manipulation gain is u(z) - u(1). It is *positive* for z < 1: by
shading its report the tenant tilts the learner's weight in a way that is worth
more than the extra payment it saves. That gain is Theta(1) per slot, hence
Slack = Theta(T) -- Theorem 1 (fragility).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

B = 4.0
S = 1.0
THETA_LO = 0.9
THETA_HI = 1.1


def phi(x):
    return 1.0 - np.exp(-x / S)


def x1_star(a):
    """Reported-welfare-optimal share for tenant 1 when it reports a (tenant 2 = 1)."""
    return B / 2.0 + (S / 2.0) * np.log(a)


def omega_star(a):
    """RULE_P weight ratio w1/w2 that realizes x1_star(a) given reports (a, 1)."""
    xs = x1_star(a)
    return xs / (a * (B - xs))


def x1_rule_p(s, omega):
    """RULE_P share for tenant 1 reporting s, weight ratio omega, tenant 2 reporting 1."""
    return B * omega * s / (omega * s + 1.0)


def abar(s, omega):
    """Per-slot allocation-value a-bar_1(s) = h_1(g_1(s, omega)) = phi(x1(s; omega))."""
    return phi(x1_rule_p(s, omega))


def utility(z, theta1: float = 1.0):
    """Tenant 1's per-slot Myerson utility at TRUE type theta1 while REPORTING z,
    with the learner's weight frozen at the value a report-trained learner converges
    to, omega*(z).  u = theta*abar(z) - p,  p = z*abar(z) - int_lo^z abar(s) ds."""
    w = omega_star(z)
    integral, _ = integrate.quad(lambda s: abar(s, w), THETA_LO, z, limit=200)
    return (theta1 - z) * abar(z, w) + integral


def gain(z, theta1: float = 1.0):
    """Per-slot manipulation gain of the persistent report z (0 at z = theta1)."""
    return utility(z, theta1) - utility(theta1, theta1)


def certified_constants(verbose: bool = False) -> dict:
    """Recompute every constant the plan certifies. These are DERIVED here, so a
    mismatch means the model in code differs from the model in the paper."""
    # optimal persistent report and its per-slot gain
    res = optimize.minimize_scalar(lambda z: -gain(z), bounds=(THETA_LO, 1.0),
                                   method="bounded", options={"xatol": 1e-10})
    z_star = float(res.x)
    peak = float(gain(z_star))

    # first-order deviation gain u'(1) and the curvature bound kappa_b
    eps = 1e-6
    u_prime_1 = float((utility(1.0 + eps) - utility(1.0 - eps)) / (2 * eps))
    zs = np.linspace(THETA_LO, 1.0, 401)
    u_vals = np.array([utility(z) for z in zs])
    u_second = np.gradient(np.gradient(u_vals, zs), zs)
    kappa_b = float(np.max(np.abs(u_second)) / 2.0)

    # rent slope in omega at omega = 1 (kappa_R) and d omega*/da at a = 1
    domega_da_1 = float((omega_star(1 + eps) - omega_star(1 - eps)) / (2 * eps))
    # kappa_R := |du/domega| at omega=1  =>  u'(1) = kappa_R * domega_da(1)
    kappa_R = float(abs(u_prime_1 / domega_da_1)) if domega_da_1 != 0 else float("nan")

    # certified lower-bound optimum of the quadratic surrogate:
    #   gain(1-d) >= |u'(1)| d - kappa_b d^2   =>  d* = |u'(1)|/(2 kappa_b),
    #   c0 = |u'(1)|^2 / (4 kappa_b)
    d_star = float(abs(u_prime_1) / (2.0 * kappa_b))
    c0 = float(abs(u_prime_1) ** 2 / (4.0 * kappa_b))

    out = {
        "W_star_per_slot": float(2.0 * (1.0 - np.exp(-2.0))),
        "omega_star_prime_1": domega_da_1,
        "kappa_R": kappa_R,
        "u_prime_1": u_prime_1,
        "z_star": z_star,
        "peak_gain": peak,
        "kappa_b": kappa_b,
        "d_star": d_star,
        "c0": c0,
    }
    if verbose:
        for k, v in out.items():
            print(f"  {k:20s} = {v:.8g}")
    return out


if __name__ == "__main__":
    print("Instance A certified constants (recomputed from the model):")
    got = certified_constants(verbose=True)
    expect = {
        "W_star_per_slot": 1.729329,
        "omega_star_prime_1": -0.5,
        "kappa_R": 0.014245,
        "u_prime_1": -0.0071227,
        "z_star": 0.94936,
        "peak_gain": 1.8430e-4,
        "kappa_b": 0.085755,
        "d_star": 0.041529,
        "c0": 1.4790e-4,
    }
    print("\ncheck vs plan:")
    ok = True
    for k, want in expect.items():
        g = got[k]
        rel = abs(g - want) / max(abs(want), 1e-12)
        good = rel < 0.02
        ok &= good
        print(f"  {k:20s} got={g: .8g}  want={want: .8g}  rel={rel:.2%}  {'OK' if good else 'MISMATCH'}")
    print("\nALL MATCH" if ok else "\nMISMATCH -- model in code != model in paper")
