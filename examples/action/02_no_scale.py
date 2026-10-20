"""
Action without matrix-scale information
=======================================

If no matrix or spectral bounds are given, felcu fits the unscaled function
g(x) = x directly on [-pi/eta, pi/eta].  You are then responsible for
rescaling H so that its spectrum lies inside that interval.

Run with:
    python examples/action/02_no_scale.py
"""

from felcu import evaluate_fourier_series, solve_fe_coefficients

result = solve_fe_coefficients("action", target_error=1e-6, m_values=range(4, 33, 2))

print(f"fitted interval = [{-result.x_max:.4f}, {result.x_max:.4f}]   (eta = {result.eta})")
print(f"m               = {result.m}")
print(f"Linf error      = {result.achieved_linf_error:.2e}")

# Spot-check the series against g(x) = x
for x in [-1.0, -0.25, 0.0, 0.5, 1.5]:
    g = evaluate_fourier_series(x, result.a0, result.ak, result.bk)
    print(f"  g({x:+.2f}) = {float(g):+.8f}")
