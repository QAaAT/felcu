"""
Complex coefficients for an LCU circuit
=======================================

An LCU circuit implements sum_k c_k e^{i k x}.  felcu returns these as
result.complex_coefficients = [c_{-m}, ..., c_0, ..., c_m].  The LCU
subnormalisation alpha is the sum of the magnitudes |c_k|.

Run with:
    python examples/exponentiation/03_lcu_coefficients.py
"""

import numpy as np

from felcu import solve_fe_coefficients

result = solve_fe_coefficients(
    "exponentiation",
    sigma_min=-1.0,
    sigma_max=1.0,
    target_error=1e-6,
    m_values=range(4, 33, 2),  # short sweep for speed; default goes to m=256
)
c = result.complex_coefficients
m = result.m

print(f"m = {m},  len(c) = {len(c)}  (= 2m + 1)")
print(f"{'k':>4}  {'|c_k|':>10}")
for k, ck in zip(range(-m, m + 1), c):
    print(f"{k:>4}  {abs(ck):>10.2e}")

# Evaluate the exponential form at a point and compare with exp(mu + x / tau)
mu = 0.5 * (result.sigma_min + result.sigma_max)
x = 0.3
g = np.sum(c * np.exp(1j * np.arange(-m, m + 1) * x))
print(f"\nsum c_k e^(ikx) at x={x}: {g.real:.8f}   (imag {g.imag:.1e})")
print(f"exp(mu + x / tau):         {np.exp(mu + x / result.tau):.8f}")
print(f"alpha = {result.alpha:.6f}")
