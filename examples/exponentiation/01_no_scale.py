"""
Exponentiation without matrix-scale information
===============================================

With no matrix or bounds, felcu fits g(x) = exp(x) on [-pi/eta, pi/eta]
(eta = 2 by default, so [-pi/2, pi/2]).

Run with:
    python examples/exponentiation/01_no_scale.py
"""

import numpy as np
from felcu import evaluate_fourier_series, solve_fe_coefficients

result = solve_fe_coefficients("exponentiation", target_error=1e-10, m_values=range(4, 33, 2))

print(f"fitted interval = [{-result.x_max:.4f}, {result.x_max:.4f}]")
print(f"basis           = {result.basis_used}")
print(f"m               = {result.m}")
print(f"Linf error      = {result.achieved_linf_error:.2e}")

x = np.linspace(-1.5, 1.5, 7)
g = evaluate_fourier_series(x, result.a0, result.ak, result.bk)
for xi, gi in zip(x, g):
    print(f"  x={xi:+.2f}  g(x)={gi:.10f}  exp(x)={np.exp(xi):.10f}")
