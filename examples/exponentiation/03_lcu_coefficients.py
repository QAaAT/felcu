"""
Complex coefficients for an LCU circuit
=======================================

An LCU circuit implements sum_k c_k e^{i k x}.  felcu gives the same c_k in
two orders:

    result.complex_coefficients = [c_{-m}, ..., c_0, ..., c_m]
        ordered by k: use it to evaluate or analyse the series.  This is the
        order stored by to_json().

    result.circuit_coefficients = [c_0, 0, c_1, c_{-1}, ..., c_m, c_{-m}]
        the order the compressed circuit consumes them in: pass this one to
        the circuit.  The extra zero is required by the compressed circuit,
        so it has 2m + 2 entries.

Circuit code that imports felcu should take result.circuit_coefficients
directly.  To paste the coefficients by hand, print
result.format_circuit_coefficients(), a bare Python list, together with
CIRCUIT_ORDER_NOTE.  The LCU subnormalisation alpha is the sum of the
magnitudes |c_k|, the same in either order.

Run with:
    python examples/exponentiation/03_lcu_coefficients.py
"""

import numpy as np

from felcu import CIRCUIT_ORDER_NOTE, solve_fe_coefficients

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

# The same coefficients in circuit order: what circuit code should consume
cc = result.circuit_coefficients
print(f"\nlen(circuit_coefficients) = {len(cc)}  (= 2m + 2, with a zero after c_0)")
print(f"sum |c_k| in circuit order = {np.sum(np.abs(cc)):.6f}  (= alpha)")

# ... and as a bare list to paste by hand
print(CIRCUIT_ORDER_NOTE)
print(result.format_circuit_coefficients())
