
#!/usr/bin/env python3
"""
gp_fir_model_fixed.py

Drop-in replacement helpers to convert FRF → FIR using the *correct* mapping
from continuous ω [rad/s] to digital Ω [rad/sample] given Ts, plus (optional)
delay compensation. Provides an API similar to the original gp_fir_model.py.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import irfft

def frf_to_fir_with_Ts(omega: np.ndarray, G: np.ndarray, L: int,
                       Ts: float, tau_comp: Optional[float] = None,
                       N_fft: Optional[int] = None) -> np.ndarray:
    """
    Convert FRF (ω [rad/s] → G(jω)) to a *causal* FIR g[n], n=0..L-1 using:
      1) Digital grid Ω_k = 2πk/N_fft, k=0..N_fft/2
      2) ω_k = Ω_k / Ts, evaluate G(jω_k)
      3) Optional delay compensation: multiply by e^{+j ω_k τ}
      4) irfft → take first L taps
    """
    omega = np.asarray(omega, dtype=float).ravel()
    G = np.asarray(G, dtype=complex).ravel()
    if N_fft is None:
        N = 1
        while N < 4*L:
            N <<= 1
        N_fft = N
    if N_fft % 2 == 1:
        N_fft += 1

    k = np.arange(N_fft//2 + 1, dtype=int)
    Omega_k = 2.0 * np.pi * k / N_fft
    omega_k = Omega_k / float(Ts)

    interp_real = interp1d(omega, np.real(G), kind='cubic', bounds_error=False,
                           fill_value=(np.real(G[0]), np.real(G[-1])))
    interp_imag = interp1d(omega, np.imag(G), kind='cubic', bounds_error=False,
                           fill_value=(np.imag(G[0]), np.imag(G[-1])))
    Hk = interp_real(omega_k) + 1j*interp_imag(omega_k)

    if (tau_comp is not None) and np.isfinite(tau_comp):
        Hk = Hk * np.exp(1j * omega_k * tau_comp)

    Hk = np.asarray(Hk, dtype=complex)
    Hk[0] = np.real(Hk[0])
    if N_fft % 2 == 0:
        Hk[-1] = np.real(Hk[-1])

    # Mild taper near Nyquist
    if len(Hk) > 8:
        m = len(Hk)
        n_roll = max(4, int(0.1*m))
        w = np.ones(m); w[-n_roll:] *= 0.5*(1.0 + np.cos(np.linspace(0, np.pi, n_roll)))
        Hk = Hk * w

    h_full = irfft(Hk, n=N_fft)
    return h_full[:L]

def estimate_delay_from_phase(omega: np.ndarray, G: np.ndarray,
                              w_lo: float = None, w_hi: float = None) -> float:
    """φ(ω) ≈ -ω τ + c → τ = -dφ/dω (LS fit)."""
    phase = np.unwrap(np.angle(G))
    w = np.asarray(omega, dtype=float)
    if w_lo is not None or w_hi is not None:
        mask = np.ones_like(w, dtype=bool)
        if w_lo is not None:
            mask &= (w >= w_lo)
        if w_hi is not None:
            mask &= (w <= w_hi)
        if np.sum(mask) >= 3:
            w = w[mask]
            phase = phase[mask]
    A = np.vstack([w, np.ones_like(w)]).T
    slope, intercept = np.linalg.lstsq(A, phase, rcond=None)[0]
    return float(-slope)
