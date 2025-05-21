#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""itgp_bode_fit.py

Iteratively-trimmed Gaussian process (ITGP) smoothing of measured frequency‑response data
for robust fitting in the presence of outliers (improved version).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from robustgp import ITGP
import warnings

warnings.filterwarnings("ignore")


def load_bode_data(filepath: Path):
    """Load three‑column (ω, |G|, arg G) data from *filepath* (CSV)."""
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase

def main():
    # Configuration
    # DEFAULT_DATAFILE = "./data_prepare/SKE2024_data16-Apr-2025_1819.dat"
    N_TEST_POINTS = 500

    # Data file path
    # --- 1) データ読み込み・前処理 ----------------------------
    DEFAULT_DIR = Path("data_prepare")
    dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    dat_files = sorted(dir_path.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in '{dir_path}'")
    omega_list, mag_list, phase_list = [], [], []
    for f in dat_files:
        w, m, p = load_bode_data(f)
        omega_list.append(w)
        mag_list.append(m)
        phase_list.append(p)
    omega = np.hstack(omega_list)
    mag   = np.hstack(mag_list)
    phase = np.hstack(phase_list)
    idx = np.argsort(omega)
    omega, mag, phase = omega[idx], mag[idx], phase[idx]
    G_meas = mag * np.exp(1j * phase)
    # 2) Prepare modelling targets --------------------------------------------
    # Log-scale input for stability
    X = np.log10(omega).reshape(-1, 1)

    # Magnitude in dB
    y_mag_db = 20.0 * np.log10(mag)

    # Unwrap phase to remove 2π discontinuities
    y_phase = np.unwrap(phase)

    # 3) Apply ITGP for magnitude ---------------------------------------------
    res_gain = ITGP(
        X, y_mag_db,
        alpha1=0.50,   # trim fraction lower
        alpha2=0.975,   # trim fraction upper
        nsh=15,
        ncc=2,
        nrw=1
    )
    gp_gain, cons_gain = res_gain.gp, res_gain.consistency

    # 4) Apply ITGP for phase -------------------------------------------------
    res_phase = ITGP(
        X, y_phase,
        alpha1=0.50,
        alpha2=0.975,
        nsh=15,
        ncc=2,
        nrw=1
    )
    gp_phase, cons_phase = res_phase.gp, res_phase.consistency

    # 5) Dense prediction grid -------------------------------------------------
    omega_test = np.logspace(
        np.log10(omega.min()),
        np.log10(omega.max()),
        N_TEST_POINTS
    )
    X_test = np.log10(omega_test).reshape(-1, 1)

    # Predict magnitude
    y_mag_pred, y_mag_std = gp_gain.predict(X_test)
    y_mag_pred = y_mag_pred.ravel()
    y_mag_std  = y_mag_std.ravel()

    y_mag_up = y_mag_pred + 1.96 * y_mag_std
    y_mag_lo = y_mag_pred - 1.96 * y_mag_std

    # Predict phase
    y_phase_pred, y_phase_std = gp_phase.predict(X_test)
    y_phase_pred = y_phase_pred.ravel()
    y_phase_std  = y_phase_std.ravel()

    # 6) Plot Bode magnitude --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(omega, y_mag_db, "b*", label="Observed (gain)")
    ax.semilogx(omega_test, y_mag_pred, "r-", lw=2, label="ITGP fit")
    ax.fill_between(
        omega_test,
        y_mag_lo,
        y_mag_up,
        color="red",
        alpha=0.25,
        label="95 % CI",
    )
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$20\,\log_{10}|G(j\omega)|$ [dB]")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("_itgp_fit.png", dpi=300)

    # 7) Plot Bode phase ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(omega, phase, "b*", label="Observed (phase)")
    ax.semilogx(omega_test, y_phase_pred, "r-", lw=2, label="ITGP fit")
    ax.fill_between(
        omega_test,
        y_phase_pred - 1.96 * y_phase_std,
        y_phase_pred + 1.96 * y_phase_std,
        color="red",
        alpha=0.25,
        label="95 % CI",
    )
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("Phase [rad]")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("_itgp_phase.png", dpi=300)

    # 8) Create Nyquist plot with MSE after Hampel filtering -------------------
    # Helper: Hampel filter for real‐valued 1D arrays
    def hampel_filter(x, window_size=7, n_sigmas=3):
        x = x.copy()
        k = 1.4826  # scale factor for Gaussian
        L = len(x)
        half_w = window_size // 2
        for i in range(L):
            start = max(i - half_w, 0)
            end   = min(i + half_w + 1, L)
            window = x[start:end]
            med = np.median(window)
            mad = k * np.median(np.abs(window - med))
            if mad and np.abs(x[i] - med) > n_sigmas * mad:
                x[i] = med
        return x

    # Original data
    G_dataset = mag * np.exp(1j * phase)

    # GP prediction at original ω for fair comparison
    y_mag_meas_pred, _   = gp_gain.predict(X)
    y_phase_meas_pred, _ = gp_phase.predict(X)
    y_mag_meas_pred   = y_mag_meas_pred.ravel()
    y_phase_meas_pred = y_phase_meas_pred.ravel()
    H_pred_meas = 10**(y_mag_meas_pred/20) * np.exp(1j * y_phase_meas_pred)

    # Hampel‐filter the measured data (do NOT use this for GP fitting)
    G_real_filt = hampel_filter(G_dataset.real)
    G_imag_filt = hampel_filter(G_dataset.imag)
    G_filt = G_real_filt + 1j * G_imag_filt

    # Compute MSE on complex Nyquist points
    mse = np.mean(np.abs(G_filt - H_pred_meas)**2)
    print(f"Nyquist MSE (after Hampel filter): {mse:.4e}")

    # Plot
    order = np.argsort(omega_test)
    plt.figure(figsize=(10, 6))
    plt.plot(G_filt.real, G_filt.imag, 'b*', markersize=6, label='Filtered Data')
    H_best = 10**(y_mag_pred/20) * np.exp(1j * y_phase_pred)
    plt.plot(
        H_best.real[order],
        H_best.imag[order],
        'r-', linewidth=2,
        label='ITGP Est.'
    )
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title(f'Nyquist Plot (MSE: {mse:.4e})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("_nyquist.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
