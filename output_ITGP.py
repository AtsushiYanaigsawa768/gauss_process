#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
itgp_bode_fit.py

Final version: Iteratively-Trimmed Gaussian Process (ITGP) smoothing of measured
Bode frequency-response data with robust handling of outliers and aligned Nyquist plot.
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
    #　データの個数を出力
    print(f"Number of data points: {len(omega)}")
    return omega, mag, phase


def main():
    # Configuration
    DEFAULT_DATAFILE = "data_prepare/SKE2024_data16-Apr-2025_1819.dat"
    N_TEST_POINTS = 500

    # 1) Load data
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_DATAFILE)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    omega_raw, mag_raw, phase_raw = load_bode_data(path)
    idx = np.argsort(omega_raw)
    omega = omega_raw[idx]
    mag   = mag_raw[idx]
    phase = phase_raw[idx]

    # 2) Prepare inputs
    X = np.log10(omega).reshape(-1, 1)  # log-scale frequency

    # 3) Model targets with centering
    # Magnitude in dB
    y_mag_db = 20.0 * np.log10(mag)
    mu_mag   = np.mean(y_mag_db)
    y_mag_c  = y_mag_db - mu_mag

    # Phase unwrapped and centered
    y_phase = np.unwrap(phase)
    mu_phase = np.mean(y_phase)
    y_phase_c = y_phase - mu_phase

    # 4) Apply ITGP for magnitude
    res_gain = ITGP(
        X, y_mag_c,
        alpha1=0.1,  # lower trim threshold (retain low-frequency points)
        alpha2=0.9,  # upper trim threshold
        nsh=2, ncc=2, nrw=1
    )
    gp_gain = res_gain.gp

    # 5) Apply ITGP for phase
    res_phase = ITGP(
        X, y_phase_c,
        alpha1=0.1,
        alpha2=0.9,
        nsh=2, ncc=2, nrw=1
    )
    gp_phase = res_phase.gp

    # 6) Dense evaluation grid (avoid extreme extrapolation)
    omega_test = np.logspace(
        np.log10(omega.min() * 1.01),
        np.log10(omega.max() * 0.99),
        N_TEST_POINTS
    )
    X_test = np.log10(omega_test).reshape(-1, 1)

    # 7) Predictions
    # Magnitude
    y_mag_pred_c, y_mag_std = gp_gain.predict(X_test)
    y_mag_pred = y_mag_pred_c.ravel() + mu_mag
    y_mag_std  = y_mag_std.ravel()
    y_mag_up   = y_mag_pred + 1.96 * y_mag_std
    y_mag_lo   = y_mag_pred - 1.96 * y_mag_std

    # Phase
    y_phase_pred_c, y_phase_std = gp_phase.predict(X_test)
    y_phase_pred = y_phase_pred_c.ravel() + mu_phase
    y_phase_std  = y_phase_std.ravel()
    # Wrap back into [-π, π]
    y_phase_wrapped = (y_phase_pred + np.pi) % (2 * np.pi) - np.pi

    # 8) Plot Bode magnitude
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(omega, y_mag_db, 'b*', label='Observed (gain)')
    ax.semilogx(omega_test, y_mag_pred, 'r-', lw=2, label='ITGP fit')
    ax.fill_between(
        omega_test,
        y_mag_lo,
        y_mag_up,
        color='red', alpha=0.25,
        label='95% CI'
    )
    ax.set_xlabel(r'$\omega$ [rad/s]')
    ax.set_ylabel(r'$20\log_{10}|G(j\omega)|$ [dB]')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig('itgp_fit.png', dpi=300)

    # 9) Plot Bode phase
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(omega, phase, 'b*', label='Observed (phase)')
    ax.semilogx(omega_test, y_phase_pred, 'r-', lw=2, label='ITGP fit')
    ax.fill_between(
        omega_test,
        y_phase_pred - 1.96 * y_phase_std,
        y_phase_pred + 1.96 * y_phase_std,
        color='red', alpha=0.25,
        label='95% CI'
    )
    ax.set_xlabel(r'$\omega$ [rad/s]')
    ax.set_ylabel('Phase [rad]')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig('itgp_phase.png', dpi=300)

    # 10) Nyquist plot
    G_dataset = mag * np.exp(1j * phase)
    H_best    = 10**(y_mag_pred/20) * np.exp(1j * y_phase_wrapped)
    order = np.argsort(omega_test)

    plt.figure(figsize=(10, 6))
    plt.plot(
        G_dataset.real, G_dataset.imag,
        'b*', markersize=6, label='Data'
    )
    plt.plot(
        H_best.real[order], H_best.imag[order],
        'r-', linewidth=2, label='ITGP Est.'
    )
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Nyquist Plot')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('itgp_nyquist.png', dpi=300)
    plt.show()
    # 二つのcsvファイルを出力
    # 一つ目：一行目に周波数ω、二行目に、|G|*exp(j*arg G) (元データ)を出力
    Path("result").mkdir(exist_ok=True)
    data_out = np.vstack([omega, mag * np.exp(1j * phase)])
    # Save as two columns: omega, real, imag
    np.savetxt("result/ITGP_data_t.csv",
               np.column_stack([omega, np.real(mag * np.exp(1j * phase)), np.imag(mag * np.exp(1j * phase))]),
               delimiter=",",
               header="omega,Re_G_data,Im_G_data",
               comments='',
               fmt=['%.10e','%.10e','%.10e'])

    # 二つ目：一行目に周波数ω、二行目に、|G|*exp(j*arg G) (ガウス過程回帰を用いた時の値)を出力
    fit_out = np.vstack([omega_test, 10**(y_mag_pred/20) * np.exp(1j * y_phase_wrapped)])
    np.savetxt("result/ITGP_data_fit_t.csv",
               np.column_stack([omega_test, np.real(10**(y_mag_pred/20) * np.exp(1j * y_phase_wrapped)), np.imag(10**(y_mag_pred/20) * np.exp(1j * y_phase_wrapped))]),
               delimiter=",",
               header="omega,Re_G_fit,Im_G_fit",
               comments='',
               fmt=['%.10e','%.10e','%.10e'])
    
    print("Data saved to result/ITGP_data_t.csv and result/ITGP_data_fit_t.csv")

if __name__ == '__main__':
    main()
