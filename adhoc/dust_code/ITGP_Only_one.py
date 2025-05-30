#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""itgp_bode_fit.py

Iteratively-trimmed Gaussian process (ITGP) smoothing of measured frequency‑response data
for robust fitting in the presence of outliers (improved version).
速度が少し遅い、精度も若干落ちる
一度しかGPを実行しないという点ではいい。
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
    N_TEST_POINTS = 500

    # Data loading code unchanged
    # DEFAULT_DIR = Path("data_prepare")
    # dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    # dat_files = sorted(dir_path.glob("*.dat"))
    # if not dat_files:
    #     raise FileNotFoundError(f"No .dat files found in '{dir_path}'")
    # omega_list, mag_list, phase_list = [], [], []
    # for f in dat_files:
    #     w, m, p = load_bode_data(f)
    #     omega_list.append(w)
    #     mag_list.append(m)
    #     phase_list.append(p)
    # omega = np.hstack(omega_list)
    # mag   = np.hstack(mag_list)
    # phase = np.hstack(phase_list)
    # idx = np.argsort(omega)
    # omega, mag, phase = omega[idx], mag[idx], phase[idx]
    # G_meas = mag * np.exp(1j * phase)

    DEFAULT_FILE = Path("data_prepare/SKE2024_data16-Apr-2025_1819.dat")
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    omega, mag, phase = load_bode_data(filepath)
    G_meas = mag * np.exp(1j * phase)
    
    # 2) Prepare modeling targets for real and imaginary parts
    X0 = np.log10(omega).reshape(-1,1)          # (N,1)
    X_real = np.hstack([X0, np.zeros_like(X0)])  # (N,2) t=0
    X_imag = np.hstack([X0, np.ones_like(X0)])   # (N,2) t=1

    X_aug = np.vstack([X_real, X_imag])         # (2N,2)
    y_aug = np.hstack([G_meas.real, G_meas.imag])# (2N,)

    # --- ITGP を一度だけ実行 ---
    res = ITGP(
        X_aug, y_aug,
        alpha1=0.50,
        alpha2=0.975,
        nsh=2,
        ncc=2,
        nrw=1
    )
    gp = res.gp

    # 5) Dense prediction grid
    omega_test = np.logspace(
        np.log10(omega.min()),
        np.log10(omega.max()),
        N_TEST_POINTS
    )
    X_test = np.log10(omega_test).reshape(-1, 1)

    # Predict real part using augmented GP on test grid
    X_test_real = np.hstack([X_test, np.zeros_like(X_test)])
    y_real_pred, y_real_std = gp.predict(X_test_real)
    y_real_pred = y_real_pred.ravel()
    y_real_std = y_real_std.ravel()

    # Predict imaginary part using augmented GP on test grid
    X_test_imag = np.hstack([X_test, np.ones_like(X_test)])
    y_imag_pred, y_imag_std = gp.predict(X_test_imag)
    y_imag_pred = y_imag_pred.ravel()
    y_imag_std = y_imag_std.ravel()
    
    # Skip Bode plots of magnitude and phase

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
    # GP prediction at original ω for fair comparison
    y_real_meas_pred, _ = gp.predict(X_real)
    y_imag_meas_pred, _ = gp.predict(X_imag)
    y_real_meas_pred = y_real_meas_pred.ravel()
    y_imag_meas_pred = y_imag_meas_pred.ravel()
    H_pred_meas = y_real_meas_pred + 1j * y_imag_meas_pred
    y_imag_meas_pred = y_imag_meas_pred.ravel()
    H_pred_meas = y_real_meas_pred + 1j * y_imag_meas_pred

    # Hampel‐filter the measured data (do NOT use this for GP fitting)
    # define dataset and apply Hampel filter to real and imaginary parts
    G_dataset   = G_meas
    G_real_filt = hampel_filter(G_dataset.real)
    G_imag_filt = hampel_filter(G_dataset.imag)
    G_filt      = G_real_filt + 1j * G_imag_filt

    # Measured real and imaginary parts for plotting
    y_real = G_dataset.real
    y_imag = G_dataset.imag
    G_imag_filt = hampel_filter(G_dataset.imag)
    G_filt = G_real_filt + 1j * G_imag_filt
    plt.figure(figsize=(8, 4))
    plt.loglog(omega, y_real, 'b.', label='Measured Real')
    plt.loglog(omega_test, y_real_pred, 'r-', label='Predicted Real')
    plt.fill_between(
        omega_test,
        y_real_pred - 2 * y_real_std,
        y_real_pred + 2 * y_real_std,
        color='r',
        alpha=0.2,
        label='±2σ'
    )
    plt.xlabel('ω (rad/s)')
    plt.ylabel('Re{G}')
    plt.title('Real Part: Measured vs Predicted')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig("_real_fit.png", dpi=300)
    plt.show()

    # Imaginary part plot: measured vs. predicted
    plt.figure(figsize=(8, 4))
    plt.loglog(omega, y_imag, 'g.', label='Measured Imag')
    plt.loglog(omega_test, y_imag_pred, 'm-', label='Predicted Imag')
    plt.fill_between(
        omega_test,
        y_imag_pred - 2 * y_imag_std,
        y_imag_pred + 2 * y_imag_std,
        color='m',
        alpha=0.2,
        label='±2σ'
    )
    plt.xlabel('ω (rad/s)')
    plt.ylabel('Im{G}')
    plt.title('Imaginary Part: Measured vs Predicted')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig("_imag_fit.png", dpi=300)
    plt.show()
    # Compute MSE on complex Nyquist points
    mse = np.mean(np.abs(G_filt - H_pred_meas)**2)
    print(f"Nyquist MSE (after Hampel filter): {mse:.4e}")

    # Plot
    order = np.argsort(omega_test)
    plt.figure(figsize=(10, 6))
    plt.plot(G_filt.real, G_filt.imag, 'b*', markersize=6, label='Filtered Data')
    H_best = y_real_pred + 1j * y_imag_pred
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
