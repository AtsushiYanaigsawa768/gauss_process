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
    # --- Configuration ------------------------------------------------------
    N_TEST_POINTS = 50_000
    TEST_FILENAMES = {
        "SKE2024_data18-Apr-2025_1205.dat",
        "SKE2024_data16-May-2025_1609.dat",
    }

    # --- Load data ----------------------------------------------------------
    DEFAULT_DIR = Path("./gp/data")
    dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    dat_files = sorted(dir_path.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in '{dir_path}'")

    # Train / test containers
    omega_tr, mag_tr, phase_tr = [], [], []
    omega_te, mag_te, phase_te = [], [], []

    for f in dat_files:
        w, m, p = load_bode_data(f)
        if f.name in TEST_FILENAMES:
            omega_te.append(w), mag_te.append(m), phase_te.append(p)
        else:
            omega_tr.append(w), mag_tr.append(m), phase_tr.append(p)

    if not omega_tr or not omega_te:
        raise RuntimeError("Both train and test sets must contain data.")

    # Stack & sort
    omega_tr = np.hstack(omega_tr)
    mag_tr   = np.hstack(mag_tr)
    phase_tr = np.hstack(phase_tr)
    idx_tr   = np.argsort(omega_tr)
    omega_tr, mag_tr, phase_tr = omega_tr[idx_tr], mag_tr[idx_tr], phase_tr[idx_tr]

    omega_te = np.hstack(omega_te)
    mag_te   = np.hstack(mag_te)
    phase_te = np.hstack(phase_te)
    idx_te   = np.argsort(omega_te)
    omega_te, mag_te, phase_te = omega_te[idx_te], mag_te[idx_te], phase_te[idx_te]

    # Complex responses
    G_tr = mag_tr * np.exp(1j * phase_tr)
    G_te = mag_te * np.exp(1j * phase_te)

    # --- Prepare GP inputs --------------------------------------------------
    X_tr = np.log10(omega_tr).reshape(-1, 1)
    X_te = np.log10(omega_te).reshape(-1, 1)

    y_real_tr, y_imag_tr = G_tr.real, G_tr.imag

    # --- ITGP fit (real & imag) --------------------------------------------
    res_real = ITGP(X_tr, y_real_tr, alpha1=0.50, alpha2=0.975, nsh=2, ncc=2, nrw=1)
    res_imag = ITGP(X_tr, y_imag_tr, alpha1=0.50, alpha2=0.975, nsh=2, ncc=2, nrw=1)
    gp_real, gp_imag = res_real.gp, res_imag.gp

    # --- Prediction grids ---------------------------------------------------
    omega_dense = np.logspace(np.log10(min(omega_tr.min(), omega_te.min())),
                              np.log10(max(omega_tr.max(), omega_te.max())),
                              N_TEST_POINTS)
    X_dense = np.log10(omega_dense).reshape(-1, 1)

    y_real_dense, _ = gp_real.predict(X_dense)
    y_imag_dense, _ = gp_imag.predict(X_dense)
    H_dense = (y_real_dense + 1j * y_imag_dense).ravel()

    # --- Predict at train / test points -------------------------------------
    y_real_tr_pred, _ = gp_real.predict(X_tr)
    y_imag_tr_pred, _ = gp_imag.predict(X_tr)
    H_tr_pred = (y_real_tr_pred + 1j * y_imag_tr_pred).ravel()

    y_real_te_pred, _ = gp_real.predict(X_te)
    y_imag_te_pred, _ = gp_imag.predict(X_te)
    H_te_pred = (y_real_te_pred + 1j * y_imag_te_pred).ravel()

    # --- Hampel filter helper ----------------------------------------------
    def hampel_filter(x, window_size=7, n_sigmas=3):
        x = x.copy()
        k = 1.4826
        L = len(x)
        half_w = window_size // 2
        for i in range(L):
            start, end = max(i - half_w, 0), min(i + half_w + 1, L)
            window = x[start:end]
            med = np.median(window)
            mad = k * np.median(np.abs(window - med))
            if mad and np.abs(x[i] - med) > n_sigmas * mad:
                x[i] = med
        return x

    # Apply Hampel filtering (real & imag separately)
    G_tr_filt = hampel_filter(G_tr.real) + 1j * hampel_filter(G_tr.imag)
    G_te_filt = hampel_filter(G_te.real) + 1j * hampel_filter(G_te.imag)

    # --- Compute MSEs -------------------------------------------------------
    mse_tr = np.mean(np.abs(G_tr_filt - H_tr_pred) ** 2)
    mse_te = np.mean(np.abs(G_te_filt - H_te_pred) ** 2)

    # --- Nyquist plot -------------------------------------------------------
    order = np.argsort(omega_dense)
    plt.figure(figsize=(10, 6))
    plt.plot(G_tr_filt.real, G_tr_filt.imag, 'b.', label='Train (filtered)')
    plt.plot(G_te_filt.real, G_te_filt.imag, 'g*', label='Test (filtered)')
    plt.plot(H_dense.real[order], H_dense.imag[order], 'r-', lw=2, label='ITGP fit')
    plt.xlabel('Re{G}')
    plt.ylabel('Im{G}')
    plt.title(f'Nyquist Plot  |  Train MSE: {mse_tr:.3e}  |  Test MSE: {mse_te:.3e}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./gp/output/_nyquist_train_test.png", dpi=300)
    plt.show()

    # --- Console output -----------------------------------------------------
    print(f"Train Nyquist MSE : {mse_tr:.3e}")
    print(f"Test  Nyquist MSE : {mse_te:.3e}")

if __name__ == "__main__":
    main()
