# -*- coding: utf-8 -*-
"""gpr_bode_fit.py

Gaussian‑process regression (GPR) smoothing of measured frequency‑response data
with proper input scaling and noise estimation.  The script reads a three‑column
text file containing (omega, |G(jw)|, arg(G(jw))) and produces a Bode‑magnitude
plot with 95 % confidence bands.

Key improvements (vs. original draft)
------------------------------------
* **Log‑frequency input** – model is trained on log10(ω) to equalise the scale
  across decades.
* **StandardScaler** – zero‑mean/unit‑variance transform for GP hyper‑parameter
  optimisation stability.
* **Physically meaningful target** – magnitude is converted to dB once,
  avoiding unit inconsistencies.
* **Flexible kernel** – Constant × RBF + WhiteKernel lets the optimiser learn
  both smoothness and heteroscedastic noise.
* **More realistic uncertainty** – predictive variance shrinks in
  data‑dense bands and grows where samples are sparse.

Usage
-----
    python gpr_bode_fit.py path/to/datafile.dat

If no path is supplied, the default file used in the original notebook is
assumed (edit *DEFAULT_DATAFILE* below).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_DATAFILE = "data_prepare/SKE2024_data16-Apr-2025_1819.dat"
N_TEST_POINTS = 500               # resolution of the dense prediction grid

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_bode_data(filepath: Path):
    """Load three‑column (ω, |G|, arg G) data from *filepath* (CSV)."""
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase


def prepare_inputs(omega):
    """Return log‑frequency X and fitted StandardScaler."""
    X_raw = np.log10(omega).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler


def build_kernel():
    """Construct GP kernel Constant × RBF + White noise."""
    const = ConstantKernel(1.0, (1e-3, 1e3))
    rbf = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    noise = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    return const * rbf + noise


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
  path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_DATAFILE)
  if not path.exists():
    raise FileNotFoundError(f"Data file not found: {path}")

  # 1) Load and sort data ----------------------------------------------------
  omega_raw, mag_raw, phase_raw = load_bode_data(path)
  idx = np.argsort(omega_raw)
  omega = omega_raw[idx]
  mag = mag_raw[idx]
  phase = phase_raw[idx]

  # 2) Convert to modelling targets -----------------------------------------
  X, scaler = prepare_inputs(omega)
  y_mag_db = 20.0 * np.log10(mag)
  y_phase = phase  # Phase in radians

  # 3) Fit GPR for magnitude -----------------------------------------------
  gpr_mag = GaussianProcessRegressor(
    kernel=build_kernel(),
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=0,
  )
  gpr_mag.fit(X, y_mag_db)

  # 4) Fit GPR for phase ----------------------------------------------------
  gpr_phase = GaussianProcessRegressor(
    kernel=build_kernel(),
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=1,
  )
  gpr_phase.fit(X, y_phase)

  # 5) Dense prediction grid -------------------------------------------------
  omega_test = np.logspace(np.log10(omega.min()), np.log10(omega.max()), N_TEST_POINTS)
  X_test = scaler.transform(np.log10(omega_test).reshape(-1, 1))
  
  # Predict magnitude
  y_mag_pred, y_mag_std = gpr_mag.predict(X_test, return_std=True)
  y_mag_up = y_mag_pred + 1.96 * y_mag_std
  y_mag_lo = y_mag_pred - 1.96 * y_mag_std
  
  # Predict phase
  y_phase_pred, y_phase_std = gpr_phase.predict(X_test, return_std=True)

  # 6) Plot Bode magnitude --------------------------------------------------
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.semilogx(omega, y_mag_db, "b*", label="Observed (gain)")
  ax.semilogx(omega_test, y_mag_pred, "r-", lw=2, label="GPR fit")
  ax.fill_between(
    omega_test,
    y_mag_lo,
    y_mag_up,
    color="red",
    alpha=0.25,
    label="95 % CI",
  )

  ax.set_xlabel(r"$\omega$ [rad/s]")
  ax.set_ylabel(r"$20\,\log_{10}|G(j\omega)|$  [dB]")
  ax.grid(True, which="both", ls=":", alpha=0.5)
  ax.legend()
  fig.tight_layout()
  fig.savefig("_gpr_fit.png", dpi=300)
  plt.show()

  # 7) Create Nyquist plot ---------------------------------------------------
  # Convert original data to complex numbers
  G_dataset = mag * np.exp(1j * phase)
  
  # Convert predictions to complex numbers
  H_best = 10**(y_mag_pred/20) * np.exp(1j * y_phase_pred)

  plt.figure(figsize=(10, 6))
  plt.plot(G_dataset.real, G_dataset.imag, 'b*', markersize=6, label='Data')
  plt.plot(H_best.real, H_best.imag, 'r-', linewidth=2, label='GPR Est.')
  plt.xlabel('Re', fontsize=16)
  plt.ylabel('Im', fontsize=16)
  plt.title('Nyquist Plot', fontsize=16)
  plt.grid(True)
  plt.legend(fontsize=12)
  plt.tight_layout()
  plt.savefig("_nyquist.png", dpi=300)
  plt.show()



if __name__ == "__main__":
    main()
