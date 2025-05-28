#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

warnings.filterwarnings("ignore")

def load_bode_data(filepath: Path):
  data = np.loadtxt(filepath, delimiter=",")
  omega, mag, phase = data
  return omega, mag, phase

def main() -> None:
  N_TEST_POINTS = 500
  TEST_FILES = {
    "SKE2024_data18-Apr-2025_1205.dat",
    "SKE2024_data16-May-2025_1609.dat",
  }

  # データ読み込み -----------------------------------------------------------
  DEFAULT_DIR = Path("data_prepare")
  dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
  dat_files = sorted(dir_path.glob("*.dat"))
  if not dat_files:
    raise FileNotFoundError(f"No .dat files found in '{dir_path}'")

  train_files = [p for p in dat_files if p.name not in TEST_FILES]
  test_files  = [p for p in dat_files if p.name in TEST_FILES]
  if not train_files or not test_files:
    raise RuntimeError("Train / Test split failed. Check file names.")

  def stack(files):
    w_l, m_l, p_l = [], [], []
    for f in files:
      w, m, p = load_bode_data(f)
      w_l.append(w); m_l.append(m); p_l.append(p)
    w = np.hstack(w_l); m = np.hstack(m_l); p = np.hstack(p_l)
    idx = np.argsort(w)
    return w[idx], m[idx], p[idx]

  # train & test raw data ---------------------------------------------------
  w_tr, mag_tr, ph_tr = stack(train_files)
  w_te, mag_te, ph_te = stack(test_files)

  G_tr = mag_tr * np.exp(1j * ph_tr)
  G_te = mag_te * np.exp(1j * ph_te)

  # log-frequency -----------------------------------------------------------
  X_tr = np.log10(w_tr).reshape(-1, 1)
  X_te = np.log10(w_te).reshape(-1, 1)

  # hampel filter -----------------------------------------------------------
  def hampel_filter(x, window_size: int = 7, n_sigmas: int = 3):
    x = x.copy(); k = 1.4826; L = len(x); half = window_size // 2
    for i in range(L):
      s, e = max(i-half, 0), min(i+half+1, L)
      win = x[s:e]; med = np.median(win)
      mad = k * np.median(np.abs(win - med))
      if mad and abs(x[i]-med) > n_sigmas * mad:
        x[i] = med
    return x

  G_tr_f = hampel_filter(G_tr.real) + 1j * hampel_filter(G_tr.imag)
  G_te_f = hampel_filter(G_te.real) + 1j * hampel_filter(G_te.imag)

  # GP モデル ---------------------------------------------------------------
  kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel()
  gp_r = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=5, normalize_y=True)
  gp_i = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=5, normalize_y=True)

  gp_r.fit(X_tr, G_tr_f.real)
  gp_i.fit(X_tr, G_tr_f.imag)

  # 連続グリッド (for smooth curve) ----------------------------------------
  w_grid = np.logspace(np.log10(min(w_tr.min(), w_te.min())),
             np.log10(max(w_tr.max(), w_te.max())),
             N_TEST_POINTS)
  X_grid = np.log10(w_grid).reshape(-1, 1)

  r_grid, _ = gp_r.predict(X_grid, return_std=True)
  i_grid, _ = gp_i.predict(X_grid, return_std=True)
  G_grid = r_grid + 1j * i_grid

  # prediction on each set --------------------------------------------------
  r_tr, _ = gp_r.predict(X_tr, return_std=True)
  i_tr, _ = gp_i.predict(X_tr, return_std=True)
  G_tr_pred = r_tr + 1j * i_tr

  r_te, _ = gp_r.predict(X_te, return_std=True)
  i_te, _ = gp_i.predict(X_te, return_std=True)
  G_te_pred = r_te + 1j * i_te

  # MSE ---------------------------------------------------------------------
  mse_tr = np.mean(np.abs(G_tr_f - G_tr_pred) ** 2)
  mse_te = np.mean(np.abs(G_te_f - G_te_pred) ** 2)
  print(f"MSE  (train): {mse_tr:.4e}")
  print(f"MSE  (test) : {mse_te:.4e}")

  # Nyquist plot ------------------------------------------------------------
  fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

  # --- train ---
  ax[0].plot(G_tr_f.real, G_tr_f.imag, 'b*', ms=6, label='Train (filtered)')
  ax[0].plot(G_grid.real, G_grid.imag, 'r-', lw=2, label='GP prediction')
  ax[0].set_title(f"Train Nyquist (MSE={mse_tr:.2e})")
  ax[0].set_xlabel('Re'); ax[0].set_ylabel('Im')
  ax[0].grid(True); ax[0].legend()

  # --- test ---
  ax[1].plot(G_te_f.real, G_te_f.imag, 'g*', ms=6, label='Test (filtered)')
  ax[1].plot(G_grid.real, G_grid.imag, 'r-', lw=2, label='GP prediction')
  ax[1].set_title(f"Test Nyquist (MSE={mse_te:.2e})")
  ax[1].set_xlabel('Re')
  ax[1].grid(True); ax[1].legend()

  plt.tight_layout()
  plt.savefig("_nyquist_train_test_gp.png", dpi=300)
  plt.show()

if __name__ == "__main__":
  main()
