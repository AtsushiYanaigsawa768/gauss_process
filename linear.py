#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")

def load_bode_data(filepath: Path):
  data = np.loadtxt(filepath, delimiter=",")
  omega, mag, phase = data
  return omega, mag, phase

def main():
  N_TEST_POINTS = 500

  # データ読み込み
  DEFAULT_DIR = Path("data_prepare")
  dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
  dat_files = sorted(dir_path.glob("*.dat"))
  if not dat_files:
    raise FileNotFoundError(f"No .dat files found in '{dir_path}'")
  omega_list, mag_list, phase_list = [], [], []
  for f in dat_files:
    w, m, p = load_bode_data(f)
    omega_list.append(w); mag_list.append(m); phase_list.append(p)
  omega = np.hstack(omega_list)
  mag   = np.hstack(mag_list)
  phase = np.hstack(phase_list)
  idx = np.argsort(omega)
  omega, mag, phase = omega[idx], mag[idx], phase[idx]
  G_meas = mag * np.exp(1j * phase)

  # モデル化対象
  X       = np.log10(omega).reshape(-1, 1)
  y_real  = G_meas.real
  y_imag  = G_meas.imag

  # --- ここから線形補間に置き換え ---
  f_real = interp1d(X.ravel(), y_real, kind='linear', fill_value='extrapolate')
  f_imag = interp1d(X.ravel(), y_imag, kind='linear', fill_value='extrapolate')

  # 予測グリッド
  omega_test = np.logspace(np.log10(omega.min()),
               np.log10(omega.max()),
               N_TEST_POINTS)
  X_test = np.log10(omega_test).reshape(-1, 1)

  # 予測値
  y_real_pred = f_real(X_test.ravel())
  y_imag_pred = f_imag(X_test.ravel())
  # 誤差はゼロとみなす
  y_real_std  = np.zeros_like(y_real_pred)
  y_imag_std  = np.zeros_like(y_imag_pred)

  # 元の測定点での予測（プロット用）
  y_real_meas_pred = f_real(X.ravel())
  y_imag_meas_pred = f_imag(X.ravel())
  H_pred_meas = y_real_meas_pred + 1j*y_imag_meas_pred

  # フィルタ関数（そのまま）
  def hampel_filter(x, window_size=7, n_sigmas=3):
    x = x.copy(); k = 1.4826
    L = len(x); half_w = window_size//2
    for i in range(L):
      start = max(i-half_w,0); end = min(i+half_w+1,L)
      window = x[start:end]
      med = np.median(window)
      mad = k*np.median(np.abs(window-med))
      if mad and abs(x[i]-med) > n_sigmas*mad:
        x[i] = med
    return x

  # Hampel‐filter したデータ
  G_dataset = mag * np.exp(1j*phase)
  G_real_filt = hampel_filter(G_dataset.real)
  G_imag_filt = hampel_filter(G_dataset.imag)
  G_filt = G_real_filt + 1j*G_imag_filt

  # プロット（省略せずそのまま）
  plt.figure(figsize=(8,4))
  plt.loglog(omega, y_real, 'b.', label='Measured Real')
  plt.loglog(omega_test, y_real_pred, 'r-', label='Interpolated Real')
  plt.fill_between(omega_test,
           y_real_pred-2*y_real_std,
           y_real_pred+2*y_real_std,
           color='r', alpha=0.2, label='±2σ')
  plt.xlabel('ω (rad/s)'); plt.ylabel('Re{G}')
  plt.title('Real Part: Measured vs Linear Interp.')
  plt.legend(); plt.grid(True,which='both',ls='--')
  plt.tight_layout(); plt.savefig("_real_fit.png",dpi=300); plt.show()

  plt.figure(figsize=(8,4))
  plt.loglog(omega, y_imag, 'g.', label='Measured Imag')
  plt.loglog(omega_test, y_imag_pred, 'm-', label='Interpolated Imag')
  plt.fill_between(omega_test,
           y_imag_pred-2*y_imag_std,
           y_imag_pred+2*y_imag_std,
           color='m', alpha=0.2, label='±2σ')
  plt.xlabel('ω (rad/s)'); plt.ylabel('Im{G}')
  plt.title('Imag Part: Measured vs Linear Interp.')
  plt.legend(); plt.grid(True,which='both',ls='--')
  plt.tight_layout(); plt.savefig("_imag_fit.png",dpi=300); plt.show()

  # MSE
  mse = np.mean(np.abs(G_filt - H_pred_meas)**2)
  print(f"Nyquist MSE (after Hampel filter): {mse:.4e}")

  # Nyquist プロット
  order = np.argsort(omega_test)
  plt.figure(figsize=(10,6))
  plt.plot(G_filt.real, G_filt.imag, 'b*', ms=6, label='Filtered Data')
  H_best = y_real_pred + 1j*y_imag_pred
  plt.plot(H_best.real[order], H_best.imag[order],
       'r-', lw=2, label='Linear Interp.')
  plt.xlabel('Re'); plt.ylabel('Im')
  plt.title(f'Nyquist Plot (MSE: {mse:.4e})')
  plt.grid(True); plt.legend()
  plt.tight_layout(); plt.savefig("_nyquist.png",dpi=300); plt.show()

  # CSV 出力
  output_data = np.column_stack((omega_test, y_real_pred, y_imag_pred))
  csv_filepath = Path("linear_predicted_G_values.csv")
  header = "omega,Re_G,Im_G"
  np.savetxt(csv_filepath, output_data,
         delimiter=",", header=header, comments='')
  print(f"Predicted G values saved to {csv_filepath}")

if __name__ == "__main__":
  main()
