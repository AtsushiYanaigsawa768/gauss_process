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

# Hampel filter that modifies data by replacing outliers with median
def hampel_filter(x, window_size: int = 7, n_sigmas: int = 3):
  x_out = x.copy() # Operate on a copy
  k = 1.4826 # k ~ 1 / scipy.stats.norm.ppf(0.75)
  L = len(x_out)
  if L == 0: return x_out

  half_window = window_size // 2
  # Small number to handle floating point comparisons when MAD is zero
  zero_mad_threshold = 1e-9 
  
  for i in range(L):
    current_val_at_i = x_out[i]

    # If current point is NaN, skip (it won't be replaced)
    if np.isnan(current_val_at_i):
        continue

    # Define window around current point using indices on x_out
    s = max(0, i - half_window)
    e = min(L, i + half_window + 1)
    window_values = x_out[s:e] # Window values from x_out (can include prior modifications)
    
    # For robust statistics, create a temporary version of the window
    # where Infs are treated as NaNs for median/MAD calculation.
    window_for_stats = window_values.copy()
    window_for_stats[np.isinf(window_for_stats)] = np.nan
    
    med = np.nanmedian(window_for_stats)

    # If median could not be computed (e.g., window_for_stats was all NaNs),
    # we cannot determine a replacement, so skip.
    if np.isnan(med):
        continue

    # If the current point itself is Inf, replace it with the computed median and move on.
    if np.isinf(current_val_at_i):
        x_out[i] = med
        continue # Proceed to the next point in x_out

    # At this point, current_val_at_i is finite.
    # Calculate Median Absolute Deviation (MAD) using window_for_stats.
    abs_devs_from_med = np.abs(window_for_stats - med) # Deviations of cleaned window values
    median_abs_dev = np.nanmedian(abs_devs_from_med) 

    # If MAD could not be computed (e.g., after Inf->NaN, window_for_stats had no valid data for MAD)
    if np.isnan(median_abs_dev):
        continue # Cannot determine threshold, so skip modifying current_val_at_i
    
    mad_scaled = k * median_abs_dev # This is the Hampel threshold scale (sigma_Hampel)

    is_outlier = False
    if mad_scaled > 0: # Standard case: compare against scaled MAD
        if np.abs(current_val_at_i - med) > n_sigmas * mad_scaled:
            is_outlier = True
    else: # mad_scaled is 0 (all valid data in window_for_stats are equal to med)
          # current_val_at_i is an outlier if it's different from med by more than a small tolerance.
        if np.abs(current_val_at_i - med) > zero_mad_threshold:
            is_outlier = True
            
    if is_outlier:
      x_out[i] = med # Replace outlier
  return x_out

# Hampel filter from reference, returns a boolean mask for non-outliers
def _hampel_filter(x: np.ndarray, win: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Return a boolean mask whose True elements are *non-outliers* according to
    the Hampel filter applied to |x|. NaNs and Infs in x are marked as False (outliers/not kept).
    """
    k = 1.4826  # scale factor for Gaussian distribution
    n = x.size
    if n == 0:
        return np.array([], dtype=bool)

    # Initialize keep: True for finite values, False for NaNs and Infs.
    # Points that are initially NaN or Inf will not be kept.
    keep = np.isfinite(x)

    # Create a version of x where non-finite values are replaced by np.nan.
    # This x_for_stats is used for calculating med and median_abs_dev robustly.
    x_for_stats = x.copy()
    x_for_stats[~keep] = np.nan # Equivalent to x_for_stats[np.isinf(x) | np.isnan(x)] = np.nan

    half_window = win // 2

    for i in range(n):
        if not keep[i]:  # If x[i] was already marked as outlier (NaN or Inf), skip.
            continue

        # At this point, x[i] is a finite number.
        # Define window indices for x_for_stats
        i_min = max(0, i - half_window)
        i_max = min(n, i + half_window + 1)
        
        window_curr_for_stats = x_for_stats[i_min:i_max]

        med = np.nanmedian(window_curr_for_stats)

        if np.isnan(med):
            # If window median is NaN (e.g., window in x_for_stats is all NaNs),
            # we cannot determine if x[i] (a finite number) is an outlier relative to this window.
            # So, x[i] is kept (keep[i] remains True).
            continue

        # Calculate Median Absolute Deviation (MAD) from median, using cleaned window data.
        abs_devs_from_med = np.abs(window_curr_for_stats - med)
        median_abs_dev = np.nanmedian(abs_devs_from_med)

        if np.isnan(median_abs_dev):
            # If MAD is NaN (e.g. window_curr_for_stats became all NaNs),
            # cannot determine threshold. x[i] (finite) is kept.
            continue
        
        sigma_hampel = k * median_abs_dev # This will be >= 0 and finite.
            
        # Test if x[i] (which is finite) is an outlier.
        if sigma_hampel > 0:
            if np.abs(x[i] - med) > n_sigmas * sigma_hampel:
                keep[i] = False  # Mark as outlier
        else: # sigma_hampel == 0 (all non-NaN values in window_curr_for_stats are identical to med)
             # Mark x[i] as outlier if it's different from med by more than a small tolerance.
             if np.abs(x[i] - med) > 1e-1: 
                 keep[i] = False 
            
    return keep

def main() -> None:
  N_TEST_POINTS = 50000
  TEST_FILES = {
    "SKE2024_data18-Apr-2025_1205.dat",
  }

  # データ読み込み -----------------------------------------------------------
  DEFAULT_DIR = Path("gp/data")
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

  # hampel filter on data ---------------------------------------------------
  # This uses the original hampel_filter to clean the data itself
  G_tr_f = hampel_filter(G_tr.real) + 1j * hampel_filter(G_tr.imag)
  G_te_f = hampel_filter(G_te.real) + 1j * hampel_filter(G_te.imag)

  # 線形補間モデル -----------------------------------------------------------
  interp_r = interp1d(X_tr.ravel(), G_tr_f.real, kind='linear', fill_value="extrapolate")
  interp_i = interp1d(X_tr.ravel(), G_tr_f.imag, kind='linear', fill_value="extrapolate")

  # 連続グリッド (for smooth curve) ----------------------------------------
  w_grid = np.logspace(np.log10(min(w_tr.min(), w_te.min())),
             np.log10(max(w_tr.max(), w_te.max())),
             N_TEST_POINTS)
  X_grid = np.log10(w_grid).reshape(-1, 1)

  r_grid = interp_r(X_grid.ravel())
  i_grid = interp_i(X_grid.ravel())
  G_grid = r_grid + 1j * i_grid

  # prediction on each set --------------------------------------------------
  r_tr = interp_r(X_tr.ravel())
  i_tr = interp_i(X_tr.ravel())
  G_tr_pred = r_tr + 1j * i_tr

  r_te = interp_r(X_te.ravel())
  i_te = interp_i(X_te.ravel())
  G_te_pred = r_te + 1j * i_te

  # MSE (modified to filter errors using _hampel_filter) ------------------
  # Calculate absolute error between filtered true data and predictions
  err_tr = np.abs(G_tr_f - G_tr_pred)
  # Replace NaN values with 0
  err_tr = np.where(np.isnan(err_tr), 0, err_tr)
  # Get mask for non-outlier errors using _hampel_filter
  keep_tr = err_tr
  # Calculate MSE on non-outlier errors
  mse_tr = np.mean(err_tr ** 2)
  mse_tr = np.sqrt(mse_tr)  # Convert to RMSE for consistency
  
  err_te = np.abs(G_te_f - G_te_pred)
  # Replace NaN values with 0
  err_te = np.where(np.isnan(err_te), 0, err_te)
  keep_te = err_te
  mse_te = np.mean(err_te ** 2)
  mse_te = np.sqrt(mse_te)  # Convert to RMSE for consistency
  print(f"MSE  (train): {mse_tr:.4e}")
  print(f"MSE  (test) : {mse_te:.4e}")

  # Nyquist plot ------------------------------------------------------------
  TITLE_FONTSIZE = 15
  AXIS_LABEL_FONTSIZE = 12
  LEGEND_FONTSIZE = 10
  
  fig, ax = plt.subplots(2, 1, figsize=(7, 10))

  # --- train ---
  ax[0].plot(G_tr_f.real, G_tr_f.imag, 'b*', ms=6, label='Train')
  ax[0].plot(G_grid.real, G_grid.imag, 'r*', lw=4, label='Linear Interpolation')
  ax[0].set_title(f"Train Nyquist (MSE={mse_tr:.2e})", fontsize=TITLE_FONTSIZE)
  ax[0].set_xlabel('Re', fontsize=AXIS_LABEL_FONTSIZE)
  ax[0].set_ylabel('Im', fontsize=AXIS_LABEL_FONTSIZE)
  ax[0].set_xlim([-0.7, 0.7]); ax[0].set_ylim([-0.6, 0.2])
  ax[0].grid(True)
  ax[0].legend(fontsize=LEGEND_FONTSIZE)

  # --- test ---
  ax[1].plot(G_te_f.real, G_te_f.imag, 'g*', ms=6, label='Test')
  ax[1].plot(G_grid.real, G_grid.imag, 'r*', lw=4, label='Linear Interpolation')
  ax[1].set_title(f"Test Nyquist (MSE={mse_te:.2e})", fontsize=TITLE_FONTSIZE)
  ax[1].set_xlabel('Re', fontsize=AXIS_LABEL_FONTSIZE)
  ax[1].set_ylabel('Im', fontsize=AXIS_LABEL_FONTSIZE)
  ax[1].set_xlim([-0.7, 0.7]); ax[1].set_ylim([-0.6, 0.2])
  ax[1].grid(True)
  ax[1].legend(fontsize=LEGEND_FONTSIZE)

  fig.tight_layout()
  fig.savefig("linear_nyquist_train_test_interp.png", dpi=300)
  
  # Output grid data to CSV -------------------------------------------------
  grid_output_data = np.column_stack((w_grid, r_grid, i_grid))
  output_csv_path = "linear_grid_predictions.csv"
  header = "omega,Re_G,Im_G"
  np.savetxt(output_csv_path, grid_output_data, delimiter=",", header=header, comments='', fmt='%e')
  print(f"Grid predictions saved to {output_csv_path}")

  plt.show()


if __name__ == "__main__":
  main()
