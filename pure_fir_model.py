#!/usr/bin/env python3
# fir_offline_predict.py (modified for simulated real-time processing)
# -----------------------------------------------------------------------
#  FRF から得た伝達関数を “そのまま” FIR に落として
#  入力 u[n] を逐次処理 → 出力 ŷ[n] を予測
# -----------------------------------------------------------------------

import numpy as np
import math
from scipy import signal
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from collections import deque

# ========== USER PARAMETERS ============================================
frf_file      = "predicted_G_values.csv"   # [ω, ReG, ImG] <- 更新例
io_file       = "data_hour.mat"            # MAT 内に [time; y; u] <- 更新例
energy_cut    = 0.95    # |g|²　累積エネルギ 95 % でタップ長を決定 <- 更新例
win_type      = "hamming"  # 窓種："boxcar", "hann", "hamming", ... <- 更新例
plot_example  = True   # True なら可視化 <- 更新例

# Parameters for simulated real-time plotting
plot_update_interval = 10   # Update plot every N samples <- 更新例
plot_display_points  = 1000  # Number of points in dynamic plot window <- 更新例
# =======================================================================

# 1) FRF をロードして等間隔周波数グリッドへ内挿 ----------------------
ω, ReG, ImG = np.loadtxt(frf_file, delimiter=",", skiprows=1).T
G_pos       = ReG + 1j * ImG
Npos        = ω.size


ω_min, ω_max = ω.min(), ω.max()
Nfft        = 2**math.ceil(math.log2(4*Npos))        # ゼロパディング大きめ
ω_uni       = np.linspace(ω_min, ω_max, Nfft//2 + 1) # 0 ～ Nyquist

G_uni = np.interp(ω_uni, ω, G_pos.real) + 1j*np.interp(ω_uni, ω, G_pos.imag)
G_full = np.concatenate([np.conj(G_uni[-2:0:-1]), G_uni])

# 2) IFFT → インパルス応答 g[n]（実数） ---------------------------------
g_full = np.real(np.fft.ifft(np.fft.ifftshift(G_full)))

Δω = ω_uni[1] - ω_uni[0]
Fs = Δω * Nfft / (2*np.pi)   # [Hz]
Ts = 1 / Fs

# 3) 全エネルギの energy_cut % を超えたところでタップ長 L を決定 -------
Ecum = np.cumsum(g_full**2)
L = np.searchsorted(Ecum/Ecum[-1], energy_cut) + 1
L = max(L, 4)
g_trunc = g_full[:L]

# 4) 窓掛けして最終 FIR 係数 h を得る -----------------------------------
w = signal.get_window(win_type, L)
h = g_trunc * w
print(f"[INFO] FIR length  L = {L}  (Ts = {Ts*1e3:.3f} ms, Fs = {Fs:.2f} Hz)")

# 5) I/O データを読み込み，必要ならリサンプル (全データロード) --------
mat_data = loadmat(io_file)
# Find the first non-dunder key assuming it's the data array
data_key = next(key for key in mat_data if not key.startswith('__'))
mat = mat_data[data_key]
time_all, y_meas_all, u_all = mat[0], mat[1], mat[2]

dt = np.mean(np.diff(time_all))
if abs(dt - Ts) > 1e-6:
    print(f"[WARN] dt ({dt:.4e}) ≠ Ts ({Ts:.4e}) → 入出力を再サンプル")
    t_fir_all = np.arange(0, time_all[-1] + Ts/2, Ts)
    u_all      = np.interp(t_fir_all, time_all, u_all)
    y_meas_all = np.interp(t_fir_all, time_all, y_meas_all)
    time_all   = t_fir_all

num_samples = len(u_all)

# Initialize for real-time processing loop
raw_y_pred_list = [] # Stores raw (unshifted) lfilter output
y_pred_shifted_accumulator = [] # Stores shifted output for final results
error_accumulator = []          # Stores error for final results

# Initial state for lfilter (FIR filter starts from zero state)
zi_state = np.zeros(L - 1)

# Real-time plotting setup
if plot_example:
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Use deques for efficient rolling plot buffers
    time_plot_deque = deque(maxlen=plot_display_points)
    y_meas_plot_deque = deque(maxlen=plot_display_points)
    y_pred_plot_deque = deque(maxlen=plot_display_points) # For shifted y_pred
    error_plot_deque = deque(maxlen=plot_display_points)

    line_meas, = ax[0].plot([], [], label="Measured y")
    line_pred, = ax[0].plot([], [], "--", label="Predicted ŷ (dynamic)")
    ax[0].set_ylabel("Output")
    ax[0].legend(loc='upper left'); ax[0].grid(True)

    line_error, = ax[1].plot([], [], color="tab:red", label="Error (dynamic)")
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("Error")
    ax[1].legend(loc='upper left'); ax[1].grid(True)
    
    plt.tight_layout()
    fig.canvas.draw()

# 6) 逐次処理ループ (Simulated real-time processing) --------------------
print("\nStarting simulated real-time processing...")
for i in range(num_samples):
    current_u_sample = u_all[i]
    current_time_sample = time_all[i]
    current_y_meas_sample = y_meas_all[i]
    
    # Apply FIR filter to the current sample
    y_sample_array, zi_state = signal.lfilter(h, [1.0], [current_u_sample], zi=zi_state)
    y_pred_raw_current = y_sample_array[0]
    raw_y_pred_list.append(y_pred_raw_current)

    # Determine the y_pred value for the current time step i, considering the original script's shift
    # y_pred_shifted[i] = 0 for i < L-1
    # y_pred_shifted[i] = raw_y_pred_list[i - (L-1)] for i >= L-1
    if i < L - 1:
        y_pred_shifted_current = 0.0
    else:
        y_pred_shifted_current = raw_y_pred_list[i - (L-1)]
    
    y_pred_shifted_accumulator.append(y_pred_shifted_current)
    
    current_error_sample = current_y_meas_sample - y_pred_shifted_current
    error_accumulator.append(current_error_sample)

    if plot_example:
        time_plot_deque.append(current_time_sample)
        y_meas_plot_deque.append(current_y_meas_sample)
        y_pred_plot_deque.append(y_pred_shifted_current)
        error_plot_deque.append(current_error_sample)

        if (i % plot_update_interval == 0 or i == num_samples - 1) and i > 0 :
            line_meas.set_data(time_plot_deque, y_meas_plot_deque)
            line_pred.set_data(time_plot_deque, y_pred_plot_deque)
            line_error.set_data(time_plot_deque, error_plot_deque)

            ax[0].set_xlim(time_plot_deque[0], time_plot_deque[-1])
            ax[0].relim(); ax[0].autoscale_view(True, False, True) # Autoscale Y
            
            ax[1].set_xlim(time_plot_deque[0], time_plot_deque[-1])
            ax[1].relim(); ax[1].autoscale_view(True, False, True) # Autoscale Y
            
            fig.canvas.draw()
            fig.canvas.flush_events()
    
    if (i + 1) % 1000 == 0: # Progress update to console
        print(f"Processed {i+1}/{num_samples} samples...")

print("Processing finished.")

# Convert accumulated lists to numpy arrays for final calculations
y_pred_final = np.array(y_pred_shifted_accumulator)
e_final = np.array(error_accumulator) # or y_meas_all - y_pred_final

# 7) 誤差指標 (全データに対して計算) ------------------------------------
rmse   = np.sqrt(np.mean(e_final**2))
nrmse  = 1 - np.linalg.norm(e_final) / np.linalg.norm(y_meas_all - y_meas_all.mean())
R2     = 1 - (e_final**2).sum() / ((y_meas_all - y_meas_all.mean())**2).sum()

print("\n=== Performance ========================================")
print(f"RMSE   : {rmse:.4g}")
print(f"NRMSE  : {nrmse*100:.2f} %")
print(f"R²     : {R2:.3f}")

# 8) 保存 & 可視化 (最終結果) ------------------------------------------
savemat("fir_realtimesim_results.mat",
        {"h": h, "rmse": rmse, "nrmse": nrmse, "R2": R2,
         "y_pred": y_pred_final, "error": e_final, "Ts": Ts})

if plot_example:
    plt.ioff() # Turn off interactive mode
    if 'fig' in locals() and plt.fignum_exists(fig.number): # Close the dynamic plot
        plt.close(fig)

    # Create the final static plot (similar to original script)
    print("\nDisplaying final summary plot.")
    Nshow_final = min(2000, len(time_all)) 

    fig_static, ax_static = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_static[0].plot(time_all[:Nshow_final], y_meas_all[:Nshow_final], label="Measured y")
    ax_static[0].plot(time_all[:Nshow_final], y_pred_final[:Nshow_final], "--", label="Predicted ŷ (final)")
    ax_static[0].set_ylabel("Output")
    ax_static[0].legend(); ax_static[0].grid(True)

    ax_static[1].plot(time_all[:Nshow_final], e_final[:Nshow_final], color="tab:red")
    ax_static[1].set_xlabel("Time [s]"); ax_static[1].set_ylabel("Error (final)")
    ax_static[1].grid(True)

    plt.tight_layout(); plt.show()
