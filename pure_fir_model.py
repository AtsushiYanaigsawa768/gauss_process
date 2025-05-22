#!/usr/bin/env python3
# fir_offline_predict.py  -----------------------------------------------
#  FRF から得た伝達関数を “そのまま” FIR に落として
#  既存の入力 u[n] へ畳み込み → 出力 ŷ[n] を予測するだけ
# -----------------------------------------------------------------------

import numpy as np
import math
from scipy import signal
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# ========== USER PARAMETERS ============================================
frf_file      = "linear_predicted_G_values.csv"   # [ω, ReG, ImG]
io_file       = "data_hour.mat"                   # MAT 内に [time; y; u]
energy_cut    = 0.99    # |g|²　累積エネルギ 99 % でタップ長を決定
win_type      = "hann"  # 窓種："boxcar", "hann", "hamming", ...
plot_example  = True    # True なら可視化
# =======================================================================

# 1) FRF をロードして等間隔周波数グリッドへ内挿 ----------------------
ω, ReG, ImG = np.loadtxt(frf_file, delimiter=",", skiprows=1).T
G_pos       = ReG + 1j * ImG
Npos        = ω.size

ω_min, ω_max = ω.min(), ω.max()
Nfft        = 2**math.ceil(math.log2(4*Npos))        # ゼロパディング大きめ
ω_uni       = np.linspace(ω_min, ω_max, Nfft//2 + 1) # 0 ～ Nyquist

# 実部・虚部を独立に線形補間（十分高密度ならスプライン不要）
G_uni = np.interp(ω_uni, ω, G_pos.real) + 1j*np.interp(ω_uni, ω, G_pos.imag)

# エルミート対称を付加してフルスペクトルを作成
G_full = np.concatenate([np.conj(G_uni[-2:0:-1]), G_uni])

# 2) IFFT → インパルス応答 g[n]（実数） ---------------------------------
g_full = np.real(np.fft.ifft(np.fft.ifftshift(G_full)))

# サンプリング周期 Ts を FRF から推定
Δω = ω_uni[1] - ω_uni[0]
Fs = Δω * Nfft / (2*np.pi)   # [Hz]
Ts = 1 / Fs

# 3) 全エネルギの energy_cut % を超えたところでタップ長 L を決定 -------
Ecum = np.cumsum(g_full**2)
L = np.searchsorted(Ecum/Ecum[-1], energy_cut) + 1   # +1 で 1-indexed
L = max(L, 4)                                        # 安全に最低 4 タップ
g_trunc = g_full[:L]

# 4) 窓掛けして最終 FIR 係数 h を得る -----------------------------------
w = signal.get_window(win_type, L)
h = g_trunc * w

print(f"[INFO] FIR length  L = {L}  (Ts = {Ts*1e3:.3f} ms, Fs = {Fs:.2f} Hz)")

# 5) I/O データを読み込み，必要ならリサンプル ---------------------------
mat = next(arr for name, arr in loadmat(io_file).items() if not name.startswith("__"))
time, y_meas, u = mat[0], mat[1], mat[2]

# I/O の時間刻み dt と FIR の Ts がずれていれば補間
dt = np.mean(np.diff(time))
if abs(dt - Ts) > 1e-6:
    print(f"[WARN] dt ({dt:.4e}) ≠ Ts ({Ts:.4e}) → 入出力を再サンプル")
    t_fir = np.arange(0, time[-1] + Ts/2, Ts)
    u      = np.interp(t_fir, time, u)
    y_meas = np.interp(t_fir, time, y_meas)
    time   = t_fir

# 6) 畳み込みで予測 ŷ[n] を生成 ----------------------------------------
y_pred = signal.lfilter(h, [1.0], u)       # forward FIR フィルタ

# 同期のため L-1 サンプル分シフト
y_pred = np.concatenate((np.zeros(L-1), y_pred[:-L+1]))

# 7) 誤差指標 ------------------------------------------------------------
e      = y_meas - y_pred
rmse   = np.sqrt(np.mean(e**2))
nrmse  = 1 - np.linalg.norm(e) / np.linalg.norm(y_meas - y_meas.mean())
R2     = 1 - (e**2).sum() / ((y_meas - y_meas.mean())**2).sum()

print("\n=== Performance ========================================")
print(f"RMSE   : {rmse:.4g}")
print(f"NRMSE  : {nrmse*100:.2f} %")
print(f"R²     : {R2:.3f}")

# 8) 保存 & 可視化 -------------------------------------------------------
savemat("fir_offline_results.mat",
        {"h": h, "rmse": rmse, "nrmse": nrmse, "R2": R2,
         "y_pred": y_pred, "error": e, "Ts": Ts})

if plot_example:
    Nshow = min(2000, len(time))        # 例として先頭 2 000 点
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(time[:Nshow], y_meas[:Nshow], label="Measured y")
    ax[0].plot(time[:Nshow], y_pred[:Nshow], "--", label="Predicted ŷ")
    ax[0].set_ylabel("Output")
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot(time[:Nshow], e[:Nshow], color="tab:red")
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("Error")
    ax[1].grid(True)

    plt.tight_layout(); plt.show()
