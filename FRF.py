"""
FIR 予測 ―― FRF (実部のみ) から FIR を生成し測定データで検証
================================================================
入力ファイル
    FRF_FILE : predicted_G_values.csv   # 列 = freq, ReG, ImG (ヘッダ付きでも可)
    IO_FILE  : ten_minitues.mat         # 行 = time ; output ; input

実行フロー
    1. FRF → 実部のみ抽出 → firwin2 で線形位相 FIR 係数 h を作成
    2. 入力信号 u を h で畳み込み → 予測 y_pred
    3. 測定出力と比較して RMSE / NRMSE / R² を算出
    4. 測定 vs 予測 を同一グラフに描画
"""

# ───────────── 0. ファイル名・パラメータ宣言 ─────────────
FRF_FILE = 'predicted_G_values.csv'
IO_FILE  = 'ten_minitues.mat'
NUMTAPS  = 101        # FIR タップ数（奇数にすると線形位相）
FONT_SZ  = 11

# ───────────── 1. 依存ライブラリ ──────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin2, lfilter
from scipy.io import loadmat
plt.rcParams.update({'font.size': FONT_SZ})

# ───────────── 2. FRF → FIR 係数 h を作成 ─────────────
try:
    frf = np.loadtxt(FRF_FILE, delimiter=',')              # ヘッダ無し
except ValueError:
    frf = np.loadtxt(FRF_FILE, delimiter=',', skiprows=1)  # ヘッダ 1 行スキップ

if frf.shape[1] < 2:
    raise ValueError('FRF_FILE は少なくとも [freq, ReG] の 2 列が必要です')

freq = np.abs(frf[:, 0])        # 周波数（0〜Fs/2）
gain = frf[:, 1]                # 実部のみ使用

# ソート & 重複排除
order = np.argsort(freq)
freq_sorted, gain_sorted = freq[order], gain[order]
uniq_f, idx              = np.unique(freq_sorted, return_index=True)
uniq_g                   = gain_sorted[idx]

# firwin2 用正規化
f_norm = uniq_f / uniq_f.max()
g_norm = uniq_g
if f_norm[0] > 0.0:             # DC が無ければ補完
    f_norm = np.insert(f_norm, 0, 0.0)
    g_norm = np.insert(g_norm, 0, g_norm[0])

h = firwin2(NUMTAPS, f_norm, g_norm)
print(f'[FRF] FIR taps (first 5) : {h[:5]}')

# ───────────── 3. ten_minitues.mat 読み込み ──────────
mat = loadmat(IO_FILE)
arr = next(v for k, v in mat.items() if not k.startswith('__'))
if arr.shape[0] < 3:
    raise ValueError('IO_FILE は [time; output; input] の 3 行が必要です')

t, y_meas, u_in = arr[0].ravel(), arr[1].ravel(), arr[2].ravel()

# ───────────── 4. 予測 & 指標 ───────────────────────
def predict(u, hcoeff):
    return lfilter(hcoeff, [1.0], u)

def metrics(y, yhat):
    rmse  = np.sqrt(np.mean((y - yhat) ** 2))
    rng   = np.ptp(y)                   # NumPy 2.x 互換
    nrmse = rmse / (rng if rng else 1)
    r2    = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return rmse, nrmse, r2

y_pred = predict(u_in, h)

delay          = NUMTAPS - 1
t_trim         = t[delay:]
y_meas_trim    = y_meas[delay:]
y_pred_trim    = y_pred[delay:]

rmse, nrmse, r2 = metrics(y_meas_trim, y_pred_trim)
print(f'[FRF] RMSE={rmse:.4g}, NRMSE={nrmse:.4g}, R²={r2:.3f}')

# ───────────── 5. 描画 ───────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(t_trim, y_meas_trim, label='Measured', color='k')
plt.plot(t_trim, y_pred_trim, '--', label=f'FRF predicted (R²={r2:.3f})',
         color='dodgerblue')
plt.xlabel('Time [s]')
plt.ylabel('Output')
plt.title(f'FIR Prediction via FRF (Order = {NUMTAPS})')
plt.grid(alpha=0.6, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
