import numpy as np
from scipy.signal import lfilter
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import math

# USER PARAMETERS
frf_file     = 'predicted_G_values.csv'  # FRF in CSV: [omega, ReG, ImG]
io_file      = 'data_hour.mat'           # recorded I/O to replay
na           = 20                         # 出力フィードバック次数
nb           = 20                         # 入力次数
model_order  = max(na, nb)               # 最大ラグ
n_ifft       = 1024                       # IFFT点数

# 1) Load FRF from CSV (この例では同定に直接使いませんが、読み込みを維持)
data = np.loadtxt(frf_file, delimiter=',', skiprows=1)
omega = data[:,0]
ReG   = data[:,1]
ImG   = data[:,2]
G_pos = ReG + 1j*ImG

# 2) Uniform frequency grid と IFFT（FIR部分のみ使用）
Npos = len(omega)
omega_min, omega_max = np.min(omega), np.max(omega)
Nfft = 2**math.ceil(math.log2(4*Npos))
omega_uni = np.linspace(omega_min, omega_max, Nfft//2+1)
G_uni = np.interp(omega_uni, omega, G_pos.real) + 1j*np.interp(omega_uni, omega, G_pos.imag)
G_full = np.concatenate([np.conj(G_uni[-2:0:-1]), G_uni])
g_ifft = np.real(np.fft.ifft(np.fft.ifftshift(G_full)))

# 3) Load I/O data
io_data = loadmat(io_file)
for name, arr in io_data.items():
    if not name.startswith('__'):
        mat = arr
        break
time = mat[0,:100000].ravel()
y    = mat[1,:100000].ravel()
u    = mat[2,:100000].ravel()
N    = len(u)

# 4) Build NARX 回帰行列 Phi と応答ベクトル Y
max_lag = max(na, nb)
rows = N - max_lag
Phi = np.zeros((rows, na + nb))
Y   = np.zeros(rows)
for k in range(max_lag, N):
    idx = k - max_lag
    # 出力フィードバック項：-y[k-i]
    Phi[idx, :na] = [-y[k - i] for i in range(1, na+1)]
    # 外部入力項：u[k-j]
    Phi[idx, na:] = [u[k - j] for j in range(1, nb+1)]
    Y[idx] = y[k]

# 5) 最小二乗で係数推定
theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
a = theta[:na]
b = theta[na:]

# 6) NARXモデルで予測（逐次再帰シミュレーション）
yhat = np.zeros(N)
for k in range(max_lag, N):
    # フィードバック部
    fb = sum(a[i] * yhat[k - i - 1] for i in range(na))
    # 入力部
    ex = sum(b[j] * u[k - j - 1] for j in range(nb))
    yhat[k] = -fb + ex

# 7) 評価指標の計算
y_true = y[max_lag:]
y_pred = yhat[max_lag:]
rmse  = np.sqrt(np.mean((y_true - y_pred)**2))
nrmse = rmse / (np.max(y_true) - np.min(y_true))
r2    = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

print(f"NARX orders: na={na}, nb={nb}")
print(f"RMSE:   {rmse:.4f}")
print(f"NRMSE:  {nrmse:.4f}")
print(f"R^2:    {r2:.4f}")

# 8) プロット
plt.figure(figsize=(8,4))
plt.plot(time[max_lag:], y_true,   label='Measured')
plt.plot(time[max_lag:], y_pred,   label='NARX Predicted', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.tight_layout()
plt.show()

# 9) 結果保存
savemat("narx_coefficients.mat", {"a": a, "b": b})
