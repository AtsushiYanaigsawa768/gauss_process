import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Hampel フィルターの実装
def hampel_filter(vals, window_size=7, n_sigmas=3):
  """
  vals: 1D numpy array
  window_size: 偶数可、窓幅
  n_sigmas: 外れ値とみなす閾値
  """
  vals = vals.copy()
  L = len(vals)
  k = window_size // 2
  for i in range(L):
    start = max(0, i - k)
    end   = min(L, i + k + 1)
    window = vals[start:end]
    med = np.median(window)
    mad = 1.4826 * np.median(np.abs(window - med))
    if mad > 0 and np.abs(vals[i] - med) > n_sigmas * mad:
      vals[i] = med
  return vals

# 1) データ読み込み
filename = 'data_prepare/SKE2024_data16-Apr-2025_1819.dat'
data = np.loadtxt(filename, delimiter=',')
omega_raw, SysGain_raw, argG_raw = data

# → Hampel フィルターをかける
SysGain_f = hampel_filter(SysGain_raw, window_size=7, n_sigmas=3)
argG_f    = hampel_filter(argG_raw,    window_size=7, n_sigmas=3)

# 2) ソート
idx = np.argsort(omega_raw)
omega    = omega_raw[idx]
SysGain  = SysGain_f[idx]
argG     = argG_f[idx]

# 3) 入力 X と出力 y を用意
X = omega.reshape(-1, 1)
y_gain  = 20 * np.log10(SysGain)
y_phase = argG


# 4) カーネル定義 + モデル構築
kernel = ConstantKernel(1.0, (1e-3, 1e3)) \
     * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
     + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))

gp_gain  = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gp_phase = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

# 5) 学習
gp_gain .fit(X, y_gain)
gp_phase.fit(X, y_phase)

# 6) 予測用 ω 帯を作成
omega_test = np.logspace(np.log10(omega.min()), np.log10(omega.max()), 500)
Xtest = omega_test.reshape(-1,1)

# 7) 予測 (平均値 + 標準偏差)
y_gain_pred,  y_gain_std  = gp_gain .predict(Xtest, return_std=True)
y_phase_pred, y_phase_std = gp_phase.predict(Xtest, return_std=True)

# 8) プロット
plt.figure(figsize=(6,4))
plt.semilogx(omega, y_gain,    'b*', label='Data (gain)')
plt.semilogx(omega_test, y_gain_pred, 'r-', label='GPR fit')
plt.fill_between(omega_test,
         y_gain_pred - 2*y_gain_std,
         y_gain_pred + 2*y_gain_std,
         alpha=0.2, color='r')
plt.xlabel('ω [rad/s]')
plt.ylabel('20 log₁₀|G(jω)| [dB]')
plt.legend()
plt.grid(True)

plt.figure(figsize=(6,4))
plt.semilogx(omega, y_phase,    'b*', label='Data (phase)')
plt.semilogx(omega_test, y_phase_pred, 'r-', label='GPR fit')
plt.fill_between(omega_test,
     y_phase_pred - 2*y_phase_std,
     y_phase_pred + 2*y_phase_std,
     alpha=0.2, color='r')
plt.xlabel('ω [rad/s]')
plt.ylabel('Phase [rad]')
plt.legend()
plt.grid(True)

# 9) Nyquist plot
G_dataset = SysGain * np.exp(1j * argG)
H_best    = 10**(y_gain_pred/20) * np.exp(1j * y_phase_pred)

plt.figure(figsize=(6,4))
plt.plot(G_dataset.real, G_dataset.imag, 'b*', label='Dataset')
plt.plot(H_best.real,    H_best.imag,    'r-', linewidth=2, label='Estimation')
plt.xlabel(r'$\mathrm{Re}\{G(j\omega)\}$', fontsize=14)
plt.ylabel(r'$\mathrm{Im}\{G(j\omega)\}$', fontsize=14)
plt.title('Nyquist Plot')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
