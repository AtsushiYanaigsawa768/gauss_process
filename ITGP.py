import numpy as np
import matplotlib.pyplot as plt
from robustgp import ITGP

# # Hampel フィルターの実装
# def hampel_filter(vals, window_size=7, n_sigmas=3):
#     vals = vals.copy()
#     L = len(vals)
#     k = window_size // 2
#     for i in range(L):
#         start = max(0, i - k)
#         end   = min(L, i + k + 1)
#         window = vals[start:end]
#         med = np.median(window)
#         mad = 1.4826 * np.median(np.abs(window - med))
#         if mad > 0 and np.abs(vals[i] - med) > n_sigmas * mad:
#             vals[i] = med
#     return vals

# 1) データ読み込み
filename = 'data_prepare/SKE2024_data16-Apr-2025_1819.dat'
data = np.loadtxt(filename, delimiter=',')
omega_raw, SysGain_raw, argG_raw = data

# # 2) Hampel フィルタ
# SysGain_f = hampel_filter(SysGain_raw, window_size=7, n_sigmas=3)
# argG_f    = hampel_filter(argG_raw,    window_size=7, n_sigmas=3)

# 3) ソート
idx = np.argsort(omega_raw)
omega   = omega_raw[idx]
# SysGain = SysGain_f[idx]
# argG    = argG_f[idx]
SysGain = SysGain_raw[idx]
argG   = argG_raw[idx]

# 4) 入力 X と出力 y を用意
X        = omega.reshape(-1, 1)
y_gain   = 20 * np.log10(SysGain)
y_phase  = argG

# 5) ITGP による頑健 GPR (トリミング)
#    α₁=0.5, α₂=0.975, nsh=2, ncc=2, nrw=1 を例に
res_gain  = ITGP(X, y_gain,  alpha1=0.5, alpha2=0.975, nsh=2, ncc=2, nrw=1)
res_phase = ITGP(X, y_phase, alpha1=0.5, alpha2=0.975, nsh=2, ncc=2, nrw=1)
gp_gain,  cons_gain  = res_gain.gp,  res_gain.consistency
gp_phase, cons_phase = res_phase.gp, res_phase.consistency

# 6) 予測用 ω 帯を作成
omega_test = np.logspace(np.log10(omega.min()),
                         np.log10(omega.max()), 500)
Xtest = omega_test.reshape(-1,1)

# 7) 予測
y_gain_pred,  y_gain_std  = gp_gain.predict(Xtest)
y_phase_pred, y_phase_std = gp_phase.predict(Xtest)

# flatten predictions to 1D arrays for plotting
y_gain_pred  = y_gain_pred.ravel()
y_gain_std   = y_gain_std.ravel()
y_phase_pred = y_phase_pred.ravel()
y_phase_std  = y_phase_std.ravel()

# 8) プロット
plt.figure(figsize=(6,4))
plt.semilogx(omega, y_gain,    'b*', label='Observed (gain)')
plt.semilogx(omega_test, y_gain_pred, 'r-', label='ITGP fit')
plt.fill_between(omega_test,
    y_gain_pred - 2*y_gain_std,
    y_gain_pred + 2*y_gain_std,
    alpha=0.2, color='r')
plt.xlabel('ω [rad/s]')
plt.ylabel('20 log₁₀|G(jω)| [dB]')
plt.legend(); plt.grid(True)

plt.figure(figsize=(6,4))
plt.semilogx(omega, y_phase,    'b*', label='Observed (phase)')
plt.semilogx(omega_test, y_phase_pred, 'r-', label='ITGP fit')
plt.fill_between(omega_test,
    y_phase_pred - 2*y_phase_std,
    y_phase_pred + 2*y_phase_std,
    alpha=0.2, color='r')
plt.xlabel('ω [rad/s]')
plt.ylabel('Phase [rad]')
plt.legend(); plt.grid(True)

# 9) Nyquist プロット
G_dataset = SysGain * np.exp(1j * argG)
H_best    = 10**(y_gain_pred/20) * np.exp(1j * y_phase_pred)

plt.figure(figsize=(6,4))
plt.plot(G_dataset.real, G_dataset.imag, 'b*', label='Data')
plt.plot(H_best.real,    H_best.imag,    'r-', linewidth=2, label='ITGP Est.')
plt.xlabel('Re'); plt.ylabel('Im')
plt.title('Nyquist Plot'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()
