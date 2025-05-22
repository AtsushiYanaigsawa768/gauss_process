import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import math

#!/usr/bin/env python3
# fir_rt_identification.py
# Real-time identification of a flexible link using an FIR model.

import matplotlib.pyplot as plt

# USER PARAMETERS
frf_file     = 'new_result/predicted_G_values.csv'  # FRF in CSV: [omega, ReG, ImG]
io_file      = 'data_flexible.mat'                  # recorded I/O to replay
lambda_factor = 0.995                               # RLS forgetting factor
energy_cut    = 0.99                                # keep ≥99% of |g| energy
plot_rate     = 100                                 # samples between plot refreshes

# 1) Load FRF from CSV
#    Assumes no header or one header row. Adjust skiprows if needed.
data = np.loadtxt(frf_file, delimiter=',', skiprows=1)
omega = data[:, 0]
ReG   = data[:, 1]
ImG   = data[:, 2]
G_pos = ReG + 1j * ImG

# Make a uniformly spaced frequency grid (required for plain IFFT)
Npos = len(omega)
omega_min = np.min(omega)
omega_max = np.max(omega)
Nfft = 2**math.ceil(math.log2(4*Npos))  # plenty of zero-padding
omega_uni = np.linspace(omega_min, omega_max, Nfft//2 + 1)

# Complex interpolation
G_uni = np.interp(omega_uni, omega, G_pos.real) + 1j * np.interp(omega_uni, omega, G_pos.imag)

# Build full Hermitian spectrum
G_full = np.concatenate([np.conj(G_uni[-2:0:-1]), G_uni])

# 2) Impulse response via IFFT
g_full = np.real(np.fft.ifft(np.fft.ifftshift(G_full)))

# Time axis
Dw = omega_uni[1] - omega_uni[0]
Fs = Dw * Nfft / (2 * np.pi)     # sampling frequency [Hz]
Ts = 1 / Fs                      # Δt

# Trim g by cumulative energy
Etotal = np.sum(np.abs(g_full)**2)
cumE = np.cumsum(np.abs(g_full)**2)
L_indices = np.where(cumE/Etotal >= energy_cut)[0]
if len(L_indices) > 0:
  L = L_indices[0] + 1
else:
  L = len(g_full)
L = max(L, 4)  # at least 4 taps

# Window (Hann) and take first L taps
w = signal.windows.hann(L)
h_init = g_full[:L] * w

print(f'[INFO] FIR length L = {L} (Δt = {Ts:.4g} s)')

# 3) Load I/O data to replay
# 3) Load I/O data to replay
io_data = loadmat(io_file)

# 全変数を表示して、最初のデータ配列を取得
for name, arr in io_data.items():
  if not name.startswith('__'):
    print(f'{name} ⇒ shape: {arr.shape}')
    mat = arr
    break

# 1列目 → time, 2列目 → output, 3列目 → input
time = mat[0,:].ravel()
y    = mat[1,:].ravel()
u    = mat[2,:].ravel()
# 100000 samples of time, output, inputだけ読み取る
time = mat[0,:100000].ravel()
y    = mat[1,:100000].ravel()
u    = mat[2,:100000].ravel()
print(time.shape, y.shape, u.shape)
# Calculate dt from time array
if len(time) > 1:
  dt_values = np.diff(time)
  dt = np.mean(dt_values)
  
  # Check if time steps are reasonably uniform
  if np.max(np.abs(dt_values - dt)) > 0.01 * dt:
    print('Warning: Non-uniform time steps detected in input data.')
else:
  dt = Ts  # Fallback to FIR sampling time

print(f"Loaded data with {len(time)} samples, dt = {dt:.4g} s")

# The rest of the code remains the same for resampling if needed
# if abs(dt - Ts) > 1e-6:
#   print(f'Warning: dt in I/O data ({dt:.4g}) ≠ FIR Ts ({Ts:.4g}). Resampling u,y.')
#   t_fir = np.arange(math.ceil(time[-1]/Ts) + 1) * Ts
#   u = np.interp(t_fir, time, u)
#   y = np.interp(t_fir, time, y)
#   dt = Ts  # now matched

N = len(u)


# 4) RLS initialisation
h = h_init.copy()               # current FIR coeffs (column)
P = 1e4 * np.eye(L)             # covariance matrix
phi = np.zeros(L)               # regressor buffer

# Preallocate arrays for speed
yhat = np.zeros(N)
err = np.zeros(N)

# 5) Real-time (offline replay) loop
plt.figure(figsize=(10, 8), num='Real-Time FIR Identification')

ax1 = plt.subplot(2, 1, 1)
h_meas, = ax1.plot([], [], 'k', label='Measured')
h_pred, = ax1.plot([], [], 'r--', label='Predicted')
ax1.grid(True)
ax1.legend()
ax1.set_xlabel('sample n')
ax1.set_ylabel('y')

ax2 = plt.subplot(2, 1, 2)
h_err, = ax2.plot([], [], 'b')
ax2.grid(True)
ax2.set_xlabel('sample n')
ax2.set_ylabel('error')

plt.ion()  # Turn on interactive mode
plt.show()

for n in range(N):
  # Update regressor buffer φ[n] = [u[n],u[n-1],...,u[n-L+1]]
  phi = np.roll(phi, 1)
  phi[0] = u[n]
  
  if n >= L - 1:
    # Prediction
    yhat[n] = np.dot(phi, h)
    err[n] = y[n] - yhat[n]
    
    # RLS update
    K = np.dot(P, phi) / (lambda_factor + np.dot(phi, np.dot(P, phi)))
    h = h + K * err[n]
    P = (P - np.outer(K, np.dot(phi, P))) / lambda_factor
  else:
    yhat[n] = 0
    err[n] = y[n]
  
  # Live plot refresh
  if (n % plot_rate == 0) or (n == N-1):
    n_range = np.arange(n+1)
    h_meas.set_data(n_range, y[:n+1])
    h_pred.set_data(n_range, yhat[:n+1])
    h_err.set_data(n_range, err[:n+1])
    
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    plt.draw()
    plt.pause(0.001)

# 6) Error metrics
rmse = np.sqrt(np.mean(err[L:]**2))
ynorm = y[L:] - np.mean(y[L:])
nrmse = 1 - np.linalg.norm(err[L:]) / np.linalg.norm(ynorm)
R2 = 1 - np.sum(err[L:]**2) / np.sum(ynorm**2)

print('\n=====  FINAL ERROR  ====================================')
print(f'RMSE   = {rmse:.4g}')
print(f'NRMSE  = {nrmse*100:.2f} %')
print(f'R^2    = {R2:.3f}')

# Keep variables in workspace for further analysis
savemat('fir_rt_results.mat', {
  'h': h, 'rmse': rmse, 'nrmse': nrmse, 'R2': R2, 
  'yhat': yhat, 'err': err, 'Ts': Ts
})

plt.ioff()
plt.show()  # Keep the plot window open