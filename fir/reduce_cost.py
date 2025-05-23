import numpy as np
from scipy import interpolate
from scipy.io import loadmat, savemat
from scipy.fft import ifft, ifftshift
import pandas as pd

import matplotlib.pyplot as plt

# USER PARAMETERS
frf_csv = './fir/data/predicted_G_values.csv'  # FRF in CSV: [omega, ReG, ImG]
io_file = './fir/data/data_hour.mat'           # recorded I/O to replay
lambda_factor = 0.995               # RLS forgetting factor
energy_cut = 0.99                   # keep ≥99% of |g| energy
plot_rate = 100                     # samples between plot refreshes

# 1) Load FRF from CSV
df = pd.read_csv(frf_csv)
M = df.values.T  # Transpose to get features as rows
omega = M[0, :]
ReG = M[1, :]
ImG = M[2, :]
G_pos = ReG + 1j * ImG

# Uniform frequency grid
Npos = len(omega)
omega_min = min(omega)
omega_max = max(omega)
Nfft = 2 ** int(np.ceil(np.log2(4 * Npos)))
omega_uni = np.linspace(omega_min, omega_max, Nfft // 2 + 1)

# Interpolation
G_uni = interpolate.pchip_interpolate(omega, G_pos, omega_uni)

# Hermitian symmetric spectrum
G_full = np.concatenate([np.conj(G_uni[Nfft//2-1:0:-1]), G_uni])

# 2) Impulse response via IFFT
g_full = np.real(ifft(ifftshift(G_full)))

# Sampling frequency
Dw = omega_uni[1] - omega_uni[0]
Fs = Dw * Nfft / (2 * np.pi)
Ts = 1 / Fs

# Trim by energy
Etotal = np.sum(np.abs(g_full) ** 2)
cumE = np.cumsum(np.abs(g_full) ** 2)
L = np.where(cumE / Etotal >= energy_cut)[0][0]
L = max(L, 4)

# Apply window
w = np.hanning(L)
h_init = g_full[:L] * w

print(f'[INFO] FIR length L = {L}  (Ts = {Ts:.4g} s)')

# 3) Load I/O data to replay
mat_data = loadmat(io_file)
var_names = [name for name in mat_data.keys() if not name.startswith('__')]
mat = mat_data[var_names[0]]

if mat.shape[0] >= 3 and mat.shape[1] > 3:
    # 3 rows: time; y; u
    time = mat[0, :].T
    y = mat[1, :].T
    u = mat[2, :].T
elif mat.shape[1] >= 3:
    # 3 columns: [time, y, u]
    time = mat[:, 0]
    y = mat[:, 1]
    u = mat[:, 2]
else:
    raise ValueError('I/O data has unexpected size')

# Calculate dt
if len(time) > 1:
    dtv = np.diff(time)
    dt = np.mean(dtv)
    if np.max(np.abs(dtv - dt)) > 0.01 * dt:
        print('Warning: Non-uniform time steps detected.')
else:
    dt = Ts
print(f'Loaded I/O: {len(time)} samples, dt = {dt:.4g} s')

# Resample if necessary
if abs(dt - Ts) > 1e-6:
    print('Resampling to FIR Ts...')
    t_old = time
    t_new = np.arange(0, np.ceil(time[-1] / Ts) + 1) * Ts
    u = np.interp(t_new, t_old, u)
    y = np.interp(t_new, t_old, y)
    dt = Ts
N = len(u)

# 4) Partial‐Update LMS initialization
h = h_init.copy()  # FIR taps
mu = 1e-3          # LMS stepsize
M_update = 10      # number of taps to update per sample
phi = np.zeros(L)

yhat = np.zeros(N)
err = np.zeros(N)
H = np.zeros((L, N))  # Store coefficient history

# 5) Real‐time loop with M‐Max partial‐update LMS
plt.figure(figsize=(10, 8))
plt.suptitle('Partial-Update LMS Identification')

ax1 = plt.subplot(3, 1, 1)
plt.grid(True)
h_meas, = plt.plot([], [], 'k', label='Measured')
h_pred, = plt.plot([], [], 'r--', label='Predicted')
plt.legend()
plt.xlabel('n')
plt.ylabel('y')

ax2 = plt.subplot(3, 1, 2)
plt.grid(True)
h_err, = plt.plot([], [], 'b')
plt.xlabel('n')
plt.ylabel('error')

ax3 = plt.subplot(3, 1, 3)
h_coef, = plt.plot(range(1, L+1), h, 'b-o', markersize=3)
plt.grid(True)
plt.xlabel('Tap index')
plt.ylabel('h')
ax3.set_title('Iteration 0')
plt.ylim([min(h_init)*1.1, max(h_init)*1.1])

plt.tight_layout()

for n in range(N):
    # update input vector
    phi = np.roll(phi, 1)
    phi[0] = u[n]
    
    # prediction and error
    yhat[n] = np.dot(phi, h)
    err[n] = y[n] - yhat[n]
    
    # partial‐update LMS
    delta = mu * phi * err[n]
    idx_sort = np.argsort(np.abs(delta))[::-1]  # Descending order
    S = idx_sort[:min(M_update, L)]
    h[S] += delta[S]
    
    # store coefficient history
    H[:, n] = h
    
    # update plots periodically
    if (n % plot_rate == 0) or (n == N - 1):
        n_plot = n + 1
        h_meas.set_data(range(1, n_plot+1), y[:n_plot])
        h_pred.set_data(range(1, n_plot+1), yhat[:n_plot])
        h_err.set_data(range(1, n_plot+1), err[:n_plot])
        h_coef.set_ydata(h)
        ax3.set_title(f'Iteration {n}')
        
        # Update axis limits
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        plt.pause(0.01)

# 6) Error metrics & save
rmse = np.sqrt(np.mean(err[L:] ** 2))
yn = y[L:] - np.mean(y[L:])
nrmse = 1 - np.linalg.norm(err[L:]) / np.linalg.norm(yn)
R2 = 1 - np.sum(err[L:] ** 2) / np.sum(yn ** 2)

print('\n===== FINAL ERROR =====')
print(f'RMSE  = {rmse:.4g}')
print(f'NRMSE = {nrmse*100:.2f} %')
print(f'R^2   = {R2:.3f}')

# Save results
savemat('pu_lms_results.mat', {
    'h': h,
    'rmse': rmse,
    'nrmse': nrmse,
    'R2': R2,
    'yhat': yhat,
    'err': err,
    'Ts': Ts
})

plt.show()