import numpy as np
from scipy.signal import lfilter
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# USER PARAMETERS
frf_file     = 'predicted_G_values.csv'  # FRF in CSV: [omega, ReG, ImG]
io_file      = 'data_hour.mat'           # recorded I/O to replay
model_order  = 50                        # FIR model order (タップ数)
n_ifft       = 1024                      # IFFT 点数（FFT 長）
train_frac   = 0.8                       # 学習用データ割合

# 1) Load FRF from CSV and compute pure FIR (optional initialization)
data = np.loadtxt(frf_file, delimiter=',', skiprows=1)
omega = data[:,0]
ReG   = data[:,1]
ImG   = data[:,2]
G_pos = ReG + 1j*ImG

# Uniform frequency grid & Hermitian symmetry for IFFT
Npos = len(omega)
omega_min, omega_max = omega.min(), omega.max()
Nfft = 2**math.ceil(math.log2(4*Npos))
omega_uni = np.linspace(omega_min, omega_max, Nfft//2 + 1)
G_uni = np.interp(omega_uni, omega, G_pos.real) + 1j*np.interp(omega_uni, omega, G_pos.imag)
G_full = np.concatenate([np.conj(G_uni[-2:0:-1]), G_uni])

# Inverse FFT to get impulse response
nfft_full = len(G_full)
h_full = np.real(np.fft.ifft(np.fft.ifftshift(G_full), n=nfft_full))
g_init = h_full[:model_order]

# 2) Load I/O data
io_data = loadmat(io_file)
for name, arr in io_data.items():
    if not name.startswith('__'):
        mat = arr
        break
time = mat[0,:10000].ravel()
y    = mat[1,:10000].ravel()
u    = mat[2,:10000].ravel()

# 3) Build regression matrix X and target vector y_target
N = len(u)
X = np.zeros((N - model_order, model_order))
for i in range(model_order):
    X[:, i] = u[model_order - i - 1 : N - i - 1]
y_target = y[model_order:]

# 4) Split into train/test and subsample for GP
split = int(len(X) * train_frac)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_target[:split], y_target[split:]
time_test = time[model_order:][split:]

# Subsample training data to prevent memory issues with GP
max_train_samples = 5000  # Limit GP training samples
if len(X_train) > max_train_samples:
    indices = np.random.choice(len(X_train), max_train_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

# 5) GP-NFIR: Gaussian Process regression on input history → output
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=5, normalize_y=True)
gpr.fit(X_train, y_train)

# 6) Predict with GP-NFIR
y_pred = gpr.predict(X_test)

# 7) Compute performance metrics
rmse  = np.sqrt(np.mean((y_test - y_pred)**2))
nrmse = rmse / (np.max(y_test) - np.min(y_test))
r2    = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

print(f"GP-NFIR (train {train_frac*100:.0f}%):")
print(f" RMSE:  {rmse:.4f}")
print(f" NRMSE: {nrmse:.4f}")
print(f" R^2:   {r2:.4f}")

# 8) Plot Measured vs Predicted
plt.figure(figsize=(8,4))
plt.plot(time_test, y_test,    label='Measured')
plt.plot(time_test, y_pred,    label='GP-NFIR Prediction', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.tight_layout()
plt.show()

# 9) Save GP model results and baseline FIR
savemat("gp_nfir_results.mat", {
    "g_init": g_init,
    "time_test": time_test,
    "y_test": y_test,
    "y_pred": y_pred
})
