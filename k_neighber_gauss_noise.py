import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, det
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import warnings
from robustgp import ITGP

warnings.filterwarnings("ignore")

"""
Paper Reference:
    "Noisy Input Gaussian Process Regression"
    https://papers.nips.cc/paper_files/paper/2011/file/a8e864d04c95572d1aece099af852d0a-Paper.pdf

"""
# # Hampel フィルターの実装
def hampel_filter(vals, window_size=7, n_sigmas=3):
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
class NIGP:
    """
    Noisy Input Gaussian Process (NIGP) implementation with k-nearest neighbors.

    Approximates input noise effects via local linearization:
      f(x_i) ≈ f(\bar{x}_i) + \nabla f(\bar{x}_i)^T (x_i - \bar{x}_i)
    leading to output noise correction:
      \tilde{\sigma}_{y,i}^2 = \sigma_y^2 + (\nabla f(\bar{x}_i))^T \Sigma_x (\nabla f(\bar{x}_i)).

    Kernel: Sum of Constant and ARD RBF kernels
      k(x, x') = const_var + \sigma_f^2 exp(-0.5 \sum_d ((x_d - x'_d)^2 / l_d^2))
    """
    def __init__(self, lengthscales, signal_var, noise_y, noise_x, const_var=1.0, k=None):
        # hyperparameters
        self.lengthscales = np.array(lengthscales)  # shape (D,)
        self.signal_var = signal_var               # scalar
        self.noise_y = noise_y                     # scalar
        self.noise_x = np.array(noise_x)           # shape (D,)
        self.const_var = const_var                 # constant kernel variance
        self.k = k                                 # number of nearest neighbors to use (None=all points)

    def _kernel(self, X1, X2):
        # Constant kernel component
        const_k = self.const_var * np.ones((X1.shape[0], X2.shape[0]))
        
        # ARD RBF kernel component
        d = (X1[:, None, :] - X2[None, :, :]) / self.lengthscales
        rbf_k = self.signal_var * np.exp(-0.5 * np.sum(d**2, axis=2))
        
        # Sum of kernels
        return const_k * rbf_k

    def _posterior_gradients(self, X, K_inv_y):
        # Compute gradients of posterior mean at training inputs:
        # \nabla f_i = sum_j \alpha_j \nabla_{x_i} k(x_i, x_j)
        N, D = X.shape
        grads = np.zeros((N, D))
        for i in range(N):
            # difference to all points
            diff = (X[i] - X) / (self.lengthscales**2)
            k_vec = self.signal_var * np.exp(-0.5 * np.sum(((X[i] - X)/self.lengthscales)**2, axis=1))
            grads[i] = (k_vec * K_inv_y) @ diff  # shape (D,)
        return grads

    def _neg_log_marginal_likelihood(self, params, X, y, grads_fixed=None):
        # params: [log lengthscales, log signal_var, log noise_y, log noise_x]
        D = X.shape[1]
        idx = 0
        self.lengthscales = np.exp(params[idx: idx + D]); idx += D
        self.signal_var = np.exp(params[idx]); idx += 1
        self.noise_y = np.exp(params[idx]); idx += 1
        self.noise_x = np.exp(params[idx: idx + D]); idx += D

        K = self._kernel(X, X)
        if grads_fixed is not None:
            # heteroscedastic noise variance per point
            corr = np.sum((grads_fixed**2) * self.noise_x[None, :], axis=1)
            noise_vec = self.noise_y + corr
        else:
            noise_vec = self.noise_y * np.ones(X.shape[0])
        K += np.diag(noise_vec)

        # Cholesky for stable inverse and logdet
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y)
        nll = 0.5 * y.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi)
        return nll

    def fit(self, X, y, iterations=3):
        # initialize hyperparameters in log-space
        D = X.shape[1]
        init = np.log(np.hstack([self.lengthscales, self.signal_var, self.noise_y, self.noise_x]))
        grads = None
        for it in range(iterations):
            # optimize marginal likelihood
            res = minimize(self._neg_log_marginal_likelihood, init,
                           args=(X, y, grads), method='L-BFGS-B')
            init = res.x
            # update hyperparameters
            _ = self._neg_log_marginal_likelihood(res.x, X, y)
            # compute posterior grads for next iteration
            K = self._kernel(X, X) + np.eye(len(y)) * self.noise_y
            L, lower = cho_factor(K, lower=True)
            alpha = cho_solve((L, lower), y)
            grads = self._posterior_gradients(X, alpha)
        return self
    
    def _find_k_nearest(self, X_train, X_test_point):
        """Find indices of k nearest neighbors for a test point"""
        distances = np.sum((X_train - X_test_point)**2, axis=1)
        if self.k is None or self.k >= len(distances):
            return np.arange(len(distances))  # Use all points
        return np.argsort(distances)[:self.k]  # Return k nearest indices

    def predict(self, X_train, y_train, X_test):
        N = X_train.shape[0]
        M = X_test.shape[0]
        mu = np.zeros(M)
        var = np.zeros(M)
        
        # Compute full training gradients first (used for all predictions)
        K_full = self._kernel(X_train, X_train)
        L_full, lower_full = cho_factor(K_full + self.noise_y * np.eye(N), lower=True)
        alpha_full = cho_solve((L_full, lower_full), y_train)
        grads_full = self._posterior_gradients(X_train, alpha_full)
        
        # For each test point, use only k nearest neighbors
        for i in range(M):
            x_test_i = X_test[i:i+1]
            
            # Find k nearest neighbors
            nn_idx = self._find_k_nearest(X_train, x_test_i[0])
            X_nn = X_train[nn_idx]
            y_nn = y_train[nn_idx]
            grads_nn = grads_full[nn_idx]
            
            # Compute corrected noise vector for these neighbors
            K_nn = self._kernel(X_nn, X_nn)
            corr = np.sum((grads_nn**2) * self.noise_x[None, :], axis=1)
            noise_vec = self.noise_y + corr
            
            # Recompute cholesky with corrected noise
            K_nn += np.diag(noise_vec)
            L_nn, lower_nn = cho_factor(K_nn, lower=True)
            alpha_nn = cho_solve((L_nn, lower_nn), y_nn)
            
            # Make prediction
            K_s = self._kernel(X_nn, x_test_i)
            mu[i] = K_s.T @ alpha_nn
            v = cho_solve((L_nn, lower_nn), K_s)
            K_ss = self._kernel(x_test_i, x_test_i)
            var[i] = np.diag(K_ss - K_s.T @ v)[0]
            
        return mu, var
def load_bode_data(filepath):
    """Load three‑column (ω, |G|, arg G) data from filepath (CSV)."""
    try:
        data = np.loadtxt(filepath, delimiter=",")
        omega, mag, phase = data
        return omega, mag, phase
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# Configuration
DEFAULT_DATAFILE = "data_prepare/SKE2024_data16-Apr-2025_1819.dat"
N_TEST_POINTS = 500

# Data file path
path = Path(DEFAULT_DATAFILE)
if not path.exists():
    print(f"Warning: Data file not found: {path}")
    # Create dummy data for testing
    omega_raw = np.logspace(-1, 2, 100)
    SysGain_raw = 10 * (1 / (1 + 1j * omega_raw / 10))
    SysGain_raw = np.abs(SysGain_raw) + 0.2 * np.random.randn(len(omega_raw))
    argG_raw = np.angle(1 / (1 + 1j * omega_raw / 10)) + 0.1 * np.random.randn(len(omega_raw))
else:
    # 1) Load data
    omega_raw, SysGain_raw, argG_raw = load_bode_data(path)

# 2) Sort data
idx = np.argsort(omega_raw)
omega = omega_raw[idx]
SysGain = SysGain_raw[idx]
argG = argG_raw[idx]

# 3) Prepare modelling targets
# Log-scale input for stability
X = np.log10(omega).reshape(-1, 1)

# Magnitude in dB
y_gain = 20.0 * np.log10(SysGain)

# Unwrap phase to remove 2π discontinuities
y_phase = np.unwrap(argG)

# 4) Apply ITGP for magnitude
res_gain = ITGP(
    X, y_gain,
    alpha1=0.3,   # trim fraction lower
    alpha2=0.9,   # trim fraction upper
    nsh=2,
    ncc=2,
    nrw=1
)
gp_gain, cons_gain = res_gain.gp, res_gain.consistency

# 5) Apply ITGP for phase
res_phase = ITGP(
    X, y_phase,
    alpha1=0.3,
    alpha2=0.9,
    nsh=2,
    ncc=2,
    nrw=1
)
gp_phase, cons_phase = res_phase.gp, res_phase.consistency

# 6) Dense prediction grid
omega_test = np.logspace(
    np.log10(omega.min()),
    np.log10(omega.max()),
    N_TEST_POINTS
)
X_test = np.log10(omega_test).reshape(-1, 1)

# Predict magnitude
y_gain_pred, y_gain_std = gp_gain.predict(X_test)
y_gain_pred = y_gain_pred.ravel()
y_gain_std = y_gain_std.ravel()

y_gain_up = y_gain_pred + 1.96 * y_gain_std
y_gain_lo = y_gain_pred - 1.96 * y_gain_std

# Predict phase
y_phase_pred, y_phase_std = gp_phase.predict(X_test)
y_phase_pred = y_phase_pred.ravel()
y_phase_std = y_phase_std.ravel()

# 7) Plot Bode magnitude
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogx(omega, y_gain, "b*", label="Observed (gain)")
ax.semilogx(omega_test, y_gain_pred, "r-", lw=2, label="ITGP fit")
ax.fill_between(
    omega_test,
    y_gain_lo,
    y_gain_up,
    color="red",
    alpha=0.25,
    label="95% CI",
)
ax.set_xlabel(r"$\omega$ [rad/s]")
ax.set_ylabel(r"$20\,\log_{10}|G(j\omega)|$ [dB]")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()

# 8) Plot Bode phase
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogx(omega, argG, "b*", label="Observed (phase)")
ax.semilogx(omega_test, y_phase_pred, "r-", lw=2, label="ITGP fit")
ax.fill_between(
    omega_test,
    y_phase_pred - 1.96 * y_phase_std,
    y_phase_pred + 1.96 * y_phase_std,
    color="red",
    alpha=0.25,
    label="95% CI",
)
ax.set_xlabel(r"$\omega$ [rad/s]")
ax.set_ylabel("Phase [rad]")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()

# 9) Create Nyquist plot
G_dataset = SysGain * np.exp(1j * argG)
H_best = 10**(y_gain_pred/20) * np.exp(1j * y_phase_pred)

# Ensure curve is plotted in frequency order
order = np.argsort(omega_test)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(G_dataset.real, G_dataset.imag, 'b*', markersize=6, label='Data')
ax.plot(
    H_best.real[order],
    H_best.imag[order],
    'r-', linewidth=2,
    label='ITGP Est.'
)
ax.set_xlabel('Re')
ax.set_ylabel('Im')
ax.set_title('Nyquist Plot')
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()
plt.plot(G_dataset.real, G_dataset.imag, 'b*', label='Data')
plt.plot(H_best.real,    H_best.imag,    'r-', linewidth=2, label='ITGP Est.')
plt.xlabel('Re'); plt.ylabel('Im')
plt.title('Nyquist Plot'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()


# # Import data
# try:
#     data = np.genfromtxt('result/merged.dat', delimiter=',')
# except:
#     try:
#         data = np.genfromtxt('/root/gauss_process/result/merged.dat', delimiter=',')
#     except:
#         # If the file doesn't exist, create some dummy data for illustration
#         print("Data file not found. Creating dummy data for illustration.")
#         omega = np.logspace(-1, 2, 100)
#         sys_gain_raw = 10 * (1 / (1 + 1j * omega / 10))
#         sys_gain_raw = np.abs(sys_gain_raw) + 0.2 * np.random.randn(len(omega))
#         arg_g_raw = np.angle(1 / (1 + 1j * omega / 10)) + 0.1 * np.random.randn(len(omega))
#         data = np.vstack((omega, sys_gain_raw, arg_g_raw))
  
# # Transpose if necessary to get data in the right shape
# if data.shape[0] == 3:
#   omega = data[0, :]
#   sys_gain_raw = data[1, :]
#   arg_g_raw = data[2, :]
# else:
#   omega = data[:, 0]
#   sys_gain_raw = data[:, 1]
#   arg_g_raw = data[:, 2]

# # Sort data by frequency
# idx = np.argsort(omega)
# omega = omega[idx]
# sys_gain_raw = sys_gain_raw[idx]
# arg_g_raw = arg_g_raw[idx]

# print(f"Number of data points: {len(omega)}")

# # Remove noise using hampel filter

# sys_gain = sys_gain_raw
# arg_g = arg_g_raw
# G = sys_gain * np.exp(1j * arg_g)


# # Gaussian Process Regression for Gain
# X = np.log10(omega).reshape(-1, 1)
# Y = np.log10(sys_gain)*20

# # Split data into training and test sets (80% test, 20% train)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=20)

# model = NIGP(lengthscales=[1.0], signal_var=1.0, noise_y=0.01, noise_x=[0.1],k=25)

# # 学習
# model.fit(X_train, Y_train, iterations=5)

# # テストデータに対する予測と訓練データに対する予測を一度に行う
# X_all = np.vstack([X_train, X_test])
# # Predict for training and test data
# mu_train, var_train = model.predict(X_train, Y_train, X_train)
# mu_test, var_test = model.predict(X_train, Y_train, X_test)

# # Calculate MSE for training and test sets
# mse_train = mean_squared_error(Y_train, mu_train)
# mse_test = mean_squared_error(Y_test, mu_test)
# print(f"Training MSE: {mse_train:.4f}")
# print(f"Test MSE: {mse_test:.4f}")

# # For timing (optional)
# calculate_time = True
# if calculate_time:
#     start = time.time()

# # Create result directory if it doesn't exist
# os.makedirs("result", exist_ok=True)

# # Generate finer grid for prediction
# omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 1000)
# X_fine = np.log10(omega_fine).reshape(-1, 1)

# # Predict using the model
# Y_pred_fine, Y_var_fine = model.predict(X_train, Y_train, X_fine)
# Y_std = np.sqrt(Y_var_fine)

# # Define filename for saving
# png_name = "gp_gain_spline_more"

# # Plot the GPR results for gain
# plt.figure(figsize=(10, 6))
# # Plot original data points
# plt.semilogx(omega, 20*np.log10(sys_gain_raw), 'b.', markersize=3, alpha=0.5, label='Raw data')
# # Plot training and test data points
# plt.semilogx(10**X_test, Y_test, 'mo', markersize=6, label='Test data')
# plt.semilogx(10**X_train, Y_train, 'ro', markersize=6, label='Training data')
# # Plot GPR prediction
# plt.semilogx(omega_fine, Y_pred_fine, 'g-', linewidth=2, label='GPR prediction')
# # Add confidence bounds (±2 standard deviations)
# plt.semilogx(omega_fine, (Y_pred_fine + 2*Y_std), 'g--', linewidth=1, alpha=0.5)
# plt.semilogx(omega_fine, (Y_pred_fine - 2*Y_std), 'g--', linewidth=1, alpha=0.5)
# # Add MSE text to plot
# plt.text(0.05, 0.05, f"Train MSE: {mse_train:.4f}\nTest MSE: {mse_test:.4f}", 
#   transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

# plt.xlabel('ω [rad/sec]', fontsize=16)
# plt.ylabel('20*log₁₀|G(jω)| ', fontsize=16)
# plt.title('Bode Gain plot with GPR (Train/Test Split)', fontsize=16)
# plt.legend(fontsize=12, loc='best')
# plt.grid(True)
# plt.savefig(f"/root/gauss_process/result/{png_name}_output.png")
# plt.close()


# if calculate_time:
#   end = time.time()
#   elapsed_time = end - start
#   print(f"Elapsed time: {elapsed_time:.2f} seconds")
