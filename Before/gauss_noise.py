import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, det
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import os
import matplotlib.pyplot as plt

"""
Paper Reference:
    "Noisy Input Gaussian Process Regression"
    https://papers.nips.cc/paper_files/paper/2011/file/a8e864d04c95572d1aece099af852d0a-Paper.pdf

"""
class NIGP:
    """
    Noisy Input Gaussian Process (NIGP) implementation.

    Approximates input noise effects via local linearization:
      f(x_i) ≈ f(\bar{x}_i) + \nabla f(\bar{x}_i)^T (x_i - \bar{x}_i)
    leading to output noise correction:
      \tilde{\sigma}_{y,i}^2 = \sigma_y^2 + (\nabla f(\bar{x}_i))^T \Sigma_x (\nabla f(\bar{x}_i)).

    Kernel: Sum of Constant and ARD RBF kernels
      k(x, x') = const_var + \sigma_f^2 exp(-0.5 \sum_d ((x_d - x'_d)^2 / l_d^2))
    """
    def __init__(self, lengthscales, signal_var, noise_y, noise_x, const_var=1.0):
        # hyperparameters
        self.lengthscales = np.array(lengthscales)  # shape (D,)
        self.signal_var = signal_var               # scalar
        self.noise_y = noise_y                     # scalar
        self.noise_x = np.array(noise_x)           # shape (D,)
        self.const_var = const_var                 # constant kernel variance

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

    def predict(self, X_train, y_train, X_test):
        # compute corrected noise vector at train
        K = self._kernel(X_train, X_train)
        L, lower = cho_factor(K + self.noise_y * np.eye(len(y_train)), lower=True)
        alpha = cho_solve((L, lower), y_train)
        grads = self._posterior_gradients(X_train, alpha)
        corr = np.sum((grads**2) * self.noise_x[None, :], axis=1)
        noise_vec = self.noise_y + corr
        # recompute cholesky with corrected noise
        K += np.diag(noise_vec)
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y_train)

        # predictions
        K_s = self._kernel(X_train, X_test)
        mu = K_s.T @ alpha
        v = cho_solve((L, lower), K_s)
        K_ss = self._kernel(X_test, X_test)
        var = np.diag(K_ss - K_s.T @ v)
        return mu, var
    
def nigp_exact_predict(X_train, y_train, X_test, 
                lengthscales, signal_var, noise_y, noise_x):
    """
    Exact moment matching predictor for NIGP (Section 4).
    - X_train: (N, D)
    - y_train: (N,)
    - X_test:  (M, D)
    - lengthscales: (D,)
    - signal_var: scalar σ_f^2
    - noise_y: scalar σ_y^2
    - noise_x: (D,) Σ_x diagonal
    Returns:
    mu:  (M,)  predictive means
    var: (M,)  predictive variances
    """
    N, D = X_train.shape
    M, _ = X_test.shape
    Λ = np.diag(lengthscales**2)
    Σ_x = np.diag(noise_x)
    
    # 1. 訓練中の D 行列（Section 2 で求めた勾配依存補正）
    #    D_ii = (∇f(x_i))^T Σ_x ∇f(x_i)
    #    ※ここでは事前に勾配 ∇f を求めて D として渡すか、
    #      省略可。仮に D=0 とすると標準的な入力不確かさ無視に帰着。
    D_diag = np.zeros(N)  # 例: すでに計算済みならここに格納

    # 2. カーネル行列 K と補正を足した後の L, α を準備
    def ard_rbf(A, B):
        diff = (A[:, None, :] - B[None, :, :]) / lengthscales
        return signal_var * np.exp(-0.5 * np.sum(diff**2, axis=2))
    
    K = ard_rbf(X_train, X_train)
    K += np.diag(noise_y + D_diag)            # σ_y^2 I + D
    L, lower = cho_factor(K, lower=True)
    alpha = cho_solve((L, lower), y_train)    # α = (K+noise)^(-1) y

    # 3. テスト点ごとに予測平均・分散を計算
    mu = np.zeros(M)
    var = np.zeros(M)
    for m in range(M):
        x_star = X_test[m]
        # (8) の q ベクトルを計算
        diff = X_train - x_star
        exp_arg = -0.5 * np.sum(diff @ np.linalg.inv(Σ_x + Λ) * diff, axis=1)
        norm_term = (signal_var * 
                    det(Σ_x @ np.linalg.inv(Λ) + np.eye(D))**(-0.5))
        q = norm_term * np.exp(exp_arg)

        mu[m] = q @ alpha  # 予測平均

        # (11) の Q 行列を計算
        k1 = ard_rbf(X_train, x_star[None, :]).ravel()
        Q = np.zeros((N, N))
        inv_term = np.linalg.inv(Λ + 0.5 * Σ_x)
        det_term = det(2 * Σ_x @ np.linalg.inv(Λ) + np.eye(D))**(-0.5)
        for i in range(N):
            for j in range(N):
                z = 0.5*(X_train[i] + X_train[j])
                exponent = (z - x_star) @ inv_term @ (z - x_star)
                Q[i, j] = k1[i] * k1[j] * det_term * np.exp(exponent)

        # (10) の予測分散
        var[m] = (signal_var
                - np.trace(cho_solve((L, lower), Q))
                + alpha @ (Q @ alpha)
                - mu[m]**2)

    return mu, var

# Import data
try:
    data = np.genfromtxt('result/merged.dat', delimiter=',')
except:
    try:
        data = np.genfromtxt('/root/gauss_process/result/merged.dat', delimiter=',')
    except:
        # If the file doesn't exist, create some dummy data for illustration
        print("Data file not found. Creating dummy data for illustration.")
        omega = np.logspace(-1, 2, 100)
        sys_gain_raw = 10 * (1 / (1 + 1j * omega / 10))
        sys_gain_raw = np.abs(sys_gain_raw) + 0.2 * np.random.randn(len(omega))
        arg_g_raw = np.angle(1 / (1 + 1j * omega / 10)) + 0.1 * np.random.randn(len(omega))
        data = np.vstack((omega, sys_gain_raw, arg_g_raw))
  
# Transpose if necessary to get data in the right shape
if data.shape[0] == 3:
  omega = data[0, :]
  sys_gain_raw = data[1, :]
  arg_g_raw = data[2, :]
else:
  omega = data[:, 0]
  sys_gain_raw = data[:, 1]
  arg_g_raw = data[:, 2]

# Sort data by frequency
idx = np.argsort(omega)
omega = omega[idx]
sys_gain_raw = sys_gain_raw[idx]
arg_g_raw = arg_g_raw[idx]

print(f"Number of data points: {len(omega)}")

# Remove noise using hampel filter

sys_gain = sys_gain_raw
arg_g = arg_g_raw
G = sys_gain * np.exp(1j * arg_g)


# Gaussian Process Regression for Gain
X = np.log10(omega).reshape(-1, 1)
Y = np.log10(sys_gain)*20

# Split data into training and test sets (80% test, 20% train)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

model = NIGP(lengthscales=[1.0], signal_var=1.0, noise_y=0.01, noise_x=[0.1])

# 学習
model.fit(X_train, Y_train, iterations=5)

# テストデータに対する予測と訓練データに対する予測を一度に行う
X_all = np.vstack([X_train, X_test])
# Predict for training and test data
mu_train, var_train = model.predict(X_train, Y_train, X_train)
mu_test, var_test = model.predict(X_train, Y_train, X_test)

# Calculate MSE for training and test sets
mse_train = mean_squared_error(Y_train, mu_train)
mse_test = mean_squared_error(Y_test, mu_test)
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# For timing (optional)
calculate_time = True
if calculate_time:
    start = time.time()

# Create result directory if it doesn't exist
os.makedirs("result", exist_ok=True)

# Generate finer grid for prediction
omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 1000)
X_fine = np.log10(omega_fine).reshape(-1, 1)

# Predict using the model
Y_pred_fine, Y_var_fine = model.predict(X_train, Y_train, X_fine)
Y_std = np.sqrt(Y_var_fine)

# Define filename for saving
png_name = "gp_gain_spline_more"

# Plot the GPR results for gain
plt.figure(figsize=(10, 6))
# Plot original data points
plt.semilogx(omega, 20*np.log10(sys_gain_raw), 'b.', markersize=3, alpha=0.5, label='Raw data')
# Plot training and test data points
plt.semilogx(10**X_test, Y_test, 'mo', markersize=6, label='Test data')
plt.semilogx(10**X_train, Y_train, 'ro', markersize=6, label='Training data')
# Plot GPR prediction
plt.semilogx(omega_fine, Y_pred_fine, 'g-', linewidth=2, label='GPR prediction')
# Add confidence bounds (±2 standard deviations)
plt.semilogx(omega_fine, (Y_pred_fine + 2*Y_std), 'g--', linewidth=1, alpha=0.5)
plt.semilogx(omega_fine, (Y_pred_fine - 2*Y_std), 'g--', linewidth=1, alpha=0.5)
# Add MSE text to plot
plt.text(0.05, 0.05, f"Train MSE: {mse_train:.4f}\nTest MSE: {mse_test:.4f}", 
  transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('ω [rad/sec]', fontsize=16)
plt.ylabel('20*log₁₀|G(jω)| ', fontsize=16)
plt.title('Bode Gain plot with GPR (Train/Test Split)', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.grid(True)
plt.savefig(f"/root/gauss_process/result/{png_name}_output.png")
plt.close()


if calculate_time:
  end = time.time()
  elapsed_time = end - start
  print(f"Elapsed time: {elapsed_time:.2f} seconds")
