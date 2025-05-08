import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy import stats
import warnings
from data_prepare.data_load import data_loader
from data_prepare.accuracy import accuracy

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Custom Gaussian Process with Laplace likelihood
class HuberGaussianProcess:
    def __init__(self, delta=1.5, n_restarts_optimizer=15):
        self.delta = delta  # Huber parameter
        self.n_restarts_optimizer = n_restarts_optimizer
        # Use Matern kernel which is more robust
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(0.1, (1e-2, 1e1), nu=2.5)
        
    def fit(self, X, y):
        # Initial fit with Gaussian likelihood
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=1e-10,
            normalize_y=True
        )
        self.gpr.fit(X, y)
        
        # Iterative robust fitting with Huber loss weights
        for i in range(3):  # Perform a few iterations of reweighting
            y_pred = self.gpr.predict(X)
            residuals = y - y_pred
            
            # Adaptively update delta based on median absolute deviation
            mad = np.median(np.abs(residuals - np.median(residuals)))
            self.delta = 1.345 * mad  # Standard value for 95% efficiency
            
            # Compute weights based on Huber loss
            weights = self._compute_huber_weights(residuals)
            
            # Refit with weighted samples
            self.gpr = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=self.n_restarts_optimizer,
                alpha=1e-10 / weights,  # Weight noise term
                normalize_y=True
            )
            self.gpr.fit(X, y)
        
        return self
        
    def _compute_huber_weights(self, residuals):
        """Compute weights based on Huber loss function"""
        # Calculate absolute residuals
        abs_residuals = np.abs(residuals)
        
        # Huber weights: 1 for |r| <= delta, delta/|r| for |r| > delta
        weights = np.ones_like(residuals)
        mask = abs_residuals > self.delta
        weights[mask] = self.delta / abs_residuals[mask]
        
        # Ensure no zero weights
        weights = np.maximum(weights, 1e-10)
        
        return weights
    
    def predict(self, X, return_std=True):
        return self.gpr.predict(X, return_std=return_std)

X_train, X_test, Y_train, Y_test, omega, sys_gain_raw = data_loader()
# Create and fit the Laplace Gaussian Process model
t_gp = HuberGaussianProcess()
t_gp.fit(X_train, Y_train)

# Create fine grid for predictions
omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 500)
X_fine = np.log10(omega_fine).reshape(-1, 1)

# Predict with GP
Y_pred_avg, Y_std_avg = t_gp.predict(X_fine)

# Calculate MSE for test set
Y_pred_test = t_gp.predict(X_test, return_std=False)
mse_avg = np.mean((Y_pred_test - Y_test)**2)

accuracy(Y_train, Y_test, Y_pred_test, Y_train_pred=None)

# Plot results
plt.figure(figsize=(10,6))
plt.semilogx(omega, 20*np.log10(sys_gain_raw), 'b.', alpha=0.5, label='Raw data')
plt.semilogx(10**X_test, Y_test, 'mo', label='Test data')
plt.semilogx(10**X_train, Y_train, 'ro', label='Train data')
plt.semilogx(omega_fine, Y_pred_avg, 'g-', label='Averaged GPR')
plt.semilogx(omega_fine, Y_pred_avg+2*Y_std_avg, 'g--', alpha=0.5)
plt.semilogx(omega_fine, Y_pred_avg-2*Y_std_avg, 'g--', alpha=0.5)
plt.text(0.05,0.05,f"Avg Test MSE: {mse_avg:.4f}", transform=plt.gca().transAxes)
plt.xlabel('ω [rad/sec]')
plt.ylabel('20*log₁₀|G(jω)|')
plt.ylim([-100, 0])  # Set y-axis limits
plt.legend()
plt.grid(True)
plt.savefig(f"/root/gauss_process/result/Huber_distribution.png")
plt.close()