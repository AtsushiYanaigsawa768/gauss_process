import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel 
import warnings

from data_prepare.data_load import data_loader
from data_prepare.accuracy import accuracy

warnings.filterwarnings("ignore")

### Kernel Setting
const = ConstantKernel()
rbf = RBF(0.1, (1e-2, 1e1))

# kernel = kernel_model[0]
kernel = const * rbf 

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1)

X_train, X_test, Y_train, Y_test,omega,sys_gain_raw = data_loader()

gpr.fit(X_train, Y_train)

# Generate finer grid for prediction
omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 500)
X_fine = np.log10(omega_fine).reshape(-1, 1)

# Predict using the GPR model
Y_pred_fine, Y_std = gpr.predict(X_fine, return_std=True)

# Predict on train and test sets
Y_pred_train = gpr.predict(X_train)
Y_pred_test = gpr.predict(X_test)

mse_train, mse_test = accuracy(Y_train, Y_test, Y_pred_test, Y_pred_train)

# Plot the GPR results for gain
# 適宜調節すること
plt.figure(figsize=(10, 6))
# Plot original data points
plt.semilogx(omega, 20*np.log10(sys_gain_raw), 'b.', markersize=3, alpha=0.5, label='Raw data')
# Plot training and test data points - convert log values back for consistent plotting
plt.semilogx(10**X_train, Y_train, 'ro', markersize=6, label='Training data')
plt.semilogx(10**X_test, Y_test, 'mo', markersize=6, label='Test data')
# Plot GPR prediction - multiply by 20 for dB scale
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
plt.savefig(f"result/sample_output.png")
plt.close()