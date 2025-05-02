import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os
import sys
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,ConstantKernel
warnings.filterwarnings("ignore")

###Hyperparameters
noise_filter = False #Must be set to False!!!
png_name = "const_RBF_noise" # Set the name of the output PNG file. PNG files will be saved in the "result" folder.

# Do not change the following two parameters.
calculate_time = True # Calculate the time of calculation especially about GPR
test_set_ratio = 0.8 # Must be set to 0.8!!!

### Kernel Setting
const = ConstantKernel()
rbf = RBF(0.1, (1e-2, 1e1))
matern = Matern(length_scale=1.0, nu=1.5)
exp_sine = ExpSineSquared(length_scale=1.0, periodicity=5.0, periodicity_bounds=(1, 10))
dot_product = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
white = WhiteKernel()
kernel_set = [const, rbf, matern, exp_sine, dot_product, white]

#### Kernel Model Set
kernel_model = [ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct() + WhiteKernel(),
           ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e1)) + WhiteKernel(),
           ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e1)) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=1.5) + WhiteKernel(),
           ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=0.5) + WhiteKernel(),
           ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(),
           ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]
#from https://github.com/hkaneko1985/dcekit/blob/master/demo_gp_kernel_design_test.py 


# kernel = kernel_model[0]
kernel = const * rbf 

#### RBF Setting
# See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for further information
# kernel = default ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
# alpha : (maybe) adding noise on the training data
# n_restarts_optimizer: 対数周辺尤度を最大化するための最適化を，異なる初期値から何回再起動するかを設定
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1)

# Function to simulate hampel filter (outlier detection and removal)
def hampel_filter(data, window_size=10, n_sigmas=3):
  """
  Apply a Hampel filter to remove outliers.
  
  Args:
    data: input data array
    window_size: size of window for median calculation
    n_sigmas: number of standard deviations to use for outlier detection
  
  Returns:
    Filtered data array
  """
  filtered_data = data.copy()
  n = len(data)
  k = window_size // 2
  
  for i in range(n):
    start = max(0, i - k)
    end = min(n, i + k + 1)
    
    window = data[start:end]
    median = np.median(window)
    mad = np.median(np.abs(window - median))  # MAD = Median Absolute Deviation
    
    # Convert MAD to standard deviation (scale factor 1.4826)
    sigma = 1.4826 * mad
    
    # Replace outliers with median
    if np.abs(data[i] - median) > n_sigmas * sigma and sigma > 0:
      filtered_data[i] = median
      
  return filtered_data

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
if noise_filter:
  sys_gain = hampel_filter(sys_gain_raw, window_size=15)
  arg_g = hampel_filter(arg_g_raw, window_size=15)
else:
  sys_gain = sys_gain_raw
  arg_g = arg_g_raw
G = sys_gain * np.exp(1j * arg_g)

if calculate_time:
  start  = time.time()

# Gaussian Process Regression for Gain
X = np.log10(omega).reshape(-1, 1)
Y = np.log10(sys_gain)*20

# Split data into training and test sets (80% train, 20% test)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

# Define the kernel for GPR
# 標準的なRBFカーネル（既存のもの）
# kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1)

gpr.fit(X_train, Y_train)

# Generate finer grid for prediction
omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 500)
X_fine = np.log10(omega_fine).reshape(-1, 1)

# Predict using the GPR model
Y_pred_fine, Y_std = gpr.predict(X_fine, return_std=True)

# Predict on train and test sets
Y_pred_train = gpr.predict(X_train)
Y_pred_test = gpr.predict(X_test)

# Calculate MSE for train and test sets
mse_train = mean_squared_error(Y_train, Y_pred_train)
mse_test = mean_squared_error(Y_test, Y_pred_test)
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Plot the GPR results for gain
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
plt.savefig(f"result/{png_name}_output.png")
plt.close()

# Plot model predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(Y_train, Y_pred_train, c='r', marker='o', label='Training data')
plt.scatter(Y_test, Y_pred_test, c='m', marker='s', label='Test data')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.xlabel('Actual values', fontsize=14)
plt.ylabel('Predicted values', fontsize=14)
plt.title('GPR Model: Predictions vs Actual Values', fontsize=16)
plt.legend(loc='best')
plt.grid(True)
plt.savefig(f"result/{png_name}_predictions.png")
plt.close()

if calculate_time:
  end = time.time()
  elapsed_time = end - start
  print(f"Elapsed time: {elapsed_time:.2f} seconds")