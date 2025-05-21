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
from data_prepare.data_load import data_loader
from data_prepare.accuracy import accuracy
import glob
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
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, alpha=0.1)
# 1) Load data
# Find all .dat files in data_prepare directory

dat_files = glob.glob('data_prepare/*.dat')
print(f"Found {len(dat_files)} .dat files")

# Load and combine all .dat files
data_list = []
for file in dat_files:
  try:
    file_data = np.loadtxt(file, delimiter=',')
    # Check and adjust the orientation if needed
    if file_data.ndim == 2 and file_data.shape[0] == 3:
        # Data is already in the correct format (3 rows)
        data_list.append(file_data)
    elif file_data.ndim == 2 and file_data.shape[1] == 3:
        # Data has 3 columns, transpose to get 3 rows
        data_list.append(file_data.T)
    else:
        print(f"Skipping {file}: incompatible shape {file_data.shape}")
        continue
    print(f"Loaded {file}: shape after processing {data_list[-1].shape}")
  except Exception as e:
    print(f"Error loading {file}: {e}")

# Make sure we have data to process
if not data_list:
    print("No valid data files could be loaded!")
    sys.exit(1)
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
# Concatenate horizontally (along columns) since data is in row format
data = np.hstack(data_list)
print(f"Combined data shape: {data.shape}")
omega_raw, sys_gain_raw, arg_g_raw = data
SysGain_f = hampel_filter(sys_gain_raw, window_size=7, n_sigmas=3)
argG_f    = hampel_filter(arg_g_raw,    window_size=7, n_sigmas=3)

# 2) Sort data by frequency
idx = np.argsort(omega_raw)
omega = omega_raw[idx]
sys_gain = SysGain_f[idx]
arg_g = argG_f  [idx]
# → Hampel フィルターをかける
# 3) Prepare input X and output y
X = omega.reshape(-1, 1)
y_gain = np.log10(sys_gain)
y_phase = arg_g

# 4) Setup GPR for gain (using the kernel already defined above)
gpr_gain = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1)
gpr_gain.fit(X, y_gain)

# 5) Setup GPR for phase
# gpr_phase = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1)
# gpr_phase.fit(X, y_phase)

# 6) Create prediction frequency grid
omega_test = np.logspace(np.log10(omega.min()), np.log10(omega.max()), 500)
X_test = omega_test.reshape(-1, 1)

# 7) Make predictions
y_gain_pred, y_gain_std = gpr_gain.predict(X_test, return_std=True)
# y_phase_pred, y_phase_std = gpr_phase.predict(X_test, return_std=True)

# 8) Plot gain
plt.figure(figsize=(10, 6))
plt.semilogx(omega, y_gain, 'b*', markersize=6, label='Observed (gain)')
plt.semilogx(omega_test, y_gain_pred, 'r-', linewidth=2, label='GPR fit')
plt.fill_between(omega_test,
        y_gain_pred - 2*y_gain_std,
        y_gain_pred + 2*y_gain_std,
        alpha=0.2, color='r')
plt.xlabel('ω [rad/s]', fontsize=16)
plt.ylabel('20 log₁₀|G(jω)| [dB]', fontsize=16)
plt.title('Bode Gain Plot with GPR', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.grid(True)
plt.ylim(-50,10)
plt.savefig(f"result/{png_name}_gain.png")
plt.close()

# # 9) Plot phase
plt.figure(figsize=(10, 6))
plt.semilogx(omega, y_phase, 'b*', markersize=6, label='Observed (phase)')
plt.semilogx(omega_test, y_phase_pred, 'r-', linewidth=2, label='GPR fit')
plt.fill_between(omega_test,
        y_phase_pred - 2*y_phase_std,
        y_phase_pred + 2*y_phase_std,
        alpha=0.2, color='r')
plt.xlabel('ω [rad/s]', fontsize=16)
plt.ylabel('Phase [rad]', fontsize=16)
plt.title('Bode Phase Plot with GPR', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.grid(True)
plt.savefig(f"result/{png_name}_phase.png")
plt.close()

# 10) Nyquist plot
G_dataset = sys_gain * np.exp(1j * arg_g)
H_best = 10**(y_gain_pred/20) * np.exp(1j * y_phase_pred)

plt.figure(figsize=(10, 6))
plt.plot(G_dataset.real, G_dataset.imag, 'b*', markersize=6, label='Data')
plt.plot(H_best.real, H_best.imag, 'r-', linewidth=2, label='GPR Est.')
plt.xlabel('Re', fontsize=16)
plt.ylabel('Im', fontsize=16)
plt.title('Nyquist Plot', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"result/{png_name}_nyquist.png")
plt.close()