import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy import stats
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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

# No noise removal
sys_gain = sys_gain_raw
arg_g = arg_g_raw
G = sys_gain * np.exp(1j * arg_g)

# Gaussian Process Regression for Gain
X = np.log10(omega).reshape(-1, 1)
Y = np.log10(sys_gain)*20

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.8 ,random_state=20
)


# Custom Gaussian Process with Student-t likelihood
class StudentTGaussianProcess:
    def __init__(self, nu=3.0, n_restarts_optimizer=25):
        self.nu = nu  # Degrees of freedom for t-distribution
        self.n_restarts_optimizer = n_restarts_optimizer
        # Use Matern kernel which is more robust
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e1))
        
    def fit(self, X, y):
        # Initial fit with Gaussian likelihood
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=1e-10,
            normalize_y=True
        )
        self.gpr.fit(X, y)
        
        # Iterative robust fitting with t-distribution weights
        for i in range(3):  # Perform a few iterations of reweighting
            y_pred = self.gpr.predict(X)
            residuals = y - y_pred
            self.scale = np.std(residuals)
            
            # Compute weights based on t-distribution
            weights = self._compute_t_weights(residuals)
            
            # Refit with weighted samples
            self.gpr = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=self.n_restarts_optimizer,
                alpha=1e-10 / weights,  # Weight noise term
                normalize_y=True
            )
            self.gpr.fit(X, y)
        
        return self
        
    def _compute_t_weights(self, residuals):
        """Compute weights based on t-distribution density relative to Gaussian"""
        # Get t-distribution PDF at residuals
        t_pdf = stats.t.pdf(residuals / self.scale, df=self.nu)
        
        # Get normal PDF at residuals for comparison
        normal_pdf = stats.norm.pdf(residuals / self.scale)
        
        # Compute weights (avoid division by zero)
        weights = np.maximum(t_pdf / (normal_pdf + 1e-10), 1e-10)
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        return weights
    
    def predict(self, X, return_std=True):
        return self.gpr.predict(X, return_std=return_std)

# Create and fit the Student-t Gaussian Process model
t_gp = StudentTGaussianProcess(nu=3.0)
t_gp.fit(X_train, Y_train)

# Create fine grid for predictions
omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 500)
X_fine = np.log10(omega_fine).reshape(-1, 1)

# Predict with GP
Y_pred_avg, Y_std_avg = t_gp.predict(X_fine)

# Calculate MSE for test set
Y_pred_test = t_gp.predict(X_test, return_std=False)
mse_avg = np.mean((Y_pred_test - Y_test)**2)
print(f"Test MSE: {mse_avg:.4f}")

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
plt.savefig(f"/root/gauss_process/result/gp_gain_avg.png")
plt.close()