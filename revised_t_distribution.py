import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gpflow
import warnings

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
    X, Y, test_size=0.8, random_state=20
)

# Convert to float32 for TensorFlow compatibility
# Convert to float64 for GPflow compatibility
X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64).reshape(-1, 1)
X_test = X_test.astype(np.float64)
Y_test = Y_test.astype(np.float64).reshape(-1, 1)
# Function to plot GPflow model predictions
def plot_model(model):
    # Create fine grid for predictions
    omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 500)
    X_fine = np.log10(omega_fine).reshape(-1, 1).astype(np.float64)
    
    # Predict with GPflow
    mean, var = model.predict_y(X_fine)
    mean = mean.numpy().flatten()
    std = np.sqrt(var.numpy().flatten())
    
    # Calculate MSE for test set
    test_mean, _ = model.predict_y(X_test)
    mse_avg = np.mean((test_mean.numpy().flatten() - Y_test.flatten())**2)
    print(f"Test MSE: {mse_avg:.4f}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.log10(omega), 20*np.log10(sys_gain_raw), 'b.', alpha=0.5, label='Raw data')
    ax.plot(X_test.flatten(), Y_test.flatten(), 'mo', label='Test data')
    ax.plot(X_train.flatten(), Y_train.flatten(), 'ro', label='Train data')
    ax.plot(np.log10(omega_fine), mean, 'g-', label='GPflow Student-t')
    ax.fill_between(np.log10(omega_fine), mean - 2*std, mean + 2*std, color='g', alpha=0.2)
    ax.text(0.05, 0.05, f"Avg Test MSE: {mse_avg:.4f}", transform=ax.transAxes)
    ax.set_xlabel('log₁₀(ω) [rad/sec]')
    ax.set_ylabel('20*log₁₀|G(jω)|')
    ax.set_ylim([-100, 0])
    ax.legend()
    ax.grid(True)
    plt.savefig(f"/root/gauss_process/result/gpflow_student_t.png")
    plt.close()

# Create and train GPflow model with Student-t likelihood
model = gpflow.models.VGP(
    data=(X_train, Y_train),
    kernel=gpflow.kernels.SquaredExponential(),
    likelihood=gpflow.likelihoods.StudentT(),
)

# Optimize the model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

# Plot the model and results
plot_model(model)
