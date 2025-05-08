import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gpflow
import warnings
import data_prepare.data_load as data_loader
from data_prepare.accuracy import accuracy
warnings.filterwarnings('ignore')

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
# Import data
X_train, X_test, Y_train, Y_test, omega, sys_gain_raw = data_loader.data_loader()
# Convert to float32 for TensorFlow compatibility
# Convert to float64 for GPflow compatibility
X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64).reshape(-1, 1)
X_test = X_test.astype(np.float64)
Y_test = Y_test.astype(np.float64).reshape(-1, 1)
# Function to plot GPflow model predictions

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
