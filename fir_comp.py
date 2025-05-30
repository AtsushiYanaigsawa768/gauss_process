import numpy as np
from scipy.signal import lfilter
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import math

# --- User Parameters ---
frf_file    = 'predicted_G_values.csv'  # FRF CSV file: [frequency, ReG, ImG]
io_file     = 'ten_minitues.mat'         # Recorded I/O data file for replay
model_order = 101                        # FIR model order (number of taps)

# --- 1. FIR Coefficient Estimation via IFFT (Original Code Base) ---
print("--- 1. FIR Coefficient Estimation via IFFT ---")
g_ifft_coeffs = np.zeros(model_order) # Initialize
try:
    # Load FRF from CSV
    data_frf = np.loadtxt(frf_file, delimiter=',', skiprows=1)
    omega_frf = data_frf[:,0]
    ReG_frf   = data_frf[:,1]
    ImG_frf   = data_frf[:,2]
    G_pos_frf = ReG_frf + 1j*ImG_frf

    # Create uniformly spaced frequency grid for IFFT
    Npos_frf = len(omega_frf)
    if Npos_frf == 0:
        raise ValueError("No data in FRF file.") # English
        
    omega_min_frf = np.min(omega_frf)
    omega_max_frf = np.max(omega_frf)
    # Determine FFT length (sufficient zero-padding, at least 1024 points, or >= 4x FRF data points)
    Nfft_frf = 2**math.ceil(math.log2(max(4*Npos_frf, 1024))) 
    omega_uni_frf = np.linspace(omega_min_frf, omega_max_frf, Nfft_frf//2 + 1)

    # Complex interpolation
    G_uni_frf_real = np.interp(omega_uni_frf, omega_frf, G_pos_frf.real)
    G_uni_frf_imag = np.interp(omega_uni_frf, omega_frf, G_pos_frf.imag)
    G_uni_frf = G_uni_frf_real + 1j * G_uni_frf_imag

    # Build full Hermitian spectrum
    if len(G_uni_frf) > 1: 
        G_neg_frf = np.conj(G_uni_frf[-2:0:-1]) 
        G_full_frf = np.concatenate([G_uni_frf[0:1], G_neg_frf, G_uni_frf[1:]]) if Nfft_frf % 2 == 1 else np.concatenate([G_uni_frf, np.conj(G_uni_frf[-2:0:-1])])
    else: 
        G_full_frf = G_uni_frf

    # Impulse response via IFFT
    g_ifft_raw = np.real(np.fft.ifft(G_full_frf)) 

    # Get impulse response coefficients according to model_order
    if len(g_ifft_raw) >= model_order:
        g_ifft_coeffs = g_ifft_raw[:model_order]
    else:
        # Zero-pad if IFFT result is shorter than model_order
        g_ifft_coeffs[:len(g_ifft_raw)] = g_ifft_raw
    print(f"Using the first {len(g_ifft_coeffs)} coefficients of the impulse response obtained from IFFT.") # English

except FileNotFoundError:
    print(f"Warning: FRF file '{frf_file}' not found. Skipping IFFT method.") # English
except Exception as e:
    print(f"Warning: An error occurred during IFFT processing: {e}. Skipping IFFT method.") # English


# --- 2. Loading I/O Data ---
print("\n--- 2. Loading I/O Data ---") # English
try:
    io_data = loadmat(io_file) # Load .mat file
    mat_data = None
    # Get data from the first non-internal variable (not starting with '__') in the .mat file
    for name, arr in io_data.items():
        if not name.startswith('__'): 
            mat_data = arr
            break
    
    if mat_data is None:
        raise ValueError(f"Data structure not found in {io_file}.") # English

    num_samples_to_use = mat_data.shape[1] # Use all samples
    
    if mat_data.shape[0] < 3:
         raise ValueError(f"Data format in {io_file} is incorrect. At least 3 rows (time, output, input) are required.") # English

    time_vec = mat_data[0, :num_samples_to_use].ravel() # Time vector
    y_meas   = mat_data[1, :num_samples_to_use].ravel() # Measured output
    u_input  = mat_data[2, :num_samples_to_use].ravel() # Input signal

    print(f"Number of loaded I/O data samples: {len(time_vec)}") # English

except FileNotFoundError:
    print(f"Error: I/O file '{io_file}' not found. Exiting script.") # English
    exit()
except Exception as e:
    print(f"Error: An error occurred while loading I/O data: {e}. Exiting script.") # English
    exit()


# --- 3. FIR Coefficient Estimation via Batch Least Squares ---
print("\n--- 3. FIR Coefficient Estimation via Batch Least Squares ---") # English
g_ls_coeffs = np.zeros(model_order) # Initialize
can_run_ls = False # Initialize flag

# Create data matrix Phi and output vector Y_ls
N_data = len(u_input)
# Number of effective data points for estimation
N_eff = N_data - (model_order - 1) 

if N_eff <= 0:
    print(f"Warning: Data length ({N_data}) is too short for model order ({model_order}). Skipping Batch Least Squares method.") # English
    can_run_ls = False
else:
    can_run_ls = True
    if N_eff < model_order:
        print(f"Warning: Number of effective data points ({N_eff}) is less than model order ({model_order}). LS estimation results may be unstable.") # English

    # Phi: (N_eff x model_order) matrix
    # Y_ls: (N_eff x 1) vector
    Phi = np.zeros((N_eff, model_order))
    # Y_ls consists of elements from (model_order-1)-th onwards from y_meas
    Y_ls = y_meas[model_order-1 : N_data] # Length N_eff

    # Construct data matrix Phi
    for i in range(N_eff):
        input_segment = u_input[i : i + model_order]
        Phi[i, :] = input_segment[::-1] # Reverse to have most recent input first

    # Calculate coefficients g_ls using least squares
    try:
        g_ls_coeffs, residuals, rank, singular_values = np.linalg.lstsq(Phi, Y_ls, rcond=None)
        print(f"FIR coefficients estimated by Batch Least Squares (first few): {g_ls_coeffs[:min(5, model_order)]}") # English
    except np.linalg.LinAlgError as e:
        print(f"Warning: A linear algebra error occurred during Batch Least Squares calculation: {e}. Skipping LS method.") # English
        can_run_ls = False


# --- 4. Output Prediction and Evaluation by Both Methods ---
print("\n--- 4. Output Prediction and Evaluation by Both Methods ---") # English

# (A) Prediction by IFFT-based model
yhat_ifft = lfilter(g_ifft_coeffs, [1.0], u_input)

# (B) Prediction by Batch Least Squares-based model (only if runnable)
if can_run_ls:
    yhat_ls = lfilter(g_ls_coeffs, [1.0], u_input)
else:
    yhat_ls = np.zeros_like(u_input) # Fill with zeros if not executed

# Align signals (ignore initial transient response due to convolution)
if N_data > model_order:
    y_trim    = y_meas[model_order:]
    t_trim    = time_vec[model_order:]
    yhat_ifft_trim = yhat_ifft[model_order:]
    if can_run_ls:
        yhat_ls_trim   = yhat_ls[model_order:]
    else:
        yhat_ls_trim = np.zeros_like(y_trim)
else:
    print("Warning: Data length is less than or equal to model order. No data for evaluation after trimming.") # English
    y_trim = np.array([])
    t_trim = np.array([])
    yhat_ifft_trim = np.array([])
    yhat_ls_trim = np.array([])


# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred, model_name_en):
    if len(y_true) == 0: 
        print(f"{model_name_en}: Evaluation data is empty. Cannot calculate performance metrics.") # English
        return float('nan'), float('nan'), float('nan')
        
    if len(y_true) != len(y_pred):
        print(f"{model_name_en}: Lengths of measured and predicted values differ. Cannot calculate performance metrics.") # English
        return float('nan'), float('nan'), float('nan')

    rmse  = np.sqrt(np.mean((y_true - y_pred)**2))
    
    y_range = np.max(y_true) - np.min(y_true)
    if y_range == 0: 
        nrmse = 0.0 if rmse < 1e-9 else float('inf') 
    else:
        nrmse = rmse / y_range

    y_mean = np.mean(y_true)
    sum_sq_total = np.sum((y_true - y_mean)**2)
    if sum_sq_total == 0: 
        r2 = 1.0 if np.allclose(y_true, y_pred) else 0.0 
    else:
        r2    = 1 - np.sum((y_true - y_pred)**2) / sum_sq_total

    print(f"\n{model_name_en} Performance:") # English
    print(f"  RMSE:   {rmse:.4f}")
    print(f"  NRMSE:  {nrmse:.4f}")
    print(f"  R^2:    {r2:.4f}")
    return rmse, nrmse, r2

# Evaluate IFFT-based model
print("\nIFFT-based Model Evaluation:") # English
metrics_ifft = calculate_metrics(y_trim, yhat_ifft_trim, "IFFT Model") # English model name

# Evaluate Batch Least Squares model
if can_run_ls:
    print("\nBatch Least Squares Model Evaluation:") # English
    metrics_ls = calculate_metrics(y_trim, yhat_ls_trim, "LS Model") # English model name
else:
    print("\nBatch Least Squares model was not executed, skipping evaluation.") # English
    metrics_ls = (float('nan'), float('nan'), float('nan'))


# --- 5. Plotting Results ---
print("\n--- 5. Plotting Results ---") # English
plt.style.use('seaborn-v0_8-whitegrid') 

if len(t_trim) > 0: 
    # Plot for IFFT-based model
    plt.figure(figsize=(12, 6))
    plt.plot(t_trim, y_trim, label='Measured', color='black', linewidth=1.0) # English
    plt.plot(t_trim, yhat_ifft_trim, label=f'IFFT Predicted\nR²={metrics_ifft[2]:.3f}', linestyle='--', color='dodgerblue', linewidth=1.2) # English
    plt.xlabel('Time', fontsize=12) # English
    plt.ylabel('Output', fontsize=12) # English
    plt.title(f'Output Prediction by IFFT Method (Model Order: {model_order})', fontsize=14) # English
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot for Batch Least Squares model (only if runnable)
    if can_run_ls:
        plt.figure(figsize=(12, 6))
        plt.plot(t_trim, y_trim, label='Measured', color='black', linewidth=1.0) # English
        plt.plot(t_trim, yhat_ls_trim, label=f'LS Predicted\nR²={metrics_ls[2]:.3f}', linestyle='--', color='crimson', linewidth=1.2) # English
        plt.xlabel('Time', fontsize=12) # English
        plt.ylabel('Output', fontsize=12) # English
        plt.title(f'Output Prediction by Batch Least Squares Method (Model Order: {model_order})', fontsize=14) # English
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
else:
    print("No data available for plotting. Skipping graph display.") # English

# --- 6. Save Coefficients (Optional) ---
# savemat("fir_coeffs_ifft.mat", {"g_ifft": g_ifft_coeffs, "model_order": model_order})
# if can_run_ls:
#    savemat("fir_coeffs_ls.mat", {"g_ls": g_ls_coeffs, "model_order": model_order})
print("\nScript execution completed.") # English