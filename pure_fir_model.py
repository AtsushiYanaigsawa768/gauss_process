import numpy as np
from scipy.signal import lfilter
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import math

# USER PARAMETERS
frf_file    = './fir/data/mini_predicted_G_values.csv'  # FRF in CSV: [omega, ReG, ImG]
io_file     = './fir/data/data_hour.mat'                  # recorded I/O to replay
model_order = 50                               # FIR model order (タップ数)
n_ifft       = 1024                           # IFFT 点数（FFT 長）

# 1) Load FRF from CSV
data = np.loadtxt(frf_file, delimiter=',', skiprows=1)
omega = data[:,0]
ReG   = data[:,1]
ImG   = data[:,2]
G_pos = ReG + 1j*ImG

# Make a uniformly spaced frequency grid (required for plain IFFT)
Npos = len(omega)
omega_min = np.min(omega)
omega_max = np.max(omega)
Nfft = 2**math.ceil(math.log2(4*Npos))  # plenty of zero-padding
omega_uni = np.linspace(omega_min, omega_max, Nfft//2 + 1)

# Complex interpolation
G_uni = np.interp(omega_uni, omega, G_pos.real) + 1j * np.interp(omega_uni, omega, G_pos.imag)

# Build full Hermitian spectrum
G_full = np.concatenate([np.conj(G_uni[-2:0:-1]), G_uni])

# 2) Impulse response via IFFT
g = np.real(np.fft.ifft(np.fft.ifftshift(G_full)))

print(g.shape)
# 4) Load I/O data and extract u,y
io_data = loadmat(io_file)
for name, arr in io_data.items():
    if not name.startswith('__'):
        mat = arr
        break
time = mat[0,:100000].ravel()
y    = mat[1,:100000].ravel()
u    = mat[2,:100000].ravel()

# 5) Predict output with pure FIR (convolution / lfilter)
yhat = lfilter(g, 1.0, u)

# 6) Align signals (初期 model_order サンプルは畳み込み遅延分を無視)
y_trim   = y[model_order:]
yhat_trim = yhat[model_order:]
t_trim    = time[model_order:]

# 7) Compute performance metrics
rmse  = np.sqrt(np.mean((y_trim - yhat_trim)**2))
nrmse = rmse / (np.max(y_trim) - np.min(y_trim))
r2    = 1 - np.sum((y_trim - yhat_trim)**2)/np.sum((y_trim - np.mean(y_trim))**2)

print(f"Model order: {model_order}")
print(f"RMSE:   {rmse:.4f}")
print(f"NRMSE:  {nrmse:.4f}")
print(f"R^2:    {r2:.4f}")

# 8) Plot measured vs predicted
plt.figure(figsize=(8,4))
plt.plot(t_trim, y_trim,   label='Measured')
plt.plot(t_trim, yhat_trim,label='Predicted', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.tight_layout()
plt.show()

# 9) Save coefficients
savemat("fir_from_ifft.mat", {"g": g})