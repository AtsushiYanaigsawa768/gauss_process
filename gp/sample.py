import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

DEFAULT_DATAFILE = "./gp/data/SKE2024_data16-Apr-2025_1819.dat"
N_TEST_POINTS = 500

def load_bode_data(filepath: Path):
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase

def prepare_inputs(omega):
    X_raw = np.log10(omega).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler

def build_kernel():
    const = ConstantKernel(1.0, (1e-3, 1e3))
    rbf   = RBF(1.0, (1e-2, 1e3))
    noise = WhiteKernel(1e-3, (1e-6, 1e1))
    return const * rbf + noise

def main():
    path = Path(sys.argv[1]) if len(sys.argv)>1 else Path(DEFAULT_DATAFILE)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # 1) Load & sort
    omega_raw, mag_raw, phase_raw = load_bode_data(path)
    idx = np.argsort(omega_raw)
    omega = omega_raw[idx]
    mag   = mag_raw[idx]
    phase = phase_raw[idx]

    # 2) 入力 X を準備
    X, scaler = prepare_inputs(omega)

    # 3) 目標：実部・虚部
    y_real = mag * np.cos(phase)
    y_imag = mag * np.sin(phase)

    # 4) GPR を学習 (real)
    gpr_real = GaussianProcessRegressor(
        kernel=build_kernel(),
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=0,
    )
    gpr_real.fit(X, y_real)

    # 5) GPR を学習 (imag)
    gpr_imag = GaussianProcessRegressor(
        kernel=build_kernel(),
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=1,
    )
    gpr_imag.fit(X, y_imag)

    # Predict on training data points for MSE calculation
    y_real_train_pred = gpr_real.predict(X)
    y_imag_train_pred = gpr_imag.predict(X)

    # Calculate MSE on complex Nyquist points
    # (comparing original data with GPR predictions at original data locations)
    G_original = y_real + 1j * y_imag
    G_pred_on_original_X = y_real_train_pred + 1j * y_imag_train_pred
    mse_complex = np.mean(np.abs(G_original - G_pred_on_original_X)**2)
    print(f"Nyquist MSE (original data vs. GPR predictions on original data): {mse_complex:.4e}")

    # 6) Dense test grid はそのまま
    omega_test = np.logspace(np.log10(omega.min()),
                             np.log10(omega.max()),
                             N_TEST_POINTS)
    X_test = scaler.transform(np.log10(omega_test).reshape(-1,1))

    # 7) 実部・虚部を予測
    y_real_pred_test, _ = gpr_real.predict(X_test, return_std=True)
    y_imag_pred_test, _ = gpr_imag.predict(X_test, return_std=True)

    # 8) Nyquist プロットのみ出力
    plt.figure(figsize=(8,6))
    plt.plot(y_real_pred_test, y_imag_pred_test, 'r-', lw=2, label='GPR Est. (on test grid)')
    plt.plot(y_real, y_imag, # Original data points
             'b*', label='Data (Original)')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title(f'Nyquist Plot (MSE: {mse_complex:.4e})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./gp/output/sample_nyquist.png", dpi=300)
    plt.show()

    # --- Save predicted data to CSV ---
    # Combine omega_test, predicted real part, and predicted imaginary part
    # The omega_test is already defined and used for predictions.
    # y_real_pred and y_imag_pred are the predictions on omega_test.
    
    output_data = np.column_stack((omega_test, y_real_pred_test, y_imag_pred_test))
    
    # Define CSV file path
    csv_filepath = Path("./gp/output/predicted_G_values.csv")
    
    # Save to CSV
    header = "omega,Re_G,Im_G"
    np.savetxt(csv_filepath, output_data, delimiter=",", header=header, comments='')
    
    print(f"Predicted G values saved to {csv_filepath}")
if __name__ == "__main__":
    main()

