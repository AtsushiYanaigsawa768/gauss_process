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

def _hampel_filter(x: np.ndarray, win: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Return a boolean mask whose True elements are *non-outliers* according to
    the Hampel filter applied to |x|.
    """
    k = 1.4826  # scale factor for Gaussian distribution
    n = x.size
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        i_min = max(i - win // 2, 0)
        i_max = min(i + win // 2 + 1, n)
        window = x[i_min:i_max]
        med = np.median(window)
        sigma = k * np.median(np.abs(window - med))
        if sigma > 0 and np.abs(x[i] - med) > n_sigmas * sigma:
            keep[i] = False
    return keep


def main():
    data_dir = Path("./gp/data")
    test_names = {
        "SKE2024_data18-Apr-2025_1205.dat",
    }

    # -------------------- load --------------------
    train_omega, train_mag, train_phase = [], [], []
    test_omega,  test_mag,  test_phase  = [], [], []

    for fp in data_dir.glob("SKE2024_data*.dat"):
        omega, mag, phase = load_bode_data(fp)
        if fp.name in test_names:
            test_omega.append(omega)
            test_mag.append(mag)
            test_phase.append(phase)
        else:
            train_omega.append(omega)
            train_mag.append(mag)
            train_phase.append(phase)

    if not train_omega or not test_omega:
        raise RuntimeError("Train / test split failed. Check data files.")

    # concat & sort
    omega_tr = np.concatenate(train_omega)
    mag_tr   = np.concatenate(train_mag)
    phase_tr = np.concatenate(train_phase)
    idx_tr   = np.argsort(omega_tr)
    omega_tr, mag_tr, phase_tr = omega_tr[idx_tr], mag_tr[idx_tr], phase_tr[idx_tr]

    omega_te = np.concatenate(test_omega)
    mag_te   = np.concatenate(test_mag)
    phase_te = np.concatenate(test_phase)
    idx_te   = np.argsort(omega_te)
    omega_te, mag_te, phase_te = omega_te[idx_te], mag_te[idx_te], phase_te[idx_te]

    # -------------------- prepare inputs --------------------
    X_tr, scaler = prepare_inputs(omega_tr)
    X_te = scaler.transform(np.log10(omega_te).reshape(-1, 1))

    y_tr_real = mag_tr * np.cos(phase_tr)
    y_tr_imag = mag_tr * np.sin(phase_tr)
    y_te_real = mag_te * np.cos(phase_te)
    y_te_imag = mag_te * np.sin(phase_te)

    # -------------------- train GP --------------------
    gpr_r = GaussianProcessRegressor(
        kernel=build_kernel(), n_restarts_optimizer=10, normalize_y=True, random_state=0
    )
    gpr_i = GaussianProcessRegressor(
        kernel=build_kernel(), n_restarts_optimizer=10, normalize_y=True, random_state=1
    )
    gpr_r.fit(X_tr, y_tr_real)
    gpr_i.fit(X_tr, y_tr_imag)

    # -------------------- predict --------------------
    y_tr_r_pred = gpr_r.predict(X_tr)
    y_tr_i_pred = gpr_i.predict(X_tr)
    y_te_r_pred = gpr_r.predict(X_te)
    y_te_i_pred = gpr_i.predict(X_te)

    # -------------------- Hampel filter on |G|, then evaluate --------------------
    G_tr_true = y_tr_real + 1j * y_tr_imag
    G_tr_pred = y_tr_r_pred + 1j * y_tr_i_pred
    keep_tr   = _hampel_filter(np.abs(G_tr_true))
    mse_tr    = np.sqrt(np.mean(np.abs(G_tr_true[keep_tr] - G_tr_pred[keep_tr]) ** 2))

    G_te_true = y_te_real + 1j * y_te_imag
    G_te_pred = y_te_r_pred + 1j * y_te_i_pred
    keep_te   = _hampel_filter(np.abs(G_te_true))
    mse_te    = np.sqrt(np.mean(np.abs(G_te_true[keep_te] - G_te_pred[keep_te]) ** 2))

    # -------------------- plot --------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    # Train plot
    ax1.plot(G_tr_true.real, G_tr_true.imag, "b.", label="Train data")
    ax1.plot(G_tr_pred.real, G_tr_pred.imag, "r.", linewidth=3, label="GP pred (train)")
    ax1.set_xlabel("Re", fontsize=14)
    ax1.set_ylabel("Im", fontsize=14)
    ax1.set_title(f"Nyquist plot - Train\nMSE_train={mse_tr:.3e}")
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.6, 0.2])
    ax1.grid(True)
    ax1.legend()

    # Test plot
    ax2.plot(G_te_true.real, G_te_true.imag, "g^", label="Test data")
    ax2.plot(G_te_pred.real, G_te_pred.imag, "ys", linewidth=3, label="GP pred (test)")
    ax2.set_xlabel("Re", fontsize=14)
    ax2.set_ylabel("Im", fontsize=14)
    ax2.set_title(f"Nyquist plot - Test\nMSE_test={mse_te:.3e}")
    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.6, 0.2])
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    out_png = Path("./gp/output/sample_nyquist_train_test.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.show()

    print(f"Train MSE : {mse_tr:.4e}")
    print(f"Test  MSE : {mse_te:.4e}")
    print(f"Figure saved to {out_png}")

if __name__ == "__main__":
    main()
