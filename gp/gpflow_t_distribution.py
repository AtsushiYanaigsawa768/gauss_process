# /root/gauss_process/gp/gpflow_t_distribution.py
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ------------------------- IO utils ------------------------- #
def load_bode_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase


def split_train_test(data_dir: Path,
                     test_filenames: List[str]) -> Tuple[Tuple[np.ndarray, ...],
                                                         Tuple[np.ndarray, ...]]:
    train, test = [], []
    for fp in data_dir.glob("SKE2024_data*.dat"):
        (test if fp.name in test_filenames else train).append(fp)

    def _concat(files: List[Path]):
        w, m, p = [], [], []
        for f in files:
            o, mag, ph = load_bode_data(f)
            w.append(o); m.append(mag); p.append(ph)
        if not w:
            return np.empty(0), np.empty(0), np.empty(0)
        w = np.concatenate(w); m = np.concatenate(m); p = np.concatenate(p)
        idx = np.argsort(w)
        return w[idx], m[idx], p[idx]

    return _concat(train), _concat(test)


# --------------------- Hampel filter mask -------------------- #
def hampel_mask(values: np.ndarray,
                win: int = 7,
                n_sigmas: float = 3.0) -> np.ndarray:
    """Sliding-window Hampel filter (True = keep)."""
    k = 1.4826          # scale for Gaussian
    n = values.size
    keep = np.ones(n, dtype=bool)
    abs_vals = np.abs(values)

    for i in range(n):
        imin, imax = max(i - win // 2, 0), min(i + win // 2 + 1, n)
        window = abs_vals[imin:imax]
        med = np.median(window)
        sigma = k * np.median(window - med)
        if sigma > 0 and abs_vals[i] - med > n_sigmas * sigma:
            keep[i] = False
    return keep


# ---------------------- GP training util --------------------- #
def make_student_t_vgp(X: tf.Tensor,
                       Y: tf.Tensor,
                       kernel=None,
                       likelihood=None,
                       maxiter: int = 1000):
    if kernel is None:
        kernel = gpflow.kernels.SquaredExponential()
    if likelihood is None:
        likelihood = gpflow.likelihoods.StudentT()
    model = gpflow.models.VGP(data=(X, Y), kernel=kernel, likelihood=likelihood)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables,
                 options={"maxiter": maxiter}, method="L-BFGS-B")
    return model


# ------------------------------ main ------------------------- #
def main():
    DATA_DIR   = Path("./gp/data")
    TEST_FILES = ["SKE2024_data18-Apr-2025_1205.dat",
                  "SKE2024_data16-May-2025_1609.dat"]

    # ---------------- load & split ---------------- #
    (w_tr, mag_tr, ph_tr), (w_te, mag_te, ph_te) = split_train_test(DATA_DIR, TEST_FILES)
    if w_tr.size == 0 or w_te.size == 0:
        raise RuntimeError("Train / test split failed. Check data files.")

    # complex transfer functions
    G_tr = mag_tr * np.exp(1j * ph_tr)
    G_te = mag_te * np.exp(1j * ph_te)

    # ------------- input scaling ------------------ #
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(np.log10(w_tr).reshape(-1, 1))
    X_te_np = scaler.transform(np.log10(w_te).reshape(-1, 1))

    Xtr_tf = tf.convert_to_tensor(X_tr_np, dtype=tf.float64)
    Xte_tf = tf.convert_to_tensor(X_te_np, dtype=tf.float64)

    Ytr_r_tf = tf.convert_to_tensor(G_tr.real.reshape(-1, 1), dtype=tf.float64)
    Ytr_i_tf = tf.convert_to_tensor(G_tr.imag.reshape(-1, 1), dtype=tf.float64)

    # ------------- GP training -------------------- #
    gp_r = make_student_t_vgp(Xtr_tf, Ytr_r_tf)
    gp_i = make_student_t_vgp(Xtr_tf, Ytr_i_tf)

    # ------------- prediction --------------------- #
    μr_tr, _ = gp_r.predict_f(Xtr_tf)
    μi_tr, _ = gp_i.predict_f(Xtr_tf)
    Ghat_tr = μr_tr.numpy().ravel() + 1j * μi_tr.numpy().ravel()

    μr_te, _ = gp_r.predict_f(Xte_tf)
    μi_te, _ = gp_i.predict_f(Xte_tf)
    Ghat_te = μr_te.numpy().ravel() + 1j * μi_te.numpy().ravel()

    # ------------- MSE with Hampel on |G| ---------
    # ω順でソート済みの G_tr, G_te に直接フィルター
    mask_tr = hampel_mask(G_tr)
    mask_te = hampel_mask(G_te)
    # エラーはフィルター後に計算
    err_tr = np.abs(G_tr - Ghat_tr)
    err_te = np.abs(G_te - Ghat_te)
    mse_tr = np.sqrt(np.mean(err_tr[mask_tr] ** 2))
    mse_te = np.sqrt(np.mean(err_te[mask_te] ** 2))

    # ------------- plotting ----------------------- #
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    # Train
    ax1.plot(G_tr.real,  G_tr.imag,  "b.", label="Train meas.")
    ax1.plot(Ghat_tr.real, Ghat_tr.imag, "c+", label="GP est. (train)", markersize=8, lw=2)
    ax1.set_xlim([-0.7, 0.7]); ax1.set_ylim([-0.6, 0.2])
    ax1.set_xlabel("Re{G}", fontsize=14); ax1.set_ylabel("Im{G}", fontsize=14)
    ax1.set_title(f"Nyquist – Train  (MSE={mse_tr:.2e})")
    ax1.grid(True, ls="--"); ax1.legend()

    # Test
    ax2.plot(G_te.real,  G_te.imag,  "g*", label="Test meas.")
    ax2.plot(Ghat_te.real, Ghat_te.imag, "r^", label="GP est. (test)", markersize=8, lw=2)
    ax2.set_xlim([-0.7, 0.7]); ax2.set_ylim([-0.6, 0.2])
    ax2.set_xlabel("Re{G}", fontsize=14); ax2.set_ylabel("Im{G}", fontsize=14)
    ax2.set_title(f"Nyquist – Test  (MSE={mse_te:.2e})")
    ax2.grid(True, ls="--"); ax2.legend()

    plt.tight_layout()
    out_png = Path("./gp/output/t_nyquist_train_test.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.show()

    print(f"Train MSE: {mse_tr:.4e}")
    print(f"Test  MSE: {mse_te:.4e}")
    print(f"Figure saved to {out_png}")

if __name__ == "__main__":
    main()
