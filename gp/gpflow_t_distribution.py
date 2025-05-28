# /d:/Coding/gauss_process/gp/gpflow_t_distribution.py
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
import warnings

warnings.filterwarnings("ignore")


# ------------------------- IO utils ------------------------- #
def load_bode_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one *.dat file (ω, |G|, ∠G)."""
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase


def split_train_test(data_dir: Path,
                     test_filenames: List[str]) -> Tuple[Tuple[np.ndarray, ...],
                                                         Tuple[np.ndarray, ...]]:
    """Read all *.dat files, split into train / test."""
    train_files, test_files = [], []
    for f in data_dir.glob("*.dat"):
        (test_files if f.name in test_filenames else train_files).append(f)

    def _concat(files: List[Path]):
        omegas, mags, phases = [], [], []
        for fp in files:
            o, m, p = load_bode_data(fp)
            omegas.append(o); mags.append(m); phases.append(p)
        return (np.concatenate(omegas),
                np.concatenate(mags),
                np.concatenate(phases))

    return _concat(train_files), _concat(test_files)


# --------------------- Hampel filter mask -------------------- #
def hampel_mask(values: np.ndarray, n_sigmas: float = 3.0) -> np.ndarray:
    """Return Boolean mask (True = keep) using global Hampel filter."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.full(values.shape, True, dtype=bool)
    thresh = n_sigmas * 1.4826 * mad
    return np.abs(values - median) <= thresh


# ---------------------- GP training util --------------------- #
def make_student_t_vgp(X: tf.Tensor, Y: tf.Tensor,
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

    # ----------- load & split data ----------- #
    (ω_tr, mag_tr, ph_tr), (ω_te, mag_te, ph_te) = split_train_test(DATA_DIR, TEST_FILES)

    # complex transfer functions
    G_tr = mag_tr * np.exp(1j * ph_tr)
    G_te = mag_te * np.exp(1j * ph_te)

    # TF tensors (log-ω as input)
    Xtr_tf = tf.convert_to_tensor(np.log10(ω_tr).reshape(-1, 1), dtype=tf.float64)
    Xte_tf = tf.convert_to_tensor(np.log10(ω_te).reshape(-1, 1), dtype=tf.float64)

    Ytr_real_tf = tf.convert_to_tensor(G_tr.real.reshape(-1, 1), dtype=tf.float64)
    Ytr_imag_tf = tf.convert_to_tensor(G_tr.imag.reshape(-1, 1), dtype=tf.float64)

    # ----------- train GPs (real / imag) ----------- #
    gp_r = make_student_t_vgp(Xtr_tf, Ytr_real_tf)
    gp_i = make_student_t_vgp(Xtr_tf, Ytr_imag_tf)

    # ----------- predictions (train & test) ----------- #
    μr_tr, _ = gp_r.predict_f(Xtr_tf)
    μi_tr, _ = gp_i.predict_f(Xtr_tf)
    Ĥ_tr = (μr_tr.numpy().ravel() + 1j * μi_tr.numpy().ravel())

    μr_te, _ = gp_r.predict_f(Xte_tf)
    μi_te, _ = gp_i.predict_f(Xte_tf)
    Ĥ_te = (μr_te.numpy().ravel() + 1j * μi_te.numpy().ravel())

    # ----------- MSE with Hampel filter ----------- #
    res_tr = np.abs(Ĥ_tr - G_tr)
    res_te = np.abs(Ĥ_te - G_te)

    mask_tr = hampel_mask(res_tr)
    mask_te = hampel_mask(res_te)

    mse_tr = np.mean(res_tr[mask_tr] ** 2)
    mse_te = np.mean(res_te[mask_te] ** 2)

    # ----------- Nyquist plot ----------- #
    plt.figure(figsize=(8, 6))

    plt.plot(G_tr.real, G_tr.imag, 'b.',  label='Train meas.')
    plt.plot(Ĥ_tr.real, Ĥ_tr.imag, 'c+', label='GP Train est.')

    plt.plot(G_te.real, G_te.imag, 'g*',  label='Test meas.')
    plt.plot(Ĥ_te.real, Ĥ_te.imag, 'r^', label='GP Test est.')

    plt.xlabel('Re{G}')
    plt.ylabel('Im{G}')
    plt.title('Nyquist (Hampel-filtered MSE)')
    plt.grid(True, ls='--')
    plt.legend()
    plt.text(0.02, 0.98,
             f'Train MSE: {mse_tr:.2e}\nTest MSE: {mse_te:.2e}',
             transform=plt.gca().transAxes,
             va='top', ha='left', bbox=dict(facecolor='w', alpha=0.7))
    plt.tight_layout()
    plt.savefig("./gp/output/nyquist_train_test.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
