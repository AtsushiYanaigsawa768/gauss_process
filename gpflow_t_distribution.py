import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
import warnings

warnings.filterwarnings("ignore")


def load_bode_data(filepath: Path):
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase


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
    opt.minimize(model.training_loss, model.trainable_variables, options={"maxiter": maxiter})
    return model


def main():
    # --- load data ---
    DEFAULT_FILE = Path("data_prepare/SKE2024_data16-Apr-2025_1819.dat")
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    omega, mag, phase = load_bode_data(filepath)
    G_meas = mag * np.exp(1j * phase)

    # inputs/targets
    X = np.log10(omega).reshape(-1, 1)
    y_real = G_meas.real.reshape(-1, 1)
    y_imag = G_meas.imag.reshape(-1, 1)

    # convert to tf.Tensor
    X_tf     = tf.convert_to_tensor(X,     dtype=tf.float64)
    Yreal_tf = tf.convert_to_tensor(y_real, dtype=tf.float64)
    Yimag_tf = tf.convert_to_tensor(y_imag, dtype=tf.float64)

    # --- train real/imag GPs ---
    gp_real = make_student_t_vgp(X_tf, Yreal_tf)
    gp_imag = make_student_t_vgp(X_tf, Yimag_tf)

    # --- prediction grid ---
    N_TEST = 500
    omega_test = np.logspace(np.log10(omega.min()), np.log10(omega.max()), N_TEST)
    Xtest = np.log10(omega_test).reshape(-1, 1)
    Xtest_tf = tf.convert_to_tensor(Xtest, dtype=tf.float64)

    # predict real
    mean_r, var_r = gp_real.predict_f(Xtest_tf)
    y_real_pred = mean_r.numpy().ravel()
    y_real_std  = np.sqrt(var_r.numpy().ravel())

    # predict imag
    mean_i, var_i = gp_imag.predict_f(Xtest_tf)
    y_imag_pred = mean_i.numpy().ravel()
    y_imag_std  = np.sqrt(var_i.numpy().ravel())

    # --- plotting (same as before) ---
    plt.figure(figsize=(8, 4))
    plt.loglog(omega, y_real, 'b.', label='Measured Real')
    plt.loglog(omega_test, y_real_pred, 'r-', label='Predicted Real')
    plt.fill_between(omega_test,
                     y_real_pred - 2 * y_real_std,
                     y_real_pred + 2 * y_real_std,
                     color='r', alpha=0.2, label='±2σ')
    plt.xlabel('ω (rad/s)'); plt.ylabel('Re{G}')
    plt.legend(); plt.grid(which='both', ls='--')
    plt.tight_layout(); plt.show()

    # prepare Nyquist data and compute mse on training set
    G_filt = G_meas
    # compute training predictions for mse
    mean_r_train, _ = gp_real.predict_f(X_tf)
    mean_i_train, _ = gp_imag.predict_f(X_tf)
    H_train = mean_r_train.numpy().ravel() + 1j * mean_i_train.numpy().ravel()
    mse = np.mean(np.abs(H_train - G_filt)**2)

    plt.figure(figsize=(8, 4))
    plt.loglog(omega, y_imag, 'g.', label='Measured Imag')
    plt.loglog(omega_test, y_imag_pred, 'm-', label='Predicted Imag')
    plt.fill_between(omega_test,
                     y_imag_pred - 2 * y_imag_std,
                     y_imag_pred + 2 * y_imag_std,
                     color='m', alpha=0.2, label='±2σ')
    plt.xlabel('ω (rad/s)'); plt.ylabel('Im{G}')
    plt.legend(); plt.grid(which='both', ls='--')
    plt.tight_layout(); plt.show()

    order = np.argsort(omega_test)
    plt.figure(figsize=(10, 6))
    plt.plot(G_filt.real, G_filt.imag, 'b*', markersize=6, label='Filtered Data')
    H_best = y_real_pred + 1j * y_imag_pred
    plt.plot(
        H_best.real[order],
        H_best.imag[order],
        'r-', linewidth=2,
        label='ITGP Est.'
    )
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title(f'Nyquist Plot (MSE: {mse:.4e})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("_nyquist.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
