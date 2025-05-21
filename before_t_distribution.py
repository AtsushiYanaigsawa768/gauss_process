#!/usr/bin/env python3
# before_t_distribution.py

import sys
from pathlib import Path
import numpy as np
# from sklearn.model_selection import train_test_split # Removed
import matplotlib.pyplot as plt
import gpflow
import warnings

warnings.filterwarnings('ignore')

def load_bode_data(filepath: Path):
    """
    3 列 CSV/DAT 形式: ω, mag, phase を読み込む
    """
    data = np.loadtxt(filepath, delimiter=",")  # カンマ区切りなら delimiter=","
    # data.shape == (N,3)
    omega = data[:, 0]
    mag   = data[:, 1]
    phase = data[:, 2]
    return omega, mag, phase

def plot_model(model, omega, sys_gain_db, X_all, Y_all):
    # Fine grid
    omega_fine = np.logspace(np.log10(omega.min()), np.log10(omega.max()), 500)
    X_fine = np.log10(omega_fine).reshape(-1,1).astype(np.float64)

    # 予測
    mean, var = model.predict_y(X_fine)
    mean = mean.numpy().flatten()
    std  = np.sqrt(var.numpy().flatten())

    # プロット
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(np.log10(omega), sys_gain_db, 'b.', alpha=0.5, label='Raw data (All used for training)')
    # ax.plot(X_test.flatten(),   Y_test.flatten(), 'mo', label='Test data') # Removed
    # ax.plot(X_train.flatten(),  Y_train.flatten(), 'ro', label='Train data') # Removed
    ax.plot(X_all.flatten(), Y_all.flatten(), 'ro', label='Training data (All data)')
    ax.plot(np.log10(omega_fine), mean, 'g-', label='GPflow Student-t')
    ax.fill_between(np.log10(omega_fine), mean-2*std, mean+2*std,
                    color='g', alpha=0.2)
    # ax.text(0.05, 0.05, f"Avg Test MSE: {mse:.4f}", transform=ax.transAxes) # Removed MSE calculation
    ax.set_xlabel('log₁₀(ω) [rad/sec]')
    ax.set_ylabel('20·log₁₀|G(jω)| [dB]')
    ax.set_ylim([-100,0])
    ax.legend(); ax.grid(True)
    plt.savefig("./result/before_gpflow_student_t_all_data.png") # Changed savefig name
    plt.close()

def main():
    # ファイル指定
    DEFAULT_FILE = Path("data_prepare/SKE2024_data16-Apr-2025_1819.dat")
    filepath = Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT_FILE
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} が見つかりません")

    # 1) データ読み込み
    omega, mag, phase = load_bode_data(filepath)
    # 2) Prepare
    X = np.log10(omega).reshape(-1,1)
    sys_gain_db = 20.0 * np.log10(mag)        # dB
    Y = sys_gain_db.reshape(-1,1)
    # 3) Split # Removed
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=0
    # )

    # 4) GPflow VGP モデル (Student-t)
    model = gpflow.models.VGP(
        data=(X, Y), # Use all data X, Y
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=gpflow.likelihoods.StudentT(),
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

    # 5) プロット
    plot_model(model, omega, sys_gain_db, X, Y) # Pass X, Y to plot_model

if __name__ == "__main__":
    main()
