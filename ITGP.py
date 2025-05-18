import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy import stats
import warnings
from robustgp import ITGP
import matplotlib.pyplot as plt
import data_prepare.data_load as data_loader
warnings.filterwarnings('ignore')

# データ読み込み
X_train, X_test, Y_train, Y_test, omega, sys_gain_raw = data_loader.data_loader(train_ratio=1)

# テストデータの有無フラグ
has_test = (X_test is not None) and (Y_test is not None)

# ITGP 実行
res = ITGP(X_train, Y_train, alpha1=0.5, alpha2=0.975, nsh=2, ncc=2, nrw=1)
gp, consistency = res.gp, res.consistency

# ファインチューニング用グリッド
omega_fine = np.logspace(np.log10(min(omega)), np.log10(max(omega)), 500)
X_fine = np.log10(omega_fine).reshape(-1, 1)

# GP 予測 (平均＋標準偏差)
Y_pred_avg, Y_std_avg = gp.predict(X_fine)

# テストデータがあれば MSE を計算
if has_test:
    Y_pred_test = gp.predict(X_test)
    mse_avg = np.mean((Y_pred_test - Y_test)**2)
    print(f"Test MSE: {mse_avg:.4f}")
else:
    mse_avg = None
    print("No test set provided; skipping MSE calculation.")

# プロット
plt.figure(figsize=(10,6))
plt.semilogx(omega, 20*np.log10(sys_gain_raw), 'b.', alpha=0.5, label='Raw data')
plt.semilogx(10**X_train, Y_train, 'ro', label='Train data')

if has_test:
    plt.semilogx(10**X_test, Y_test, 'mo', label='Test data')

plt.semilogx(omega_fine, Y_pred_avg, 'g-', label='Averaged GPR')
plt.semilogx(omega_fine, Y_pred_avg + 2*Y_std_avg, 'g--', alpha=0.5)
plt.semilogx(omega_fine, Y_pred_avg - 2*Y_std_avg, 'g--', alpha=0.5)

# テキスト表示
txt = f"Avg Test MSE: {mse_avg:.4f}" if mse_avg is not None else "No test set"
plt.text(0.05, 0.05, txt, transform=plt.gca().transAxes)

plt.xlabel('ω [rad/sec]')
plt.ylabel('20*log₁₀|G(jω)|')
plt.ylim([-100, 0])
plt.legend()
plt.grid(True)
plt.savefig("/root/gauss_process/result/gp_gain_avg.png")
plt.close()