import numpy as np
from numpy import genfromtxt, hstack, vstack, logspace, abs, angle
from numpy.random import randn
from sklearn.model_selection import train_test_split
from pathlib import Path
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def data_loader(
    data_dir: str = "data_prepare",
    file_pattern: str = "*.csv",
    train_ratio: float = 0.2,
    random_state: int = 20
):
    # 1) CSVファイルをすべて読み込んで横に連結
    files = sorted(Path(data_dir).glob(file_pattern))
    data_list = [np.genfromtxt(f, delimiter=',') for f in files]
    data = np.hstack(data_list)  # shape = (3, total_columns)

    # 転置チェック（shapeが(3,N)でなければ転置）
    if data.shape[0] != 3:
        data = data.T

    # A→omega, B→sys_gain_raw, C→arg_g_raw
    omega, sys_gain_raw, arg_g_raw = data

    # 2) omega の小さい順にソート
    idx_sort = np.argsort(omega)
    omega = omega[idx_sort]
    sys_gain_raw = sys_gain_raw[idx_sort]
    arg_g_raw = arg_g_raw[idx_sort]

    # 3) 同一の omega 値でグループ化
    unique_vals, counts = np.unique(omega, return_counts=True)
    groups = []
    start = 0
    for cnt in counts:
        groups.append(np.arange(start, start + cnt))
        start += cnt

    # 4) 各グループの i 番目を集めてセットを作成
    num_sets = int(np.min(counts))
    print(f"Number of sets: {num_sets}")

    sets = []
    for i in range(num_sets):
        idxs = [g[i] for g in groups]
        sets.append(idxs)

    # 5) セット単位で訓練／テストに分割
    train_sets, test_sets = train_test_split(
        sets, train_size=train_ratio, random_state=random_state
    )
    print(f"Train sets: {len(train_sets)}, Test sets: {len(test_sets)}")

    # フラットなインデックスリストに戻す
    train_idx = np.concatenate(train_sets)
    test_idx  = np.concatenate(test_sets)

    # 6) 前処理（log変換）して X, Y を作成
    X_all = np.log10(omega).reshape(-1, 1)
    Y_all = np.log10(sys_gain_raw) * 20

    X_train = X_all[train_idx]
    X_test  = X_all[test_idx]
    Y_train = Y_all[train_idx]
    Y_test  = Y_all[test_idx]

    return X_train, X_test, Y_train, Y_test, omega, sys_gain_raw,arg_g_raw

if __name__ == "__main__":
    # 1) データを読み込む
    X_train, X_test, Y_train, Y_test, omega, sys_gain_raw, arg_g_raw = data_loader(
        data_dir="result",
        file_pattern="*.dat",
        train_ratio=1.0,      # ここでは全データを使う
        random_state=0
    )

    # 2) 複素周波数応答 G(jω) を組み立て
    G = sys_gain_raw * np.exp(1j * arg_g_raw)
    s = 1j * omega

    # 3) 特異点周波数 b2 を探索（振幅最小点の ω^2 + 0.01）
    n60 = int(0.6 * omega.size)
    zpid = np.argmin(np.abs(G[:n60]))
    b1 = 1.0
    b2 = omega[zpid]**2 + 0.01

    # 4) モデル定義
    def model(p, s):
        num = b1 * s**2 + b2
        den = p[0]*s**4 + p[1]*s**3 + p[2]*s**2 + p[3]*s + p[4]
        return num / den

    # 5) 残差関数
    def residuals(p):
        D = model(p, s) - G
        return np.concatenate([D.real, D.imag])

    # 6) パラメータ推定
    best_p = None
    best_cost = np.inf
    for _ in range(30):
        p0 = np.random.rand(5) * 1e5
        res = least_squares(residuals, p0, method='lm')
        if res.cost < best_cost:
            best_cost, best_p = res.cost, res.x

    print("Best parameters:", best_p)

    # 7) Nyquist プロット
    G_fit = model(best_p, s)

    plt.figure()
    plt.plot(G.real, G.imag, 'b*', label='Data')
    plt.plot(G_fit.real, G_fit.imag, 'r-', lw=2, label='Fit')
    plt.xlabel('Re G(jω)')
    plt.ylabel('Im G(jω)')
    plt.title('Nyquist Plot')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()
