import numpy as np
from numpy import genfromtxt, hstack, vstack, logspace, abs, angle
from numpy.random import randn
from sklearn.model_selection import train_test_split
from pathlib import Path

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

    return X_train, X_test, Y_train, Y_test, omega, sys_gain_raw
if __name__ == "__main__":
  X_train, X_test, Y_train, Y_test, omega, sys_gain_raw = data_loader()
  # print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
  # print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
  # print(f"omega: {omega.shape}, sys_gain_raw: {sys_gain_raw.shape}")