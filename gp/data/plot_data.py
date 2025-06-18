import pandas as pd
import numpy as np
from pathlib import Path

# /d:/Coding/gauss_process/data_prepare/plot_data.py

import matplotlib.pyplot as plt

def main():
  # 同じフォルダから .dat ファイルを読み込む（double precision）
  # フォルダー内のすべての .dat ファイルを読み込み，
  # 各ファイルの 0 行目→omega，1 行目→sys_gain，2 行目→sys_phase を抽出し，末尾に連結する
  # データ分割比率 (0〜1 の間)
  train_ratio = 1.0  # 例: 80% を train, 20% を test

  dat_dir = Path(__file__).parent
  dat_paths = sorted(dat_dir.glob("*.dat"))
  k = len(dat_paths)
  num_train = int(train_ratio * k)

  # ファイルを分割
  train_paths = dat_paths[:num_train]
  test_paths  = [] if train_ratio == 1.0 else dat_paths[num_train:]

  # リスト初期化
  train_omega_list, train_gain_list, train_phase_list = [], [], []
  test_omega_list,  test_gain_list,  test_phase_list  = [], [], []

  def load_lists(path_list, omega_l, gain_l, phase_l):
    for p in path_list:
      df = pd.read_csv(
        p,
        sep=r'[\s,]+',
        engine='python',
        header=None,
        dtype=np.float64
      )
      omega_l.append(df.iloc[0].to_numpy())
      gain_l.append(df.iloc[1].to_numpy())
      phase_l.append(df.iloc[2].to_numpy())

  # train/test の読み込み
  load_lists(train_paths, train_omega_list, train_gain_list, train_phase_list)
  if test_paths:
    load_lists(test_paths, test_omega_list, test_gain_list, test_phase_list)

  # numpy 配列に変換
  omega_train = np.hstack(train_omega_list)
  gain_train  = np.hstack(train_gain_list)
  phase_train = np.hstack(train_phase_list)

  z_train = gain_train * np.exp(1j * phase_train)
  mask_train = np.abs(z_train) < 0.8

  Y_train = np.column_stack((z_train.real[mask_train], z_train.imag[mask_train]))
  X_train = omega_train[mask_train]

  # テストセットがある場合
  if test_paths:
    omega_test = np.hstack(test_omega_list)
    gain_test  = np.hstack(test_gain_list)
    phase_test = np.hstack(test_phase_list)

    z_test = gain_test * np.exp(1j * phase_test)
    mask_test = np.abs(z_test) < 0.8

    Y_test = np.column_stack((z_test.real[mask_test], z_test.imag[mask_test]))
    X_test = omega_test[mask_test]
  else:
    Y_test = np.empty((0, 2))
    X_test = np.empty((0,))
  # 確認
  # # 各配列を 1 次元に連結
  omega     = np.hstack(train_omega_list + test_omega_list)
  sys_gain  = np.hstack(train_gain_list + test_gain_list)
  sys_phase = np.hstack(train_phase_list + test_phase_list)

  # 複素数を complex128（倍精度 complex）で作成
  z = sys_gain * np.exp(1j * sys_phase)

  # zが-1, -1に近づいている点は除外
  z = z[np.abs(z) < 0.8]

  # 実部と虚部
  real = z.real
  imag = z.imag

  # X のデータを 2 種類の説明変数 (実部, 虚部) として構築
  # 形状: (サンプル数, 2)
  X = np.column_stack((real, imag))

  # プロット
  plt.figure(figsize=(6,6))
  plt.scatter(X[:,0], X[:,1], c='blue', s=10)
  plt.xlabel("Real(sys_gain·e^{i·phase})")
  plt.ylabel("Imag(sys_gain·e^{i·phase})")
  plt.title("Real vs Imag")
  plt.grid(True)
  plt.axis('equal')
  plt.show()
  return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
  main()