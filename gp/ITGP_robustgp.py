#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITGP_robustgp.py

Iteratively-trimmed Gaussian process (ITGP) smoothing of measured
frequency-response data with outlier-robust processing.

‐ StandardScaler で入力を前処理
‐ Hampel フィルタで MSE 計算時の外れ値を除外
‐ 訓練／テストを縦に並べた Nyquist 図
"""

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from robustgp import ITGP

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#                               ハイパーパラメータ
# --------------------------------------------------------------------------- #
N_TEST_POINTS = 50000
TEST_FILENAMES = {
    "SKE2024_data18-Apr-2025_1205.dat",
}


# --------------------------------------------------------------------------- #
#                                ユーティリティ
# --------------------------------------------------------------------------- #
def load_bode_data(filepath: Path):
    """3 列 (ω, |G|, arg G) CSV を読み込む。"""
    omega, mag, phase = np.loadtxt(filepath, delimiter=",")
    return omega, mag, phase


def prepare_inputs(omega: np.ndarray):
    """log10(ω) を StandardScaler でスケールして返す。"""
    X_raw = np.log10(omega).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler

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


# --------------------------------------------------------------------------- #
#                                    main
# --------------------------------------------------------------------------- #
def main():
    # -------------------- データ読み込み --------------------
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./gp/data")

    train_om, train_mag, train_ph = [], [], []
    test_om, test_mag, test_ph = [], [], []

    for fp in sorted(data_dir.glob("SKE2024_data*.dat")):
        om, mag, ph = load_bode_data(fp)
        if fp.name in TEST_FILENAMES:
            test_om.append(om), test_mag.append(mag), test_ph.append(ph)
        else:
            train_om.append(om), train_mag.append(mag), train_ph.append(ph)

    if not train_om or not test_om:
        raise RuntimeError("Train / test split failed. Check data files.")



    # 結合 & 昇順ソート
    omega_tr = np.concatenate(train_om)
    mag_tr = np.concatenate(train_mag)
    phase_tr = np.concatenate(train_ph)
    idx_tr = np.argsort(omega_tr)
    omega_tr, mag_tr, phase_tr = omega_tr[idx_tr], mag_tr[idx_tr], phase_tr[idx_tr]

    omega_te = np.concatenate(test_om)
    mag_te = np.concatenate(test_mag)
    phase_te = np.concatenate(test_ph)
    idx_te = np.argsort(omega_te)
    omega_te, mag_te, phase_te = omega_te[idx_te], mag_te[idx_te], phase_te[idx_te]

    # -------------------- 入力前処理 --------------------
    X_tr, scaler = prepare_inputs(omega_tr)
    X_te = scaler.transform(np.log10(omega_te).reshape(-1, 1))

    # 対象出力 (実部・虚部)
    y_tr_r = mag_tr * np.cos(phase_tr)
    y_tr_i = mag_tr * np.sin(phase_tr)
    y_te_r = mag_te * np.cos(phase_te)
    y_te_i = mag_te * np.sin(phase_te)
    


    # -------------------- ITGP フィット --------------------
    res_r = ITGP(X_tr, y_tr_r, alpha1=0.50, alpha2=0.975, nsh=2, ncc=2, nrw=1)
    res_i = ITGP(X_tr, y_tr_i, alpha1=0.50, alpha2=0.975, nsh=2, ncc=2, nrw=1)
    gp_r, gp_i = res_r.gp, res_i.gp

    # -------------------- 予測 --------------------
    # 訓練・テスト
    y_tr_r_pred, _ = gp_r.predict(X_tr)
    y_tr_r_pred = y_tr_r_pred.ravel()
    y_tr_i_pred, _ = gp_i.predict(X_tr)
    y_tr_i_pred = y_tr_i_pred.ravel()
    y_te_r_pred, _ = gp_r.predict(X_te)
    y_te_r_pred = y_te_r_pred.ravel()
    y_te_i_pred, _ = gp_i.predict(X_te)
    y_te_i_pred = y_te_i_pred.ravel()

    # 高密度グリッド
    omega_dense = np.logspace(np.log10(min(omega_tr.min(), omega_te.min())),
                              np.log10(max(omega_tr.max(), omega_te.max())),
                              N_TEST_POINTS)
    X_dense = scaler.transform(np.log10(omega_dense).reshape(-1, 1))
    y_dense_r, _ = gp_r.predict(X_dense)
    y_dense_i, _ = gp_i.predict(X_dense)
    H_dense = (y_dense_r + 1j * y_dense_i).ravel()

    # -------------------- MSE (Hampel 除外) --------------------
    # 複素ゲイン（訓練・テスト）
    G_tr_true = y_tr_r + 1j * y_tr_i
    G_tr_pred = y_tr_r_pred + 1j * y_tr_i_pred
    G_te_true = y_te_r + 1j * y_te_i
    G_te_pred = y_te_r_pred + 1j * y_te_i_pred

    # ω順にソート済みの G_true の絶対値に Hampel フィルタ
    mask_tr = _hampel_filter(G_tr_true)
    mask_te = _hampel_filter(G_te_true)

    # フィルタ後に誤差を計算
    err_tr = np.abs(G_tr_true - G_tr_pred)
    err_te = np.abs(G_te_true - G_te_pred)
    mse_tr = np.sqrt(np.mean(err_tr[mask_tr] ** 2))
    mse_te = np.sqrt(np.mean(err_te[mask_te] ** 2))
    # -------------------- Nyquist プロット --------------------
    order = np.argsort(omega_dense)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    # Train
    ax1.plot(G_tr_true.real, G_tr_true.imag, "b.", label="Train data")
    ax1.plot(G_tr_pred.real, G_tr_pred.imag, "r.", label="ITGP pred")
    ax1.set_title(f"Nyquist ‑ Train | MSE={mse_tr:.3e}")
    ax1.set_xlabel("Re{G}", fontsize=14)
    ax1.set_ylabel("Im{G}", fontsize=14)
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.6, 0.2])
    ax1.grid(True)
    ax1.legend()

    # Test
    ax2.plot(G_te_true.real, G_te_true.imag, "g^", label="Test data")
    ax2.plot(G_te_pred.real, G_te_pred.imag, "ys", label="ITGP pred")
    ax2.set_title(f"Nyquist ‑ Test | MSE={mse_te:.3e}")
    ax2.set_xlabel("Re{G}", fontsize=14)
    ax2.set_ylabel("Im{G}", fontsize=14)
    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.6, 0.2])
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    out_png = Path("./gp/output/test_itgp_nyquist_train_test.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.show()

    # -------------------- Console --------------------
    print(f"Train MSE : {mse_tr:.4e}")
    print(f"Test  MSE : {mse_te:.4e}")
    print(f"Figure saved to {out_png}")


if __name__ == "__main__":
    main()
