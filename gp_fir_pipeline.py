#!/usr/bin/env python3
"""
End-to-end pipeline:

GP (FRF estimation) → CSV export → Nyquist plot → FIR model build →
time-domain evaluation → results export.

This module provides two entrypoints:
- gp(config): a lightweight function-style template you can call with params
- run_complete_pipeline(...): convenience wrapper used by test_pipeline.py

Notes:
- FRF inputs are .dat files in gp/data (rows: omega, |G|, phase).
- If no real I/O .mat data is provided for FIR evaluation, a synthetic
  I/O set is generated to verify the pipeline end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat
from scipy.signal import lfilter
from scipy.fft import ifft, ifftshift


# =========================
# Utility: Hampel filtering
# =========================
def _hampel_keep_mask(x: np.ndarray, win: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Return boolean mask (True = keep) using Hampel filter on |x|.
    NaNs/Infs are marked False.
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return np.array([], dtype=bool)

    mag = np.abs(x.astype(np.complex128))
    keep = np.isfinite(mag)
    z = mag.copy()
    z[~keep] = np.nan

    half = win // 2
    k = 1.4826  # scale factor
    for i in range(n):
        if not keep[i]:
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        wv = z[lo:hi]
        med = np.nanmedian(wv)
        if np.isnan(med):
            keep[i] = False
            continue
        mad = np.nanmedian(np.abs(wv - med))
        sigma = k * mad
        if sigma > 0:
            if np.abs(mag[i] - med) > n_sigmas * sigma:
                keep[i] = False
        else:
            if np.abs(mag[i] - med) > 1e-6:
                keep[i] = False
    return keep


# ======================================
# Data loading and basic preprocessing
# ======================================
def _load_bode_dat(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(fp, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase


def _stack_bode(files: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w_l, m_l, p_l = [], [], []
    for f in files:
        w, m, p = _load_bode_dat(f)
        w_l.append(w); m_l.append(m); p_l.append(p)
    w = np.hstack(w_l); m = np.hstack(m_l); p = np.hstack(p_l)
    idx = np.argsort(w)
    return w[idx], m[idx], p[idx]


# ==================
# GP model variants
# ==================
def _predict_linear_interp(
    X_train: np.ndarray,
    G_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    """Baseline: linear interpolation on log10(omega).
    Returns complex predictions at X_eval.
    """
    f_r = interp1d(X_train, G_train.real, kind="linear", fill_value="extrapolate")
    f_i = interp1d(X_train, G_train.imag, kind="linear", fill_value="extrapolate")
    return f_r(X_eval) + 1j * f_i(X_eval)


def _predict_gpr(
    X_train: np.ndarray,
    G_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    """Sklearn GPR on real/imag parts independently.
    Keeps dependencies light and works well for smooth FRF.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train.reshape(-1, 1))
    Xs_eval = scaler.transform(X_eval.reshape(-1, 1))

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-6, 1e1))
    gpr_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=0)
    gpr_i = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=1)

    gpr_r.fit(Xs_train, G_train.real)
    gpr_i.fit(Xs_train, G_train.imag)

    r_pred = gpr_r.predict(Xs_eval)
    i_pred = gpr_i.predict(Xs_eval)
    return r_pred + 1j * i_pred


# =======================
# FIR from FRF utilities
# =======================
def _frf_to_impulse_response(
    omega: np.ndarray,
    G_pos: np.ndarray,
    energy_cut: float = 0.99,
) -> Tuple[np.ndarray, float]:
    """Construct real impulse response h via IFFT from one-sided FRF samples.

    Returns (h_init, Ts)
    """
    Npos = len(omega)
    w_min, w_max = float(np.min(omega)), float(np.max(omega))
    Nfft = 2 ** int(np.ceil(np.log2(4 * Npos)))
    w_uni = np.linspace(w_min, w_max, Nfft // 2 + 1)

    # PCHIP would be nicer; linear is acceptable for now
    G_uni_r = np.interp(w_uni, omega, G_pos.real)
    G_uni_i = np.interp(w_uni, omega, G_pos.imag)
    G_uni = G_uni_r + 1j * G_uni_i

    # Hermitian full spectrum: [neg freqs] + [pos freqs]
    G_full = np.concatenate([np.conj(G_uni[Nfft // 2 - 1:0:-1]), G_uni])

    g_full = np.real(ifft(ifftshift(G_full)))

    Dw = w_uni[1] - w_uni[0]
    Fs = Dw * Nfft / (2 * np.pi)
    Ts = 1.0 / Fs

    # Trim h by cumulative energy
    Etotal = float(np.sum(np.abs(g_full) ** 2))
    cumE = np.cumsum(np.abs(g_full) ** 2)
    idx = np.where(cumE / Etotal >= energy_cut)[0]
    L = int(idx[0]) if idx.size > 0 else len(g_full)
    L = max(L, 4)

    # Simple Hann window on the kept part
    w = np.hanning(L)
    h_init = g_full[:L] * w
    return h_init.astype(float), float(Ts)


# ======================
# Configuration dataclass
# ======================
@dataclass
class GPConfig:
    # Input
    data_dir: Path = Path("./gp/data")
    file_glob: str = "SKE2024_data*.dat"
    test_filenames: Optional[List[str]] = None  # if None, use ratio
    test_ratio: float = 0.2

    # GP method: 'gpr' | 'linear'
    method: str = "gpr"
    n_grid: int = 2000

    # Output
    out_dir: Path = Path("./gp/output")
    save_csv_path: Optional[Path] = None  # e.g., Path('fir/data/predicted_G_values.csv')
    figure_name: str = "gp_results.png"

    # FIR stage
    fir_io_mat: Optional[Path] = None  # e.g., Path('fir/data/data_hour.mat')
    energy_cut: float = 0.99
    lms_mu: float = 1e-3
    lms_partial_m: int = 10


def gp(config: GPConfig) -> Dict[str, object]:
    """Run the GP → CSV → Nyquist stage, then FIR evaluation if requested.

    Returns a dict with paths and metrics.
    """
    out = {}
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load train/test Bode data ----------
    all_files = sorted(config.data_dir.glob(config.file_glob))
    if not all_files:
        raise FileNotFoundError(f"No .dat files found in {config.data_dir} with pattern {config.file_glob}")

    if config.test_filenames:
        test_set = set(config.test_filenames)
        train_files = [f for f in all_files if f.name not in test_set]
        test_files = [f for f in all_files if f.name in test_set]
    else:
        # simple split by ratio
        k = len(all_files)
        n_test = max(1, int(round(k * config.test_ratio)))
        test_files = all_files[:n_test]
        train_files = all_files[n_test:]

    if not train_files or not test_files:
        raise RuntimeError("Train/Test split failed. Please check files or ratio.")

    w_tr, mag_tr, ph_tr = _stack_bode(train_files)
    w_te, mag_te, ph_te = _stack_bode(test_files)

    G_tr = mag_tr * np.exp(1j * ph_tr)
    G_te = mag_te * np.exp(1j * ph_te)

    X_tr = np.log10(w_tr)
    X_te = np.log10(w_te)

    # ---------- Choose model ----------
    if config.method.lower() == "linear":
        predict_fn = _predict_linear_interp
    else:
        predict_fn = _predict_gpr

    # grid for smooth plot and CSV
    w_min = float(min(w_tr.min(), w_te.min()))
    w_max = float(max(w_tr.max(), w_te.max()))
    w_grid = np.logspace(math.log10(w_min), math.log10(w_max), config.n_grid)
    X_grid = np.log10(w_grid)

    # fit/predict
    G_grid = predict_fn(X_tr, G_tr, X_grid)
    G_tr_pred = predict_fn(X_tr, G_tr, X_tr)
    G_te_pred = predict_fn(X_tr, G_tr, X_te)

    # ---------- Metrics (Hampel on |G|) ----------
    keep_tr = _hampel_keep_mask(G_tr)
    keep_te = _hampel_keep_mask(G_te)
    rmse_tr = float(np.sqrt(np.mean(np.abs(G_tr[keep_tr] - G_tr_pred[keep_tr]) ** 2)))
    rmse_te = float(np.sqrt(np.mean(np.abs(G_te[keep_te] - G_te_pred[keep_te]) ** 2)))

    # ---------- Save CSV ----------
    gp_csv = out_dir / "gp_predictions.csv"
    csv_arr = np.column_stack([w_grid, G_grid.real, G_grid.imag])
    np.savetxt(gp_csv, csv_arr, delimiter=",", header="omega,Re_G,Im_G", comments="", fmt="% .8e")
    out["gp_csv"] = str(gp_csv)

    # also save to fir/data/predicted_G_values.csv for compatibility
    if config.save_csv_path is not None:
        config.save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(config.save_csv_path, csv_arr, delimiter=",", header="omega,Re_G,Im_G", comments="", fmt="% .8e")
        out["fir_predicted_csv"] = str(config.save_csv_path)

    # ---------- Nyquist plot ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    ax1.plot(G_tr.real, G_tr.imag, "b.", label="Train data", markersize=3)
    ax1.plot(G_grid.real, G_grid.imag, "r-", label=f"{config.method.upper()} pred", linewidth=2)
    ax1.set_title(f"Nyquist - Train (RMSE={rmse_tr:.3e})")
    ax1.set_xlabel("Re")
    ax1.set_ylabel("Im")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend()

    ax2.plot(G_te.real, G_te.imag, "g.", label="Test data", markersize=3)
    ax2.plot(G_grid.real, G_grid.imag, "r-", label=f"{config.method.upper()} pred", linewidth=2)
    ax2.set_title(f"Nyquist - Test (RMSE={rmse_te:.3e})")
    ax2.set_xlabel("Re")
    ax2.set_ylabel("Im")
    ax2.grid(True)
    ax2.axis("equal")
    ax2.legend()

    fig.tight_layout()
    fig_path = out_dir / config.figure_name
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    out["gp_figure"] = str(fig_path)

    # ---------- Save metrics ----------
    metrics = {"rmse_train": rmse_tr, "rmse_test": rmse_te, "method": config.method}
    with open(out_dir / "gp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    out["gp_metrics_json"] = str(out_dir / "gp_metrics.json")

    # ---------- Optional FIR evaluation ----------
    fir_out = None
    if config.energy_cut is not None:
        # Build FIR from FRF and evaluate
        h_init, Ts = _frf_to_impulse_response(w_grid, G_grid, energy_cut=config.energy_cut)
        fir_out = _run_fir_eval(
            h_init=h_init,
            Ts=Ts,
            io_mat=config.fir_io_mat,
            out_dir=out_dir,
            lms_mu=config.lms_mu,
            lms_partial_m=config.lms_partial_m,
        )
        out.update({f"fir_{k}": v for k, v in fir_out.items()})

    out.update(metrics)
    return out


def _load_io_mat(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = loadmat(mat_path)
    var_names = [k for k in d.keys() if not k.startswith("__")]
    if not var_names:
        raise ValueError("No variables found in MAT file")
    arr = d[var_names[0]]
    if arr.ndim == 2 and arr.shape[0] >= 3:
        t = arr[0, :].ravel()
        y = arr[1, :].ravel()
        u = arr[2, :].ravel()
        return t, u, y
    if arr.ndim == 2 and arr.shape[1] >= 3:
        t = arr[:, 0].ravel()
        y = arr[:, 1].ravel()
        u = arr[:, 2].ravel()
        return t, u, y
    raise ValueError("Unexpected MAT array shape; expected [3,N] or [N,3]")


def _run_fir_eval(
    h_init: np.ndarray,
    Ts: float,
    io_mat: Optional[Path],
    out_dir: Path,
    lms_mu: float = 1e-3,
    lms_partial_m: int = 10,
) -> Dict[str, object]:
    """Evaluate a FIR model from an initial impulse response.

    If io_mat is None or missing, synthesize a test I/O dataset to validate
    the end-to-end flow.
    """
    out: Dict[str, object] = {}

    if io_mat is not None and io_mat.exists():
        t, u, y = _load_io_mat(io_mat)
        # resample not implemented; assume sampling close enough for demo
    else:
        # Synthesize I/O using the provided h_init
        N = 5000
        rng = np.random.default_rng(0)
        u = rng.standard_normal(N).astype(float)
        y_clean = lfilter(h_init, [1.0], u)
        y = (y_clean + 0.02 * np.std(y_clean) * rng.standard_normal(N)).astype(float)
        t = np.arange(N) * Ts

    N = len(u)
    L = len(h_init)
    phi = np.zeros(L)
    h = h_init.copy()
    yhat = np.zeros(N)
    err = np.zeros(N)

    for n in range(N):
        # update regressor [u[n], ..., u[n-L+1]]
        phi = np.roll(phi, 1)
        phi[0] = u[n]
        # prediction
        yhat[n] = float(np.dot(phi, h))
        err[n] = y[n] - yhat[n]
        # partial-update LMS on top-M taps by gradient magnitude
        delta = lms_mu * phi * err[n]
        idx = np.argsort(np.abs(delta))[::-1][: min(lms_partial_m, L)]
        h[idx] += delta[idx]

    # Metrics after initial transient (ignore first L samples)
    e_tail = err[L:]
    y_tail = y[L:]
    rmse = float(np.sqrt(np.mean(e_tail**2)))
    nrmse = float(1.0 - np.linalg.norm(e_tail) / (np.linalg.norm(y_tail - np.mean(y_tail)) + 1e-12))
    r2 = float(1.0 - np.sum(e_tail**2) / (np.sum((y_tail - np.mean(y_tail))**2) + 1e-12))

    # Save plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(t, y, "k", label="Measured y")
    ax1.plot(t, yhat, "r--", label="Predicted yhat")
    ax1.set_title("Output vs Prediction")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("y")
    ax1.legend(); ax1.grid(True)

    ax2.plot(t, err, "b", label="error")
    ax2.set_title(f"Error (RMSE={rmse:.3e}, NRMSE={nrmse*100:.2f}%, R2={r2:.3f})")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("e")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fir_fig = out_dir / "fir_results.png"
    fig.savefig(fir_fig, dpi=300)
    plt.close(fig)

    # Save coefficients and summary
    coef_path = out_dir / "fir_coefficients.npz"
    np.savez(coef_path, h=h, h_init=h_init, Ts=Ts)

    summary = {"rmse": rmse, "nrmse": nrmse, "r2": r2, "L": int(L), "Ts": float(Ts)}
    with open(out_dir / "fir_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out.update({
        "results_png": str(fir_fig),
        "coeff_npz": str(coef_path),
        **summary,
    })
    return out


# ==============================
# Convenience wrapper for tests
# ==============================
def run_complete_pipeline(
    input_data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    method: str = "gpr",
) -> Dict[str, object]:
    """Convenience wrapper used by test_pipeline.py.

    - input_data_path: optional directory containing .dat files; defaults to gp/data
    - output_dir: directory to place outputs (png/csv/npz/json)
    - method: 'gpr' or 'linear'
    """
    if output_dir is None:
        output_dir = Path("./gp/output")
    else:
        output_dir = Path(output_dir)
    data_dir = Path(input_data_path) if input_data_path is not None else Path("./gp/data")

    cfg = GPConfig(
        data_dir=data_dir,
        method=method,
        out_dir=output_dir,
        save_csv_path=Path("./fir/data/predicted_G_values.csv"),
        # Use one known file as test if present
        test_filenames=["SKE2024_data18-Apr-2025_1205.dat"],
        # FIR stage: try to use provided MAT file if exists, else synthetic
        fir_io_mat=Path("./fir/data/data_hour.mat") if Path("./fir/data/data_hour.mat").exists() else None,
    )

    result = gp(cfg)
    return result

