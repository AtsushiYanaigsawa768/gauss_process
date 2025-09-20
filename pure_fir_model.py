"""
Kernel-regularized FIR identification (time-domain) with DC, SS(2), and SI kernels.

Predict the current output from input/output sequences by estimating a finite-length
impulse response g via Tikhonov/GP regularization. Hyperparameters are selected by
maximizing the GP marginal likelihood using low-rank identities (no N x N matrices).

Outputs: figures and metrics similar to gp_fir_pipeline.py
- fir_results.png: y vs yhat and error plot
- fir_coefficients.npz: stores g_hat, kernel matrix, and L
- fir_metrics.json: summary (RMSE, FIT, NRMSE, R2, L)

Usage (CLI):
  python pure_fir_model.py --io ./fir/data/data_hour.mat --kernel dc --out ./test_output

Notes:
- Indices use causal convention with g[0] multiplying u[n]. This is a 0-lag FIR form
  common in signal processing. The guide’s τ≥1 form corresponds to g shifted by 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize


# =====================
# Helper: I/O data load
# =====================
def _load_io_mat(io_mat: Path, nmax: Optional[int] = 100_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    io_data = loadmat(io_mat)
    mat = None
    for name, arr in io_data.items():
        if not name.startswith("__"):
            mat = arr
            break
    if mat is None:
        raise ValueError("MAT file does not contain a numeric array")
    if mat.ndim != 2 or (mat.shape[0] != 3 and mat.shape[1] != 3):
        raise ValueError("Expected a 3xN or Nx3 matrix with rows [time; y; u]")
    if mat.shape[0] == 3:
        t_all = mat[0, :].ravel()
        y_all = mat[1, :].ravel()
        u_all = mat[2, :].ravel()
    else:
        t_all = mat[:, 0].ravel()
        y_all = mat[:, 1].ravel()
        u_all = mat[:, 2].ravel()
    if nmax is not None and len(u_all) > nmax:
        t_all = t_all[:nmax]
        y_all = y_all[:nmax]
        u_all = u_all[:nmax]
    return t_all.astype(float), u_all.astype(float), y_all.astype(float)


# =============================
# Regressor matrix (Toeplitz U)
# =============================
def _build_regressor(u: np.ndarray, L: int) -> np.ndarray:
    """Build regressor U with rows phi[n] = [u[n], u[n-1], ..., u[n-L+1]].
    Pads with zeros for negative indices. Shape: (N, L).
    """
    N = len(u)
    u_pad = np.concatenate([np.zeros(L - 1, dtype=float), u.astype(float)])
    U = sliding_window_view(u_pad, window_shape=L)[:N, ::-1]
    return U


# ====================
# Kernel constructions
# ====================
def _kernel_dc(L: int, c: float, lam: float, rho: float) -> np.ndarray:
    """Diagonal/Correlated kernel: k(i,j) = c * lam^{(i+j)/2} * rho^{|i-j|}.
    0 < lam < 1, rho in [-1, 1]. Indices i,j = 0..L-1.
    """
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    K = (lam ** ((i + j) / 2.0)) * (np.power(np.abs(rho), np.abs(i - j)))
    # Preserve sign of rho for odd |i-j| if rho < 0
    if rho < 0:
        K *= ((-1.0) ** (np.abs(i - j)))
    return c * K


def _kernel_ss2(L: int, c: float, lam: float) -> np.ndarray:
    """Second-order stable spline via construction g = C H w.

    H: lower-triangular with H[i,j] = lam^{i-j} for i >= j
    C: cumulative sum (strictly lower + diag ones)
    K = c * (C H) (C H)^T
    """
    idx = np.arange(L)
    H = np.tril(lam ** (idx[:, None] - idx[None, :]))
    H[H == np.inf] = 0.0
    H[np.isnan(H)] = 0.0
    C = np.tril(np.ones((L, L)))
    CH = C @ H
    return c * (CH @ CH.T)


def _kernel_si(L: int, r: float, omega: float, q: float, barlam: float) -> np.ndarray:
    """Simulation-Induced kernel (DT, D=0).

    State model (2nd-order oscillatory nominal):
        z_{t+1} = A z_t + B delta(t) + B b(t) w(t),    z_0 ~ N(0, q I)
        g(t)    = C z_t
    with A = r * R(omega), R rotation matrix.
    b(t) = barlam^{t/2} (so b(t)^2 = barlam^t).
    B = [1, 0]^T, C = [1, 0].
    Returns K(t,s) = Cov(g(t), g(s)), for t,s = 0..L-1.
    """
    # Build A, B, C
    A = r * np.array([[math.cos(omega), -math.sin(omega)],
                      [math.sin(omega),  math.cos(omega)]], dtype=float)
    B = np.array([[1.0], [0.0]], dtype=float)
    C = np.array([[1.0, 0.0]], dtype=float)

    # Precompute powers of A, and sequences
    A_pows: List[np.ndarray] = [np.eye(2)]
    for _ in range(1, L + 1):
        A_pows.append(A_pows[-1] @ A)

    # g_series[n] = C A^n B  (n >= 0)
    g_series = np.zeros(L, dtype=float)
    for n in range(L):
        g_series[n] = float(C @ (A_pows[n] @ B))

    # v_series[n] = C A^n  (1x2)
    v_series = [C @ A_pows[n] for n in range(L + 1)]

    # K initialization
    K = np.zeros((L, L), dtype=float)

    # z0 term: C A^t Q (A^s)^T C^T with Q = q I
    for t in range(L):
        for s in range(L):
            K[t, s] += q * float(v_series[t] @ v_series[s].T)

    # process noise term: sum_{k=0}^{min(t,s)-1} b(k)^2 C A^{t-1-k} B B^T (A^{s-1-k})^T C^T
    # with b(k)^2 = barlam^k
    b2 = barlam ** np.arange(L)
    for t in range(L):
        for s in range(L):
            m = min(t, s)
            if m == 0:
                continue
            # sum over k = 0..m-1 of b2[k] * g_series[t-1-k] * g_series[s-1-k]
            acc = 0.0
            for k in range(m):
                acc += b2[k] * g_series[t - 1 - k] * g_series[s - 1 - k]
            K[t, s] += acc
    return K


# ==============================
# Hyperparameter transformations
# ==============================
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _build_kernel(L: int, kernel: str, theta: np.ndarray) -> np.ndarray:
    """Dispatch to build K for the selected kernel and (unconstrained) theta."""
    kname = kernel.lower()
    if kname == "dc":
        # theta = [log_c, unconstrained lam_logit, unconstrained rho_atanh]
        c = math.exp(theta[0])
        lam = _sigmoid(theta[1])  # (0,1)
        rho = math.tanh(theta[2])  # (-1,1)
        return _kernel_dc(L, c, lam, rho)
    elif kname == "ss":
        # theta = [log_c, lam_logit]
        c = math.exp(theta[0])
        lam = _sigmoid(theta[1])
        lam = min(max(lam, 1e-6), 1 - 1e-6)
        return _kernel_ss2(L, c, lam)
    elif kname == "si":
        # theta = [r_logit, omega_raw, log_q, barlam_logit]
        r = _sigmoid(theta[0])  # (0,1)
        r = min(max(r, 1e-4), 1 - 1e-6)
        # map omega_raw -> (0, pi)
        omega = math.pi * _sigmoid(theta[1])
        omega = min(max(omega, 1e-3), math.pi - 1e-3)
        q = math.exp(theta[2])
        barlam = _sigmoid(theta[3])
        barlam = min(max(barlam, 1e-6), 1 - 1e-6)
        return _kernel_si(L, r, omega, q, barlam)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


# =========================================
# Marginal likelihood via low-rank identities
# =========================================
def _nll(theta_all: np.ndarray,
         G: np.ndarray,
         b: np.ndarray,
         yy: float,
         N: int,
         L: int,
         kernel: str,
         jitter: float = 1e-10) -> float:
    """Negative log marginal likelihood (up to constant)."""
    # split theta into kernel params and log_sigma2
    if kernel.lower() == "dc":
        kdim = 3
    elif kernel.lower() == "ss":
        kdim = 2
    elif kernel.lower() == "si":
        kdim = 4
    else:
        raise ValueError("kernel")
    theta_k = theta_all[:kdim]
    log_sig2 = theta_all[kdim]
    sig2 = math.exp(log_sig2)

    # Build K and its Cholesky (K = Lk Lk^T)
    K = _build_kernel(L, kernel, theta_k)
    Lk = cholesky(K + jitter * np.eye(L), lower=True, check_finite=False)

    # Symmetric S: I + (1/sig2) Lk^T G Lk
    Ssym = np.eye(L) + (Lk.T @ (G @ Lk)) / max(sig2, 1e-30)
    try:
        Ls = cholesky(Ssym + jitter * np.eye(L), lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        Ls = cholesky(Ssym + 1e-6 * np.eye(L), lower=True, check_finite=False)

    # w = Lk^T b ; solve Ssym x = w
    w = Lk.T @ b
    x = cho_solve((Ls, True), w, check_finite=False)

    # y^T C^{-1} y = (1/sig2) * (yy - (1/sig2) w^T x)
    quad = (yy - (w @ x) / max(sig2, 1e-30)) / max(sig2, 1e-30)

    # log det C = N log(sig2) + log det Ssym
    logdetS = 2.0 * np.sum(np.log(np.diag(Ls)))
    nll = 0.5 * (quad + N * math.log(max(sig2, 1e-30)) + logdetS)
    return float(nll)


def _posterior_mean_g(theta_all: np.ndarray,
                      G: np.ndarray,
                      b: np.ndarray,
                      L: int,
                      kernel: str) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return (g_hat, sigma2, K) for given params."""
    if kernel.lower() == "dc":
        kdim = 3
    elif kernel.lower() == "ss":
        kdim = 2
    elif kernel.lower() == "si":
        kdim = 4
    else:
        raise ValueError("kernel")
    theta_k = theta_all[:kdim]
    sig2 = math.exp(theta_all[kdim])
    K = _build_kernel(L, kernel, theta_k)
    Lk = cholesky(K + 1e-10 * np.eye(L), lower=True, check_finite=False)
    # Ssym = I + (1/sig2) Lk^T G Lk
    Ssym = np.eye(L) + (Lk.T @ (G @ Lk)) / max(sig2, 1e-30)
    Ls = cholesky(Ssym + 1e-10 * np.eye(L), lower=True, check_finite=False)
    # g_hat = (1/sig2) * Lk * Ssym^{-1} * (Lk^T b)
    w = Lk.T @ b
    x = cho_solve((Ls, True), w, check_finite=False)
    g_hat = (Lk @ x) / max(sig2, 1e-30)
    return g_hat.reshape(-1), sig2, K


# ======================
# Public API / Experiment
# ======================
@dataclass
class FIRConfig:
    io_mat: Path = Path("./fir/data/data_hour.mat")
    out_dir: Path = Path("./test_output")
    kernel: str = "dc"  # 'dc' | 'ss' | 'si'
    L: int = 128        # FIR length (effective memory)
    multi_starts: int = 3
    optimize: bool = True
    maxiter: int = 200
    demean: bool = True


def run_fir_experiment(cfg: FIRConfig) -> Dict[str, object]:
    """Run kernel-regularized FIR identification and export figures + metrics."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Load I/O (fallback to synthetic if file missing)
    if Path(cfg.io_mat).exists():
        t, u_raw, y_raw = _load_io_mat(cfg.io_mat)
    else:
        print("[Warning] I/O MAT file not found, using synthetic data")
        # Synthetic dataset for end-to-end validation
        N = 5000
        Ts = 1.0
        t = np.arange(N, dtype=float) * Ts
        rng = np.random.default_rng(0)
        u_raw = rng.standard_normal(N).astype(float)
        # True sparse/decaying impulse
        L0 = min(cfg.L, 64)
        g0 = np.exp(-0.05 * np.arange(L0)) * (np.sin(0.2 * np.arange(L0)))
        y_clean = np.convolve(u_raw, g0, mode="full")[:N]
        y_raw = (y_clean + 0.02 * np.std(y_clean) * rng.standard_normal(N)).astype(float)
    N = len(u_raw)
    if cfg.demean:
        u_mean, y_mean = float(np.mean(u_raw)), float(np.mean(y_raw))
        u = u_raw - u_mean
        y = y_raw - y_mean
    else:
        u, y = u_raw.copy(), y_raw.copy()

    # Build regressor summaries (avoid N x N)
    U = _build_regressor(u, cfg.L)
    G = U.T @ U  # L x L
    b = U.T @ y  # L
    yy = float(y @ y)

    # Initialize theta per kernel
    kname = cfg.kernel.lower()
    if kname == "dc":
        theta0 = np.array([math.log(1.0),  # log_c
                           math.log(0.98 / (1 - 0.98)),  # lam_logit
                           np.arctanh(0.8)])             # rho_atanh
    elif kname == "ss":
        theta0 = np.array([math.log(1.0),  # log_c
                           math.log(0.96 / (1 - 0.96))])  # lam_logit
    elif kname == "si":
        theta0 = np.array([math.log(0.98 / (1 - 0.98)),  # r_logit
                           math.log(0.25),               # omega_raw ~ sigmoid^-1(0.25*pi)
                           math.log(1e-2),               # log_q
                           math.log(0.96 / (1 - 0.96))]) # barlam_logit
    else:
        raise ValueError("Unknown kernel")
    # Add noise param
    theta0 = np.concatenate([theta0, np.array([math.log(1e-2)])])

    # Multi-starts around theta0
    best = (float("inf"), theta0.copy())
    rng = np.random.default_rng(0)
    n_starts = max(1, int(cfg.multi_starts))
    for s in range(n_starts):
        if s == 0:
            th_init = theta0.copy()
        else:
            th_init = theta0 + 0.2 * rng.standard_normal(theta0.shape)

        res = minimize(
            lambda th: _nll(th, G, b, yy, N, cfg.L, cfg.kernel),
            th_init,
            method="L-BFGS-B",
            options={"maxiter": cfg.maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxcor": 20},
        )
        if res.fun < best[0]:
            best = (res.fun, res.x.copy())

    theta_hat = best[1]

    # Posterior mean g
    g_hat, sig2_hat, K_hat = _posterior_mean_g(theta_hat, G, b, cfg.L, cfg.kernel)

    # Predict
    yhat = U @ g_hat
    if cfg.demean:
        yhat = yhat + y_mean

    # Metrics (ignore initial L-1 samples)
    e = (y_raw - yhat)
    e_tail = e[cfg.L - 1:]
    y_tail = y_raw[cfg.L - 1:]
    rmse = float(np.sqrt(np.mean(e_tail ** 2)))
    nrmse = float(1.0 - np.linalg.norm(e_tail) / (np.linalg.norm(y_tail - np.mean(y_tail)) + 1e-12))
    r2 = float(1.0 - np.sum(e_tail ** 2) / (np.sum((y_tail - np.mean(y_tail)) ** 2) + 1e-12))
    fit = float(100.0 * (1.0 - np.linalg.norm(e_tail) / (np.linalg.norm(y_tail - np.mean(y_tail)) + 1e-12)))

    # Figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(t, y_raw, label="Measured y", color="k")
    ax1.plot(t, yhat, label="Predicted yhat", color="r", linestyle="--")
    ax1.set_title(f"Output vs Prediction ({cfg.kernel.upper()} kernel)")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("y")
    ax1.legend(); ax1.grid(True)

    ax2.plot(t, e, label="error", color="b")
    ax2.set_title(f"Error  RMSE={rmse:.3e}, FIT={fit:.2f}%, R2={r2:.3f}")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("e")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig_path = cfg.out_dir / "fir_results.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # Impulse response plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.stem(np.arange(cfg.L), g_hat, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax.set_title(f"Estimated impulse response g (L={cfg.L})")
    ax.set_xlabel("lag k")
    ax.set_ylabel("g[k]")
    ax.grid(True)
    fig2.tight_layout()
    fig2_path = cfg.out_dir / "fir_impulse.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # Save artifacts
    np.savez(cfg.out_dir / "fir_coefficients.npz", g_hat=g_hat, K=K_hat, kernel=cfg.kernel, L=cfg.L)
    metrics = {
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
        "fit_percent": fit,
        "L": int(cfg.L),
        "sigma2": float(sig2_hat),
        "kernel": cfg.kernel,
    }
    with open(cfg.out_dir / "fir_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "results_png": str(fig_path),
        "impulse_png": str(fig2_path),
        "coeff_npz": str(cfg.out_dir / "fir_coefficients.npz"),
        **metrics,
    }


def _parse_argv() -> Optional[FIRConfig]:
    import argparse
    p = argparse.ArgumentParser(description="Kernel-regularized FIR identification")
    p.add_argument("--io", type=str, default="./fir/data/data_hour.mat", help="MAT file with [time;y;u]")
    p.add_argument("--out", type=str, default="./test_output_test", help="Output directory")
    p.add_argument("--kernel", type=str, default="dc", choices=["dc", "ss", "si"], help="Kernel type")
    p.add_argument("--L", type=int, default=128, help="FIR length (memory)")
    p.add_argument("--starts", type=int, default=3, help="Number of multi-starts for ML")
    p.add_argument("--noopt", action="store_true", help="Disable hyperparameter optimization")
    args = p.parse_args()
    return FIRConfig(
        io_mat=Path(args.io),
        out_dir=Path(args.out),
        kernel=args.kernel,
        L=int(args.L),
        multi_starts=int(args.starts),
        optimize=(not args.noopt),
    )


if __name__ == "__main__":
    cfg = _parse_argv()
    res = run_fir_experiment(cfg)
    print(json.dumps(res, indent=2))
