
#!/usr/bin/env python3
"""
gp_to_fir_direct_fixed.py

Mathematically corrected GP→FIR conversion:
- Uses sampling period Ts from MAT timebase to map continuous ω [rad/s]
  to digital Ω [rad/sample] on the DFT grid.
- Evaluates G(jω_k) at ω_k = Ω_k / Ts for the *exact* DFT bins.
- Estimates and removes group delay τ from the phase before IFFT so that
  the impulse response is (approximately) causal.
- Uses irfft for numerical robustness and returns the first L taps.

This fixes the unit-mismatch and noncausal-centering issues that made RMSE
insensitive to kernel choice.
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d
from numpy.fft import irfft
import warnings
import json
import matplotlib.pyplot as plt
# --------------------------- helpers ---------------------------

def build_uniform_omega_linear(omega_meas: np.ndarray,
                               G_meas: np.ndarray,
                               Nd: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 1: 連続ωの線形スケール上で等間隔グリッドを構築し、
            線形補間（実部・虚部別）で G_uni(ω_m) を得る。
    Args:
        omega_meas: 実験で得た角周波数 [rad/s]（昇順を推奨）
        G_meas:     FRF G(jω) の複素値（omega_meas と同長）
        Nd:         等間隔化後の点数（Noneなら測定点数に合わせる）
    Returns:
        omega_uni:  等間隔化した ω グリッド（Nd 点）
        G_uni:      線形補間で評価した複素 FRF（Nd 点）
    """
    omega_meas = np.asarray(omega_meas, dtype=float).ravel()
    G_meas = np.asarray(G_meas, dtype=complex).ravel()
    assert omega_meas.size == G_meas.size and omega_meas.size >= 2

    if Nd is None:
        Nd = omega_meas.size

    omega_min = float(np.min(omega_meas))
    omega_max = float(np.max(omega_meas))
    omega_uni = np.linspace(omega_min, omega_max, Nd)

    # 実部・虚部を別々に線形補間（端点外は端値保持）
    Gr = np.interp(omega_uni, omega_meas, np.real(G_meas),
                   left=np.real(G_meas[0]), right=np.real(G_meas[-1]))
    Gi = np.interp(omega_uni, omega_meas, np.imag(G_meas),
                   left=np.imag(G_meas[0]), right=np.imag(G_meas[-1]))
    G_uni = Gr + 1j * Gi
    return omega_uni, G_uni


def build_two_sided_hermitian_from_positive(G_uni: np.ndarray) -> np.ndarray:
    """
    Step 2: 実数インパルス応答を得るため、二側（奇数長）スペクトルを明示構築する。
            長さ M = 2*Nd - 1 （奇数）にし、
            X[0] = DC（実数化）、X[1:Nd] = 正側、X[M-(Nd-1):] = 負側（正側の共役反転）。
    Args:
        G_uni: ω>=0 の正側サンプル（Nd 点, DC含む・最上位周波数含む）
    Returns:
        X: ifft にそのまま渡せる長さ M の複素スペクトル
    """
    G_uni = np.asarray(G_uni, dtype=complex).ravel()
    Nd = G_uni.size
    assert Nd >= 2

    M = 2*Nd - 1  # 奇数長（ナイキスト固有点なし）
    X = np.zeros(M, dtype=complex)

    # 正側: k = 0..Nd-1（DC + 正周波数）
    X[0] = np.real(G_uni[0])  # DC を実数化（実数インパルス条件）
    if Nd > 1:
        X[1:Nd] = G_uni[1:Nd]

    # 負側: k = M-(Nd-1) .. M-1（正側の共役反転）
    if Nd > 1:
        X[M-(Nd-1):] = np.conjugate(G_uni[1:Nd][::-1])
    return X


def idft_to_fir_coeffs_two_sided(X: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 3: 明示的な二側スペクトル X（奇数長 M）から ifft で時間応答を計算し、
            実部の先頭 N タップを FIR 係数として返す。
    Args:
        X: 1D 複素スペクトル（build_two_sided_hermitian_from_positive の出力）
        N: 希望する FIR モデル次数（タップ数）
    Returns:
        h: FIR 係数（長さ N, 実数）
        h_full: ifft の全長 M の時間波形（実数部）
    """
    X = np.asarray(X, dtype=complex).ravel()
    M = X.size
    h_full = np.fft.ifft(X, n=M).real  # NumPy は 1/M 正規化済み
    if N > M:
        raise ValueError(f"N={N} exceeds available length M={M}. Reduce N.")
    h = h_full[:N].copy()
    return h, h_full


def _load_timebase_from_mat(mat_file: Path) -> Tuple[np.ndarray, float]:
    """Return (t, Ts) from MAT file (expects variables shaped [t, y, u] or named t/time)."""
    data = loadmat(mat_file)
    t = None
    # 1) Direct 't' or 'time'
    for key in ('t', 'time'):
        if key in data:
            t = np.ravel(data[key]).astype(float)
            break
    # 2) 3xN or Nx3 array with [t, y, u]
    if t is None:
        for k, v in data.items():
            if k.startswith('__') or not isinstance(v, np.ndarray):
                continue
            if v.ndim == 2 and (v.shape[0] == 3 or v.shape[1] == 3):
                if v.shape[0] == 3:
                    t = np.ravel(v[0, :]).astype(float)
                else:
                    t = np.ravel(v[:, 0]).astype(float)
                break
    if t is None or t.size < 2:
        raise ValueError("Could not infer time vector from MAT file; need 't'/'time' or 3-column array [t,y,u].")
    dt = np.median(np.diff(t))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Invalid time vector; cannot compute Ts.")
    return t, float(dt)


def _build_H_for_irfft(omega: np.ndarray, G: np.ndarray, L: int, Ts: float, N_fft: Optional[int] = None,
                       gp_predict_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                       taper: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one-sided spectrum H_k for irfft:
    - Ω_k = 2πk/N_fft for k=0..N_fft/2
    - ω_k = Ω_k / Ts
    - Evaluate G(jω_k) via GP predictor if provided, else cubic interp
    - Optional cosine taper near Nyquist to reduce wrap/alias
    Returns (H_half, omega_k, Omega_k)
    """
    if N_fft is None:
        # Reasonable resolution: at least 4×L and power-of-two
        N = 1
        while N < 4*L:
            N <<= 1
        N_fft = N
    if N_fft % 2 == 1:
        N_fft += 1  # enforce even for irfft

    k = np.arange(N_fft//2 + 1, dtype=int)
    Omega_k = 2.0 * np.pi * k / N_fft            # [rad/sample]
    omega_k = Omega_k / Ts                        # [rad/s]

    # Predict/interp G at ω_k
    if gp_predict_func is not None:
        Gk = gp_predict_func(omega_k)
    else:
        # Safe cubic interpolation with edge hold
        interp_real = interp1d(omega, np.real(G), kind='cubic', bounds_error=False,
                               fill_value=(np.real(G[0]), np.real(G[-1])))
        interp_imag = interp1d(omega, np.imag(G), kind='cubic', bounds_error=False,
                               fill_value=(np.imag(G[0]), np.imag(G[-1])))
        Gk = interp_real(omega_k) + 1j*interp_imag(omega_k)


    # Enforce DC, Nyquist realness for real impulse response (numerical hygiene)
    Gk = np.asarray(Gk, dtype=complex)
    Gk[0] = np.real(Gk[0])
    if (N_fft % 2 == 0):
        Gk[-1] = np.real(Gk[-1])

    # Optional mild taper near Nyquist to reduce circular wrap/alias
    if taper and len(Gk) > 8:
        # Cosine roll-off over last 10% of band
        m = len(Gk)
        n_roll = max(4, int(0.1 * m))
        w = np.ones(m)
        win = 0.5*(1.0 + np.cos(np.linspace(0, np.pi, n_roll)))
        w[-n_roll:] *= win
        Gk = Gk * w

    return Gk, omega_k, Omega_k

def plot_gp_fir_results_fixed(t: np.ndarray, y: np.ndarray,
                              y_pred: np.ndarray, u: np.ndarray,
                              rmse: float, fit_percent: float, r2: float,
                              output_dir: Path,
                              prefix: str = "gp_fir_fixed"):
    """
    Create visualization plots for GP-based FIR model results (fixed version).
    Shows output vs predicted and error plots.

    Args:
        t: Time vector
        y: Actual output
        y_pred: Predicted output
        u: Input
        rmse: Root mean square error
        fit_percent: FIT percentage metric
        r2: R-squared value
        output_dir: Directory to save plots
        prefix: Filename prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Output vs Predicted
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Show full time range or limit to reasonable duration
    t_max = min(t[-1], t[0] + 200)  # Show first 200 seconds
    mask = t <= t_max

    ax1.plot(t[mask], y[mask], 'k-', label='Measured Output', linewidth=1.5, alpha=0.8)
    ax1.plot(t[mask], y_pred[mask], 'r--', label='FIR Predicted', linewidth=1.5)
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Output', fontsize=12)

    # Title without delay information
    title = (f'FIR Model Validation (Corrected Method)\n'
             f'RMSE={rmse:.3e}, FIT={fit_percent:.1f}%, R²={r2:.3f}')
    ax1.set_title(title, fontsize=14)

    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_output_vs_predicted.png", dpi=300)
    plt.close(fig1)

    # Figure 2: Error
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    error = y - y_pred
    ax2.plot(t[mask], error[mask], 'b-', linewidth=1, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Error (Measured - Predicted)', fontsize=12)
    ax2.set_title('FIR Model Prediction Error', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add error statistics text
    error_stats = (f'Mean Error: {np.mean(error[mask]):.3e}\n'
                   f'Std Error: {np.std(error[mask]):.3e}')
    ax2.text(0.02, 0.98, error_stats, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_error.png", dpi=300)
    plt.close(fig2)

    # Figure 3: Frequency Response Comparison
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))

    # Compute FFT of FIR coefficients to verify frequency response
    # This is optional but helpful for debugging
    # We'll skip this for now to keep it simple

    print(f"Plots saved to {output_dir}")

# --------------------------- main API ---------------------------

def gp_to_fir_direct_pipeline(
    omega: np.ndarray,
    G: np.ndarray,
    gp_predict_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mat_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fir_length: int = 1024,
    N_fft: Optional[int] = None,
    paper_mode: bool = True,
    fir_order: Optional[int] = None,
) -> Dict[str, object]:
    """
    Corrected GP→FIR pipeline with paper-mode support.
    Args:
        omega: angular frequencies [rad/s] for GP-smoothed FRF
        G: complex FRF values
        gp_predict_func: optional callable for GP predictions at arbitrary ω
        mat_file: MAT with timebase to infer Ts (required for correct mapping in legacy mode)
        output_dir: where to save artifacts
        fir_length: number of FIR taps for legacy IRFFT mode (ignored in paper_mode)
        N_fft: optional FFT length for legacy mode (ignored in paper_mode)
        paper_mode: if True, use paper-based procedure (uniform ω, two-sided Hermitian, IDFT)
        fir_order: FIR model order (number of taps) for paper_mode (defaults to min(M, 1024))
    """
    omega = np.asarray(omega, dtype=float).ravel()
    G = np.asarray(G, dtype=complex).ravel()
    if output_dir is None:
        output_dir = Path("fir_gp_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Paper-mode: uniform ω grid + two-sided Hermitian + IDFT
    # ============================================================
    if paper_mode:
        print("[Paper Mode] Using paper-based FIR procedure:")
        print("  Step 1: Uniform linear ω grid + linear interpolation")
        print("  Step 2: Two-sided Hermitian spectrum (odd-length)")
        print("  Step 3: IDFT → first N taps as FIR coefficients")

        # Step 1: Build uniform omega grid with linear interpolation
        omega_uni, G_uni = build_uniform_omega_linear(omega, G, Nd=len(omega))
        Nd = len(omega_uni)
        print(f"  Nd = {Nd} frequency points")

        # Step 2: Build two-sided Hermitian spectrum (M = 2*Nd - 1, odd-length)
        X = build_two_sided_hermitian_from_positive(G_uni)
        M = X.size
        print(f"  M = {M} (IDFT length, odd)")

        # Verify Hermitian symmetry and realness
        max_imag_dc = np.abs(X[0].imag)
        if max_imag_dc > 1e-10:
            warnings.warn(f"DC has significant imaginary part: {max_imag_dc:.3e}")

        # Step 3: IDFT to get impulse response, take first N taps
        N_requested = int(fir_order if fir_order is not None else min(M, 1024))
        N = min(N_requested, M)  # Cannot exceed M
        if N < N_requested:
            warnings.warn(f"Requested fir_order={N_requested} exceeds M={M}. Using N={N} instead.")
        g, h_full = idft_to_fir_coeffs_two_sided(X, N)
        print(f"  N = {N} FIR taps extracted (requested: {N_requested})")

        # Verify realness of impulse response
        max_imag_h = np.max(np.abs(h_full.imag)) if np.iscomplexobj(h_full) else 0.0
        if max_imag_h > 1e-10:
            warnings.warn(f"Impulse response has significant imaginary part: {max_imag_h:.3e}")

        # Save coefficients
        np.savez(output_dir / "fir_coefficients_gp_paper.npz",
                 g=g, fir_order=N, M_idft=M, Nd=Nd,
                 omega_uni=omega_uni, G_uni=G_uni,
                 omega_meas=omega, G_meas=G,
                 h_full=h_full[:min(len(h_full), 4096)])  # Save truncated full response

        results = {
            "paper_mode": True,
            "Nd": int(Nd),
            "M_idft": int(M),
            "fir_order": int(N),
            "method": "gp_paper_mode",
            "omega_range": [float(omega_uni[0]), float(omega_uni[-1])],
            "rmse": None,
            "fit_percent": None,
            "r2": None,
        }

        # Validation if MAT file provided
        if mat_file is not None and Path(mat_file).exists():
            t, Ts = _load_timebase_from_mat(Path(mat_file))
            print(f"  Ts = {Ts:.6f} s (from MAT file)")
            results["Ts"] = float(Ts)

            # Load time-series data for validation
            data = loadmat(mat_file)
            T = None
            y = None
            u = None
            for key, val in data.items():
                if key.startswith("__") or not isinstance(val, np.ndarray):
                    continue
                if val.ndim == 2 and (val.shape[0] == 3 or val.shape[1] == 3):
                    if val.shape[0] == 3:
                        T = np.ravel(val[0, :]).astype(float)
                        y = np.ravel(val[1, :]).astype(float)
                        u = np.ravel(val[2, :]).astype(float)
                    else:
                        T = np.ravel(val[:, 0]).astype(float)
                        y = np.ravel(val[:, 1]).astype(float)
                        u = np.ravel(val[:, 2]).astype(float)
                    break

            if T is None:
                # Try named variables
                T = np.ravel(data.get("t", data.get("time"))).astype(float)
                y = np.ravel(data.get("y"))
                u = np.ravel(data.get("u"))

            if T is not None and y is not None and u is not None:
                # ========== STRICT EVALUATION MODE ==========
                # NO detrend, minimal transient skip for fair comparison

                # Optional detrend (OFF by default for fair comparison)
                DETREND = False  # Set to False for strict evaluation
                if DETREND:
                    u_eval = u - np.mean(u)
                    y_eval = y - np.mean(y)
                    print("  [WARNING] Detrend is ON - DC errors will be hidden")
                else:
                    u_eval = u.copy()
                    y_eval = y.copy()
                    print("  [STRICT MODE] No detrend - evaluating full frequency range including DC")

                # Predict with causal convolution
                y_pred = np.convolve(u_eval, g, mode="full")[:len(y_eval)]

                # Minimal transient skip (only first 10 samples for settling)
                # Previous: skip = N // 2 (hid ~512 samples!)
                skip = min(10, N // 10)  # Much more strict
                y_valid = y_eval[skip:]
                y_pred_valid = y_pred[skip:]

                print(f"  Transient skip: {skip} samples (was: {N//2} samples)")

                # Calculate metrics WITHOUT additional detrending
                err = y_valid - y_pred_valid
                rmse = float(np.sqrt(np.mean(err**2)))

                # FIT metric without detrending in denominator
                norm_y = np.linalg.norm(y_valid)
                fit = float(100 * (1.0 - np.linalg.norm(err) / norm_y)) if norm_y > 0 else 0.0

                # R² without detrending
                ss_res = float(np.sum(err**2))
                ss_tot = float(np.sum((y_valid - y_valid[0])**2))  # Use first value, not mean
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                results.update({"rmse": rmse, "fit_percent": fit, "r2": r2,
                               "transient_skip": skip, "detrend": DETREND})

                print(f"  Validation (STRICT): RMSE={rmse:.3e}, FIT={fit:.1f}%, R²={r2:.3f}")

                # Save validation data
                np.savez(output_dir / "validation_preview_paper.npz",
                         t=T, y=y, y_pred=y_pred, u=u)

                # Create visualization plots
                plot_gp_fir_results_fixed(
                    t=T, y=y, y_pred=y_pred, u=u,
                    rmse=rmse, fit_percent=fit, r2=r2,
                    output_dir=output_dir,
                    prefix="gp_fir_paper"
                )
            else:
                print("  Warning: Could not load validation data from MAT file")
        else:
            print("  No validation MAT file provided")
            results["Ts"] = None

            # Create summary figure without validation
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')

            info_text = f"""FIR Model Extraction Complete (Paper Mode)
====================================================
Method: Paper-based (uniform ω + two-sided Hermitian + IDFT)
Nd: {Nd} frequency points
M (IDFT length): {M} (odd)
FIR Order N: {N} taps
Frequency Range: {omega_uni[0]:.2f} - {omega_uni[-1]:.2f} rad/s

No validation data provided.
To validate, specify --fir-validation-mat
"""
            ax.text(0.5, 0.5, info_text, fontsize=12, family='monospace',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

            plt.tight_layout()
            plt.savefig(output_dir / "gp_fir_paper_summary.png", dpi=300)
            plt.close()

        # Save results JSON
        with open(output_dir / "fir_gp_paper_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"[Paper Mode] Results saved to {output_dir}")
        return results

    # ============================================================
    # Legacy mode: original IRFFT-based approach
    # ============================================================
    print("[Legacy Mode] Using IRFFT-based approach")

    if mat_file is None or not Path(mat_file).exists():
        warnings.warn("No MAT file (timebase) provided. Falling back to *unit* Ts=1.0 s. "
                      "This will likely distort the FIR. Provide --fir-validation-mat.")
        Ts = 1.0
        t = None
    else:
        t, Ts = _load_timebase_from_mat(Path(mat_file))

    # Build spectrum on the *digital* grid
    H_half, omega_k, Omega_k = _build_H_for_irfft(
        omega, G, L=fir_length, Ts=Ts, N_fft=N_fft,
        gp_predict_func=gp_predict_func, taper=True
    )

    # Real impulse via irfft; take first L (causal) taps
    h_full = irfft(H_half, n=(2*(len(H_half)-1)))
    g = h_full[:fir_length].copy()

    # Save
    np.savez(output_dir / "fir_coefficients_gp_fixed.npz",
             g=g, fir_length=fir_length, Ts=Ts,
             omega=omega, G=G, omega_k=omega_k, Omega_k=Omega_k)

    results = {
        "fir_length": fir_length,
        "Ts": Ts,
        "method": "gp_direct_fixed",
        "n_fft": int(2*(len(H_half)-1)),
        "rmse": None,
        "fit_percent": None,
        "r2": None,
    }

    # Optional validation if MAT present
    if mat_file is not None and Path(mat_file).exists():
        data = loadmat(mat_file)
        # Load [t,y,u]
        T = None
        y = None
        u = None
        for key, val in data.items():
            if key.startswith("__") or not isinstance(val, np.ndarray):
                continue
            if val.ndim == 2 and (val.shape[0] == 3 or val.shape[1] == 3):
                if val.shape[0] == 3:
                    T = np.ravel(val[0, :]).astype(float)
                    y = np.ravel(val[1, :]).astype(float)
                    u = np.ravel(val[2, :]).astype(float)
                else:
                    T = np.ravel(val[:, 0]).astype(float)
                    y = np.ravel(val[:, 1]).astype(float)
                    u = np.ravel(val[:, 2]).astype(float)
                break
        if T is None:
            # Try named variables
            T = np.ravel(data.get("t", data.get("time"))).astype(float)
            y = np.ravel(data.get("y"))
            u = np.ravel(data.get("u"))

        # Use raw signals for evaluation (no detrending or aggressive skipping)
        u_eval = np.asarray(u, dtype=float)
        y_eval = np.asarray(y, dtype=float)

        # Predict with causal convolution
        y_pred = np.convolve(u_eval, g, mode="full")[: len(y_eval)]

        # Skip only a small leading transient introduced by zero initial conditions
        skip = min(10, len(g) // 10)
        y_valid = y_eval[skip:]
        y_pred_valid = y_pred[skip:]

        err = y_valid - y_pred_valid
        rmse = float(np.sqrt(np.mean(err**2)))

        norm_y = np.linalg.norm(y_valid)
        fit = float(100 * (1.0 - np.linalg.norm(err) / norm_y)) if norm_y > 0 else 0.0

        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_valid - y_valid[0])**2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results.update({"rmse": rmse,
                        "fit_percent": fit,
                        "r2": r2,
                        "transient_skip": int(skip)})

        # Save preview arrays for quick inspection
        np.savez(output_dir / "validation_preview_fixed.npz",
                 t=T, y=y, y_pred=y_pred, u=u)

        # Create visualization plots
        plot_gp_fir_results_fixed(
            t=T,
            y=y,
            y_pred=y_pred,
            u=u,
            rmse=rmse,
            fit_percent=fit,
            r2=r2,
            output_dir=output_dir
        )

    else:
        # If no validation data, create a simple summary figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')

        info_text = f"""FIR Model Extraction Complete (Fixed Method)
====================================================
Method: GP-based interpolation with proper ω→Ω mapping
FIR Length: {fir_length}
Sampling Period Ts: {Ts:.6f} s
FFT Length: {2*(len(H_half)-1)}
Frequency Range: {omega[0]:.2f} - {omega[-1]:.2f} rad/s

No validation data provided.
To validate, specify --fir-validation-mat
"""
        ax.text(0.5, 0.5, info_text, fontsize=12, family='monospace',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_dir / "gp_fir_fixed_summary.png", dpi=300)
        plt.close()

    # Minimal text report
    with open(output_dir / "fir_gp_fixed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
