#!/usr/bin/env python3
"""
frequency_response_matched.py

Goal
----
Make the Python FRF/Nyquist estimation *match* the MATLAB/Simulink pipeline ("program 1")
given raw time-series logs from "program 3" (External mode).

Key alignments implemented:
1) Frequency grid identical to MATLAB (log grid; excludes upper endpoint):
    logs = f_low + (f_up - f_low)/N_d * (0..N_d-1);  f = 10**logs;  ω = 2πf
2) Synchronous demodulation via *time-weighted* trapezoidal integration (handles non-uniform dt).
3) FRF aggregation across multiple records using the cross-power estimator:
    G(ω) = (Σ_k Y_k * conj(U_k)) / (Σ_k |U_k|^2)
4) MATLAB phase convention parity for saved `P = [ω; |G|; atan2(-ImG, -ReG)]`.

It also supports reading frequency grids from a previous MATLAB file if available
(keys: 'frequency','freq' [Hz], 'omega','w' [rad/s], or 'P' with ω in first row).

Expected inputs
---------------
- One or more .mat files that contain *either*:
  (A) `t`, `u`, `y` variables (as saved by your program 3). `y` may be 1D or 2D;
     use `--y-col` to select a specific column (default 0).
  (B) A 3xN or Nx3 numeric array named `output` or any top-level variable shaped like that,
     with rows/cols ordered [t, y, u] (time, output, input).

Usage
-----
python frequency_response_matched.py logs_io/*.mat \
  --nd 100 --f-low -1.0 --f-up 2.3 --dropseconds 30 \
  --out-prefix runA --out-dir output --n-files 10

python src\frequency_response.py input/*.mat --n-files 1

This writes into the 'output' directory:
  runA_bode_mag.png, runA_bode_phase.png, runA_nyquist.png,
  runA_frf.csv, runA_frf.mat, and (with --save-matlab) runA_frf.dat
"""

import argparse
import csv
import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat


# --------------------------- I/O helpers ---------------------------

def resolve_mat_files(targets: Sequence[str], recursive: bool = False) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for target in targets:
        p = Path(target)
        if p.exists():
            pool = p.rglob("*.mat") if (recursive and p.is_dir()) else ([p] if p.is_file() else [])
        else:
            pool = [Path(x) for x in glob.glob(target, recursive=recursive)]
        for item in pool:
            if item.is_file():
                rp = item.resolve()
                if rp not in seen:
                    seen.add(rp)
                    files.append(rp)
    return sorted(files)


def _ravel1d(x) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x).squeeze()
        arr = arr.astype(float).ravel()
        return arr if arr.size > 0 and np.isfinite(arr).all() else None
    except Exception:
        return None


def load_time_u_y(mat_path: Path, y_col: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract time, input, output vectors from a MAT file.

    Accepts either:
     - separate variables `t`, `u`, `y` (y may be 1D or 2D; choose column by --y-col);
     - a 3xN or Nx3 numeric array where rows/cols are [t, y, u] (time, output, input);
       prefer the variable named `output`, but will scan any top-level 2D numeric.
    Returns (t, u, y) as 1D float arrays.
    """
    data = loadmat(mat_path)

    # Path (A): explicit t/u/y
    t = _ravel1d(data.get("t"))
    u = _ravel1d(data.get("u"))
    y_raw = data.get("y")
    if t is not None and u is not None and y_raw is not None:
        y_arr = np.asarray(y_raw)
        if y_arr.ndim == 1:
            y = _ravel1d(y_arr)
        elif y_arr.ndim == 2:
            if y_arr.shape[1] == 1:
                y = _ravel1d(y_arr[:, 0])
            else:
                # select column
                if not (0 <= y_col < y_arr.shape[1]):
                    raise ValueError(f"--y-col {y_col} out of range for y with shape {y_arr.shape}")
                y = _ravel1d(y_arr[:, y_col])
        else:
            y = None
        if y is not None and len(t) > 1 and len(u) == len(t) and len(y) == len(t):
            return t, u, y

    # Path (B): 3xN or Nx3 array, prefer "output", else scan
    def _try_matrix(arr):
        arr = np.asarray(arr)
        if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
            return None
        if arr.shape[0] == 3:
            t, y, u = arr[0], arr[1], arr[2]
        elif arr.shape[1] == 3:
            t, y, u = arr[:, 0], arr[:, 1], arr[:, 2]
        else:
            return None
        return _ravel1d(t), _ravel1d(u), _ravel1d(y)

    if "output" in data:
        candidate = _try_matrix(data["output"])
        if candidate is not None:
            t, u, y = candidate
            if t is not None and u is not None and y is not None:
                return t, u, y

    for key, value in data.items():
        if key.startswith("__"):
            continue
        candidate = _try_matrix(value)
        if candidate is not None:
            t, u, y = candidate
            if t is not None and u is not None and y is not None:
                return t, u, y

    raise RuntimeError(
        f"Could not locate compatible [t,u,y] in {mat_path}.\n"
        f"Expected either t/u/y variables (with y 1D or 2D) or a 3xN (or Nx3) array with rows/cols [t,y,u]."
    )


def try_load_frequency_from_mat(mat_path: Path) -> Optional[np.ndarray]:
    """Try to load an analysis frequency vector (Hz or rad/s) from a MAT file.
    Recognizes:
     - 'frequency' (Hz), 'freq' (Hz), 'omega' or 'w' (rad/s),
     - 'P' where the first row is ω (rad/s) as saved by this script/MATLAB postprocess.
    Returns angular frequency ω [rad/s] if something is found; otherwise None.
    """
    try:
        data = loadmat(mat_path)
    except Exception:
        return None

    def _as_1d(x):
        arr = np.asarray(x).squeeze()
        return arr.astype(float).ravel() if arr.size > 0 and np.isfinite(arr).all() else None

    for key in ("omega", "w"):
        if key in data:
            out = _as_1d(data[key])
            if out is not None:
                return out
    for key in ("frequency", "freq"):
        if key in data:
            hz = _as_1d(data[key])
            if hz is not None:
                return 2.0 * np.pi * hz
    if "P" in data:
        P = np.asarray(data["P"])
        if P.ndim == 2 and P.shape[0] >= 1:
            w = np.ravel(P[0, :]).astype(float)
            if np.isfinite(w).all():
                return w
    return None


# --------------------------- Core math ---------------------------

def matlab_freq_grid(n_points: int, f_low_log10: float, f_up_log10: float) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate MATLAB: frequency = 10.^[f_low : step : f_up - step], where step = (f_up - f_low)/N_d"""
    step = (f_up_log10 - f_low_log10) / n_points
    logs = f_low_log10 + step * np.arange(n_points, dtype=np.float64)  # 0..N_d-1
    freqs_hz = np.power(10.0, logs)
    omega = 2.0 * np.pi * freqs_hz
    return freqs_hz, omega


def describe_timebase(t: np.ndarray) -> Tuple[float, float, bool]:
    """Return (median dt, span T, is_monotone)."""
    if t.size <= 1:
        return float("nan"), 0.0, True
    diffs = np.diff(t)
    return float(np.median(diffs)), float(t[-1] - t[0]), bool(np.all(diffs > 0))


def synchronous_coefficients_trapz(
    t: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    drop_seconds: float = 0.0,
    subtract_mean: bool = True,
) -> np.ndarray:
    """Time-weighted synchronous demodulation using trapezoidal integration.

    C_x(ω) = (2/T) ∫ x(t) e^{-jωt} dt
         = (2/T) [ ∫ x cos(ωt) dt  - j ∫ x sin(ωt) dt ]

    - Handles non-uniform dt robustly.
    - For uniform dt, scales to match continuous-time amplitude.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if drop_seconds and drop_seconds > 0.0:
        t0 = t[0] + drop_seconds
        keep = t >= t0
        t = t[keep]
        x = x[keep]

    if t.size < 2:
        raise ValueError("Not enough samples after transient drop.")

    if subtract_mean:
        x = x - np.mean(x)

    T = float(t[-1] - t[0])
    if T <= 0:
        raise ValueError("Non-positive record length.")

    coeffs = np.empty(w.shape, dtype=np.complex128)
    scale = 2.0 / T
    for i, wi in enumerate(w):
        cos_wt = np.cos(wi * t)
        sin_wt = np.sin(wi * t)
        real = scale * np.trapz(x * cos_wt, t)
        imag = -scale * np.trapz(x * sin_wt, t)  # minus to match MATLAB atan2(-Im, -Re) convention
        coeffs[i] = real + 1j * imag
    return coeffs


# --------------------------- Estimation & saving ---------------------------

def frf_cross_power_average(
    U_list: List[np.ndarray], Y_list: List[np.ndarray], eps: float = 1e-15
) -> np.ndarray:
    """G(ω) = (Σ Y_k * conj(U_k)) / (Σ |U_k|^2) with NaN where denominator is ~0."""
    if not U_list or not Y_list or len(U_list) != len(Y_list):
        raise ValueError("Empty or mismatched lists for cross-power averaging.")
    U_stack = np.vstack(U_list)
    Y_stack = np.vstack(Y_list)
    num = np.sum(Y_stack * np.conj(U_stack), axis=0)
    den = np.sum(np.abs(U_stack) ** 2, axis=0)
    G = np.full(den.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ok = den > eps
    G[ok] = num[ok] / den[ok]
    return G


def save_plots(prefix: Path, w: np.ndarray, G: np.ndarray) -> None:
    base = str(prefix)
    mag_db = 20.0 * np.log10(np.abs(G))
    phase_unwrapped = np.unwrap(np.angle(G))

    fig1, ax1 = plt.subplots()
    ax1.semilogx(w, mag_db, marker="*", linestyle="None")
    ax1.set_xlabel(r"$\omega$ [rad/s]")
    ax1.set_ylabel(r"$20\log_{10}|G(j\omega)|$ [dB]")
    ax1.set_title("Bode Gain plot")
    ax1.grid(True, which="both")
    fig1.tight_layout()
    fig1.savefig(base + "_bode_mag.png", dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(np.real(G), np.imag(G), marker="*", linestyle="None")
    ax2.set_xlabel(r"$\mathrm{Re}(G(j\omega))$")
    ax2.set_ylabel(r"$\mathrm{Im}(G(j\omega))$")
    ax2.set_title("Nyquist Plot")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(base + "_nyquist.png", dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    ax3.semilogx(w, phase_unwrapped, marker="*", linestyle="None")
    ax3.set_xlabel(r"$\omega$ [rad/s]")
    ax3.set_ylabel("phase [rad]")
    ax3.set_title("Bode Phase plot")
    ax3.grid(True, which="both")
    fig3.tight_layout()
    fig3.savefig(base + "_bode_phase.png", dpi=200)
    plt.close(fig3)


def save_csv(prefix: Path, w: np.ndarray, G: np.ndarray) -> None:
    base = str(prefix)
    freqs_hz = w / (2.0 * np.pi)
    with open(base + "_frf.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["omega_rad_s", "freq_Hz", "ReG", "ImG", "absG", "phase_rad"])
        for wi, fi, Gi in zip(w, freqs_hz, G):
            if not np.isfinite(Gi):
                continue
            writer.writerow([wi, fi, np.real(Gi), np.imag(Gi), np.abs(Gi), np.angle(Gi)])


def save_mat(prefix: Path, w: np.ndarray, G: np.ndarray) -> None:
    """Save MATLAB .mat with basic variables and MATLAB-compatible P."""
    base = str(prefix)
    freqs_hz = w / (2.0 * np.pi)
    gain = np.abs(G)
    phase_matlab = np.arctan2(-np.imag(G), -np.real(G))  # MATLAB plotting convention
    P = np.vstack((w, gain, phase_matlab))
    savemat(base + "_frf.mat", {"w": w, "f": freqs_hz, "G": G, "P": P})


def save_matlab_dat(prefix: Path, w: np.ndarray, G: np.ndarray) -> None:
    """Optional ASCII export of P (3xN) for MATLAB compatibility."""
    base = str(prefix)
    gain = np.abs(G)
    phase_matlab = np.arctan2(-np.imag(G), -np.real(G))
    P = np.vstack((w, gain, phase_matlab))
    np.savetxt(base + "_frf.dat", P, delimiter=",")


def save_readme(prefix: Path, out_dir: Path) -> None:
    """Write a small README describing output formats (Japanese)."""
    base = str(prefix)
    readme_path = out_dir / (Path(base).name + "_README.txt")
    text = f"""出力ファイルの形式

1) {Path(base).name}_frf.mat (MATLAB 形式)
   - w: 1xN (double) 角周波数 [rad/s]
   - f: 1xN (double) 周波数 [Hz] = w / (2π)
   - G: 1xN (complex double) FRF = Y/U
   - P: 3xN (double) 行ベクトルの縦結合 [w; |G|; atan2(-Im(G), -Re(G))]（MATLAB 互換の位相）

2) {Path(base).name}_frf.csv (テキスト)
   ヘッダ: omega_rad_s,freq_Hz,ReG,ImG,absG,phase_rad
   - omega_rad_s: 角周波数 [rad/s]
   - freq_Hz: 周波数 [Hz]
   - ReG/ImG: FRF の実部/虚部
   - absG: |G|
   - phase_rad: np.angle(G) による位相 [rad]

3) 図
   - {Path(base).name}_bode_mag.png: ゲイン（dB）
   - {Path(base).name}_bode_phase.png: 位相（rad, unwrap 済み）
   - {Path(base).name}_nyquist.png: ナイキスト

4) オプション (--save-matlab 指定時)
   - {Path(base).name}_frf.dat: P(3xN) を CSV 風に出力（MATLAB スクリプト互換）

出力先ディレクトリ: {out_dir.resolve()}
"""
    readme_path.write_text(text, encoding="utf-8")


# --------------------------- Main ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FRF estimation aligned to MATLAB logic (trapz demod + cross-power averaging)."
    )
    parser.add_argument("mat_files", nargs="+", help="MAT files/dirs/globs containing t/u/y or 3xN [t,y,u].")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories.")
    parser.add_argument("--nd", type=int, default=100, help="Number of frequency points (N_d).")
    parser.add_argument("--f-low", type=float, default=-1.0, help="log10 lower frequency bound.")
    parser.add_argument("--f-up", type=float, default=2.3, help="log10 upper frequency bound.")
    parser.add_argument("--dropseconds", type=float, default=0.0, help="Seconds to discard from start of each record.")
    parser.add_argument("--keep-mean", action="store_true", help="Keep DC (no mean subtraction).")
    parser.add_argument("--y-col", type=int, default=0, help="Which column of y to use if y is 2D (default 0).")
    parser.add_argument("--out-prefix", default="matched", help="Output file prefix (no extension).")
    parser.add_argument("--out-dir", default="output", help="Output directory (default: output).")
    parser.add_argument("--save-matlab", action="store_true", help="Also write *_frf.dat (ASCII P matrix).")
    parser.add_argument("--freq-from-mat", default=None,
                        help="MAT file to read frequency grid from (keys: frequency,freq,omega,w,P[0,:]).")
    parser.add_argument("--n-files", type=int, default=None,
                        help="Use only the first N MAT files after expansion/sort.")

    args = parser.parse_args()
    if args.nd <= 0:
        parser.error("--nd must be positive.")
    if args.f_up <= args.f_low:
        parser.error("--f-up must be greater than --f-low.")
    if args.dropseconds < 0:
        parser.error("--dropseconds must be >= 0.")
    if args.n_files is not None and args.n_files <= 0:
        parser.error("--n-files must be positive when provided.")

    # Expand MAT inputs
    mat_files = resolve_mat_files(args.mat_files, recursive=args.recursive)
    if not mat_files:
        parser.error("No MAT files found for the requested input.")

    # Limit to first N files if requested
    if args.n_files is not None and args.n_files < len(mat_files):
        print(f"Limiting to first {args.n_files} MAT file(s) out of {len(mat_files)}.")
        mat_files = mat_files[:args.n_files]

    # Build frequency grid
    w_from_arg = None
    if args.freq_from_mat:
        w_from_arg = try_load_frequency_from_mat(Path(args.freq_from_mat))

    w_auto = None
    if w_from_arg is None:
        for mp in mat_files:
            w_auto = try_load_frequency_from_mat(mp)
            if w_auto is not None:
                print(f"Detected ω grid from '{mp.name}' (len={w_auto.size}).")
                break

    if w_from_arg is not None:
        omega = np.asarray(w_from_arg, dtype=float).ravel()
        print(f"Using ω grid loaded from --freq-from-mat (len={omega.size}).")
    elif w_auto is not None:
        omega = np.asarray(w_auto, dtype=float).ravel()
        print(f"Using ω grid auto-detected from input (len={omega.size}).")
    else:
        _, omega = matlab_freq_grid(args.nd, args.f_low, args.f_up)
        print(f"Using MATLAB log grid with N_d={args.nd}, 10^[{args.f_low}, {args.f_up}).")

    # Accumulate cross-power statistics
    subtract_mean = not args.keep_mean
    U_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    total_samples = 0
    total_seconds = 0.0

    for mat_path in mat_files:
        t, u, y = load_time_u_y(mat_path, y_col=args.y_col)
        dt, T, is_mono = describe_timebase(t)
        total_samples += t.size
        total_seconds += max(T, 0.0)

        if not is_mono:
            order = np.argsort(t)
            t = t[order]; u = u[order]; y = y[order]

        U = synchronous_coefficients_trapz(t, u, omega, args.dropseconds, subtract_mean)
        Y = synchronous_coefficients_trapz(t, y, omega, args.dropseconds, subtract_mean)
        U_list.append(U); Y_list.append(Y)

        dt_str = f"{dt:.6g}" if np.isfinite(dt) else "n/a"
        dur_str = f"{T:.2f}"
        print(f"{mat_path.name}: {t.size} samples, dt~{dt_str} s, span~{dur_str} s, "
              f"transient_dropped={args.dropseconds:.2f}s")

    hours = total_seconds / 3600.0 if total_seconds else 0.0
    print(f"Accumulated {len(mat_files)} file(s): {total_samples} samples, {total_seconds:.2f} s (~{hours:.2f} h).")

    # Cross-power average
    G = frf_cross_power_average(U_list, Y_list)

    # Sort by ω (defensive)
    order = np.argsort(omega)
    omega = omega[order]
    G = G[order]

    # Output paths (default to 'output' directory)
    out_dir = Path(args.out_dir) if args.out_dir else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / Path(args.out_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figures and data
    save_plots(prefix_path, omega, G)
    save_csv(prefix_path, omega, G)
    save_mat(prefix_path, omega, G)
    if args.save_matlab:
        save_matlab_dat(prefix_path, omega, G)
    save_readme(prefix_path, out_dir)

    print(f"Outputs written with prefix '{prefix_path}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
