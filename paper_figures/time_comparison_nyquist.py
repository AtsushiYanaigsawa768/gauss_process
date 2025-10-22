#!/usr/bin/env python3
"""
time_comparison_nyquist.py

Generate Nyquist plots comparing different data durations for paper.

This script creates four Nyquist plots using logarithmic frequency grid:
1. 10 minutes: 1 MAT file, 600 seconds
2. 30 minutes: 1 MAT file, 1800 seconds
3. 1 hour: 1 MAT file, full duration (~3600 seconds)
4. 10 hours: 10 MAT files, full duration each

All plots use synchronous demodulation method (matching generate_nyquist_figures.py).

Usage:
    python paper_figures/time_comparison_nyquist.py input/*.mat
    python paper_figures/time_comparison_nyquist.py input/*.mat --nd 100
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (8, 6),
    'lines.linewidth': 2,
    'lines.markersize': 8,
})


def load_time_u_y(mat_path: Path, y_col: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract time, input, output vectors from a MAT file.

    Returns (t, u, y) as 1D float arrays.
    """
    data = loadmat(mat_path)

    # Try to extract t, u, y variables
    t = data.get('t', None)
    u = data.get('u', None)
    y = data.get('y', None)

    # Helper to convert to 1D array
    def _ravel1d(x):
        try:
            arr = np.asarray(x).squeeze()
            arr = arr.astype(float).ravel()
            return arr if arr.size > 0 and np.isfinite(arr).all() else None
        except Exception:
            return None

    # Path (A): explicit t/u/y
    t = _ravel1d(t)
    u = _ravel1d(u)
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
                    raise ValueError(f"y_col {y_col} out of range for y with shape {y_arr.shape}")
                y = _ravel1d(y_arr[:, y_col])
        else:
            y = None
        if y is not None and len(t) > 1 and len(u) == len(t) and len(y) == len(t):
            return t, u, y

    # Path (B): 3xN or Nx3 array
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

    raise RuntimeError(f"Could not locate compatible [t,u,y] in {mat_path}")


def matlab_freq_grid(n_points: int, f_low_log10: float, f_up_log10: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replicate MATLAB logarithmic frequency grid.

    Returns:
        (frequencies [Hz], omega [rad/s])
    """
    step = (f_up_log10 - f_low_log10) / n_points
    logs = f_low_log10 + step * np.arange(n_points, dtype=np.float64)
    freqs_hz = np.power(10.0, logs)
    omega = 2.0 * np.pi * freqs_hz
    return freqs_hz, omega


def synchronous_coefficients_trapz(
    t: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    drop_seconds: float = 0.0,
    subtract_mean: bool = True,
) -> np.ndarray:
    """
    Time-weighted synchronous demodulation using trapezoidal integration.

    This is the method used in frequency_response.py.

    Returns:
        C_x(ω) = (2/T) ∫ x(t) e^{-jωt} dt
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
        imag = -scale * np.trapz(x * sin_wt, t)
        coeffs[i] = real + 1j * imag
    return coeffs


def compute_frf_log_grid(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    n_points: int = 100,
    f_low_log10: float = -1.0,
    f_up_log10: float = 2.3,
    drop_seconds: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency response using logarithmic frequency grid.

    This replicates the frequency_response.py method.

    Returns:
        (omega [rad/s], G(jω))
    """
    # Generate logarithmic frequency grid
    freqs_hz, omega = matlab_freq_grid(n_points, f_low_log10, f_up_log10)

    # Synchronous demodulation
    U = synchronous_coefficients_trapz(t, u, omega, drop_seconds, subtract_mean=True)
    Y = synchronous_coefficients_trapz(t, y, omega, drop_seconds, subtract_mean=True)

    # Compute transfer function
    eps = 1e-15
    G = np.full(omega.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ok = np.abs(U) > eps
    G[ok] = Y[ok] / U[ok]

    return omega, G


def compute_frf_multiple_files(
    mat_files: List[Path],
    n_files: int = 1,
    n_points: int = 100,
    f_low_log10: float = -1.0,
    f_up_log10: float = 2.3,
    drop_seconds: float = 0.0,
    time_duration: float = None,
    y_col: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency response from multiple MAT files using cross-power aggregation.

    Args:
        mat_files: List of MAT file paths
        n_files: Number of files to use
        n_points: Number of frequency points
        f_low_log10: log10 lower frequency bound
        f_up_log10: log10 upper frequency bound
        drop_seconds: Seconds to drop from start
        time_duration: Duration to use from each file (None = full duration)
        y_col: Column index for output if y is 2D

    Returns:
        (omega [rad/s], G(jω))
    """
    # Generate frequency grid (same for all files)
    freqs_hz, omega = matlab_freq_grid(n_points, f_low_log10, f_up_log10)

    # Accumulate cross-power and input power
    sum_YU_conj = np.zeros(omega.shape, dtype=np.complex128)
    sum_U_power = np.zeros(omega.shape, dtype=np.float64)

    for i, mat_file in enumerate(mat_files[:n_files]):
        print(f"  Processing file {i+1}/{n_files}: {mat_file.name}")

        # Load data
        t, u, y = load_time_u_y(mat_file, y_col=y_col)

        # Apply time duration limit if specified
        if time_duration is not None:
            t_start = t[0]
            t_end = t_start + time_duration
            mask = (t >= t_start) & (t <= t_end)
            t = t[mask]
            u = u[mask]
            y = y[mask]

            if len(t) < 2:
                print(f"  Warning: Not enough samples after time duration limit for {mat_file.name}")
                continue

        # Compute synchronous coefficients
        U = synchronous_coefficients_trapz(t, u, omega, drop_seconds, subtract_mean=True)
        Y = synchronous_coefficients_trapz(t, y, omega, drop_seconds, subtract_mean=True)

        # Accumulate cross-power and input power
        sum_YU_conj += Y * np.conj(U)
        sum_U_power += np.abs(U) ** 2

    # Compute aggregated transfer function
    eps = 1e-15
    G = np.full(omega.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ok = sum_U_power > eps
    G[ok] = sum_YU_conj[ok] / sum_U_power[ok]

    return omega, G


def plot_nyquist(omega: np.ndarray, G: np.ndarray, output_path: Path,
                title: str = 'Nyquist Plot', save_eps: bool = True,
                auto_range: bool = True, percentile: float = 99.0) -> None:
    """
    Generate Nyquist plot in publication style.

    Args:
        omega: Angular frequency [rad/s]
        G: Complex transfer function
        output_path: Output file path (without extension)
        title: Plot title
        save_eps: Whether to save EPS version
        auto_range: Automatically adjust plot range based on percentiles
        percentile: Percentile for auto-ranging (default: 99.0)
    """
    # Filter out NaN values
    valid = np.isfinite(G)
    G_valid = G[valid]
    omega_valid = omega[valid]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Nyquist (markers only, matching frequency_response.py style)
    ax.plot(np.real(G_valid), np.imag(G_valid), marker="*", linestyle="None",
            markersize=10, color='blue', label='Measured Data')

    # Labels and formatting
    ax.set_xlabel(r"$\mathrm{Re}(G(j\omega))$", fontsize=20)
    ax.set_ylabel(r"$\mathrm{Im}(G(j\omega))$", fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.grid(True, linestyle='--')
    ax.legend(loc='best')

    # Auto-range based on percentiles to exclude outliers
    if auto_range and len(G_valid) > 10:
        real_parts = np.real(G_valid)
        imag_parts = np.imag(G_valid)

        # Use percentiles to determine reasonable plot range
        real_low = np.percentile(real_parts, 100 - percentile)
        real_high = np.percentile(real_parts, percentile)
        imag_low = np.percentile(imag_parts, 100 - percentile)
        imag_high = np.percentile(imag_parts, percentile)

        # Add margin (10%)
        real_margin = 0.1 * (real_high - real_low)
        imag_margin = 0.1 * (imag_high - imag_low)

        ax.set_xlim(real_low - real_margin, real_high + real_margin)
        ax.set_ylim(imag_low - imag_margin, imag_high + imag_margin)

    # Use equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save PNG
    png_path = str(output_path) + '.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {png_path}")

    # Save EPS
    if save_eps:
        eps_path = str(output_path) + '.eps'
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Saved: {eps_path}")

    plt.close(fig)


def main():
    """Main function to generate time comparison figures."""
    parser = argparse.ArgumentParser(
        description='Generate Nyquist plots comparing different data durations'
    )

    parser.add_argument(
        'mat_files',
        nargs='+',
        help='MAT files containing time-domain data (need at least 10 files)'
    )
    parser.add_argument(
        '--nd',
        type=int,
        default=100,
        help='Number of frequency points for log grid (default: 100)'
    )
    parser.add_argument(
        '--f-low',
        type=float,
        default=-1.0,
        help='log10 lower frequency bound (default: -1.0)'
    )
    parser.add_argument(
        '--f-up',
        type=float,
        default=2.3,
        help='log10 upper frequency bound (default: 2.3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='paper_figures',
        help='Output directory (default: paper_figures)'
    )
    parser.add_argument(
        '--no-eps',
        action='store_true',
        help='Do not save EPS files'
    )

    args = parser.parse_args()

    # Convert to Path objects and sort
    mat_files = sorted([str(Path(f).resolve()) for f in args.mat_files])

    if len(mat_files) < 10:
        print(f"Warning: Only {len(mat_files)} MAT files found. Need at least 10 for the 10-hour condition.")
        print("Will proceed with available files for 10-minute and 1-hour conditions.")

    print(f"\n{'='*80}")
    print("Time Comparison Nyquist Plot Generation")
    print(f"{'='*80}")
    print(f"Total MAT files available: {len(mat_files)}")
    print(f"First file: {Path(mat_files[0]).name}")
    print(f"Frequency points (nd): {args.nd}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Condition 1: 10 minutes (600 seconds)
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 1: 10 Minutes (1 file, 600 seconds)")
    print(f"{'='*60}")

    omega_10min, G_10min = compute_frf_multiple_files(
        mat_files=[Path(f) for f in mat_files],
        n_files=1,
        n_points=args.nd,
        f_low_log10=args.f_low,
        f_up_log10=args.f_up,
        drop_seconds=0.0,
        time_duration=600.0
    )

    n_valid_10min = np.sum(np.isfinite(G_10min))
    print(f"Valid frequency points: {n_valid_10min}/{len(omega_10min)}")
    print(f"Frequency range: {omega_10min[0]:.3f} - {omega_10min[-1]:.3f} rad/s")
    print(f"              = {omega_10min[0]/(2*np.pi):.3f} - {omega_10min[-1]/(2*np.pi):.3f} Hz")

    # Generate plot
    output_path_10min = output_dir / 'nyquist_10min'
    plot_nyquist(
        omega_10min, G_10min,
        output_path_10min,
        title='Nyquist Plot (10 Minutes)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Condition 2: 30 minutes (1800 seconds)
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 2: 30 Minutes (1 file, 1800 seconds)")
    print(f"{'='*60}")

    omega_30min, G_30min = compute_frf_multiple_files(
        mat_files=[Path(f) for f in mat_files],
        n_files=1,
        n_points=args.nd,
        f_low_log10=args.f_low,
        f_up_log10=args.f_up,
        drop_seconds=0.0,
        time_duration=1800.0
    )

    n_valid_30min = np.sum(np.isfinite(G_30min))
    print(f"Valid frequency points: {n_valid_30min}/{len(omega_30min)}")
    print(f"Frequency range: {omega_30min[0]:.3f} - {omega_30min[-1]:.3f} rad/s")
    print(f"              = {omega_30min[0]/(2*np.pi):.3f} - {omega_30min[-1]/(2*np.pi):.3f} Hz")

    # Generate plot
    output_path_30min = output_dir / 'nyquist_30min'
    plot_nyquist(
        omega_30min, G_30min,
        output_path_30min,
        title='Nyquist Plot (30 Minutes)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Condition 3: 1 hour (full file, ~3600 seconds)
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 3: 1 Hour (1 file, full duration)")
    print(f"{'='*60}")

    omega_1hour, G_1hour = compute_frf_multiple_files(
        mat_files=[Path(f) for f in mat_files],
        n_files=1,
        n_points=args.nd,
        f_low_log10=args.f_low,
        f_up_log10=args.f_up,
        drop_seconds=0.0,
        time_duration=None  # Use full duration
    )

    n_valid_1hour = np.sum(np.isfinite(G_1hour))
    print(f"Valid frequency points: {n_valid_1hour}/{len(omega_1hour)}")
    print(f"Frequency range: {omega_1hour[0]:.3f} - {omega_1hour[-1]:.3f} rad/s")
    print(f"              = {omega_1hour[0]/(2*np.pi):.3f} - {omega_1hour[-1]/(2*np.pi):.3f} Hz")

    # Generate plot
    output_path_1hour = output_dir / 'nyquist_1hour'
    plot_nyquist(
        omega_1hour, G_1hour,
        output_path_1hour,
        title='Nyquist Plot (1 Hour)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Condition 4: 10 hours (10 files, full duration each)
    # ========================================
    if len(mat_files) >= 10:
        print(f"\n{'='*60}")
        print("Condition 4: 10 Hours (10 files, full duration each)")
        print(f"{'='*60}")

        omega_10hour, G_10hour = compute_frf_multiple_files(
            mat_files=[Path(f) for f in mat_files],
            n_files=10,
            n_points=args.nd,
            f_low_log10=args.f_low,
            f_up_log10=args.f_up,
            drop_seconds=0.0,
            time_duration=None
        )

        n_valid_10hour = np.sum(np.isfinite(G_10hour))
        print(f"Valid frequency points: {n_valid_10hour}/{len(omega_10hour)}")
        print(f"Frequency range: {omega_10hour[0]:.3f} - {omega_10hour[-1]:.3f} rad/s")
        print(f"              = {omega_10hour[0]/(2*np.pi):.3f} - {omega_10hour[-1]/(2*np.pi):.3f} Hz")

        # Generate plot
        output_path_10hour = output_dir / 'nyquist_10hour'
        plot_nyquist(
            omega_10hour, G_10hour,
            output_path_10hour,
            title='Nyquist Plot (10 Hours)',
            save_eps=not args.no_eps
        )
    else:
        print(f"\n{'='*60}")
        print(f"Skipping Condition 4: Only {len(mat_files)} files available (need 10)")
        print(f"{'='*60}")

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"\nGenerated files:")
    print(f"  1. {output_path_10min}.png (and .eps) - 10 Minutes")
    print(f"  2. {output_path_30min}.png (and .eps) - 30 Minutes")
    print(f"  3. {output_path_1hour}.png (and .eps) - 1 Hour")
    if len(mat_files) >= 10:
        print(f"  4. {output_path_10hour}.png (and .eps) - 10 Hours")
    print(f"\nAll plots use:")
    print(f"  - Logarithmic frequency grid (nd={args.nd})")
    print(f"  - Synchronous demodulation method (direct processing)")
    print(f"  - Cross-power aggregation for multiple files")
    print(f"  - Publication-quality styling (300 DPI PNG + EPS)")
    print(f"  - Markers only (star markers, no lines)")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
