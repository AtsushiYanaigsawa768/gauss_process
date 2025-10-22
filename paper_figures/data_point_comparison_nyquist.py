#!/usr/bin/env python3
"""
data_point_comparison_nyquist.py

Generate Nyquist plots comparing different numbers of frequency data points for paper.

This script creates four Nyquist plots from 1 hour of data (1 MAT file, full duration)
with varying numbers of frequency points:
1. 10 points: nd = 10
2. 30 points: nd = 30
3. 50 points: nd = 50
4. 100 points: nd = 100

All plots use synchronous demodulation method (matching generate_nyquist_figures.py).

Usage:
    python paper_figures/data_point_comparison_nyquist.py input/input_test_20250912_165937.mat
    python paper_figures/data_point_comparison_nyquist.py input/*.mat
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
    """Main function to generate data point comparison figures."""
    parser = argparse.ArgumentParser(
        description='Generate Nyquist plots comparing different numbers of frequency points'
    )

    parser.add_argument(
        'mat_file',
        nargs='?',
        help='MAT file containing time-domain data (1 hour). If not specified, uses first file from input/'
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

    # Determine input file
    if args.mat_file is None:
        # Use first file from input/ directory
        input_dir = Path('input')
        mat_files = sorted(input_dir.glob('*.mat'))
        if not mat_files:
            print("Error: No MAT files found in input/ directory")
            return 1
        mat_path = str(mat_files[0].resolve())
        print(f"Using default input file: {Path(mat_path).name}")
    else:
        mat_path = str(Path(args.mat_file).resolve())
        if not Path(mat_path).exists():
            print(f"Error: File not found: {mat_path}")
            return 1

    mat_files = [mat_path]

    print(f"\n{'='*80}")
    print("Data Point Comparison Nyquist Plot Generation")
    print(f"{'='*80}")
    print(f"Input file: {Path(mat_path).name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Comparing: 10, 30, 50, 100 frequency points")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    print(f"Loading data from: {Path(mat_path).name}")
    t, u, y = load_time_u_y(Path(mat_path), y_col=0)

    dt = np.median(np.diff(t))
    duration = t[-1] - t[0]
    print(f"  Samples: {len(t)}")
    print(f"  Duration: {duration:.2f} s")
    print(f"  Sample rate: {1.0/dt:.2f} Hz")

    # ========================================
    # Condition 1: 10 data points
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 1: 10 Frequency Points (nd=10)")
    print(f"{'='*60}")

    omega_10pt, G_10pt = compute_frf_log_grid(
        t, u, y,
        n_points=10,
        f_low_log10=-1.0,
        f_up_log10=2.3,
        drop_seconds=0.0
    )

    n_valid_10pt = np.sum(np.isfinite(G_10pt))
    print(f"Valid frequency points: {n_valid_10pt}/{len(omega_10pt)}")
    print(f"Frequency range: {omega_10pt[0]:.3f} - {omega_10pt[-1]:.3f} rad/s")
    print(f"              = {omega_10pt[0]/(2*np.pi):.3f} - {omega_10pt[-1]/(2*np.pi):.3f} Hz")

    # Generate plot
    output_path_10pt = output_dir / 'nyquist_10points'
    plot_nyquist(
        omega_10pt, G_10pt,
        output_path_10pt,
        title='Nyquist Plot (10 Frequency Points)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Condition 2: 30 data points
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 2: 30 Frequency Points (nd=30)")
    print(f"{'='*60}")

    omega_30pt, G_30pt = compute_frf_log_grid(
        t, u, y,
        n_points=30,
        f_low_log10=-1.0,
        f_up_log10=2.3,
        drop_seconds=0.0
    )

    n_valid_30pt = np.sum(np.isfinite(G_30pt))
    print(f"Valid frequency points: {n_valid_30pt}/{len(omega_30pt)}")
    print(f"Frequency range: {omega_30pt[0]:.3f} - {omega_30pt[-1]:.3f} rad/s")
    print(f"              = {omega_30pt[0]/(2*np.pi):.3f} - {omega_30pt[-1]/(2*np.pi):.3f} Hz")

    # Generate plot
    output_path_30pt = output_dir / 'nyquist_30points'
    plot_nyquist(
        omega_30pt, G_30pt,
        output_path_30pt,
        title='Nyquist Plot (30 Frequency Points)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Condition 3: 50 data points
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 3: 50 Frequency Points (nd=50)")
    print(f"{'='*60}")

    omega_50pt, G_50pt = compute_frf_log_grid(
        t, u, y,
        n_points=50,
        f_low_log10=-1.0,
        f_up_log10=2.3,
        drop_seconds=0.0
    )

    n_valid_50pt = np.sum(np.isfinite(G_50pt))
    print(f"Valid frequency points: {n_valid_50pt}/{len(omega_50pt)}")
    print(f"Frequency range: {omega_50pt[0]:.3f} - {omega_50pt[-1]:.3f} rad/s")
    print(f"              = {omega_50pt[0]/(2*np.pi):.3f} - {omega_50pt[-1]/(2*np.pi):.3f} Hz")

    # Generate plot
    output_path_50pt = output_dir / 'nyquist_50points'
    plot_nyquist(
        omega_50pt, G_50pt,
        output_path_50pt,
        title='Nyquist Plot (50 Frequency Points)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Condition 4: 100 data points
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 4: 100 Frequency Points (nd=100)")
    print(f"{'='*60}")

    omega_100pt, G_100pt = compute_frf_log_grid(
        t, u, y,
        n_points=100,
        f_low_log10=-1.0,
        f_up_log10=2.3,
        drop_seconds=0.0
    )

    n_valid_100pt = np.sum(np.isfinite(G_100pt))
    print(f"Valid frequency points: {n_valid_100pt}/{len(omega_100pt)}")
    print(f"Frequency range: {omega_100pt[0]:.3f} - {omega_100pt[-1]:.3f} rad/s")
    print(f"              = {omega_100pt[0]/(2*np.pi):.3f} - {omega_100pt[-1]/(2*np.pi):.3f} Hz")

    # Generate plot
    output_path_100pt = output_dir / 'nyquist_100points'
    plot_nyquist(
        omega_100pt, G_100pt,
        output_path_100pt,
        title='Nyquist Plot (100 Frequency Points)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"\nGenerated files:")
    print(f"  1. {output_path_10pt}.png (and .eps) - 10 Points")
    print(f"  2. {output_path_30pt}.png (and .eps) - 30 Points")
    print(f"  3. {output_path_50pt}.png (and .eps) - 50 Points")
    print(f"  4. {output_path_100pt}.png (and .eps) - 100 Points")
    print(f"\nAll plots use:")
    print(f"  - 1 hour of data (1 MAT file, full duration)")
    print(f"  - Logarithmic frequency grid")
    print(f"  - Synchronous demodulation method (direct processing)")
    print(f"  - Publication-quality styling (300 DPI PNG + EPS)")
    print(f"  - Markers only (star markers, no lines)")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
