#!/usr/bin/env python3
"""
generate_nyquist_figures.py

Generate publication-quality Nyquist plots for paper.

This script creates two Nyquist plots from a single MAT file:
1. Logarithmic frequency grid (using synchronous demodulation method from frequency_response.py)
2. Linear frequency grid (using FFT method from fourier_transform.py)

Both plots use consistent styling matching the frequency_response.py format.

Usage:
    python paper_figures/generate_nyquist_figures.py input/input_test_20250912_165937.mat
    python paper_figures/generate_nyquist_figures.py input/*.mat --n-files 1
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft, fftfreq
from scipy.signal import get_window

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
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


def compute_frf_fft(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    n_freq: int = None,
    window: str = 'none',  # Changed default to 'none' for testing
    drop_seconds: float = 0.0,
    f_min: float = 0.05,
    f_max: float = 200.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency response using FFT.

    This replicates the fourier_transform.py method with frequency filtering
    to avoid DC component issues.

    Args:
        t: Time array
        u: Input signal
        y: Output signal
        n_freq: Number of FFT points
        window: Window function name
        drop_seconds: Seconds to drop from start
        f_min: Minimum frequency to include [Hz] (default: 0.05)
        f_max: Maximum frequency to include [Hz] (default: 200.0)

    Returns:
        (omega [rad/s], G(jω))
    """
    # Apply time windowing
    if drop_seconds > 0:
        mask = t >= (t[0] + drop_seconds)
        t = t[mask]
        u = u[mask]
        y = y[mask]

    # Sample rate
    dt = np.mean(np.diff(t))
    n = len(u)

    # Remove mean
    u = u - np.mean(u)
    y = y - np.mean(y)

    # Apply window
    if window is not None and window != 'none':
        win = get_window(window, n)
        u = u * win
        y = y * win
        # Compensate for window power loss
        win_correction = np.sum(win) / n
        u = u / win_correction
        y = y / win_correction

    # Compute FFT
    if n_freq is None:
        n_freq = n

    # Compute FFT with standard normalization
    # Note: Window function is disabled to match synchronous demodulation amplitude scale
    # With Hann window, the amplitude compensation was incorrectly applied
    scale_factor = 1.0 / n

    U_spectrum = fft(u, n=n_freq) * scale_factor
    Y_spectrum = fft(y, n=n_freq) * scale_factor
    frequencies = fftfreq(n_freq, dt)

    # Keep only positive frequencies in valid range
    # Filter out DC component (very low freq) and frequencies above f_max
    valid_mask = (frequencies >= f_min) & (frequencies <= f_max)
    frequencies = frequencies[valid_mask]
    U_spectrum = U_spectrum[valid_mask]
    Y_spectrum = Y_spectrum[valid_mask]

    # Compute transfer function with input-power-based filtering
    # Only use frequencies where the input has significant power
    U_power = np.abs(U_spectrum) ** 2

    # Determine threshold based on median power
    if len(U_power) > 0:
        power_median = np.median(U_power)
        # Keep frequencies where power is at least 1% of median
        power_threshold = 0.01 * power_median
        significant = U_power > power_threshold
    else:
        significant = np.ones_like(frequencies, dtype=bool)

    # Compute transfer function only at significant frequencies
    G = np.full_like(frequencies, np.nan + 1j * np.nan, dtype=np.complex128)
    G[significant] = Y_spectrum[significant] / U_spectrum[significant]

    # Convert to angular frequency
    omega = 2.0 * np.pi * frequencies

    return omega, G


def plot_nyquist(
    omega: np.ndarray,
    G: np.ndarray,
    output_path: Path,
    title: str = 'Nyquist Plot',
    save_eps: bool = True,
    auto_range: bool = True,
    percentile: float = 99.0
) -> None:
    """
    Generate Nyquist plot in publication style.

    Uses the same style as frequency_response.py:
    - Markers only (no lines)
    - Star markers
    - Grid enabled

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
    ax.set_xlabel(r"$\mathrm{Re}(G(j\omega))$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{Im}(G(j\omega))$", fontsize=14)
    ax.set_title(title, fontsize=16)
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
    """Main function to generate paper figures."""
    parser = argparse.ArgumentParser(
        description='Generate publication-quality Nyquist plots for paper'
    )

    parser.add_argument(
        'mat_file',
        nargs='?',
        default=None,
        help='MAT file containing time-domain data (default: first file in input/)'
    )
    parser.add_argument(
        '--n-files',
        type=int,
        default=1,
        help='Number of files to use (default: 1)'
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
        help='log10 lower frequency bound for log grid (default: -1.0)'
    )
    parser.add_argument(
        '--f-up',
        type=float,
        default=2.3,
        help='log10 upper frequency bound for log grid (default: 2.3)'
    )
    parser.add_argument(
        '--drop-seconds',
        type=float,
        default=0.0,
        help='Seconds to drop from start of data (default: 0.0)'
    )
    parser.add_argument(
        '--y-col',
        type=int,
        default=0,
        help='Column index if y is 2D (default: 0)'
    )
    parser.add_argument(
        '--no-eps',
        action='store_true',
        help='Do not save EPS files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='paper_figures',
        help='Output directory (default: paper_figures)'
    )
    parser.add_argument(
        '--fft-f-min',
        type=float,
        default=0.05,
        help='Minimum frequency for FFT method [Hz] (default: 0.05)'
    )
    parser.add_argument(
        '--fft-f-max',
        type=float,
        default=200.0,
        help='Maximum frequency for FFT method [Hz] (default: 200.0)'
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
        mat_path = mat_files[0]
        print(f"Using default input file: {mat_path}")
    else:
        mat_path = Path(args.mat_file)
        if not mat_path.exists():
            print(f"Error: File not found: {mat_path}")
            return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from: {mat_path.name}")
    t, u, y = load_time_u_y(mat_path, y_col=args.y_col)

    dt = np.median(np.diff(t))
    duration = t[-1] - t[0]
    print(f"  Samples: {len(t)}")
    print(f"  Duration: {duration:.2f} s")
    print(f"  Sample rate: {1.0/dt:.2f} Hz")
    print(f"  Median dt: {dt:.6f} s")

    # ========================================
    # Method 1: Logarithmic frequency grid
    # ========================================
    print(f"\n{'='*60}")
    print("Method 1: Logarithmic Frequency Grid (Synchronous Demodulation)")
    print(f"{'='*60}")

    omega_log, G_log = compute_frf_log_grid(
        t, u, y,
        n_points=args.nd,
        f_low_log10=args.f_low,
        f_up_log10=args.f_up,
        drop_seconds=args.drop_seconds
    )

    # Count valid points
    n_valid_log = np.sum(np.isfinite(G_log))
    G_log_mag = np.abs(G_log[np.isfinite(G_log)])
    mag_median_log = np.median(G_log_mag) if len(G_log_mag) > 0 else 0.0
    print(f"Frequency points: {len(omega_log)} ({n_valid_log} valid)")
    print(f"Frequency range: {omega_log[0]:.3f} - {omega_log[-1]:.3f} rad/s")
    print(f"              = {omega_log[0]/(2*np.pi):.3f} - {omega_log[-1]/(2*np.pi):.3f} Hz")
    print(f"G magnitude median: {mag_median_log:.3e}")

    # Generate Nyquist plot
    output_path_log = output_dir / 'nyquist_log_scale'
    plot_nyquist(
        omega_log, G_log,
        output_path_log,
        title='Nyquist Plot (Logarithmic Frequency Grid)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Method 2: Linear frequency grid (FFT)
    # ========================================
    print(f"\n{'='*60}")
    print("Method 2: Linear Frequency Grid (FFT)")
    print(f"{'='*60}")
    print(f"Frequency filtering: {args.fft_f_min:.2f} - {args.fft_f_max:.2f} Hz")

    omega_fft, G_fft = compute_frf_fft(
        t, u, y,
        n_freq=None,  # Use full FFT resolution
        window='none',  # Disable windowing for amplitude matching
        drop_seconds=args.drop_seconds,
        f_min=args.fft_f_min,
        f_max=args.fft_f_max
    )

    # Print statistics for debugging
    G_fft_mag = np.abs(G_fft)
    valid_finite = np.isfinite(G_fft)
    if np.any(valid_finite):
        mag_median = np.median(G_fft_mag[valid_finite])
        mag_min = np.min(G_fft_mag[valid_finite])
        mag_max = np.max(G_fft_mag[valid_finite])
        n_valid = np.sum(valid_finite)
        print(f"G magnitude statistics: median={mag_median:.3e}, min={mag_min:.3e}, max={mag_max:.3e}")
        print(f"Valid points: {n_valid} out of {len(G_fft)}")

    # Downsample FFT results to match log grid resolution for visualization
    # (Keep the same number of points for fair comparison)
    n_downsample = args.nd
    indices = np.linspace(0, len(omega_fft)-1, n_downsample, dtype=int)
    omega_fft_ds = omega_fft[indices]
    G_fft_ds = G_fft[indices]

    n_valid_fft = np.sum(np.isfinite(G_fft_ds))
    print(f"Frequency points: {len(omega_fft_ds)} (downsampled from {len(omega_fft)})")
    print(f"                 ({n_valid_fft} valid)")
    print(f"Frequency range: {omega_fft_ds[0]:.3f} - {omega_fft_ds[-1]:.3f} rad/s")
    print(f"              = {omega_fft_ds[0]/(2*np.pi):.3f} - {omega_fft_ds[-1]/(2*np.pi):.3f} Hz")

    # Generate Nyquist plot (matching frequency_response.py style)
    output_path_fft = output_dir / 'nyquist_linear_scale'
    plot_nyquist(
        omega_fft_ds, G_fft_ds,
        output_path_fft,
        title='Nyquist Plot (Linear Frequency Grid)',
        save_eps=not args.no_eps
    )

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Input file: {mat_path}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"\nGenerated files:")
    print(f"  1. {output_path_log}.png (and .eps)")
    print(f"  2. {output_path_fft}.png (and .eps)")
    print(f"\nBoth Nyquist plots use consistent styling:")
    print(f"  - Markers only (no lines)")
    print(f"  - Star markers for data points")
    print(f"  - Frequency direction indicators (red=low, green=high)")
    print(f"  - Equal axis aspect ratio")

    return 0


if __name__ == "__main__":
    sys.exit(main())
