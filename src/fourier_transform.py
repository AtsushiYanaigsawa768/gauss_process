#!/usr/bin/env python3
"""
fourier_transform.py

Fourier Transform-based frequency domain analysis for system identification.
This module provides an alternative to FRF estimation by performing direct
Fourier transformation of time-domain data and processing on a linear frequency scale.

Key features:
1. Direct FFT computation from time-domain signals
2. Linear frequency scale processing (not logarithmic)
3. Compatible output format with the unified pipeline
4. Handles multiple input files and time windowing

Usage:
    python src/fourier_transform.py input/*.mat --n-files 1 --nd 100 --out-dir output
    python src/fourier_transform.py input/*.mat --time-duration 60.0 --window hann
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.signal import get_window
from scipy.fft import fft, fftfreq


def load_time_data(mat_path: Path, y_col: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time-domain data from MAT file.

    Args:
        mat_path: Path to MAT file
        y_col: Column index if y is 2D (default: 0)

    Returns:
        Tuple of (time, input, output) arrays
    """
    data = loadmat(mat_path)

    # Try to extract t, u, y variables
    t = data.get('t', None)
    u = data.get('u', None)
    y = data.get('y', None)

    # Alternative: look for 3xN or Nx3 array
    if t is None or u is None or y is None:
        for key, value in data.items():
            if key.startswith('__'):
                continue
            arr = np.asarray(value)
            if arr.ndim == 2:
                if arr.shape[0] == 3:
                    t, y, u = arr[0], arr[1], arr[2]
                    break
                elif arr.shape[1] == 3:
                    t, y, u = arr[:, 0], arr[:, 1], arr[:, 2]
                    break

    # Ensure all are 1D arrays
    t = np.asarray(t).squeeze()
    u = np.asarray(u).squeeze()
    y = np.asarray(y).squeeze()

    # Handle 2D y array
    if y.ndim == 2:
        y = y[:, y_col]

    # Validate
    if t.size != u.size or t.size != y.size:
        raise ValueError(f"Inconsistent array sizes: t={t.size}, u={u.size}, y={y.size}")

    return t.astype(float), u.astype(float), y.astype(float)


def apply_time_window(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    drop_seconds: float = 0.0,
    time_duration: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply time windowing to the signals.

    Args:
        t: Time array
        u: Input signal
        y: Output signal
        drop_seconds: Seconds to drop from start (transient removal)
        time_duration: Duration to use (None = use all)

    Returns:
        Windowed (t, u, y) arrays
    """
    # Drop initial transient
    if drop_seconds > 0:
        mask = t >= (t[0] + drop_seconds)
        t = t[mask]
        u = u[mask]
        y = y[mask]

    # Limit duration if specified
    if time_duration is not None:
        t_end = t[0] + time_duration
        mask = t <= t_end
        t = t[mask]
        u = u[mask]
        y = y[mask]

    return t, u, y


def compute_fourier_transform(
    t: np.ndarray,
    signal: np.ndarray,
    n_freq: int = None,
    window: str = 'hann',
    detrend: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Fourier transform of a signal.

    Args:
        t: Time array
        signal: Signal to transform
        n_freq: Number of frequency points (None = auto)
        window: Window function name
        detrend: Whether to remove mean before FFT

    Returns:
        (frequencies, complex spectrum)
    """
    # Sample rate and number of samples
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    n = len(signal)

    # Remove mean if requested
    if detrend:
        signal = signal - np.mean(signal)

    # Apply window
    if window is not None:
        win = get_window(window, n)
        signal = signal * win
        # Compensate for window power loss
        win_correction = np.sum(win) / n
        signal = signal / win_correction

    # Compute FFT
    if n_freq is None:
        n_freq = n

    spectrum = fft(signal, n=n_freq)
    frequencies = fftfreq(n_freq, dt)

    # Keep only positive frequencies
    positive_mask = frequencies >= 0
    frequencies = frequencies[positive_mask]
    spectrum = spectrum[positive_mask]

    # Scale by dt for physical units
    spectrum = spectrum * dt

    return frequencies, spectrum


def compute_transfer_function(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    n_freq: int = None,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer function G(f) = Y(f) / U(f) using Fourier transforms.

    Args:
        t: Time array
        u: Input signal
        y: Output signal
        n_freq: Number of frequency points
        window: Window function

    Returns:
        (frequencies [Hz], complex transfer function)
    """
    # Compute Fourier transforms
    freq_u, U_spectrum = compute_fourier_transform(t, u, n_freq, window)
    freq_y, Y_spectrum = compute_fourier_transform(t, y, n_freq, window)

    # Ensure frequencies match
    assert np.allclose(freq_u, freq_y), "Frequency arrays do not match"

    # Compute transfer function with regularization
    eps = 1e-10
    G = Y_spectrum / (U_spectrum + eps)

    # Set to zero where input spectrum is too small
    small_input = np.abs(U_spectrum) < (eps * 10)
    G[small_input] = 0.0 + 0.0j

    return freq_u, G


def linear_frequency_grid(
    f_min: float,
    f_max: float,
    n_points: int
) -> np.ndarray:
    """
    Create a linear frequency grid (not logarithmic).

    Args:
        f_min: Minimum frequency [Hz]
        f_max: Maximum frequency [Hz]
        n_points: Number of points

    Returns:
        Linear frequency array [Hz]
    """
    return np.linspace(f_min, f_max, n_points)


def interpolate_to_grid(
    f_original: np.ndarray,
    G_original: np.ndarray,
    f_grid: np.ndarray
) -> np.ndarray:
    """
    Interpolate complex transfer function to new frequency grid.

    Args:
        f_original: Original frequencies
        G_original: Original transfer function
        f_grid: Target frequency grid

    Returns:
        Interpolated transfer function
    """
    # Interpolate real and imaginary parts separately
    real_interp = np.interp(f_grid, f_original, np.real(G_original))
    imag_interp = np.interp(f_grid, f_original, np.imag(G_original))

    return real_interp + 1j * imag_interp


def average_multiple_estimates(
    G_list: List[np.ndarray]
) -> np.ndarray:
    """
    Average multiple transfer function estimates.

    Args:
        G_list: List of complex transfer functions

    Returns:
        Averaged transfer function
    """
    # Simple averaging in complex domain
    G_stack = np.vstack(G_list)
    return np.mean(G_stack, axis=0)


def save_results(
    frequencies: np.ndarray,
    G: np.ndarray,
    output_prefix: Path,
    save_plots: bool = True
) -> None:
    """
    Save Fourier transform results to files.

    Args:
        frequencies: Frequency array [Hz]
        G: Complex transfer function
        output_prefix: Output file prefix (no extension)
        save_plots: Whether to generate plots
    """
    # Convert to angular frequency for compatibility
    omega = 2.0 * np.pi * frequencies

    # Save CSV (compatible with unified_pipeline.py)
    df = pd.DataFrame({
        'omega_rad_s': omega,
        'freq_Hz': frequencies,
        'ReG': np.real(G),
        'ImG': np.imag(G),
        'absG': np.abs(G),
        'phase_rad': np.angle(G)
    })
    csv_path = str(output_prefix) + '_fft.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved FFT results to: {csv_path}")

    # Save MAT file
    mat_data = {
        'f': frequencies,
        'omega': omega,
        'G': G,
        'method': 'fourier_transform'
    }
    mat_path = str(output_prefix) + '_fft.mat'
    savemat(mat_path, mat_data)

    # Generate plots if requested
    if save_plots:
        plot_fourier_results(frequencies, G, output_prefix)


def plot_fourier_results(
    frequencies: np.ndarray,
    G: np.ndarray,
    output_prefix: Path
) -> None:
    """
    Generate plots of Fourier transform results.

    Args:
        frequencies: Frequency array [Hz]
        G: Complex transfer function
        output_prefix: Output file prefix
    """
    # Magnitude plot (linear scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequencies, np.abs(G), 'b-', linewidth=2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|G(f)|')
    ax.set_title('Transfer Function Magnitude (Linear Scale)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(frequencies[0], frequencies[-1])
    plt.tight_layout()
    plt.savefig(str(output_prefix) + '_magnitude_linear.png', dpi=300)
    plt.close()

    # Phase plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequencies, np.unwrap(np.angle(G)), 'r-', linewidth=2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [rad]')
    ax.set_title('Transfer Function Phase')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(frequencies[0], frequencies[-1])
    plt.tight_layout()
    plt.savefig(str(output_prefix) + '_phase.png', dpi=300)
    plt.close()

    # Nyquist plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(np.real(G), np.imag(G), 'g-', linewidth=2)
    ax.plot(np.real(G[0]), np.imag(G[0]), 'ro', markersize=8, label='Start')
    ax.plot(np.real(G[-1]), np.imag(G[-1]), 'rs', markersize=8, label='End')
    ax.set_xlabel('Real{G(f)}')
    ax.set_ylabel('Imag{G(f)}')
    ax.set_title('Nyquist Plot')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_prefix) + '_nyquist.png', dpi=300)
    plt.close()

    # Power spectrum plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Magnitude spectrum in dB
    mag_db = 20.0 * np.log10(np.abs(G) + 1e-10)
    ax1.plot(frequencies, mag_db, 'b-', linewidth=2)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('Transfer Function Magnitude Spectrum')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(frequencies[0], frequencies[-1])

    # Phase spectrum
    ax2.plot(frequencies, np.angle(G), 'r-', linewidth=2)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [rad]')
    ax2.set_title('Transfer Function Phase Spectrum')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(frequencies[0], frequencies[-1])

    plt.tight_layout()
    plt.savefig(str(output_prefix) + '_spectrum.png', dpi=300)
    plt.close()


def main():
    """Main function for Fourier transform analysis."""
    parser = argparse.ArgumentParser(
        description='Fourier Transform-based frequency analysis for system identification'
    )

    # Input arguments
    parser.add_argument(
        'mat_files',
        nargs='+',
        help='MAT files containing time-domain data'
    )

    # Processing options
    parser.add_argument(
        '--n-files',
        type=int,
        default=None,
        help='Number of files to process (default: all)'
    )
    parser.add_argument(
        '--nd',
        type=int,
        default=100,
        help='Number of frequency points (default: 100)'
    )
    parser.add_argument(
        '--f-min',
        type=float,
        default=0.0,
        help='Minimum frequency [Hz] (default: 0.0)'
    )
    parser.add_argument(
        '--f-max',
        type=float,
        default=None,
        help='Maximum frequency [Hz] (default: Nyquist frequency)'
    )
    parser.add_argument(
        '--drop-seconds',
        type=float,
        default=0.0,
        help='Seconds to drop from start (default: 0.0)'
    )
    parser.add_argument(
        '--time-duration',
        type=float,
        default=None,
        help='Time duration to use [s] (default: all)'
    )
    parser.add_argument(
        '--window',
        type=str,
        default='hann',
        choices=['hann', 'hamming', 'blackman', 'bartlett', 'none'],
        help='Window function (default: hann)'
    )
    parser.add_argument(
        '--y-col',
        type=int,
        default=0,
        help='Column index if y is 2D (default: 0)'
    )

    # Output options
    parser.add_argument(
        '--out-dir',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--out-prefix',
        type=str,
        default='fourier',
        help='Output file prefix (default: fourier)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    args = parser.parse_args()

    # Resolve input files
    mat_files = []
    for pattern in args.mat_files:
        mat_files.extend(glob.glob(pattern))

    if not mat_files:
        print("Error: No MAT files found")
        return 1

    # Limit number of files if specified
    if args.n_files is not None:
        mat_files = mat_files[:args.n_files]

    print(f"Processing {len(mat_files)} file(s)")

    # Process each file
    G_list = []
    f_max_actual = 0.0

    for i, mat_file in enumerate(mat_files):
        print(f"\nProcessing {Path(mat_file).name}...")

        # Load data
        t, u, y = load_time_data(Path(mat_file), args.y_col)

        # Apply time window
        t, u, y = apply_time_window(t, u, y, args.drop_seconds, args.time_duration)

        print(f"  Samples: {len(t)}")
        print(f"  Duration: {t[-1] - t[0]:.2f} s")
        print(f"  Sample rate: {1.0/np.mean(np.diff(t)):.2f} Hz")

        # Compute transfer function
        window = None if args.window == 'none' else args.window
        freq, G = compute_transfer_function(t, u, y, window=window)

        # Update maximum frequency
        f_max_actual = max(f_max_actual, freq[-1])

        G_list.append((freq, G))

    # Determine frequency grid
    if args.f_max is None:
        f_max = f_max_actual
    else:
        f_max = min(args.f_max, f_max_actual)

    f_grid = linear_frequency_grid(args.f_min, f_max, args.nd)
    print(f"\nFrequency grid: {args.f_min:.2f} - {f_max:.2f} Hz ({args.nd} points)")

    # Interpolate all estimates to common grid
    G_interp_list = []
    for freq, G in G_list:
        G_interp = interpolate_to_grid(freq, G, f_grid)
        G_interp_list.append(G_interp)

    # Average if multiple files
    if len(G_interp_list) > 1:
        print("Averaging multiple estimates...")
        G_avg = average_multiple_estimates(G_interp_list)
    else:
        G_avg = G_interp_list[0]

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    output_prefix = output_dir / args.out_prefix
    save_results(f_grid, G_avg, output_prefix, save_plots=not args.no_plots)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())