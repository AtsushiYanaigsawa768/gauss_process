#!/usr/bin/env python3
"""
time_comparison_nyquist.py

Generate Nyquist plots comparing different data durations for paper.

This script creates three Nyquist plots using logarithmic frequency grid:
1. 10 minutes: 1 MAT file, 600 seconds
2. 1 hour: 1 MAT file, full duration (~3600 seconds)
3. 10 hours: 10 MAT files, full duration each

All plots use synchronous demodulation method (matching frequency_response.py).

Usage:
    python paper_figures/time_comparison_nyquist.py input/*.mat
    python paper_figures/time_comparison_nyquist.py input/*.mat --nd 100
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd

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


def run_frequency_response(mat_files: List[str], output_dir: Path,
                          n_files: int = 1, time_duration: float = None,
                          nd: int = 100, prefix: str = 'frf') -> Path:
    """
    Run frequency_response.py and return path to output CSV.

    This mimics the unified_pipeline.py approach.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        'src/frequency_response.py',
        *mat_files[:n_files],  # Use only the first n_files
        '--n-files', str(n_files),
        '--out-dir', str(output_dir),
        '--out-prefix', prefix,
        '--nd', str(nd)
    ]

    if time_duration is not None and n_files == 1:
        cmd.extend(['--time-duration', str(time_duration)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"frequency_response.py failed with code {result.returncode}")

    # Print stdout for debugging
    if result.stdout:
        print(result.stdout)

    frf_csv = output_dir / f'{prefix}_frf.csv'
    if not frf_csv.exists():
        raise RuntimeError(f"Expected output file not found: {frf_csv}")

    return frf_csv


def load_frf_data(frf_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load frequency response data from CSV.

    Returns:
        (omega [rad/s], G complex)
    """
    df = pd.read_csv(frf_file)

    required_cols = ['omega_rad_s', 'ReG', 'ImG']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"FRF file must contain columns: {required_cols}")

    omega = df['omega_rad_s'].values
    G_complex = df['ReG'].values + 1j * df['ImG'].values

    return omega, G_complex


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

    # Temporary directory for intermediate FRF CSV files
    temp_dir = output_dir / 'temp_frf'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Condition 1: 10 minutes (600 seconds)
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 1: 10 Minutes (1 file, 600 seconds)")
    print(f"{'='*60}")

    frf_csv_10min = run_frequency_response(
        mat_files=mat_files,
        output_dir=temp_dir,
        n_files=1,
        time_duration=600.0,
        nd=args.nd,
        prefix='10min'
    )

    omega_10min, G_10min = load_frf_data(frf_csv_10min)

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
    # Condition 2: 1 hour (full file, ~3600 seconds)
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 2: 1 Hour (1 file, full duration)")
    print(f"{'='*60}")

    frf_csv_1hour = run_frequency_response(
        mat_files=mat_files,
        output_dir=temp_dir,
        n_files=1,
        time_duration=None,  # Use full duration
        nd=args.nd,
        prefix='1hour'
    )

    omega_1hour, G_1hour = load_frf_data(frf_csv_1hour)

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
    # Condition 3: 10 hours (10 files, full duration each)
    # ========================================
    if len(mat_files) >= 10:
        print(f"\n{'='*60}")
        print("Condition 3: 10 Hours (10 files, full duration each)")
        print(f"{'='*60}")

        frf_csv_10hour = run_frequency_response(
            mat_files=mat_files,
            output_dir=temp_dir,
            n_files=10,
            time_duration=None,
            nd=args.nd,
            prefix='10hour'
        )

        omega_10hour, G_10hour = load_frf_data(frf_csv_10hour)

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
        print(f"Skipping Condition 3: Only {len(mat_files)} files available (need 10)")
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
    print(f"  2. {output_path_1hour}.png (and .eps) - 1 Hour")
    if len(mat_files) >= 10:
        print(f"  3. {output_path_10hour}.png (and .eps) - 10 Hours")
    print(f"\nAll plots use:")
    print(f"  - Logarithmic frequency grid (nd={args.nd})")
    print(f"  - Synchronous demodulation method")
    print(f"  - Publication-quality styling (300 DPI PNG + EPS)")
    print(f"  - Markers only (star markers, no lines)")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
