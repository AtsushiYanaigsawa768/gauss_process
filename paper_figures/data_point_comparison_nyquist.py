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

All plots use synchronous demodulation method (matching frequency_response.py).

Usage:
    python paper_figures/data_point_comparison_nyquist.py input/input_test_20250912_165937.mat
    python paper_figures/data_point_comparison_nyquist.py input/*.mat
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
                          nd: int, prefix: str = 'frf') -> Path:
    """
    Run frequency_response.py and return path to output CSV.

    Uses 1 file with full duration (1 hour).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        'src/frequency_response.py',
        mat_files[0],  # Use only the first file
        '--n-files', '1',
        '--out-dir', str(output_dir),
        '--out-prefix', prefix,
        '--nd', str(nd)
    ]

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

    # Temporary directory for intermediate FRF CSV files
    temp_dir = output_dir / 'temp_frf_datapoints'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Condition 1: 10 data points
    # ========================================
    print(f"\n{'='*60}")
    print("Condition 1: 10 Frequency Points (nd=10)")
    print(f"{'='*60}")

    frf_csv_10pt = run_frequency_response(
        mat_files=mat_files,
        output_dir=temp_dir,
        nd=10,
        prefix='10pt'
    )

    omega_10pt, G_10pt = load_frf_data(frf_csv_10pt)

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

    frf_csv_30pt = run_frequency_response(
        mat_files=mat_files,
        output_dir=temp_dir,
        nd=30,
        prefix='30pt'
    )

    omega_30pt, G_30pt = load_frf_data(frf_csv_30pt)

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

    frf_csv_50pt = run_frequency_response(
        mat_files=mat_files,
        output_dir=temp_dir,
        nd=50,
        prefix='50pt'
    )

    omega_50pt, G_50pt = load_frf_data(frf_csv_50pt)

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

    frf_csv_100pt = run_frequency_response(
        mat_files=mat_files,
        output_dir=temp_dir,
        nd=100,
        prefix='100pt'
    )

    omega_100pt, G_100pt = load_frf_data(frf_csv_100pt)

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
    print(f"  - Synchronous demodulation method")
    print(f"  - Publication-quality styling (300 DPI PNG + EPS)")
    print(f"  - Markers only (star markers, no lines)")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
