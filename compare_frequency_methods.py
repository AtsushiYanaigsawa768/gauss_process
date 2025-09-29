#!/usr/bin/env python3
"""
compare_frequency_methods.py

Compare the results from FRF and Fourier transform methods side by side.
This script loads the results from both methods and creates comparison plots.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_frequency_data(csv_path: Path) -> pd.DataFrame:
    """Load frequency domain data from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def compare_methods(frf_dir: str = "output_test_frf", fourier_dir: str = "output_test_fourier"):
    """Compare FRF and Fourier transform results."""
    print("Comparing Frequency Analysis Methods")
    print("="*60)

    # Find FRF results
    frf_path = Path(frf_dir)
    frf_csv = frf_path / "unified_frf.csv"
    if not frf_csv.exists():
        print(f"Error: FRF results not found at {frf_csv}")
        return 1

    # Find Fourier results
    fourier_path = Path(fourier_dir)
    fourier_csv = fourier_path / "unified_fft.csv"
    if not fourier_csv.exists():
        print(f"Error: Fourier results not found at {fourier_csv}")
        return 1

    # Load data
    print("Loading FRF data...")
    frf_df = load_frequency_data(frf_csv)

    print("Loading Fourier transform data...")
    fourier_df = load_frequency_data(fourier_csv)

    # Extract data
    omega_frf = frf_df['omega_rad_s'].values
    G_frf = frf_df['ReG'].values + 1j * frf_df['ImG'].values

    omega_fourier = fourier_df['omega_rad_s'].values
    G_fourier = fourier_df['ReG'].values + 1j * fourier_df['ImG'].values

    print(f"\nFRF points: {len(omega_frf)}")
    print(f"Fourier points: {len(omega_fourier)}")
    print(f"FRF frequency range: {omega_frf[0]/(2*np.pi):.2f} - {omega_frf[-1]/(2*np.pi):.2f} Hz")
    print(f"Fourier frequency range: {omega_fourier[0]/(2*np.pi):.2f} - {omega_fourier[-1]/(2*np.pi):.2f} Hz")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Magnitude comparison
    ax = axes[0, 0]
    ax.loglog(omega_frf/(2*np.pi), np.abs(G_frf), 'b.-', label='FRF', markersize=8, alpha=0.7)
    ax.loglog(omega_fourier/(2*np.pi), np.abs(G_fourier), 'r.-', label='Fourier', markersize=8, alpha=0.7)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|G|')
    ax.set_title('Magnitude Comparison (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase comparison
    ax = axes[0, 1]
    ax.semilogx(omega_frf/(2*np.pi), np.unwrap(np.angle(G_frf)), 'b.-', label='FRF', markersize=8, alpha=0.7)
    ax.semilogx(omega_fourier/(2*np.pi), np.unwrap(np.angle(G_fourier)), 'r.-', label='Fourier', markersize=8, alpha=0.7)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [rad]')
    ax.set_title('Phase Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Nyquist comparison
    ax = axes[1, 0]
    ax.plot(np.real(G_frf), np.imag(G_frf), 'b.-', label='FRF', markersize=6, alpha=0.7)
    ax.plot(np.real(G_fourier), np.imag(G_fourier), 'r.-', label='Fourier', markersize=6, alpha=0.7)
    ax.set_xlabel('Real{G}')
    ax.set_ylabel('Imag{G}')
    ax.set_title('Nyquist Plot Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Linear magnitude comparison (Fourier feature)
    ax = axes[1, 1]
    freq_frf = omega_frf / (2 * np.pi)
    freq_fourier = omega_fourier / (2 * np.pi)
    ax.plot(freq_frf, np.abs(G_frf), 'b.-', label='FRF', markersize=8, alpha=0.7)
    ax.plot(freq_fourier, np.abs(G_fourier), 'r.-', label='Fourier', markersize=8, alpha=0.7)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|G|')
    ax.set_title('Magnitude Comparison (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('frequency_methods_comparison.png', dpi=300)
    plt.show()

    # Compute differences
    print("\n" + "="*60)
    print("Key Differences:")
    print("="*60)

    # Frequency scale difference
    print(f"\n1. Frequency Scale:")
    print(f"   - FRF uses logarithmic frequency spacing")
    print(f"   - Fourier uses linear frequency spacing")

    # Interpolate to common grid for comparison
    common_omega = np.linspace(
        max(omega_frf[0], omega_fourier[0]),
        min(omega_frf[-1], omega_fourier[-1]),
        100
    )

    G_frf_interp = np.interp(common_omega, omega_frf, np.abs(G_frf))
    G_fourier_interp = np.interp(common_omega, omega_fourier, np.abs(G_fourier))

    # Compute RMS difference
    rms_diff = np.sqrt(np.mean((G_frf_interp - G_fourier_interp)**2))
    rel_diff = rms_diff / np.mean(G_frf_interp)

    print(f"\n2. Magnitude Difference:")
    print(f"   - RMS difference: {rms_diff:.3e}")
    print(f"   - Relative difference: {rel_diff*100:.2f}%")

    print("\n3. Processing Characteristics:")
    print("   - FRF: Cross-power spectral estimation with windowing")
    print("   - Fourier: Direct FFT with optional windowing")

    print("="*60)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare FRF and Fourier transform results")
    parser.add_argument('--frf-dir', default='output_test_frf', help='FRF output directory')
    parser.add_argument('--fourier-dir', default='output_test_fourier', help='Fourier output directory')

    args = parser.parse_args()

    sys.exit(compare_methods(args.frf_dir, args.fourier_dir))