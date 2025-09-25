#!/usr/bin/env python3
"""
gp_fir_model.py

Integration module that converts GP-smoothed frequency response functions
to FIR filter coefficients and validates the resulting model.

This bridges the gap between GP regression in frequency domain and
time-domain FIR model identification.

Key features:
- Convert GP-predicted FRF to FIR coefficients via IFFT
- Handle complex interpolation and Hermitian symmetry
- Validate FIR model against time-series data
- Compare with kernel-regularized FIR from pure_fir_model.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
from scipy import signal


# =====================
# FIR Conversion Functions
# =====================

def frf_to_fir(omega: np.ndarray, G: np.ndarray, L: int,
               method: str = 'ifft', window: Optional[str] = None) -> np.ndarray:
    """Convert frequency response function to FIR coefficients.

    Args:
        omega: Angular frequencies [rad/s]
        G: Complex frequency response
        L: Desired FIR filter length
        method: 'ifft' or 'frequency_sampling'
        window: Window function name (e.g., 'hamming', 'hann')

    Returns:
        g: FIR coefficients of length L
    """
    if method == 'ifft':
        # Method 1: IFFT with proper interpolation
        g = _frf_to_fir_ifft(omega, G, L, window)
    elif method == 'frequency_sampling':
        # Method 2: Frequency sampling design
        g = _frf_to_fir_freq_sampling(omega, G, L)
    else:
        raise ValueError(f"Unknown method: {method}")

    return g


def _frf_to_fir_ifft(omega: np.ndarray, G: np.ndarray, L: int,
                     window: Optional[str] = None) -> np.ndarray:
    """Convert FRF to FIR using IFFT with Hermitian symmetry."""

    # Ensure omega starts from 0
    if omega[0] > 1e-10:
        omega = np.concatenate([[0], omega])
        G = np.concatenate([[G[0]], G])  # DC extrapolation

    # Determine FFT size (at least 2*L for good resolution)
    N = max(2 * L, 512)

    # Create uniform frequency grid
    omega_uniform = np.linspace(0, omega[-1], N // 2)

    # Interpolate G to uniform grid
    interp_real = interp1d(omega, np.real(G), kind='cubic', fill_value='extrapolate')
    interp_imag = interp1d(omega, np.imag(G), kind='cubic', fill_value='extrapolate')

    G_uniform = interp_real(omega_uniform) + 1j * interp_imag(omega_uniform)

    # Build full frequency response with Hermitian symmetry
    G_full = np.zeros(N, dtype=complex)
    G_full[:N//2] = G_uniform
    G_full[N//2] = np.real(G_uniform[-1])  # Nyquist must be real
    G_full[N//2+1:] = np.conj(G_uniform[-2:0:-1])  # Mirror and conjugate

    # IFFT to get impulse response
    h = np.fft.ifft(G_full).real

    # Truncate to desired length
    g = h[:L]

    # Apply window if specified
    if window is not None:
        w = signal.get_window(window, L)
        g = g * w

    return g


def _frf_to_fir_freq_sampling(omega: np.ndarray, G: np.ndarray, L: int) -> np.ndarray:
    """Convert FRF to FIR using frequency sampling method."""

    # Create frequency sampling grid
    N = 2 * L  # Typically use N = 2L for frequency sampling
    k = np.arange(N // 2 + 1)
    omega_k = 2 * np.pi * k / N * (omega[-1] / np.pi)  # Scale to match omega range

    # Interpolate G to sampling frequencies
    interp_real = interp1d(omega, np.real(G), kind='cubic', bounds_error=False, fill_value=0)
    interp_imag = interp1d(omega, np.imag(G), kind='cubic', bounds_error=False, fill_value=0)

    G_k = interp_real(omega_k) + 1j * interp_imag(omega_k)

    # Build full DFT samples
    G_full = np.zeros(N, dtype=complex)
    G_full[:N//2+1] = G_k
    G_full[N//2+1:] = np.conj(G_k[-2:0:-1])

    # IDFT to get impulse response
    h = np.fft.ifft(G_full).real

    # Use first L samples as FIR coefficients
    g = h[:L]

    return g


# =====================
# Validation Functions
# =====================

def validate_fir_model(g: np.ndarray, u: np.ndarray, y: np.ndarray,
                      start_idx: int = 0) -> Dict[str, float]:
    """Validate FIR model against measured input/output data.

    Args:
        g: FIR coefficients
        u: Input signal
        y: Output signal
        start_idx: Start index for validation (skip initial transient)

    Returns:
        Dictionary with validation metrics
    """
    L = len(g)
    N = len(u)

    # Compute predicted output
    y_pred = np.convolve(u, g, mode='full')[:N]

    # Skip initial transient
    y_val = y[start_idx:]
    y_pred_val = y_pred[start_idx:]

    # Compute metrics
    e = y_val - y_pred_val
    rmse = float(np.sqrt(np.mean(e**2)))

    # Normalized RMS error (FIT percentage)
    y_mean = np.mean(y_val)
    fit = float(100 * (1 - np.linalg.norm(e) / np.linalg.norm(y_val - y_mean)))

    # R-squared
    ss_res = np.sum(e**2)
    ss_tot = np.sum((y_val - y_mean)**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Peak error
    peak_error = float(np.max(np.abs(e)))

    return {
        'rmse': rmse,
        'fit_percent': fit,
        'r2': r2,
        'peak_error': peak_error,
        'fir_length': L
    }


def plot_fir_validation(t: np.ndarray, u: np.ndarray, y: np.ndarray,
                       g: np.ndarray, output_prefix: Path, title_suffix: str = ""):
    """Create validation plots for FIR model."""
    # Compute predicted output
    y_pred = np.convolve(u, g, mode='full')[:len(u)]

    # Validation metrics
    L = len(g)
    metrics = validate_fir_model(g, u, y, start_idx=L)

    # Figure 1: Output comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(t, y, 'k-', label='Measured', linewidth=1.5, alpha=0.7)
    ax1.plot(t, y_pred, 'r--', label='FIR Predicted', linewidth=1.5)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Output')
    ax1.set_title(f'FIR Model Validation{title_suffix} (L={L}, FIT={metrics["fit_percent"]:.1f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zoom to interesting region
    if len(t) > 1000:
        ax1.set_xlim([t[L], min(t[L] + (t[-1] - t[0]) * 0.2, t[-1])])

    ax2.plot(t, y - y_pred, 'b-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title(f'Error (RMSE={metrics["rmse"]:.3e}, R²={metrics["r2"]:.3f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_validation.png", dpi=300)
    plt.close()

    # Figure 2: Impulse response
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    k = np.arange(len(g))
    ax1.stem(k, g, linefmt='C0-', markerfmt='C0o', basefmt='k-')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'FIR Impulse Response{title_suffix}')
    ax1.grid(True, alpha=0.3)

    # Frequency response of FIR
    w, h = signal.freqz(g, worN=512)
    ax2.loglog(w, np.abs(h), 'b-', linewidth=2)
    ax2.set_xlabel('Normalized Frequency [rad/sample]')
    ax2.set_ylabel('|H(e^{jω})|')
    ax2.set_title('FIR Frequency Response')
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_impulse.png", dpi=300)
    plt.close()

    return metrics


# =====================
# GP-FIR Pipeline
# =====================

@dataclass
class GPFIRConfig:
    """Configuration for GP-FIR conversion."""
    frf_csv: Path  # GP-smoothed FRF from unified_pipeline.py
    io_mat: Optional[Path] = None  # Time-series data for validation
    L: int = 128  # FIR length
    method: str = 'ifft'  # Conversion method
    window: Optional[str] = None  # Window function
    out_dir: Path = Path('gp_fir_output')


def run_gp_fir_pipeline(config: GPFIRConfig) -> Dict[str, object]:
    """Convert GP-smoothed FRF to FIR and validate if time-series data available."""
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # Load GP-smoothed FRF
    print(f"Loading GP-smoothed FRF from {config.frf_csv}")
    frf_df = pd.read_csv(config.frf_csv)

    omega = frf_df['omega_rad_s'].values
    G = frf_df['ReG'].values + 1j * frf_df['ImG'].values

    # Convert to FIR
    print(f"Converting FRF to FIR (L={config.L}, method={config.method})")
    g = frf_to_fir(omega, G, config.L, method=config.method, window=config.window)

    # Save FIR coefficients
    fir_path = config.out_dir / 'gp_fir_coefficients.npz'
    np.savez(fir_path, g=g, L=config.L, omega=omega, G=G, method=config.method)
    print(f"FIR coefficients saved to {fir_path}")

    results = {
        'fir_length': config.L,
        'method': config.method,
        'window': config.window,
        'fir_path': str(fir_path)
    }

    # Validate against time-series data if available
    if config.io_mat and config.io_mat.exists():
        print(f"Validating FIR model using {config.io_mat}")

        # Load I/O data
        io_data = loadmat(config.io_mat)
        # Try different possible variable names
        for var_name in ['t', 'time']:
            if var_name in io_data:
                t = io_data[var_name].ravel()
                break
        else:
            # Try to extract from matrix format
            for key, value in io_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.shape[0] == 3 or value.shape[1] == 3:
                        if value.shape[0] == 3:
                            t = value[0, :].ravel()
                            y = value[1, :].ravel()
                            u = value[2, :].ravel()
                        else:
                            t = value[:, 0].ravel()
                            y = value[:, 1].ravel()
                            u = value[:, 2].ravel()
                        break
            else:
                raise ValueError(f"Could not find time-series data in {config.io_mat}")

        # Ensure proper data types
        t = t.astype(float)
        u = u.astype(float)
        y = y.astype(float)

        # Remove mean if present
        u = u - np.mean(u)
        y = y - np.mean(y)

        # Validate
        metrics = plot_fir_validation(t, u, y, g, config.out_dir / 'gp_fir',
                                    title_suffix=" (GP-smoothed)")
        results.update(metrics)

        # Compare with different FIR lengths
        compare_fir_lengths(omega, G, u, y, t, config.out_dir)

    # Create summary plot of GP->FIR process
    plot_gp_fir_summary(omega, G, g, config.out_dir)

    # Save results
    with open(config.out_dir / 'gp_fir_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def compare_fir_lengths(omega: np.ndarray, G: np.ndarray, u: np.ndarray,
                       y: np.ndarray, t: np.ndarray, output_dir: Path):
    """Compare performance for different FIR lengths."""
    L_values = [32, 64, 128, 256, 512]
    metrics_list = []

    for L in L_values:
        if L > len(u) // 4:  # Skip if too long
            continue

        g = frf_to_fir(omega, G, L, method='ifft')
        metrics = validate_fir_model(g, u, y, start_idx=L)
        metrics['L'] = L
        metrics_list.append(metrics)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    L_vals = [m['L'] for m in metrics_list]
    fit_vals = [m['fit_percent'] for m in metrics_list]
    rmse_vals = [m['rmse'] for m in metrics_list]

    ax1.plot(L_vals, fit_vals, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('FIR Length L')
    ax1.set_ylabel('FIT [%]')
    ax1.set_title('Model Fit vs FIR Length')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(L_vals, rmse_vals, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('FIR Length L')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs FIR Length')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fir_length_comparison.png', dpi=300)
    plt.close()


def plot_gp_fir_summary(omega: np.ndarray, G: np.ndarray, g: np.ndarray, output_dir: Path):
    """Create summary plot showing GP->FIR conversion process."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Original FRF magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(omega, np.abs(G), 'b-', linewidth=2)
    ax1.set_xlabel(r'$\omega$ [rad/s]')
    ax1.set_ylabel('|G(jω)|')
    ax1.set_title('GP-Smoothed FRF Magnitude')
    ax1.grid(True, which='both', alpha=0.3)

    # 2. Original FRF phase
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(omega, np.unwrap(np.angle(G)), 'r-', linewidth=2)
    ax2.set_xlabel(r'$\omega$ [rad/s]')
    ax2.set_ylabel('Phase [rad]')
    ax2.set_title('GP-Smoothed FRF Phase')
    ax2.grid(True, which='both', alpha=0.3)

    # 3. Nyquist plot
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(np.real(G), np.imag(G), 'g-', linewidth=2)
    ax3.set_xlabel('Real{G(jω)}')
    ax3.set_ylabel('Imag{G(jω)}')
    ax3.set_title('Nyquist Plot of GP-Smoothed FRF')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # 4. FIR impulse response
    ax4 = fig.add_subplot(gs[2, 0])
    k = np.arange(len(g))
    ax4.stem(k[:50], g[:50], linefmt='C0-', markerfmt='C0o', basefmt='k-')
    ax4.set_xlabel('Sample k')
    ax4.set_ylabel('g[k]')
    ax4.set_title(f'FIR Impulse Response (first 50 of {len(g)} samples)')
    ax4.grid(True, alpha=0.3)

    # 5. FIR frequency response vs original
    ax5 = fig.add_subplot(gs[2, 1])
    w_fir, h_fir = signal.freqz(g, worN=512)
    # Convert to same frequency scale
    omega_fir = w_fir * (omega[-1] / np.pi)
    ax5.loglog(omega, np.abs(G), 'b-', linewidth=2, label='GP-Smoothed')
    ax5.loglog(omega_fir, np.abs(h_fir), 'r--', linewidth=2, label='FIR Model')
    ax5.set_xlabel(r'$\omega$ [rad/s]')
    ax5.set_ylabel('|H(jω)|')
    ax5.set_title('FRF Comparison')
    ax5.legend()
    ax5.grid(True, which='both', alpha=0.3)

    plt.suptitle('GP to FIR Model Conversion Summary', fontsize=14)
    plt.savefig(output_dir / 'gp_fir_summary.png', dpi=300)
    plt.close()


# =====================
# Integration with pure_fir_model
# =====================

def compare_with_kernel_fir(config: GPFIRConfig, kernel_type: str = 'dc') -> Dict[str, object]:
    """Compare GP-based FIR with kernel-regularized FIR from pure_fir_model."""
    try:
        from pure_fir_model import FIRConfig, run_fir_experiment
    except ImportError:
        print("Warning: pure_fir_model.py not found, skipping comparison")
        return {}

    if not config.io_mat or not config.io_mat.exists():
        print("No I/O data available for comparison")
        return {}

    # Run kernel-regularized FIR
    fir_config = FIRConfig(
        io_mat=config.io_mat,
        out_dir=config.out_dir / f'kernel_{kernel_type}',
        kernel=kernel_type,
        L=config.L,
        multi_starts=3,
        optimize=True
    )

    print(f"Running kernel-regularized FIR with {kernel_type} kernel...")
    kernel_results = run_fir_experiment(fir_config)

    # Load kernel FIR coefficients
    kernel_fir_data = np.load(config.out_dir / f'kernel_{kernel_type}' / 'fir_coefficients.npz')
    g_kernel = kernel_fir_data['g_hat']

    # Load GP FIR coefficients
    gp_fir_data = np.load(config.out_dir / 'gp_fir_coefficients.npz')
    g_gp = gp_fir_data['g']

    # Compare impulse responses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    k = np.arange(max(len(g_gp), len(g_kernel)))
    ax1.stem(k[:len(g_gp)], g_gp, linefmt='C0-', markerfmt='C0o',
             basefmt='k-', label=f'GP-based (L={len(g_gp)})')
    ax1.stem(k[:len(g_kernel)], g_kernel, linefmt='C1-', markerfmt='C1s',
             basefmt='k-', label=f'{kernel_type.upper()} kernel (L={len(g_kernel)})')
    ax1.set_xlabel('Sample k')
    ax1.set_ylabel('g[k]')
    ax1.set_title('FIR Impulse Response Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Frequency response comparison
    w_gp, h_gp = signal.freqz(g_gp, worN=512)
    w_kernel, h_kernel = signal.freqz(g_kernel, worN=512)

    ax2.loglog(w_gp, np.abs(h_gp), 'b-', linewidth=2, label='GP-based FIR')
    ax2.loglog(w_kernel, np.abs(h_kernel), 'r--', linewidth=2,
               label=f'{kernel_type.upper()} kernel FIR')
    ax2.set_xlabel('Normalized Frequency [rad/sample]')
    ax2.set_ylabel('|H(e^{jω})|')
    ax2.set_title('FIR Frequency Response Comparison')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.out_dir / 'fir_comparison.png', dpi=300)
    plt.close()

    comparison = {
        'gp_fit': kernel_results.get('fit_percent', 0),
        'kernel_fit': kernel_results.get('fit_percent', 0),
        'gp_rmse': kernel_results.get('rmse', 0),
        'kernel_rmse': kernel_results.get('rmse', 0),
        'kernel_type': kernel_type
    }

    return comparison


# =====================
# Main Function
# =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert GP-smoothed FRF to FIR model"
    )
    parser.add_argument('--frf-csv', type=str, required=True,
                      help='CSV file with GP-smoothed FRF')
    parser.add_argument('--io-mat', type=str, default=None,
                      help='MAT file with time-series I/O data for validation')
    parser.add_argument('--L', type=int, default=128,
                      help='FIR filter length (default: 128)')
    parser.add_argument('--method', type=str, default='ifft',
                      choices=['ifft', 'frequency_sampling'],
                      help='FRF to FIR conversion method')
    parser.add_argument('--window', type=str, default=None,
                      choices=['hamming', 'hann', 'blackman', None],
                      help='Window function for FIR design')
    parser.add_argument('--out-dir', type=str, default='gp_fir_output',
                      help='Output directory')
    parser.add_argument('--compare-kernel', type=str, default=None,
                      choices=['dc', 'ss', 'si'],
                      help='Compare with kernel-regularized FIR')

    args = parser.parse_args()

    config = GPFIRConfig(
        frf_csv=Path(args.frf_csv),
        io_mat=Path(args.io_mat) if args.io_mat else None,
        L=args.L,
        method=args.method,
        window=args.window,
        out_dir=Path(args.out_dir)
    )

    # Run main pipeline
    results = run_gp_fir_pipeline(config)
    print(f"\nGP-FIR conversion complete. Results: {json.dumps(results, indent=2)}")

    # Optional comparison
    if args.compare_kernel:
        comparison = compare_with_kernel_fir(config, args.compare_kernel)
        if comparison:
            print(f"\nComparison with {args.compare_kernel} kernel FIR:")
            print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()