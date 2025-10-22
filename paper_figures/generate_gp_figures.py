#!/usr/bin/env python3
"""
generate_gp_figures.py

Generate publication-quality figures from GP regression results with FIR model extraction.

This script runs unified_pipeline.py with specific parameters optimized for paper figures:
- FRF method (log scale frequency)
- nd = 100 frequency points
- 1 hour data (n_files=1, time_duration=None)
- Matern 5/2 kernel
- Grid search for hyperparameter optimization
- Outputs 3 types of figures in PNG (300 DPI) + EPS format
- Optional FIR model extraction and validation

Usage:
    python paper_figures/generate_gp_figures.py input/*.mat --output-dir paper_figures/gp_results
    python paper_figures/generate_gp_figures.py input/*.mat --kernel matern52 --nd 100
    python paper_figures/generate_gp_figures.py input/*.mat --extract-fir --fir-validation-mat input/test.mat
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import from unified_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from unified_pipeline import (
    create_kernel, GaussianProcessRegressor, load_frf_data,
    run_frequency_response, configure_plot_style,
    generate_validation_data_from_mat
)

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
})


def plot_gp_real_part(omega: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                     y_std: np.ndarray, output_path: Path, save_eps: bool = True):
    """Plot GP regression results for real part."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot data and predictions
    ax.plot(omega / (2 * np.pi), y_true, 'k.', markersize=10, label='Measured Data', alpha=0.6)
    ax.plot(omega / (2 * np.pi), y_pred, 'r-', linewidth=3.0, label='GP Mean')

    # Plot confidence interval
    if y_std is not None:
        ax.fill_between(omega / (2 * np.pi),
                        y_pred - 2 * y_std,
                        y_pred + 2 * y_std,
                        alpha=0.2, color='red', label='95% Confidence')

    ax.set_xlabel('Frequency [Hz]', fontsize=20)
    ax.set_ylabel('Real{G(jω)}', fontsize=20)
    ax.set_title('GP Regression: Real Part', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

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


def plot_gp_imag_part(omega: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                     y_std: np.ndarray, output_path: Path, save_eps: bool = True):
    """Plot GP regression results for imaginary part."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot data and predictions
    ax.plot(omega / (2 * np.pi), y_true, 'k.', markersize=10, label='Measured Data', alpha=0.6)
    ax.plot(omega / (2 * np.pi), y_pred, 'b-', linewidth=3.0, label='GP Mean')

    # Plot confidence interval
    if y_std is not None:
        ax.fill_between(omega / (2 * np.pi),
                        y_pred - 2 * y_std,
                        y_pred + 2 * y_std,
                        alpha=0.2, color='blue', label='95% Confidence')

    ax.set_xlabel('Frequency [Hz]', fontsize=20)
    ax.set_ylabel('Imag{G(jω)}', fontsize=20)
    ax.set_title('GP Regression: Imaginary Part', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

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


def plot_gp_nyquist(omega: np.ndarray, G_true: np.ndarray, G_pred: np.ndarray,
                   G_std_real: np.ndarray, G_std_imag: np.ndarray,
                   output_path: Path, save_eps: bool = True):
    """Plot Nyquist diagram with GP predictions."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot measured data
    ax.plot(np.real(G_true), np.imag(G_true), 'k.', markersize=12,
            label='Measured Data', alpha=0.6)

    # Plot GP mean
    ax.plot(np.real(G_pred), np.imag(G_pred), 'r-', linewidth=3.5,
            label='GP Mean')

    # Plot confidence ellipses at selected frequencies
    if G_std_real is not None and G_std_imag is not None:
        n_ellipses = min(15, len(omega))
        indices = np.linspace(0, len(omega)-1, n_ellipses, dtype=int)
        for i in indices:
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_x = np.real(G_pred[i]) + 2*G_std_real[i] * np.cos(theta)
            ellipse_y = np.imag(G_pred[i]) + 2*G_std_imag[i] * np.sin(theta)
            ax.plot(ellipse_x, ellipse_y, 'r-', alpha=0.15, linewidth=1.5)

    ax.set_xlabel('Real{G(jω)}', fontsize=20)
    ax.set_ylabel('Imag{G(jω)}', fontsize=20)
    ax.set_title('Nyquist Plot with GP Regression', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.axis('equal')

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
    """Main function to generate GP regression figures."""
    parser = argparse.ArgumentParser(
        description='Generate publication-quality GP regression figures'
    )

    parser.add_argument(
        'mat_files',
        nargs='+',
        help='MAT files for frequency response analysis'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='matern52',
        choices=['rbf', 'matern', 'matern12', 'matern32', 'matern52', 'rq'],
        help='Kernel type (default: matern52)'
    )
    parser.add_argument(
        '--nd',
        type=int,
        default=100,
        help='Number of frequency points (default: 100)'
    )
    parser.add_argument(
        '--n-files',
        type=int,
        default=1,
        help='Number of MAT files to use (default: 1)'
    )
    parser.add_argument(
        '--time-duration',
        type=float,
        default=None,
        help='Time duration in seconds (None = full duration, default: None)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='paper_figures/gp_results',
        help='Output directory (default: paper_figures/gp_results)'
    )
    parser.add_argument(
        '--no-eps',
        action='store_true',
        help='Do not save EPS files'
    )
    parser.add_argument(
        '--no-grid-search',
        action='store_true',
        help='Disable grid search (use gradient-based optimization instead)'
    )
    parser.add_argument(
        '--extract-fir',
        action='store_true',
        help='Extract FIR model coefficients from GP predictions'
    )
    parser.add_argument(
        '--fir-length',
        type=int,
        default=1024,
        help='FIR filter length (default: 1024)'
    )
    parser.add_argument(
        '--fir-validation-mat',
        type=str,
        default=None,
        help='MAT file with [time, output, input] for FIR validation'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("GP Regression Figure Generation for Paper")
    print(f"{'='*80}")
    print(f"Kernel: {args.kernel}")
    print(f"Frequency points (nd): {args.nd}")
    print(f"Data files: {args.n_files}")
    print(f"Time duration: {args.time_duration if args.time_duration else 'Full (1 hour)'}")
    print(f"Grid search: {'Enabled' if not args.no_grid_search else 'Disabled'}")
    print(f"Output directory: {args.output_dir}")
    if args.fir_validation_mat:
        print(f"Validation file: {args.fir_validation_mat}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for FRF data
    temp_dir = output_dir / 'temp_frf'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training data files (exclude validation file if specified)
    training_mat_files = args.mat_files.copy() if isinstance(args.mat_files, list) else list(args.mat_files)

    if args.fir_validation_mat is not None:
        validation_mat_path = Path(args.fir_validation_mat).resolve()

        # Filter out validation file from training files
        original_count = len(training_mat_files)
        training_mat_files = [
            f for f in training_mat_files
            if Path(f).resolve() != validation_mat_path
        ]
        excluded_count = original_count - len(training_mat_files)

        if excluded_count > 0:
            print(f"\n{'='*70}")
            print(f"DATA SEPARATION: Excluded {excluded_count} file(s) from training data")
            print(f"  Validation file: {validation_mat_path}")
            print(f"  Training files remaining: {len(training_mat_files)}")
            print(f"{'='*70}\n")

        if len(training_mat_files) == 0:
            raise ValueError(
                f"\n{'='*70}\n"
                f"ERROR: Training and validation cannot use the same file!\n"
                f"{'='*70}\n"
                f"  Validation file: {validation_mat_path.name}\n"
                f"  Training files: {[Path(f).name for f in args.mat_files]}\n"
                f"\n"
                f"Solutions:\n"
                f"  1. Use DIFFERENT files for training and validation\n"
                f"  2. Or use multiple files and exclude one for validation\n"
                f"{'='*70}\n"
            )

    # Step 1: Run frequency response analysis
    print(f"\n{'='*60}")
    print("Step 1: Frequency Response Analysis (FRF)")
    print(f"{'='*60}")

    frf_csv = run_frequency_response(
        mat_files=training_mat_files,
        output_dir=temp_dir,
        n_files=args.n_files,
        time_duration=args.time_duration,
        nd=args.nd
    )

    # Step 2: Load FRF data
    print(f"\n{'='*60}")
    print("Step 2: Loading Frequency Response Data")
    print(f"{'='*60}")

    frf_df = load_frf_data(frf_csv)
    omega = frf_df['omega_rad_s'].values
    G_complex = frf_df['ReG'].values + 1j * frf_df['ImG'].values

    print(f"Loaded {len(omega)} frequency points")
    print(f"Frequency range: {omega[0]/(2*np.pi):.3f} - {omega[-1]/(2*np.pi):.3f} Hz")

    # Step 3: Prepare data for GP
    print(f"\n{'='*60}")
    print("Step 3: GP Regression Setup")
    print(f"{'='*60}")

    X = omega.reshape(-1, 1)

    # Normalize data
    X_scaler = StandardScaler()
    X_normalized = X_scaler.fit_transform(X)

    y_real_scaler = StandardScaler()
    y_real = y_real_scaler.fit_transform(np.real(G_complex).reshape(-1, 1)).ravel()

    y_imag_scaler = StandardScaler()
    y_imag = y_imag_scaler.fit_transform(np.imag(G_complex).reshape(-1, 1)).ravel()

    print(f"Data normalized using StandardScaler")

    # Step 3.5: Generate validation data if available and grid search is enabled
    use_grid_search = not args.no_grid_search
    validation_X_real = None
    validation_y_real = None
    validation_X_imag = None
    validation_y_imag = None

    if use_grid_search and args.fir_validation_mat is not None:
        print(f"\n{'='*70}")
        print(f"GRID SEARCH VALIDATION DATA (TEST DATA)")
        print(f"{'='*70}")
        print(f"  Test/Validation file: {args.fir_validation_mat}")
        print(f"  Test data points: 150 (fixed, independent of training nd={args.nd})")
        print(f"  NOTE: This file is EXCLUDED from training data")
        print(f"  NOTE: This is the same file used for FIR model evaluation")
        print(f"{'='*70}")
        try:
            # Generate validation data from FIR validation mat file
            # Use 150 points for grid search validation (higher resolution than training)
            # Validation data is always returned in raw (non-normalized) scale
            nd_validation = 150
            omega_val, G_real_val, G_imag_val = generate_validation_data_from_mat(
                args.fir_validation_mat,
                nd=nd_validation,
                freq_method='frf',
            )

            # Prepare validation inputs (same transformation as training data)
            X_val = omega_val.reshape(-1, 1)

            # Normalize X using the SAME scaler as training data
            validation_X_real = X_scaler.transform(X_val)
            validation_X_imag = X_scaler.transform(X_val)

            # Keep Y in RAW (original) scale - do NOT normalize
            # GP predictions will be denormalized before RMSE evaluation
            validation_y_real = G_real_val
            validation_y_imag = G_imag_val

            print(f"  Validation data prepared: {len(validation_y_real)} points")
        except Exception as e:
            print(f"  Warning: Could not load validation data: {e}")
            print(f"  Proceeding with NLL-based grid search instead")

    # Step 4: Create kernel
    print(f"\nCreating {args.kernel} kernel...")
    kernel_real = create_kernel(args.kernel)
    kernel_imag = create_kernel(args.kernel)

    # Step 5: Fit GP for real part
    print(f"\n{'='*60}")
    print("Step 4: GP Regression - Real Part")
    print(f"{'='*60}")

    gp_real = GaussianProcessRegressor(kernel=kernel_real, noise_variance=1e-6)
    gp_real.fit(
        X_normalized, y_real,
        optimize=True,
        n_restarts=3,
        use_grid_search=use_grid_search,
        max_grid_combinations=5000,
        validation_X=validation_X_real,
        validation_y=validation_y_real,
        y_scaler=y_real_scaler
    )

    y_real_pred, y_real_std = gp_real.predict(X_normalized, return_std=True)

    # Denormalize
    y_real_pred = y_real_scaler.inverse_transform(y_real_pred.reshape(-1, 1)).ravel()
    y_real_std = y_real_std * y_real_scaler.scale_
    y_real_orig = y_real_scaler.inverse_transform(y_real.reshape(-1, 1)).ravel()

    # Calculate metrics
    residuals_real = y_real_orig - y_real_pred
    rmse_real = np.sqrt(np.mean(residuals_real**2))
    r2_real = 1 - np.sum(residuals_real**2) / np.sum((y_real_orig - np.mean(y_real_orig))**2)

    print(f"Real part RMSE: {rmse_real:.6e}")
    print(f"Real part R²: {r2_real:.6f}")

    # Step 6: Fit GP for imaginary part
    print(f"\n{'='*60}")
    print("Step 5: GP Regression - Imaginary Part")
    print(f"{'='*60}")

    gp_imag = GaussianProcessRegressor(kernel=kernel_imag, noise_variance=1e-6)
    gp_imag.fit(
        X_normalized, y_imag,
        optimize=True,
        n_restarts=3,
        use_grid_search=use_grid_search,
        max_grid_combinations=5000,
        validation_X=validation_X_imag,
        validation_y=validation_y_imag,
        y_scaler=y_imag_scaler
    )

    y_imag_pred, y_imag_std = gp_imag.predict(X_normalized, return_std=True)

    # Denormalize
    y_imag_pred = y_imag_scaler.inverse_transform(y_imag_pred.reshape(-1, 1)).ravel()
    y_imag_std = y_imag_std * y_imag_scaler.scale_
    y_imag_orig = y_imag_scaler.inverse_transform(y_imag.reshape(-1, 1)).ravel()

    # Calculate metrics
    residuals_imag = y_imag_orig - y_imag_pred
    rmse_imag = np.sqrt(np.mean(residuals_imag**2))
    r2_imag = 1 - np.sum(residuals_imag**2) / np.sum((y_imag_orig - np.mean(y_imag_orig))**2)

    print(f"Imaginary part RMSE: {rmse_imag:.6e}")
    print(f"Imaginary part R²: {r2_imag:.6f}")

    # Step 7: Generate figures
    print(f"\n{'='*60}")
    print("Step 6: Generating Figures")
    print(f"{'='*60}")

    # Figure 1: Real part
    print("\nGenerating Figure 1: Real Part...")
    plot_gp_real_part(
        omega, y_real_orig, y_real_pred, y_real_std,
        output_dir / 'gp_real',
        save_eps=not args.no_eps
    )

    # Figure 2: Imaginary part
    print("Generating Figure 2: Imaginary Part...")
    plot_gp_imag_part(
        omega, y_imag_orig, y_imag_pred, y_imag_std,
        output_dir / 'gp_imag',
        save_eps=not args.no_eps
    )

    # Figure 3: Nyquist plot
    print("Generating Figure 3: Nyquist Plot...")
    G_pred = y_real_pred + 1j * y_imag_pred
    plot_gp_nyquist(
        omega, G_complex, G_pred, y_real_std, y_imag_std,
        output_dir / 'gp_nyquist',
        save_eps=not args.no_eps
    )

    # Step 8: FIR Model Extraction (if requested)
    if args.extract_fir:
        print(f"\n{'='*60}")
        print("Step 7: FIR Model Extraction")
        print(f"{'='*60}")

        # Get validation MAT file if specified
        validation_mat = None
        if args.fir_validation_mat:
            validation_mat = Path(args.fir_validation_mat)
            if not validation_mat.exists():
                print(f"Warning: Validation MAT file not found: {validation_mat}")
                validation_mat = None
            else:
                print(f"  Validation file: {validation_mat}")
                print(f"  FIR length: {args.fir_length}")
                print(f"  NOTE: Time-domain RMSE will be computed using FULL duration")

        # Create a GP prediction function for better interpolation
        def gp_predict_at_omega(omega_new):
            """Predict using the fitted GP models at new frequencies."""
            X_new = omega_new.reshape(-1, 1).copy()
            X_new_normalized = X_scaler.transform(X_new)

            # Predict with GPs
            y_real_new = gp_real.predict(X_new_normalized)
            y_imag_new = gp_imag.predict(X_new_normalized)

            # Denormalize
            y_real_new = y_real_scaler.inverse_transform(y_real_new.reshape(-1, 1)).ravel()
            y_imag_new = y_imag_scaler.inverse_transform(y_imag_new.reshape(-1, 1)).ravel()

            return y_real_new + 1j * y_imag_new

        # Use GP-direct method for FIR extraction
        try:
            from gp_to_fir_direct_fixed import gp_to_fir_direct_pipeline as gp_to_fir_direct_pipeline_fixed
            print("Using GP-based FIR extraction...")
            fir_results = gp_to_fir_direct_pipeline_fixed(
                omega=omega,
                G=G_pred,
                gp_predict_func=gp_predict_at_omega,
                mat_file=validation_mat,
                output_dir=output_dir,
                N_fft=None,
                fir_length=args.fir_length
            )

            print(f"FIR extraction complete.")

            # Additional evaluation with Wave.mat if it exists
            wave_mat_path = Path('Wave.mat')
            if wave_mat_path.exists() and validation_mat != wave_mat_path:
                print(f"\n{'='*60}")
                print("Additional FIR Evaluation with Wave.mat")
                print(f"  Using: {wave_mat_path.absolute()}")
                print(f"{'='*60}")

                try:
                    from scipy.io import loadmat

                    # Extract FIR coefficients from results
                    if 'fir_coefficients' in fir_results:
                        g = fir_results['fir_coefficients']
                        N = len(g)

                        # Load Wave.mat data
                        wave_data = loadmat(wave_mat_path)
                        T = None
                        y = None
                        u = None

                        # Try to extract [t, y, u] from Wave.mat
                        for key, val in wave_data.items():
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
                            T = np.ravel(wave_data.get("t", wave_data.get("time"))).astype(float)
                            y = np.ravel(wave_data.get("y"))
                            u = np.ravel(wave_data.get("u"))

                        if T is not None and y is not None and u is not None:
                            # Evaluation mode
                            DETREND = False
                            if DETREND:
                                u_eval = u - np.mean(u)
                                y_eval = y - np.mean(y)
                            else:
                                u_eval = u.copy()
                                y_eval = y.copy()

                            # Predict with causal convolution
                            y_pred = np.convolve(u_eval, g, mode="full")[:len(y_eval)]

                            # Minimal transient skip
                            skip = min(10, N // 10)
                            y_valid = y_eval[skip:]
                            y_pred_valid = y_pred[skip:]

                            # Calculate metrics
                            err = y_valid - y_pred_valid
                            rmse = float(np.sqrt(np.mean(err**2)))

                            norm_y = np.linalg.norm(y_valid)
                            fit = float(100 * (1.0 - np.linalg.norm(err) / norm_y)) if norm_y > 0 else 0.0

                            ss_res = float(np.sum(err**2))
                            ss_tot = float(np.sum((y_valid - y_valid[0])**2))
                            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                            # Create visualization plots
                            from gp_to_fir_direct_fixed import plot_gp_fir_results_fixed
                            plot_gp_fir_results_fixed(
                                t=T, y=y, y_pred=y_pred, u=u,
                                rmse=rmse, fit_percent=fit, r2=r2,
                                output_dir=output_dir,
                                prefix="gp_fir_wave"
                            )

                            print(f"  Wave.mat Validation: RMSE={rmse:.3e}, FIT={fit:.1f}%, R²={r2:.3f}")
                            print(f"  Results saved to {output_dir}")
                        else:
                            print("  Warning: Could not load time-series data from Wave.mat")
                    else:
                        print("  Warning: FIR coefficients not found in results")

                except Exception as e:
                    print(f"  Error during Wave.mat evaluation: {str(e)}")

        except ImportError:
            print("Warning: gp_to_fir_direct_fixed module is not available")
        except Exception as e:
            print(f"Error during FIR extraction: {str(e)}")

    # Step 9: Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"\nGenerated files:")
    print(f"  1. gp_real.png (and .eps) - Real Part GP Regression")
    print(f"  2. gp_imag.png (and .eps) - Imaginary Part GP Regression")
    print(f"  3. gp_nyquist.png (and .eps) - Nyquist Plot with GP")
    if args.extract_fir:
        print(f"  4. FIR model coefficients and validation plots")
    print(f"\nKernel: {args.kernel}")
    print(f"Real part - RMSE: {rmse_real:.6e}, R²: {r2_real:.6f}")
    print(f"Imaginary part - RMSE: {rmse_imag:.6e}, R²: {r2_imag:.6f}")
    print(f"\nOptimized kernel parameters:")
    print(f"  Real: {gp_real.kernel.get_params()}")
    print(f"  Imag: {gp_imag.kernel.get_params()}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
