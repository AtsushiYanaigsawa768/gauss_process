"""
Main entry point for system identification pipeline.
Implements the workflow described in Method.tex:
1. Load raw input/output data
2. Convert to frequency domain
3. Apply GP regression for interpolation
4. Perform IFFT to generate FIR model
5. Validate predictions against actual data
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path to import from gp folder
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import load_mat_files
from freq_domain import convert_to_frequency_domain, plot_nyquist
from gp_regressor import GPFrequencyRegressor
from fir_model import create_fir_model, validate_fir_model
from visualization import plot_gp_interpolation, plot_fir_results


# ==========================================
# Configuration Constants
# ==========================================

class Config:
    """Configuration constants for the system identification pipeline."""

    # Input/Output paths
    INPUT_DIR = Path("F:/Code/gauss_process/input")
    OUTPUT_DIR = Path("F:/Code/gauss_process/output")

    # Data loading parameters
    NUM_MAT_FILES = 50  # Number of .mat files to use (set to None to use all)
    MAX_SAMPLES_PER_FILE = 1_000_000  # Limit samples per file for memory efficiency

    # Frequency domain parameters
    NUM_FREQ_POINTS = 1000  # Number of frequency points for initial FFT analysis
    FREQ_MIN = 0.1  # Minimum frequency [Hz]
    FREQ_MAX = 200.0  # Maximum frequency [Hz]

    # GP regression parameters
    NUM_TRAINING_POINTS = 1000  # Number of frequency points for GP training
    NUM_INTERPOLATION_POINTS = 50000  # Dense grid for interpolation
    GP_KERNEL = "rbf"  # Options: rbf, matern12, matern32, matern52, exp, tc, dc, di, ss, ss2, hf, stable_spline
    GP_NOISE = 1e-3  # Noise level for GP
    GP_OPTIMIZE = True  # Whether to optimize hyperparameters
    GP_MAXITER = 500  # Maximum iterations for optimization

    # FIR model parameters
    FIR_LENGTH = 1024  # Length of FIR filter (must be power of 2 for FFT efficiency)

    # Output file naming
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PREFIX = f"sysid_{GP_KERNEL}_{TIMESTAMP}"

    # Output files
    NYQUIST_PLOT = OUTPUT_DIR / f"{RESULTS_PREFIX}_nyquist.png"
    GP_INTERP_PLOT = OUTPUT_DIR / f"{RESULTS_PREFIX}_gp_interpolation.png"
    FIR_RESULTS_PLOT = OUTPUT_DIR / f"{RESULTS_PREFIX}_fir_results.png"
    METRICS_CSV = OUTPUT_DIR / f"{RESULTS_PREFIX}_metrics.csv"
    FIR_COEFFS_NPZ = OUTPUT_DIR / f"{RESULTS_PREFIX}_fir_coefficients.npz"
    CONFIG_JSON = OUTPUT_DIR / f"{RESULTS_PREFIX}_config.json"

    # Processing parameters
    DEMEAN = True  # Whether to remove mean from signals
    USE_WINDOW = True  # Whether to apply window function in FFT
    WINDOW_TYPE = "hanning"  # Window type for FFT


def save_config(config: Config):
    """Save configuration to JSON file."""
    config_dict = {
        "timestamp": config.TIMESTAMP,
        "input_dir": str(config.INPUT_DIR),
        "output_dir": str(config.OUTPUT_DIR),
        "num_mat_files": config.NUM_MAT_FILES,
        "max_samples_per_file": config.MAX_SAMPLES_PER_FILE,
        "num_freq_points": config.NUM_FREQ_POINTS,
        "freq_range": [config.FREQ_MIN, config.FREQ_MAX],
        "num_training_points": config.NUM_TRAINING_POINTS,
        "num_interpolation_points": config.NUM_INTERPOLATION_POINTS,
        "gp_kernel": config.GP_KERNEL,
        "gp_noise": config.GP_NOISE,
        "gp_optimize": config.GP_OPTIMIZE,
        "fir_length": config.FIR_LENGTH,
        "demean": config.DEMEAN,
        "use_window": config.USE_WINDOW,
        "window_type": config.WINDOW_TYPE
    }

    with open(config.CONFIG_JSON, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {config.CONFIG_JSON}")


def main():
    """Main pipeline execution."""
    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(Config)

    print("=" * 60)
    print("System Identification Pipeline")
    print("=" * 60)
    print(f"Timestamp: {Config.TIMESTAMP}")
    print(f"GP Kernel: {Config.GP_KERNEL}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print()

    # Step 1: Load raw data
    print("Step 1: Loading raw data...")
    time_data, input_data, output_data = load_mat_files(
        Config.INPUT_DIR,
        num_files=Config.NUM_MAT_FILES,
        max_samples_per_file=Config.MAX_SAMPLES_PER_FILE
    )
    print(f"Loaded data shape: time={time_data.shape}, input={input_data.shape}, output={output_data.shape}")

    dt = float(np.median(np.diff(time_data)))
    if dt <= 0:
        raise ValueError("Non-positive sampling interval detected in time data.")
    sampling_freq = 1.0 / dt
    nyquist_freq = 0.5 * sampling_freq
    freq_min_effective = max(Config.FREQ_MIN, 1e-9)
    freq_max_requested = Config.FREQ_MAX
    freq_max_effective = min(freq_max_requested, nyquist_freq)
    if freq_max_requested > nyquist_freq + 1e-9:
        print(
            f"Warning: freq_max {freq_max_requested:.3f} Hz exceeds Nyquist {nyquist_freq:.3f} Hz. "
            f"Using {freq_max_effective:.3f} Hz instead."
        )
    if freq_max_effective <= freq_min_effective:
        raise ValueError("Effective frequency range is invalid. Adjust FREQ_MIN/FREQ_MAX or sampling parameters.")
    print(f"Estimated sampling frequency: {sampling_freq:.3f} Hz (Nyquist {nyquist_freq:.3f} Hz)")

    # Step 2: Convert to frequency domain
    print("\nStep 2: Converting to frequency domain...")
    freq_points, freq_response = convert_to_frequency_domain(
        time_data, input_data, output_data,
        num_freq=Config.NUM_FREQ_POINTS,
        freq_min=freq_min_effective,
        freq_max=freq_max_effective,
        demean=Config.DEMEAN,
        use_window=Config.USE_WINDOW,
        window_type=Config.WINDOW_TYPE
    )
    print(f"Frequency points: {len(freq_points)}")

    # Plot Nyquist diagram
    print("Plotting Nyquist diagram...")
    plot_nyquist(freq_response, Config.NYQUIST_PLOT)

    # Step 3: GP regression for interpolation
    print("\nStep 3: Applying GP regression...")
    gp_model = GPFrequencyRegressor(
        kernel=Config.GP_KERNEL,
        noise=Config.GP_NOISE,
        optimize=Config.GP_OPTIMIZE,
        maxiter=Config.GP_MAXITER
    )

    # Select training points
    num_available = len(freq_points)
    if num_available < 2:
        raise ValueError("Insufficient frequency points for GP training.")
    num_train = min(Config.NUM_TRAINING_POINTS, num_available)
    train_indices = np.linspace(0, num_available - 1, num_train)
    train_indices = np.round(train_indices).astype(int)
    train_indices = np.clip(train_indices, 0, num_available - 1)
    train_indices = np.unique(train_indices)
    if train_indices.size < 2:
        train_indices = np.arange(num_available)
    freq_train = freq_points[train_indices]
    response_train = freq_response[train_indices]

    # Fit GP model
    gp_model.fit(freq_train, response_train)

    # Interpolate to dense grid
    freq_dense = np.logspace(np.log10(freq_min_effective), np.log10(freq_max_effective),
                            Config.NUM_INTERPOLATION_POINTS)
    response_interp = gp_model.predict(freq_dense)

    # Plot GP interpolation
    print("Plotting GP interpolation results...")
    plot_gp_interpolation(
        freq_train, response_train,
        freq_dense, response_interp,
        Config.GP_INTERP_PLOT
    )

    # Step 4: Create FIR model via IFFT
    print("\nStep 4: Creating FIR model...")
    fir_coeffs = create_fir_model(
        freq_dense, response_interp,
        fir_length=Config.FIR_LENGTH,
        sampling_freq=sampling_freq
    )
    print(f"FIR coefficients shape: {fir_coeffs.shape}")

    # Save FIR coefficients
    np.savez(Config.FIR_COEFFS_NPZ,
             coefficients=fir_coeffs,
             frequencies=freq_dense,
             response=response_interp,
             sampling_frequency=sampling_freq)

    # Step 5: Validate FIR model
    print("\nStep 5: Validating FIR model...")
    # Use a portion of data for validation
    val_samples = min(50000, len(input_data))
    val_input = input_data[:val_samples]
    val_output = output_data[:val_samples]
    val_time = time_data[:val_samples]

    predicted_output, metrics = validate_fir_model(
        fir_coeffs, val_input, val_output,
        demean=Config.DEMEAN
    )

    print(f"Validation metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  NRMSE: {metrics['nrmse']:.6f}")
    print(f"  R2: {metrics['r2']:.6f}")
    print(f"  FIT%: {metrics['fit_percent']:.2f}%")

    # Plot validation results
    print("Plotting FIR validation results...")
    plot_fir_results(
        val_time, val_output, predicted_output,
        fir_coeffs, metrics,
        Config.FIR_RESULTS_PLOT
    )

    # Save metrics to CSV
    print("\nSaving metrics...")
    import pandas as pd
    metrics_df = pd.DataFrame([{
        'timestamp': Config.TIMESTAMP,
        'kernel': Config.GP_KERNEL,
        'num_freq_points': len(freq_points),
        'num_training_points': int(train_indices.size),
        'fir_length': Config.FIR_LENGTH,
        'freq_min_used': freq_min_effective,
        'freq_max_used': freq_max_effective,
        'sampling_frequency': sampling_freq,
        'sampling_period': dt,
        **metrics
    }])
    metrics_df.to_csv(Config.METRICS_CSV, index=False)

    print(f"\nPipeline completed successfully!")
    print(f"Results saved to: {Config.OUTPUT_DIR}")

    return metrics


if __name__ == "__main__":
    metrics = main()