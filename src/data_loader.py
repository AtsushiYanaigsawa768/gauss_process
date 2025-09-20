"""
Data loader module for loading .mat files containing input/output data.
Handles loading multiple files and combining data as needed.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from scipy.io import loadmat


def load_single_mat_file(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single .mat file and extract time, input, and output data.

    Parameters:
    -----------
    file_path : Path
        Path to the .mat file

    Returns:
    --------
    time : np.ndarray
        Time vector (1D)
    input_data : np.ndarray
        Input signal (1D)
    output_data : np.ndarray
        Output signal (1D)
    """
    # Load the .mat file
    data = loadmat(str(file_path))

    # The data is stored under the key 'output' with shape (3, N)
    if 'output' not in data:
        raise ValueError(f"Expected 'output' key in {file_path}, found keys: {list(data.keys())}")

    output_matrix = data['output']

    # Validate shape
    if output_matrix.shape[0] != 3:
        raise ValueError(f"Expected 3 rows in output matrix, got {output_matrix.shape[0]}")

    # Extract data (row 0: time, row 1: input, row 2: output)
    time = output_matrix[0, :].astype(float)
    input_data = output_matrix[1, :].astype(float)
    output_data = output_matrix[2, :].astype(float)

    return time, input_data, output_data


def load_mat_files(
    input_dir: Path,
    num_files: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
    file_pattern: str = "*.mat"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load multiple .mat files from a directory and combine the data.

    Parameters:
    -----------
    input_dir : Path
        Directory containing .mat files
    num_files : Optional[int]
        Number of files to load (None = all files)
    max_samples_per_file : Optional[int]
        Maximum samples to use from each file (None = all samples)
    file_pattern : str
        File pattern to match (default: "*.mat")

    Returns:
    --------
    time : np.ndarray
        Combined time vector
    input_data : np.ndarray
        Combined input signal
    output_data : np.ndarray
        Combined output signal
    """
    # Get list of .mat files
    mat_files = sorted(list(input_dir.glob(file_pattern)))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {input_dir}")

    # Limit number of files if specified
    if num_files is not None:
        mat_files = mat_files[:num_files]

    print(f"Loading {len(mat_files)} .mat files from {input_dir}")

    # Lists to store data from each file
    time_list = []
    input_list = []
    output_list = []

    # Load each file
    for i, file_path in enumerate(mat_files):
        print(f"  Loading {i+1}/{len(mat_files)}: {file_path.name}")

        # Load single file
        time, input_data, output_data = load_single_mat_file(file_path)

        # Limit samples if specified
        if max_samples_per_file is not None and len(time) > max_samples_per_file:
            time = time[:max_samples_per_file]
            input_data = input_data[:max_samples_per_file]
            output_data = output_data[:max_samples_per_file]

        # Append to lists
        time_list.append(time)
        input_list.append(input_data)
        output_list.append(output_data)

    # Combine data from all files
    if len(mat_files) == 1:
        # Single file - return as is
        return time_list[0], input_list[0], output_list[0]

    else:
        # Multiple files - concatenate with time offset
        combined_time = []
        combined_input = []
        combined_output = []

        time_offset = 0.0

        for time, input_data, output_data in zip(time_list, input_list, output_list):
            # Add time offset to maintain continuity
            combined_time.append(time + time_offset)
            combined_input.append(input_data)
            combined_output.append(output_data)

            # Update offset for next file
            if len(time) > 0:
                time_offset = combined_time[-1][-1] + (time[1] - time[0]) if len(time) > 1 else 0.002

        # Concatenate all data
        combined_time = np.concatenate(combined_time)
        combined_input = np.concatenate(combined_input)
        combined_output = np.concatenate(combined_output)

        return combined_time, combined_input, combined_output


def validate_data(time: np.ndarray, input_data: np.ndarray, output_data: np.ndarray) -> None:
    """
    Validate the loaded data for consistency.

    Parameters:
    -----------
    time : np.ndarray
        Time vector
    input_data : np.ndarray
        Input signal
    output_data : np.ndarray
        Output signal

    Raises:
    -------
    ValueError
        If data is invalid or inconsistent
    """
    # Check shapes
    if time.shape != input_data.shape or time.shape != output_data.shape:
        raise ValueError(f"Shape mismatch: time={time.shape}, input={input_data.shape}, output={output_data.shape}")

    # Check for NaN or Inf values
    if np.any(np.isnan(time)) or np.any(np.isinf(time)):
        raise ValueError("Time vector contains NaN or Inf values")

    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
        raise ValueError("Input data contains NaN or Inf values")

    if np.any(np.isnan(output_data)) or np.any(np.isinf(output_data)):
        raise ValueError("Output data contains NaN or Inf values")

    # Check time monotonicity
    if not np.all(np.diff(time) > 0):
        raise ValueError("Time vector is not strictly monotonic")

    print("Data validation passed!")


def get_data_statistics(time: np.ndarray, input_data: np.ndarray, output_data: np.ndarray) -> dict:
    """
    Calculate basic statistics of the loaded data.

    Parameters:
    -----------
    time : np.ndarray
        Time vector
    input_data : np.ndarray
        Input signal
    output_data : np.ndarray
        Output signal

    Returns:
    --------
    stats : dict
        Dictionary containing data statistics
    """
    dt = np.median(np.diff(time))
    duration = time[-1] - time[0]

    stats = {
        'num_samples': len(time),
        'duration': duration,
        'sampling_time': dt,
        'sampling_frequency': 1.0 / dt if dt > 0 else np.nan,
        'input_mean': np.mean(input_data),
        'input_std': np.std(input_data),
        'input_min': np.min(input_data),
        'input_max': np.max(input_data),
        'output_mean': np.mean(output_data),
        'output_std': np.std(output_data),
        'output_min': np.min(output_data),
        'output_max': np.max(output_data)
    }

    return stats


if __name__ == "__main__":
    # Test the data loader
    input_dir = Path("F:/Code/gauss_process/input")

    # Load first 3 files with limited samples
    time, input_data, output_data = load_mat_files(
        input_dir,
        num_files=3,
        max_samples_per_file=10000
    )

    # Validate data
    validate_data(time, input_data, output_data)

    # Print statistics
    stats = get_data_statistics(time, input_data, output_data)
    print("\nData statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")