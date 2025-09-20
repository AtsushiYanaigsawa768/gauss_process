"""
Frequency domain conversion module.
Implements FFT-based frequency response estimation and Nyquist plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pathlib import Path


def convert_to_frequency_domain(
    time: np.ndarray,
    input_data: np.ndarray,
    output_data: np.ndarray,
    num_freq: int = 100,
    freq_min: float = 0.1,
    freq_max: float = 200.0,
    demean: bool = True,
    use_window: bool = True,
    window_type: str = "hanning"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time-domain input/output data to frequency domain.
    Extracts frequency response at specified frequency points.

    Parameters:
    -----------
    time : np.ndarray
        Time vector
    input_data : np.ndarray
        Input signal
    output_data : np.ndarray
        Output signal
    num_freq : int
        Number of frequency points to extract
    freq_min : float
        Minimum frequency [Hz]
    freq_max : float
        Maximum frequency [Hz]
    demean : bool
        Whether to remove mean from signals
    use_window : bool
        Whether to apply window function
    window_type : str
        Type of window ('hanning', 'hamming', 'blackman')

    Returns:
    --------
    frequencies : np.ndarray
        Frequency points [Hz]
    freq_response : np.ndarray
        Complex frequency response G(jw)
    """
    # Copy data to avoid modifying original
    u = input_data.copy()
    y = output_data.copy()

    # Remove mean if requested
    if demean:
        u = u - np.mean(u)
        y = y - np.mean(y)

    # Calculate sampling time
    dt = np.median(np.diff(time))
    fs = 1.0 / dt  # Sampling frequency
    N = len(u)

    # Apply window if requested
    if use_window:
        if window_type == "hanning":
            window = np.hanning(N)
        elif window_type == "hamming":
            window = np.hamming(N)
        elif window_type == "blackman":
            window = np.blackman(N)
        else:
            window = np.ones(N)

        # Normalize window to preserve signal power
        window = window / np.sqrt(np.mean(window**2))
        u = u * window
        y = y * window

    # Compute FFT
    U = np.fft.fft(u)
    Y = np.fft.fft(y)
    freq_fft = np.fft.fftfreq(N, dt)

    # Select positive frequencies only
    pos_idx = freq_fft > 0
    freq_pos = freq_fft[pos_idx]
    U_pos = U[pos_idx]
    Y_pos = Y[pos_idx]

    # Create target frequency grid (log-spaced)
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_freq)

    # Method 1: Direct FFT-based estimation at closest frequencies
    # This is more suitable for multi-sine inputs as described in Method.tex
    freq_response = np.zeros(num_freq, dtype=complex)

    for i, f_target in enumerate(frequencies):
        # Find closest frequency in FFT result
        idx = np.argmin(np.abs(freq_pos - f_target))

        # Skip if frequency is too far from target
        if np.abs(freq_pos[idx] - f_target) / f_target > 0.1:
            # Use interpolation for better accuracy
            freq_response[i] = _interpolate_frequency_response(
                freq_pos, U_pos, Y_pos, f_target
            )
        else:
            # Direct estimation: G(jw) = Y(jw) / U(jw)
            if np.abs(U_pos[idx]) > 1e-10:
                freq_response[i] = Y_pos[idx] / U_pos[idx]
            else:
                freq_response[i] = 0.0 + 0.0j

    # Alternative Method 2: Spectral estimation using periodogram method
    # This provides smoother estimates but is commented out as Method.tex
    # specifically mentions multi-sine wave extraction
    """
    freq_response_smooth = estimate_frequency_response_welch(
        u, y, fs, frequencies
    )
    """

    return frequencies, freq_response


def _interpolate_frequency_response(
    freq: np.ndarray,
    U: np.ndarray,
    Y: np.ndarray,
    f_target: float
) -> complex:
    """
    Interpolate frequency response at target frequency.

    Uses linear interpolation in log-frequency domain for magnitude
    and linear interpolation for phase.
    """
    # Find surrounding frequencies
    idx_low = np.where(freq <= f_target)[0]
    idx_high = np.where(freq >= f_target)[0]

    if len(idx_low) == 0 or len(idx_high) == 0:
        return 0.0 + 0.0j

    idx1 = idx_low[-1]
    idx2 = idx_high[0]

    if idx1 == idx2:
        # Exact match
        if np.abs(U[idx1]) > 1e-10:
            return Y[idx1] / U[idx1]
        else:
            return 0.0 + 0.0j

    # Linear interpolation in log-frequency domain
    f1, f2 = freq[idx1], freq[idx2]
    alpha = (np.log(f_target) - np.log(f1)) / (np.log(f2) - np.log(f1))

    # Compute frequency responses at surrounding points
    G1 = Y[idx1] / U[idx1] if np.abs(U[idx1]) > 1e-10 else 0.0
    G2 = Y[idx2] / U[idx2] if np.abs(U[idx2]) > 1e-10 else 0.0

    # Interpolate magnitude in log scale and phase linearly
    mag1, phase1 = np.abs(G1), np.angle(G1)
    mag2, phase2 = np.abs(G2), np.angle(G2)

    # Handle phase unwrapping
    phase_diff = phase2 - phase1
    if phase_diff > np.pi:
        phase2 -= 2 * np.pi
    elif phase_diff < -np.pi:
        phase2 += 2 * np.pi

    # Interpolate
    log_mag = (1 - alpha) * np.log(mag1 + 1e-20) + alpha * np.log(mag2 + 1e-20)
    phase = (1 - alpha) * phase1 + alpha * phase2

    return np.exp(log_mag) * np.exp(1j * phase)


def estimate_frequency_response_welch(
    input_data: np.ndarray,
    output_data: np.ndarray,
    fs: float,
    frequencies: np.ndarray,
    nperseg: Optional[int] = None
) -> np.ndarray:
    """
    Estimate frequency response using Welch's method.
    This provides smoother estimates for noisy data.

    Parameters:
    -----------
    input_data : np.ndarray
        Input signal
    output_data : np.ndarray
        Output signal
    fs : float
        Sampling frequency
    frequencies : np.ndarray
        Target frequencies
    nperseg : Optional[int]
        Segment length for Welch's method

    Returns:
    --------
    freq_response : np.ndarray
        Complex frequency response
    """
    from scipy import signal

    if nperseg is None:
        nperseg = min(len(input_data) // 8, 8192)

    # Compute cross-spectral density and input power spectral density
    f_welch, Pxy = signal.csd(input_data, output_data, fs=fs, nperseg=nperseg)
    _, Pxx = signal.welch(input_data, fs=fs, nperseg=nperseg)

    # Frequency response: G = Pxy / Pxx
    G_welch = Pxy / (Pxx + 1e-12)

    # Interpolate to target frequencies
    freq_response = np.interp(frequencies, f_welch, G_welch)

    return freq_response


def extract_multisine_components(
    time: np.ndarray,
    signal: np.ndarray,
    target_frequencies: np.ndarray,
    method: str = "fft"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract amplitude and phase of specific frequency components from a signal.
    Useful for analyzing multi-sine wave inputs as described in Method.tex.

    Parameters:
    -----------
    time : np.ndarray
        Time vector
    signal : np.ndarray
        Signal to analyze
    target_frequencies : np.ndarray
        Frequencies to extract [Hz]
    method : str
        Method to use ('fft' or 'goertzel')

    Returns:
    --------
    amplitudes : np.ndarray
        Amplitude at each target frequency
    phases : np.ndarray
        Phase at each target frequency [radians]
    """
    if method == "fft":
        # Use FFT method
        dt = np.median(np.diff(time))
        N = len(signal)
        fft_result = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(N, dt)

        amplitudes = np.zeros(len(target_frequencies))
        phases = np.zeros(len(target_frequencies))

        for i, freq in enumerate(target_frequencies):
            # Find closest frequency bin
            idx = np.argmin(np.abs(fft_freq - freq))
            complex_amp = fft_result[idx] * 2.0 / N
            amplitudes[i] = np.abs(complex_amp)
            phases[i] = np.angle(complex_amp)

    elif method == "goertzel":
        # Use Goertzel algorithm (more efficient for few frequencies)
        amplitudes, phases = _goertzel_multiple(signal, target_frequencies, time)

    else:
        raise ValueError(f"Unknown method: {method}")

    return amplitudes, phases


def _goertzel_multiple(
    signal: np.ndarray,
    frequencies: np.ndarray,
    time: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Goertzel algorithm for extracting specific frequency components.
    More efficient than FFT when only few frequencies are needed.
    """
    dt = np.median(np.diff(time))
    fs = 1.0 / dt
    N = len(signal)

    amplitudes = np.zeros(len(frequencies))
    phases = np.zeros(len(frequencies))

    for i, freq in enumerate(frequencies):
        # Normalized frequency
        omega = 2.0 * np.pi * freq / fs
        coeff = 2.0 * np.cos(omega)

        # Goertzel iteration
        s_prev = 0.0
        s_prev2 = 0.0

        for n in range(N):
            s = signal[n] + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s

        # Final calculations
        real = s_prev - s_prev2 * np.cos(omega)
        imag = s_prev2 * np.sin(omega)

        amplitudes[i] = 2.0 * np.sqrt(real**2 + imag**2) / N
        phases[i] = np.arctan2(imag, real)

    return amplitudes, phases


def plot_nyquist(
    freq_response: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Nyquist Diagram",
    show: bool = False
) -> None:
    """
    Plot Nyquist diagram of frequency response.

    Parameters:
    -----------
    freq_response : np.ndarray
        Complex frequency response
    save_path : Optional[Path]
        Path to save the figure
    title : str
        Plot title
    show : bool
        Whether to display the plot
    """
    plt.figure(figsize=(10, 8))

    # Extract real and imaginary parts
    real = np.real(freq_response)
    imag = np.imag(freq_response)

    # Main plot
    plt.plot(real, imag, 'b-', linewidth=2, label='G(jw)')
    plt.plot(real, imag, 'ro', markersize=4, alpha=0.5)

    # Mark start and end points
    if len(real) > 0:
        plt.plot(real[0], imag[0], 'go', markersize=10, label='Start (lowest freq)')
        plt.plot(real[-1], imag[-1], 'rs', markersize=10, label='End (highest freq)')

    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')

    # Add critical point
    plt.plot(-1, 0, 'kx', markersize=12, markeredgewidth=3, label='Critical point (-1, 0)')

    # Formatting
    plt.xlabel('Real', fontsize=12)
    plt.ylabel('Imaginary', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(fontsize=10)

    # Add arrows to show direction
    n_arrows = min(10, len(real) - 1)
    arrow_indices = np.linspace(0, len(real) - 2, n_arrows, dtype=int)

    for idx in arrow_indices:
        dx = real[idx + 1] - real[idx]
        dy = imag[idx + 1] - imag[idx]
        plt.arrow(real[idx], imag[idx], dx * 0.3, dy * 0.3,
                 head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Nyquist plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_loader import load_mat_files

    # Load sample data
    input_dir = Path("F:/Code/gauss_process/input")
    time, input_data, output_data = load_mat_files(input_dir, num_files=1, max_samples_per_file=50000)

    # Convert to frequency domain
    frequencies, freq_response = convert_to_frequency_domain(
        time, input_data, output_data,
        num_freq=100,
        freq_min=0.1,
        freq_max=200.0
    )

    print(f"Extracted {len(frequencies)} frequency points")
    print(f"Frequency range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")

    # Plot Nyquist diagram
    plot_nyquist(freq_response, title="Test Nyquist Diagram", show=True)
