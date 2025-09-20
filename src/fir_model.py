"""
FIR model creation and validation module.
Implements FIR coefficient extraction from frequency response using IFFT.
Based on the methodology described in Method.tex.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.signal import convolve



def create_fir_model(
    frequencies: np.ndarray,
    freq_response: np.ndarray,
    fir_length: int = 1024,
    sampling_freq: Optional[float] = None,
    method: str = "ifft"
) -> np.ndarray:
    """
    Create FIR model coefficients from frequency response data.

    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency points [Hz] (positive frequencies only)
    freq_response : np.ndarray
        Complex frequency response G(jw)
    fir_length : int
        Desired length of FIR filter
    sampling_freq : Optional[float]
        Sampling frequency of the original time-domain data [Hz].
        If None, it is estimated from the maximum provided frequency.
    method : str
        Method to use ("ifft" or "interpolation")

    Returns:
    --------
    fir_coeffs : np.ndarray
        FIR filter coefficients h[n]
    """
    if method == "ifft":
        return _create_fir_ifft(frequencies, freq_response, fir_length, sampling_freq)
    elif method == "interpolation":
        return _create_fir_interpolation(frequencies, freq_response, fir_length, sampling_freq)
    else:
        raise ValueError(f"Unknown method: {method}")



def _create_fir_ifft(
    frequencies: np.ndarray,
    freq_response: np.ndarray,
    fir_length: int,
    sampling_freq: Optional[float]
) -> np.ndarray:
    """
    Create FIR coefficients using an inverse FFT with a Hermitian spectrum.
    """
    freqs = np.asarray(frequencies, dtype=float).ravel()
    response = np.asarray(freq_response, dtype=complex).ravel()

    if freqs.size != response.size:
        raise ValueError("frequencies and freq_response must have the same length")

    valid = freqs >= 0.0
    freqs = freqs[valid]
    response = response[valid]

    if freqs.size < 2:
        raise ValueError("At least two positive frequency samples are required")

    freqs, unique_idx = np.unique(freqs, return_index=True)
    response = response[unique_idx]

    if sampling_freq is None:
        sampling_freq = float(freqs[-1] * 2.0)

    if sampling_freq <= 0:
        raise ValueError("sampling_freq must be positive")

    nyquist = sampling_freq / 2.0
    if freqs[-1] > nyquist * (1.0 + 1e-9):
        raise ValueError("Maximum frequency exceeds Nyquist limit for the provided sampling frequency")

    n_fft_target = max(2 * fir_length, 4 * freqs.size, 8)
    n_fft = int(2 ** np.ceil(np.log2(n_fft_target)))
    freq_uniform = np.linspace(0.0, nyquist, n_fft // 2 + 1)

    magnitude = np.abs(response)
    phase = np.unwrap(np.angle(response))

    mag_uniform = np.interp(freq_uniform, freqs, magnitude, left=magnitude[0], right=magnitude[-1])
    phase_uniform = np.interp(freq_uniform, freqs, phase, left=phase[0], right=phase[-1])

    positive_spectrum = mag_uniform * np.exp(1j * phase_uniform)
    positive_spectrum[0] = complex(positive_spectrum[0].real, 0.0)
    if n_fft % 2 == 0:
        positive_spectrum[-1] = complex(positive_spectrum[-1].real, 0.0)

    impulse_response = np.fft.irfft(positive_spectrum, n=n_fft)
    impulse_response = np.real_if_close(impulse_response)

    return impulse_response[:fir_length]



def _create_fir_interpolation(
    frequencies: np.ndarray,
    freq_response: np.ndarray,
    fir_length: int,
    sampling_freq: Optional[float]
) -> np.ndarray:
    """
    Alternative method using direct frequency sampling and a smoothing window.
    """
    freqs = np.asarray(frequencies, dtype=float).ravel()
    response = np.asarray(freq_response, dtype=complex).ravel()

    if freqs.size == 0:
        return np.zeros(fir_length, dtype=float)

    if sampling_freq is None:
        sampling_freq = float(np.max(freqs) * 2.0)

    if sampling_freq <= 0:
        raise ValueError("sampling_freq must be positive")

    t = np.arange(fir_length, dtype=float) / sampling_freq

    h = np.zeros(fir_length, dtype=float)
    weight = 1.0 / freqs.size
    for f, G in zip(freqs, response):
        omega_t = 2.0 * np.pi * f * t
        h += weight * (np.real(G) * np.cos(omega_t) - np.imag(G) * np.sin(omega_t))

    window = np.hanning(fir_length)
    return h * window


def validate_fir_model(
    fir_coeffs: np.ndarray,
    input_signal: np.ndarray,
    true_output: np.ndarray,
    demean: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Validate FIR model by comparing predicted output with true output.

    Parameters:
    -----------
    fir_coeffs : np.ndarray
        FIR filter coefficients
    input_signal : np.ndarray
        Input signal
    true_output : np.ndarray
        True output signal
    demean : bool
        Whether to remove mean from signals

    Returns:
    --------
    predicted_output : np.ndarray
        Predicted output using FIR model
    metrics : dict
        Dictionary containing RMSE, NRMSE, R², and FIT%
    """
    # Copy signals to avoid modification
    u = input_signal.copy()
    y_true = true_output.copy()

    # Remove mean if requested
    if demean:
        u_mean = np.mean(u)
        y_mean = np.mean(y_true)
        u = u - u_mean
        y_true = y_true - y_mean
    else:
        y_mean = 0

    # Apply FIR filter using convolution
    y_pred = convolve(u, fir_coeffs, mode='same')

    # Add mean back if it was removed
    if demean:
        y_pred = y_pred + y_mean
        y_true = y_true + y_mean

    # Calculate metrics (ignore transient at beginning)
    L = len(fir_coeffs)
    e = y_true - y_pred
    e_tail = e[L:]
    y_tail = y_true[L:]

    # RMSE
    rmse = float(np.sqrt(np.mean(e_tail**2)))

    # NRMSE (Normalized RMSE)
    y_range = np.max(y_tail) - np.min(y_tail)
    nrmse = rmse / (y_range + 1e-10)

    # R² (Coefficient of determination)
    ss_res = np.sum(e_tail**2)
    ss_tot = np.sum((y_tail - np.mean(y_tail))**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    # FIT% (as used in system identification)
    fit = 100.0 * (1.0 - np.linalg.norm(e_tail) / (np.linalg.norm(y_tail - np.mean(y_tail)) + 1e-10))

    metrics = {
        'rmse': rmse,
        'nrmse': nrmse,
        'r2': r2,
        'fit_percent': fit
    }

    return y_pred, metrics


def analyze_fir_stability(fir_coeffs: np.ndarray) -> Dict[str, float]:
    """
    Analyze stability properties of FIR filter.

    Parameters:
    -----------
    fir_coeffs : np.ndarray
        FIR filter coefficients

    Returns:
    --------
    analysis : dict
        Dictionary containing stability metrics
    """
    # FIR filters are always stable, but we can analyze properties
    analysis = {}

    # Energy of impulse response
    analysis['energy'] = float(np.sum(fir_coeffs**2))

    # Decay rate estimation (for stable systems)
    # Find envelope decay
    envelope = np.abs(fir_coeffs)
    nonzero_idx = np.where(envelope > 1e-6 * np.max(envelope))[0]

    if len(nonzero_idx) > 1:
        # Fit exponential decay to envelope
        t = nonzero_idx
        log_envelope = np.log(envelope[nonzero_idx] + 1e-10)

        # Linear regression for log(envelope) = log(A) - alpha*t
        p = np.polyfit(t, log_envelope, 1)
        decay_rate = -p[0]
        analysis['decay_rate'] = float(decay_rate)
    else:
        analysis['decay_rate'] = np.inf

    # Effective length (where most energy is concentrated)
    cumsum_energy = np.cumsum(fir_coeffs**2)
    total_energy = cumsum_energy[-1]
    idx_95 = np.where(cumsum_energy >= 0.95 * total_energy)[0]
    analysis['effective_length'] = int(idx_95[0]) if len(idx_95) > 0 else len(fir_coeffs)

    return analysis


def frequency_response_from_fir(
    fir_coeffs: np.ndarray,
    frequencies: np.ndarray,
    fs: float = None
) -> np.ndarray:
    """
    Compute frequency response of FIR filter at specified frequencies.

    Parameters:
    -----------
    fir_coeffs : np.ndarray
        FIR filter coefficients
    frequencies : np.ndarray
        Frequency points [Hz]
    fs : float
        Sampling frequency (if None, estimated from max frequency)

    Returns:
    --------
    freq_response : np.ndarray
        Complex frequency response
    """
    if fs is None:
        fs = 2 * np.max(frequencies)

    # Compute frequency response using DFT formula
    N = len(fir_coeffs)
    freq_response = np.zeros(len(frequencies), dtype=complex)

    for i, f in enumerate(frequencies):
        # Normalized frequency
        omega = 2 * np.pi * f / fs
        # DFT at specific frequency
        n = np.arange(N)
        freq_response[i] = np.sum(fir_coeffs * np.exp(-1j * omega * n))

    return freq_response


if __name__ == "__main__":
    # Test FIR model creation
    import matplotlib.pyplot as plt

    # Create synthetic frequency response
    frequencies = np.logspace(-1, 2, 100)  # 0.1 to 100 Hz
    omega_n = 10.0
    zeta = 0.3
    s = 1j * 2 * np.pi * frequencies
    G = omega_n**2 / (s**2 + 2*zeta*omega_n*s + omega_n**2)

    # Create FIR model
    fir_coeffs = create_fir_model(frequencies, G, fir_length=256, sampling_freq=2 * frequencies[-1])

    # Analyze FIR properties
    analysis = analyze_fir_stability(fir_coeffs)
    print("FIR Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.6f}")

    # Verify frequency response
    G_fir = frequency_response_from_fir(fir_coeffs, frequencies)

    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Impulse response
    ax1.stem(range(len(fir_coeffs)), fir_coeffs, basefmt=' ')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('h[n]')
    ax1.set_title('FIR Impulse Response')
    ax1.grid(True, alpha=0.3)

    # Magnitude comparison
    ax2.loglog(frequencies, np.abs(G), 'b-', label='Original')
    ax2.loglog(frequencies, np.abs(G_fir), 'r--', label='FIR approximation')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    # Phase comparison
    ax3.semilogx(frequencies, np.angle(G) * 180/np.pi, 'b-', label='Original')
    ax3.semilogx(frequencies, np.angle(G_fir) * 180/np.pi, 'r--', label='FIR approximation')
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Phase [deg]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fir_model_test.png", dpi=150)
    plt.close()

    print("\nTest completed. Results saved to fir_model_test.png")