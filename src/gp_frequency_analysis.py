#!/usr/bin/env python3
"""
gp_frequency_analysis.py

Advanced frequency-domain analysis tools for Gaussian Process regression.
Provides specialized methods for analyzing frequency response functions using GP.

Features:
- Multi-output GP for complex-valued frequency response
- Spectral mixture kernels for frequency domain
- Stability analysis and constraint handling
- Causal system identification
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from unified_pipeline import (
    Kernel, CombinedKernel, RBFKernel, MaternKernel,
    GaussianProcessRegressor, create_kernel
)


# =====================
# Specialized Frequency-Domain Kernels
# =====================

class SpectralMixtureKernel(Kernel):
    """Spectral Mixture kernel for modeling quasi-periodic patterns."""

    def __init__(self, n_components: int = 1, weights: Optional[np.ndarray] = None,
                 means: Optional[np.ndarray] = None, variances: Optional[np.ndarray] = None):
        self.n_components = n_components

        # Initialize parameters
        if weights is None:
            weights = np.ones(n_components) / n_components
        if means is None:
            means = np.random.rand(n_components) * 2.0
        if variances is None:
            variances = np.ones(n_components) * 0.5

        super().__init__(weights=weights, means=means, variances=variances)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        # Compute pairwise differences
        diffs = X1[:, None, :] - X2[None, :, :]  # Shape: (n1, n2, d)

        K = np.zeros((X1.shape[0], X2.shape[0]))

        for q in range(self.n_components):
            # Gaussian component
            gauss = np.exp(-2 * np.pi**2 * np.sum(diffs**2 * self.params['variances'][q], axis=2))

            # Cosine component
            cos_term = np.cos(2 * np.pi * np.sum(diffs * self.params['means'][q], axis=2))

            K += self.params['weights'][q] * gauss * cos_term

        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        # Weights bounds
        for _ in range(self.n_components):
            bounds.append((1e-3, 1.0))
        # Means bounds
        for _ in range(self.n_components):
            bounds.append((0.0, 10.0))
        # Variances bounds
        for _ in range(self.n_components):
            bounds.append((1e-3, 10.0))
        return bounds

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.params['weights'],
            self.params['means'],
            self.params['variances']
        ])

    def set_params(self, params: np.ndarray) -> None:
        n = self.n_components
        self.params['weights'] = params[:n]
        self.params['means'] = params[n:2*n]
        self.params['variances'] = params[2*n:3*n]

    @property
    def n_params(self) -> int:
        return 3 * self.n_components


class ResonanceKernel(Kernel):
    """Kernel for modeling resonant behavior in frequency domain."""

    def __init__(self, resonance_freq: float = 1.0, damping: float = 0.1, amplitude: float = 1.0):
        super().__init__(resonance_freq=resonance_freq, damping=damping, amplitude=amplitude)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        omega1 = X1.ravel()
        omega2 = X2.ravel()

        # Resonance kernel based on second-order system response
        w0 = self.params['resonance_freq']
        zeta = self.params['damping']

        # Frequency response magnitude at each frequency
        H1 = 1.0 / np.sqrt((w0**2 - omega1**2)**2 + (2*zeta*w0*omega1)**2)
        H2 = 1.0 / np.sqrt((w0**2 - omega2**2)**2 + (2*zeta*w0*omega2)**2)

        # Kernel is product of frequency responses
        K = self.params['amplitude'] * H1[:, None] * H2[None, :]
        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (0.1, 100.0),   # resonance_freq
            (0.01, 2.0),    # damping
            (1e-3, 1e3),    # amplitude
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['resonance_freq'], self.params['damping'], self.params['amplitude']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['resonance_freq'] = params[0]
        self.params['damping'] = params[1]
        self.params['amplitude'] = params[2]

    @property
    def n_params(self) -> int:
        return 3


# =====================
# Multi-Output GP for Complex Data
# =====================

class ComplexGaussianProcess:
    """GP for complex-valued outputs using coupled real/imaginary modeling."""

    def __init__(self, kernel_real: Kernel, kernel_imag: Kernel,
                 kernel_cross: Optional[Kernel] = None, noise_variance: float = 1e-6):
        self.kernel_real = kernel_real
        self.kernel_imag = kernel_imag
        self.kernel_cross = kernel_cross
        self.noise_variance = noise_variance

        self.X_train = None
        self.y_train_complex = None
        self.K_inv = None
        self.alpha = None

    def fit(self, X: np.ndarray, y_complex: np.ndarray, optimize: bool = True):
        """Fit multi-output GP to complex data."""
        self.X_train = X.copy()
        self.y_train_complex = y_complex.copy()

        n = len(X)

        # Stack real and imaginary parts
        y_stacked = np.concatenate([np.real(y_complex), np.imag(y_complex)])

        # Build block covariance matrix
        K_rr = self.kernel_real(X)
        K_ii = self.kernel_imag(X)

        if self.kernel_cross is not None:
            K_ri = self.kernel_cross(X)
            K_ir = K_ri.T
        else:
            K_ri = np.zeros((n, n))
            K_ir = np.zeros((n, n))

        # Block matrix
        K = np.block([[K_rr, K_ri],
                      [K_ir, K_ii]])

        # Add noise
        K += self.noise_variance * np.eye(2*n)

        # Compute inverse
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L, y_stacked)
            self.alpha = np.linalg.solve(L.T, self.alpha)
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ y_stacked

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict complex values at test points."""
        n_train = len(self.X_train)
        n_test = len(X)

        # Compute cross-covariances
        K_star_rr = self.kernel_real(X, self.X_train)
        K_star_ii = self.kernel_imag(X, self.X_train)

        if self.kernel_cross is not None:
            K_star_ri = self.kernel_cross(X, self.X_train)
            K_star_ir = self.kernel_cross(self.X_train, X).T
        else:
            K_star_ri = np.zeros((n_test, n_train))
            K_star_ir = np.zeros((n_test, n_train))

        # Stack for prediction
        K_star = np.block([[K_star_rr, K_star_ri],
                          [K_star_ii, K_star_ir]])

        # Mean prediction
        mean_stacked = K_star @ self.alpha
        mean_real = mean_stacked[:n_test]
        mean_imag = mean_stacked[n_test:]
        mean_complex = mean_real + 1j * mean_imag

        if return_std:
            # Predictive variance
            K_star_star_rr = self.kernel_real(X)
            K_star_star_ii = self.kernel_imag(X)

            var_real = np.diag(K_star_star_rr) - np.sum((K_star_rr @ self.K_inv[:n_train, :n_train]) * K_star_rr, axis=1)
            var_imag = np.diag(K_star_star_ii) - np.sum((K_star_ii @ self.K_inv[n_train:, n_train:]) * K_star_ii, axis=1)

            std_real = np.sqrt(np.maximum(var_real, 0))
            std_imag = np.sqrt(np.maximum(var_imag, 0))

            return mean_complex, (std_real, std_imag)

        return mean_complex


# =====================
# Frequency Domain Analysis Tools
# =====================

class FrequencyDomainAnalyzer:
    """Tools for analyzing GP results in frequency domain."""

    @staticmethod
    def compute_coherence(G_true: np.ndarray, G_pred: np.ndarray,
                         window_size: Optional[int] = None) -> np.ndarray:
        """Compute magnitude-squared coherence between true and predicted FRF."""
        if window_size is None:
            window_size = max(len(G_true) // 10, 1)

        # Compute cross-spectral density
        f, Pxy = signal.csd(G_true, G_pred, nperseg=window_size, return_onesided=False)
        f, Pxx = signal.welch(G_true, nperseg=window_size, return_onesided=False)
        f, Pyy = signal.welch(G_pred, nperseg=window_size, return_onesided=False)

        # Coherence
        coherence = np.abs(Pxy)**2 / (Pxx * Pyy)
        coherence = np.nan_to_num(coherence, nan=0.0)

        return coherence

    @staticmethod
    def estimate_delay(omega: np.ndarray, G: np.ndarray) -> float:
        """Estimate time delay from phase response."""
        phase = np.unwrap(np.angle(G))

        # Linear fit to phase vs frequency
        A = np.vstack([omega, np.ones_like(omega)]).T
        delay, _ = np.linalg.lstsq(A, -phase, rcond=None)[0]

        return delay

    @staticmethod
    def check_stability(G: np.ndarray, omega: np.ndarray) -> Dict[str, Union[bool, float]]:
        """Check stability criteria for frequency response."""
        # Nyquist stability check (simplified)
        # Count encirclements of -1+0j
        real_g = np.real(G)
        imag_g = np.imag(G)

        # Check if trajectory passes through -1+0j
        distances = np.sqrt((real_g + 1)**2 + imag_g**2)
        min_distance = np.min(distances)

        # Gain and phase margins
        mag_db = 20 * np.log10(np.abs(G))
        phase_deg = np.angle(G) * 180 / np.pi

        # Find gain margin (phase = -180°)
        phase_crossings = np.where(np.diff(np.sign(phase_deg + 180)))[0]
        if len(phase_crossings) > 0:
            gain_margin_db = -mag_db[phase_crossings[0]]
        else:
            gain_margin_db = np.inf

        # Find phase margin (|G| = 1 or 0 dB)
        mag_crossings = np.where(np.diff(np.sign(mag_db)))[0]
        if len(mag_crossings) > 0:
            phase_margin_deg = 180 + phase_deg[mag_crossings[0]]
        else:
            phase_margin_deg = np.inf

        return {
            'min_distance_to_critical': float(min_distance),
            'gain_margin_db': float(gain_margin_db),
            'phase_margin_deg': float(phase_margin_deg),
            'appears_stable': min_distance > 0.1 and gain_margin_db > 0 and phase_margin_deg > 0
        }

    @staticmethod
    def compute_sensitivity_functions(G: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute sensitivity and complementary sensitivity functions."""
        # Sensitivity: S = 1/(1 + G)
        S = 1.0 / (1.0 + G)

        # Complementary sensitivity: T = G/(1 + G)
        T = G / (1.0 + G)

        return {
            'sensitivity': S,
            'complementary_sensitivity': T,
            'loop_gain': G
        }


# =====================
# Causal GP for System Identification
# =====================

class CausalGaussianProcess(GaussianProcessRegressor):
    """GP with causality constraints for system identification."""

    def __init__(self, kernel: Kernel, noise_variance: float = 1e-6, enforce_causality: bool = True):
        super().__init__(kernel, noise_variance)
        self.enforce_causality = enforce_causality

    def predict_impulse_response(self, n_samples: int = 100) -> np.ndarray:
        """Predict impulse response from frequency response GP."""
        # Create dense frequency grid
        omega_max = np.max(self.X_train)
        omega_dense = np.linspace(0, omega_max, n_samples)

        # Predict on dense grid
        G_pred = self.predict(omega_dense.reshape(-1, 1))

        # Convert to time domain via IFFT
        # Ensure Hermitian symmetry for real impulse response
        n_fft = 2 * n_samples
        G_full = np.zeros(n_fft, dtype=complex)
        G_full[:n_samples] = G_pred
        G_full[n_samples:] = np.conj(G_pred[-2:0:-1])  # Mirror and conjugate

        # IFFT
        impulse_response = np.fft.ifft(G_full).real[:n_samples]

        if self.enforce_causality:
            # Zero out non-causal part
            impulse_response[:0] = 0

        return impulse_response


# =====================
# Visualization Extensions
# =====================

def plot_gp_diagnostics(omega: np.ndarray, G_true: np.ndarray, G_pred: np.ndarray,
                       G_std: Optional[Tuple[np.ndarray, np.ndarray]], output_dir: Path):
    """Create comprehensive diagnostic plots for GP frequency response."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = FrequencyDomainAnalyzer()

    # 1. Coherence plot
    coherence = analyzer.compute_coherence(G_true, G_pred)
    if len(coherence) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        freq_coherence = np.linspace(0, np.max(omega), len(coherence))
        ax.plot(freq_coherence, coherence, 'b-', linewidth=2)
        ax.set_xlabel('Frequency [rad/s]')
        ax.set_ylabel('Magnitude-Squared Coherence')
        ax.set_title('Coherence between Measured and GP-Predicted FRF')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig(output_dir / 'gp_coherence.png', dpi=300)
        plt.close()

    # 2. Stability analysis
    stability = analyzer.check_stability(G_pred, omega)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Nyquist plot with stability info
    ax1.plot(np.real(G_true), np.imag(G_true), 'k.', markersize=6, label='Measured', alpha=0.6)
    ax1.plot(np.real(G_pred), np.imag(G_pred), 'r-', linewidth=2, label='GP Predicted')
    ax1.plot(-1, 0, 'rx', markersize=15, markeredgewidth=3, label='Critical Point')

    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'g--', alpha=0.3, label='Unit Circle')

    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.set_title(f'Nyquist Plot (GM={stability["gain_margin_db"]:.1f}dB, PM={stability["phase_margin_deg"]:.1f}°)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Sensitivity functions
    sens_funcs = analyzer.compute_sensitivity_functions(G_pred)
    ax2.loglog(omega, np.abs(sens_funcs['sensitivity']), 'b-', linewidth=2, label='|S|')
    ax2.loglog(omega, np.abs(sens_funcs['complementary_sensitivity']), 'r-', linewidth=2, label='|T|')
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(y=np.sqrt(2), color='k', linestyle=':', alpha=0.5, label='3dB')
    ax2.set_xlabel(r'$\omega$ [rad/s]')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Sensitivity Functions')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'gp_stability_analysis.png', dpi=300)
    plt.close()

    # 3. Error analysis across frequency
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    error_mag = np.abs(np.abs(G_true) - np.abs(G_pred))
    error_phase = np.angle(G_true) - np.angle(G_pred)
    error_phase = np.angle(np.exp(1j * error_phase))  # Wrap to [-π, π]

    ax1.loglog(omega, error_mag, 'b.', markersize=6, alpha=0.6)
    ax1.set_xlabel(r'$\omega$ [rad/s]')
    ax1.set_ylabel('Magnitude Error')
    ax1.set_title('Absolute Magnitude Error vs Frequency')
    ax1.grid(True, which='both', alpha=0.3)

    ax2.semilogx(omega, error_phase * 180/np.pi, 'r.', markersize=6, alpha=0.6)
    ax2.set_xlabel(r'$\omega$ [rad/s]')
    ax2.set_ylabel('Phase Error [deg]')
    ax2.set_title('Phase Error vs Frequency')
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'gp_error_analysis.png', dpi=300)
    plt.close()

    # Save diagnostics summary
    diagnostics = {
        'stability': stability,
        'delay_estimate_seconds': float(analyzer.estimate_delay(omega, G_pred)),
        'mean_coherence': float(np.mean(coherence)) if len(coherence) > 1 else 1.0,
        'mean_magnitude_error': float(np.mean(error_mag)),
        'rms_phase_error_deg': float(np.sqrt(np.mean(error_phase**2)) * 180/np.pi)
    }

    import json
    with open(output_dir / 'gp_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)

    return diagnostics


# =====================
# Example Usage
# =====================

def example_advanced_gp_analysis():
    """Example of using advanced GP analysis tools."""

    # Generate synthetic frequency response data
    omega = np.logspace(-1, 2, 50)
    # Second-order system with resonance
    w0 = 10.0  # Natural frequency
    zeta = 0.3  # Damping ratio
    G_true = 1.0 / (1 - (omega/w0)**2 + 2j*zeta*(omega/w0))
    G_noisy = G_true + 0.01 * (np.random.randn(len(omega)) + 1j*np.random.randn(len(omega)))

    # 1. Fit complex GP with coupled kernels
    X = omega.reshape(-1, 1)

    # Use resonance kernel for both real and imaginary parts
    kernel_real = ResonanceKernel(resonance_freq=w0, damping=zeta)
    kernel_imag = ResonanceKernel(resonance_freq=w0, damping=zeta)

    complex_gp = ComplexGaussianProcess(kernel_real, kernel_imag, noise_variance=1e-4)
    complex_gp.fit(X, G_noisy, optimize=False)  # Use known parameters

    G_pred, (std_real, std_imag) = complex_gp.predict(X, return_std=True)

    # 2. Analyze results
    analyzer = FrequencyDomainAnalyzer()
    stability = analyzer.check_stability(G_pred, omega)
    print(f"Stability analysis: {stability}")

    # 3. Create diagnostic plots
    output_dir = Path('gp_diagnostics_example')
    plot_gp_diagnostics(omega, G_noisy, G_pred, (std_real, std_imag), output_dir)

    print(f"Advanced GP analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    example_advanced_gp_analysis()