#!/usr/bin/env python3
"""
unified_pipeline.py

A comprehensive pipeline that:
1. Runs frequency_response.py on input .mat files
2. Loads the resulting frequency response data
3. Applies Gaussian Process regression with extensible kernels
4. Visualizes the results

The kernel architecture is designed to be highly extensible for adding new kernels.

Usage:
    python src/unified_pipeline.py input/*.mat --kernel rbf --out-dir gp_output
    python src/unified_pipeline.py --use-existing output/matched_frf.csv --kernel matern --nu 2.5
    python src/unified_pipeline.py input/*.mat --n-files 1 
      --kernel rbf --normalize --log-frequency 
      --extract-fir --fir-length 1024 
      --fir-validation-mat input/input_test_20250912_165937.mat 
      --out-dir output_complete
    python src/unified_pipeline.py input/*.mat --n-files 1       --kernel rbf --normalize --log-frequency       --extract-fir --fir-length 1024       --fir-validation-mat input/input_test_20250912_165937.mat       --out-dir output_complete
"""

import argparse
import json
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Import FIR extraction modules
gp_to_fir_linear_pipeline = None

try:
    from gp_to_fir_direct import gp_to_fir_direct_pipeline
except ImportError:
    gp_to_fir_direct_pipeline = None


# =====================
# Kernel Base Classes
# =====================

class Kernel(ABC):
    """Abstract base class for all GP kernels."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.bounds = self._get_default_bounds()

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix K(X1, X2). If X2 is None, compute K(X1, X1)."""
        pass

    @abstractmethod
    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        """Return default bounds for hyperparameter optimization."""
        pass

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Get current hyperparameters as array."""
        pass

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Set hyperparameters from array."""
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of hyperparameters."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class CombinedKernel(Kernel):
    """Base class for combined kernels (sum, product)."""

    def __init__(self, kernels: List[Kernel]):
        self.kernels = kernels
        super().__init__()

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        for kernel in self.kernels:
            bounds.extend(kernel.bounds)
        return bounds

    def get_params(self) -> np.ndarray:
        params = []
        for kernel in self.kernels:
            params.extend(kernel.get_params())
        return np.array(params)

    def set_params(self, params: np.ndarray) -> None:
        idx = 0
        for kernel in self.kernels:
            n = kernel.n_params
            kernel.set_params(params[idx:idx+n])
            idx += n

    @property
    def n_params(self) -> int:
        return sum(k.n_params for k in self.kernels)


# =====================
# Standard GP Kernels
# =====================

class RBFKernel(Kernel):
    """Radial Basis Function (Squared Exponential) kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        super().__init__(length_scale=length_scale, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        # Compute squared distances
        dists = cdist(X1, X2, metric='sqeuclidean')

        # RBF kernel
        K = self.params['variance'] * np.exp(-0.5 * dists / (self.params['length_scale']**2))
        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e3),  # length_scale
            (1e-3, 1e3),  # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['length_scale'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['length_scale'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2


class MaternKernel(Kernel):
    """Matern kernel with parameter nu."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, nu: float = 1.5):
        super().__init__(length_scale=length_scale, variance=variance, nu=nu)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        dists = cdist(X1, X2)
        dists = dists / self.params['length_scale']
        nu = self.params['nu']

        if nu == 0.5:
            # Exponential kernel
            K = np.exp(-dists)
        elif nu == 1.5:
            K = (1.0 + np.sqrt(3.0) * dists) * np.exp(-np.sqrt(3.0) * dists)
        elif nu == 2.5:
            K = (1.0 + np.sqrt(5.0) * dists + 5.0/3.0 * dists**2) * np.exp(-np.sqrt(5.0) * dists)
        else:
            # General case (requires special functions)
            from scipy.special import kv, gamma
            K = dists**nu
            K *= kv(nu, np.sqrt(2.0 * nu) * dists)
            K *= 2.0**(1.0 - nu) / gamma(nu)
            K[dists == 0] = 1.0

        K *= self.params['variance']
        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e3),  # length_scale
            (1e-3, 1e3),  # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['length_scale'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['length_scale'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2


class RationalQuadraticKernel(Kernel):
    """Rational Quadratic kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, alpha: float = 1.0):
        super().__init__(length_scale=length_scale, variance=variance, alpha=alpha)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        dists_sq = cdist(X1, X2, metric='sqeuclidean')
        K = self.params['variance'] * (1.0 + dists_sq / (2.0 * self.params['alpha'] * self.params['length_scale']**2))**(-self.params['alpha'])
        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e3),  # length_scale
            (1e-3, 1e3),  # variance
            (1e-3, 1e2),  # alpha
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['length_scale'], self.params['variance'], self.params['alpha']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['length_scale'] = params[0]
        self.params['variance'] = params[1]
        self.params['alpha'] = params[2]

    @property
    def n_params(self) -> int:
        return 3


# =====================
# Frequency-Domain Kernels
# =====================

class TCKernel(Kernel):
    """Tuned/Correlated kernel from Chen, Ohlsson, Ljung 2012."""

    def __init__(self, beta: float = 0.95, rho: float = 0.8, variance: float = 1.0):
        super().__init__(beta=beta, rho=rho, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        # X should be frequency (omega) values
        omega1 = X1.ravel()
        omega2 = X2.ravel()

        # TC kernel: k(ω_i, ω_j) = c * β^(ω_i + ω_j) * ρ^|ω_i - ω_j|
        omega_sum = omega1[:, None] + omega2[None, :]
        omega_diff = np.abs(omega1[:, None] - omega2[None, :])

        K = self.params['variance'] * (self.params['beta'] ** omega_sum) * (self.params['rho'] ** omega_diff)
        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (0.01, 0.99),  # beta
            (0.01, 0.99),  # rho
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['beta'], self.params['rho'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['beta'] = params[0]
        self.params['rho'] = params[1]
        self.params['variance'] = params[2]

    @property
    def n_params(self) -> int:
        return 3


class ExponentialDecayKernel(Kernel):
    """Exponential decay kernel for frequency domain."""

    def __init__(self, decay_rate: float = 0.1, variance: float = 1.0):
        super().__init__(decay_rate=decay_rate, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        omega1 = X1.ravel()
        omega2 = X2.ravel()

        # Exponential decay based on frequency magnitude
        omega_prod = omega1[:, None] * omega2[None, :]
        K = self.params['variance'] * np.exp(-self.params['decay_rate'] * omega_prod)
        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 10.0),  # decay_rate
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['decay_rate'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['decay_rate'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2


# =====================
# Kernel Combinations
# =====================

class SumKernel(CombinedKernel):
    """Sum of multiple kernels."""

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        K = np.zeros_like(self.kernels[0](X1, X2))
        for kernel in self.kernels:
            K += kernel(X1, X2)
        return K


class ProductKernel(CombinedKernel):
    """Product of multiple kernels."""

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        K = np.ones_like(self.kernels[0](X1, X2))
        for kernel in self.kernels:
            K *= kernel(X1, X2)
        return K


# =====================
# GP Regression Class
# =====================

@dataclass
class GPConfig:
    """Configuration for GP regression."""
    kernel_type: str = 'rbf'
    kernel_params: Dict = field(default_factory=dict)
    noise_variance: float = 1e-6
    optimize: bool = True
    n_restarts: int = 3
    normalize_inputs: bool = True
    normalize_outputs: bool = True


class GaussianProcessRegressor:
    """Gaussian Process Regressor with extensible kernel support."""

    def __init__(self, kernel: Kernel, noise_variance: float = 1e-6):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        self.X_scaler = None
        self.y_scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True, n_restarts: int = 3):
        """Fit the GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()

        if optimize:
            self._optimize_hyperparameters(n_restarts)

        # Compute kernel matrix and its inverse
        K = self.kernel(self.X_train)
        K += self.noise_variance * np.eye(K.shape[0])

        # Cholesky decomposition for stable inversion
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L, self.y_train)
            self.alpha = np.linalg.solve(L.T, self.alpha)
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Fallback to SVD if Cholesky fails
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ self.y_train

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict mean and optionally standard deviation at test points."""
        K_star = self.kernel(X, self.X_train)
        mean = K_star @ self.alpha

        if return_std:
            K_star_star = self.kernel(X)
            var = np.diag(K_star_star) - np.sum((K_star @ self.K_inv) * K_star, axis=1)
            std = np.sqrt(np.maximum(var, 0))
            return mean, std

        return mean

    def _optimize_hyperparameters(self, n_restarts: int):
        """Optimize kernel hyperparameters by maximizing log marginal likelihood."""
        def neg_log_marginal_likelihood(params):
            self.kernel.set_params(params[:-1])
            self.noise_variance = np.exp(params[-1])  # Log-transform noise

            K = self.kernel(self.X_train)
            K += self.noise_variance * np.eye(K.shape[0])

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, self.y_train)
                alpha = np.linalg.solve(L.T, alpha)

                # Negative log marginal likelihood
                nll = 0.5 * (self.y_train @ alpha)
                nll += np.sum(np.log(np.diag(L)))
                nll += 0.5 * len(self.y_train) * np.log(2 * np.pi)

                return nll
            except np.linalg.LinAlgError:
                return 1e10

        # Multiple random restarts
        best_params = None
        best_nll = np.inf

        # Get bounds for optimization
        bounds = self.kernel.bounds + [(np.log(1e-10), np.log(1e-1))]  # Noise variance bounds (log scale)

        for _ in range(n_restarts):
            # Random initialization
            init_params = []
            for low, high in bounds:
                if low > 0 and high / low > 100:  # Log scale for large ranges
                    init_params.append(np.exp(np.random.uniform(np.log(low), np.log(high))))
                else:
                    init_params.append(np.random.uniform(low, high))
            init_params = np.array(init_params)
            init_params[-1] = np.log(self.noise_variance)  # Convert noise to log scale

            # Add current parameters as one of the starting points
            if _ == 0:
                init_params[:-1] = self.kernel.get_params()

            result = minimize(neg_log_marginal_likelihood, init_params, bounds=bounds, method='L-BFGS-B')

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x

        # Set optimal parameters
        self.kernel.set_params(best_params[:-1])
        self.noise_variance = np.exp(best_params[-1])


# =====================
# Kernel Factory
# =====================

def create_kernel(kernel_type: str, **kwargs) -> Kernel:
    """Factory function to create kernels by name."""
    kernel_map = {
        'rbf': RBFKernel,
        'matern': MaternKernel,
        'rq': RationalQuadraticKernel,
        'tc': TCKernel,
        'exp': ExponentialDecayKernel,
    }

    if kernel_type not in kernel_map:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available: {list(kernel_map.keys())}")

    return kernel_map[kernel_type](**kwargs)


# =====================
# Data Loading Functions
# =====================

def load_frf_data(frf_file: Path) -> pd.DataFrame:
    """Load frequency response function data from CSV."""
    df = pd.read_csv(frf_file)
    required_cols = ['omega_rad_s', 'ReG', 'ImG', 'absG', 'phase_rad']

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"FRF file must contain columns: {required_cols}")

    return df


def run_frequency_response(mat_files: List[str], output_dir: Path, n_files: int = 1) -> Path:
    """Run frequency_response.py and return path to output CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        'src/frequency_response.py',
        *mat_files,
        '--n-files', str(n_files),
        '--out-dir', str(output_dir),
        '--out-prefix', 'unified'
    ]

    print(f"Running frequency_response.py with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running frequency_response.py:\n{result.stderr}")
        raise RuntimeError(f"frequency_response.py failed with code {result.returncode}")

    frf_csv = output_dir / 'unified_frf.csv'
    if not frf_csv.exists():
        raise RuntimeError(f"Expected output file not found: {frf_csv}")

    return frf_csv


# =====================
# Visualization Functions
# =====================

def plot_gp_results(omega: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                   y_std: Optional[np.ndarray], title: str, output_path: Path):
    """Plot GP regression results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: True vs Predicted
    ax1.semilogx(omega, y_true, 'k.', markersize=8, label='Measured', alpha=0.6)
    ax1.semilogx(omega, y_pred, 'r-', linewidth=2, label='GP mean')

    if y_std is not None:
        ax1.fill_between(omega, y_pred - 2*y_std, y_pred + 2*y_std,
                        alpha=0.3, color='red', label='95% confidence')

    ax1.set_xlabel(r'$\omega$ [rad/s]')
    ax1.set_ylabel(title)
    ax1.set_title(f'GP Regression: {title}')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    # Plot 2: Residuals
    residuals = y_true - y_pred
    ax2.semilogx(omega, residuals, 'b.', markersize=6, alpha=0.6)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$\omega$ [rad/s]')
    ax2.set_ylabel('Residual')
    ax2.set_title('Prediction Residuals')
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Calculate metrics
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)

    return {'rmse': rmse, 'r2': r2}


def plot_complex_gp(omega: np.ndarray, G_true: np.ndarray, G_pred: np.ndarray,
                   G_std_real: Optional[np.ndarray], G_std_imag: Optional[np.ndarray],
                   output_prefix: Path):
    """Plot complex-valued GP results (magnitude/phase and Nyquist)."""
    mag_true = np.abs(G_true)
    mag_pred = np.abs(G_pred)
    phase_true = np.angle(G_true)
    phase_pred = np.angle(G_pred)

    # Bode magnitude plot
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(omega, mag_true, 'k.', markersize=8, label='Measured', alpha=0.6)
    ax.loglog(omega, mag_pred, 'r-', linewidth=2, label='GP mean')
    ax.set_xlabel(r'$\omega$ [rad/s]')
    ax.set_ylabel('|G(jω)|')
    ax.set_title('Bode Magnitude Plot')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_prefix) + '_bode_mag_gp.png', dpi=300)
    plt.close()

    # Bode phase plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(omega, np.unwrap(phase_true), 'k.', markersize=8, label='Measured', alpha=0.6)
    ax.semilogx(omega, np.unwrap(phase_pred), 'r-', linewidth=2, label='GP mean')
    ax.set_xlabel(r'$\omega$ [rad/s]')
    ax.set_ylabel('Phase [rad]')
    ax.set_title('Bode Phase Plot')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_prefix) + '_bode_phase_gp.png', dpi=300)
    plt.close()

    # Nyquist plot
    fig3, ax = plt.subplots(figsize=(8, 8))
    ax.plot(np.real(G_true), np.imag(G_true), 'k.', markersize=8, label='Measured', alpha=0.6)
    ax.plot(np.real(G_pred), np.imag(G_pred), 'r-', linewidth=2, label='GP mean')

    if G_std_real is not None and G_std_imag is not None:
        # Plot confidence ellipses at selected frequencies
        n_ellipses = min(20, len(omega))
        indices = np.linspace(0, len(omega)-1, n_ellipses, dtype=int)
        for i in indices:
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_x = np.real(G_pred[i]) + 2*G_std_real[i] * np.cos(theta)
            ellipse_y = np.imag(G_pred[i]) + 2*G_std_imag[i] * np.sin(theta)
            ax.plot(ellipse_x, ellipse_y, 'r-', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('Real{G(jω)}')
    ax.set_ylabel('Imag{G(jω)}')
    ax.set_title('Nyquist Plot with GP Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(str(output_prefix) + '_nyquist_gp.png', dpi=300)
    plt.close()


# =====================
# Main Pipeline Function
# =====================

def run_gp_pipeline(config: argparse.Namespace):
    """Run the complete frequency response -> GP regression pipeline."""
    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get FRF data
    if config.use_existing:
        print(f"Using existing FRF data from: {config.use_existing}")
        frf_csv = Path(config.use_existing)
    else:
        print("Running frequency_response.py...")
        frf_csv = run_frequency_response(config.mat_files, output_dir, config.n_files)

    # Step 2: Load FRF data
    print("Loading FRF data...")
    frf_df = load_frf_data(frf_csv)

    omega = frf_df['omega_rad_s'].values
    G_complex = frf_df['ReG'].values + 1j * frf_df['ImG'].values

    # Prepare data for GP
    X = omega.reshape(-1, 1)

    # Option to work with log-frequency
    if config.log_frequency:
        X_gp = np.log10(X)
    else:
        X_gp = X

    # Step 3: Create kernel
    print(f"Creating {config.kernel} kernel...")
    kernel_params = {}

    # Parse kernel-specific parameters
    if config.kernel == 'matern' and config.nu is not None:
        kernel_params['nu'] = config.nu

    kernel = create_kernel(config.kernel, **kernel_params)

    # Step 4: Fit GP models
    results = {}

    # Option 1: Separate GPs for real and imaginary parts
    if config.gp_mode == 'separate':
        print("Fitting separate GPs for real and imaginary parts...")

        # Normalize if requested
        if config.normalize:
            X_scaler = StandardScaler()
            X_gp_normalized = X_scaler.fit_transform(X_gp)

            y_real_scaler = StandardScaler()
            y_real = y_real_scaler.fit_transform(np.real(G_complex).reshape(-1, 1)).ravel()

            y_imag_scaler = StandardScaler()
            y_imag = y_imag_scaler.fit_transform(np.imag(G_complex).reshape(-1, 1)).ravel()
        else:
            X_gp_normalized = X_gp
            y_real = np.real(G_complex)
            y_imag = np.imag(G_complex)

        # Fit real part
        gp_real = GaussianProcessRegressor(kernel=create_kernel(config.kernel, **kernel_params),
                                         noise_variance=config.noise_variance)
        gp_real.fit(X_gp_normalized, y_real, optimize=config.optimize, n_restarts=config.n_restarts)

        y_real_pred, y_real_std = gp_real.predict(X_gp_normalized, return_std=True)

        if config.normalize:
            y_real_pred = y_real_scaler.inverse_transform(y_real_pred.reshape(-1, 1)).ravel()
            y_real_std = y_real_std * y_real_scaler.scale_
            y_real_orig = y_real_scaler.inverse_transform(y_real.reshape(-1, 1)).ravel()
        else:
            y_real_orig = y_real

        # Fit imaginary part
        gp_imag = GaussianProcessRegressor(kernel=create_kernel(config.kernel, **kernel_params),
                                         noise_variance=config.noise_variance)
        gp_imag.fit(X_gp_normalized, y_imag, optimize=config.optimize, n_restarts=config.n_restarts)

        y_imag_pred, y_imag_std = gp_imag.predict(X_gp_normalized, return_std=True)

        if config.normalize:
            y_imag_pred = y_imag_scaler.inverse_transform(y_imag_pred.reshape(-1, 1)).ravel()
            y_imag_std = y_imag_std * y_imag_scaler.scale_
            y_imag_orig = y_imag_scaler.inverse_transform(y_imag.reshape(-1, 1)).ravel()
        else:
            y_imag_orig = y_imag

        # Plot results
        metrics_real = plot_gp_results(omega, y_real_orig, y_real_pred, y_real_std,
                                     'Real{G(jω)}', output_dir / 'gp_real.png')
        metrics_imag = plot_gp_results(omega, y_imag_orig, y_imag_pred, y_imag_std,
                                     'Imag{G(jω)}', output_dir / 'gp_imag.png')

        # Complex plots
        G_pred = y_real_pred + 1j * y_imag_pred
        plot_complex_gp(omega, G_complex, G_pred, y_real_std, y_imag_std, output_dir / 'gp_complex')

        results['real'] = metrics_real
        results['imag'] = metrics_imag
        results['kernel_params'] = {
            'real': gp_real.kernel.get_params().tolist(),
            'imag': gp_imag.kernel.get_params().tolist(),
            'noise_real': float(gp_real.noise_variance),
            'noise_imag': float(gp_imag.noise_variance)
        }

    # Option 2: GP on magnitude and phase
    elif config.gp_mode == 'polar':
        print("Fitting GPs for magnitude and phase...")

        mag = np.abs(G_complex)
        phase = np.unwrap(np.angle(G_complex))

        # Log transform magnitude
        log_mag = np.log(mag)

        # Normalize if requested
        if config.normalize:
            X_scaler = StandardScaler()
            X_gp_normalized = X_scaler.fit_transform(X_gp)

            mag_scaler = StandardScaler()
            y_mag = mag_scaler.fit_transform(log_mag.reshape(-1, 1)).ravel()

            phase_scaler = StandardScaler()
            y_phase = phase_scaler.fit_transform(phase.reshape(-1, 1)).ravel()
        else:
            X_gp_normalized = X_gp
            y_mag = log_mag
            y_phase = phase

        # Fit magnitude (in log scale)
        gp_mag = GaussianProcessRegressor(kernel=create_kernel(config.kernel, **kernel_params),
                                        noise_variance=config.noise_variance)
        gp_mag.fit(X_gp_normalized, y_mag, optimize=config.optimize, n_restarts=config.n_restarts)

        y_mag_pred, y_mag_std = gp_mag.predict(X_gp_normalized, return_std=True)

        if config.normalize:
            y_mag_pred = mag_scaler.inverse_transform(y_mag_pred.reshape(-1, 1)).ravel()
            y_mag_std = y_mag_std * mag_scaler.scale_

        mag_pred = np.exp(y_mag_pred)

        # Fit phase
        gp_phase = GaussianProcessRegressor(kernel=create_kernel(config.kernel, **kernel_params),
                                          noise_variance=config.noise_variance)
        gp_phase.fit(X_gp_normalized, y_phase, optimize=config.optimize, n_restarts=config.n_restarts)

        y_phase_pred, y_phase_std = gp_phase.predict(X_gp_normalized, return_std=True)

        if config.normalize:
            y_phase_pred = phase_scaler.inverse_transform(y_phase_pred.reshape(-1, 1)).ravel()
            y_phase_std = y_phase_std * phase_scaler.scale_

        # Plot results
        metrics_mag = plot_gp_results(omega, mag, mag_pred, None,
                                    '|G(jω)|', output_dir / 'gp_magnitude.png')
        metrics_phase = plot_gp_results(omega, phase, y_phase_pred, y_phase_std,
                                      'Phase [rad]', output_dir / 'gp_phase.png')

        # Complex plots
        G_pred = mag_pred * np.exp(1j * y_phase_pred)
        plot_complex_gp(omega, G_complex, G_pred, None, None, output_dir / 'gp_complex')

        results['magnitude'] = metrics_mag
        results['phase'] = metrics_phase
        results['kernel_params'] = {
            'magnitude': gp_mag.kernel.get_params().tolist(),
            'phase': gp_phase.kernel.get_params().tolist(),
            'noise_mag': float(gp_mag.noise_variance),
            'noise_phase': float(gp_phase.noise_variance)
        }

    # Save results summary
    results['config'] = {
        'kernel': config.kernel,
        'gp_mode': config.gp_mode,
        'normalize': config.normalize,
        'log_frequency': config.log_frequency,
        'optimize': config.optimize,
    }

    with open(output_dir / 'gp_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"GP regression complete. Results saved to {output_dir}")

    # Save GP-smoothed FRF for downstream use (e.g., FIR identification)
    if config.gp_mode == 'separate':
        smoothed_df = pd.DataFrame({
            'omega_rad_s': omega,
            'freq_Hz': omega / (2 * np.pi),
            'ReG': y_real_pred,
            'ImG': y_imag_pred,
            'absG': np.abs(y_real_pred + 1j * y_imag_pred),
            'phase_rad': np.angle(y_real_pred + 1j * y_imag_pred)
        })
    else:  # polar mode
        smoothed_df = pd.DataFrame({
            'omega_rad_s': omega,
            'freq_Hz': omega / (2 * np.pi),
            'ReG': np.real(G_pred),
            'ImG': np.imag(G_pred),
            'absG': mag_pred,
            'phase_rad': y_phase_pred
        })

    smoothed_csv = output_dir / 'gp_smoothed_frf.csv'
    smoothed_df.to_csv(smoothed_csv, index=False)
    print(f"GP-smoothed FRF saved to {smoothed_csv}")

    # Step 4: FIR model extraction (if requested)
    if config.extract_fir:
        print("\n" + "="*70)
        print("STEP 4: Extracting FIR model coefficients")
        print("="*70)

        # Use GP-smoothed frequency response
        if config.gp_mode == 'separate':
            G_smoothed = y_real_pred + 1j * y_imag_pred
        else:  # polar mode
            G_smoothed = G_pred

        # Get validation MAT file if specified
        validation_mat = None
        if config.fir_validation_mat:
            validation_mat = Path(config.fir_validation_mat)
            if not validation_mat.exists():
                print(f"Warning: Validation MAT file not found: {validation_mat}")
                validation_mat = None

        # Create a GP prediction function for better interpolation
        def gp_predict_at_omega(omega_new):
            """Predict using the fitted GP models at new frequencies."""
            X_new = omega_new.reshape(-1, 1)

            # Apply same transformations as during fitting
            if config.log_frequency:
                X_new_gp = np.log10(X_new)
            else:
                X_new_gp = X_new

            if config.normalize:
                X_new_normalized = X_scaler.transform(X_new_gp)
            else:
                X_new_normalized = X_new_gp

            # Predict with GPs
            if config.gp_mode == 'separate':
                y_real_new = gp_real.predict(X_new_normalized)
                y_imag_new = gp_imag.predict(X_new_normalized)

                # Denormalize if needed
                if config.normalize:
                    y_real_new = y_real_scaler.inverse_transform(y_real_new.reshape(-1, 1)).ravel()
                    y_imag_new = y_imag_scaler.inverse_transform(y_imag_new.reshape(-1, 1)).ravel()

                return y_real_new + 1j * y_imag_new
            else:
                # Polar mode
                y_mag_new = gp_mag.predict(X_new_normalized)
                y_phase_new = gp_phase.predict(X_new_normalized)

                if config.normalize:
                    y_mag_new = mag_scaler.inverse_transform(y_mag_new.reshape(-1, 1)).ravel()
                    y_phase_new = phase_scaler.inverse_transform(y_phase_new.reshape(-1, 1)).ravel()

                mag_new = np.exp(y_mag_new)
                return mag_new * np.exp(1j * y_phase_new)

        # Use GP-direct method
        if gp_to_fir_direct_pipeline is not None:
            print("Using GP-based interpolation for FIR extraction...")
            fir_output_dir = output_dir / 'fir_gp'
            fir_results = gp_to_fir_direct_pipeline(
                omega=omega,
                G=G_smoothed,
                gp_predict_func=gp_predict_at_omega,
                mat_file=validation_mat,
                output_dir=fir_output_dir,
                fir_length=config.fir_length
            )

            # Add FIR results to overall results
            results['fir_extraction'] = fir_results

            # Update saved results
            with open(output_dir / 'gp_results.json', 'w') as f:
                json.dump(results, f, indent=2)

            print(f"FIR extraction complete. Results saved to {fir_output_dir}")
        else:
            print("Warning: gp_to_fir_direct module not available")


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for frequency response analysis and GP regression"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('mat_files', nargs='*', default=[],
                           help='MAT files for frequency response analysis')
    input_group.add_argument('--use-existing', type=str,
                           help='Use existing FRF CSV file instead of running frequency_response.py')

    # Frequency response options
    parser.add_argument('--n-files', type=int, default=1,
                      help='Number of MAT files to process (default: 1)')

    # GP options
    parser.add_argument('--kernel', type=str, default='rbf',
                      choices=['rbf', 'matern', 'rq', 'tc', 'exp'],
                      help='Kernel type (default: rbf)')
    parser.add_argument('--nu', type=float, default=None,
                      help='Nu parameter for Matern kernel (default: 1.5)')
    parser.add_argument('--gp-mode', type=str, default='separate',
                      choices=['separate', 'polar'],
                      help='GP mode: separate (real/imag) or polar (mag/phase)')
    parser.add_argument('--noise-variance', type=float, default=1e-6,
                      help='Initial noise variance (default: 1e-6)')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize inputs and outputs')
    parser.add_argument('--log-frequency', action='store_true',
                      help='Use log-frequency as GP input')
    parser.add_argument('--optimize', action='store_true', default=True,
                      help='Optimize hyperparameters (default: True)')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                      help='Disable hyperparameter optimization')
    parser.add_argument('--n-restarts', type=int, default=3,
                      help='Number of optimization restarts (default: 3)')

    # Output options
    parser.add_argument('--out-dir', type=str, default='gp_output',
                      help='Output directory (default: gp_output)')

    # FIR extraction options
    parser.add_argument('--extract-fir', action='store_true',
                      help='Extract FIR model coefficients using linear interpolation method')
    parser.add_argument('--fir-length', type=int, default=1024,
                      help='FIR filter length (default: 1024)')
    parser.add_argument('--fir-validation-mat', type=str, default=None,
                      help='MAT file with [time, output, input] for FIR validation')

    args = parser.parse_args()

    # Validate inputs
    if not args.use_existing and not args.mat_files:
        parser.error("Either provide MAT files or use --use-existing")

    # Run pipeline
    run_gp_pipeline(args)


if __name__ == "__main__":
    main()