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
    python src/unified_pipeline.py input/*.mat --n-files 1 --nd 50
      --kernel rbf --normalize --log-frequency
      --extract-fir --fir-length 1024
      --fir-validation-mat input/input_test_20250912_165937.mat
      --out-dir output_complete
    python src/unified_pipeline.py input/*.mat --n-files 1 --nd 100 --freq-method fourier
      --kernel rbf --normalize --extract-fir --fir-length 1024
      --fir-validation-mat input/input_test_20250912_165937.mat
      --out-dir output_fourier
"""

import argparse
import glob
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

def _prepare_kernel_inputs(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float).reshape(-1)

def _heaviside(x: np.ndarray) -> np.ndarray:
    return (x >= 0.0).astype(float)

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

class ExponentialKernel(Kernel):
    """First-order stable spline kernel with Heaviside support."""

    def __init__(self, omega: float = 1.0, variance: float = 1.0):
        super().__init__(omega=omega, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        x1 = _prepare_kernel_inputs(X1)
        x2 = _prepare_kernel_inputs(X2)
        H1 = _heaviside(x1)
        H2 = _heaviside(x2)
        sum_grid = x1[:, None] + x2[None, :]
        K = np.exp(-self.params['omega'] * sum_grid)
        K *= H1[:, None] * H2[None, :]
        return self.params['variance'] * K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e2),   # omega
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['omega'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['omega'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2

class TCKernel(Kernel):
    """Turned Correlated kernel with Heaviside support."""

    def __init__(self, omega: float = 1.0, variance: float = 1.0):
        super().__init__(omega=omega, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        x1 = _prepare_kernel_inputs(X1)
        x2 = _prepare_kernel_inputs(X2)
        H1 = _heaviside(x1)
        H2 = _heaviside(x2)
        max_grid = np.maximum.outer(x1, x2)
        K = np.exp(-self.params['omega'] * max_grid)
        K *= H1[:, None] * H2[None, :]
        return self.params['variance'] * K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e2),   # omega
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['omega'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['omega'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2

class DCKernel(Kernel):
    """Diagonal correlated kernel."""

    def __init__(self, alpha: float = 0.9, beta: float = 1.0, rho: float = 0.5):
        super().__init__(alpha=alpha, beta=beta, rho=rho)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        i = np.rint(_prepare_kernel_inputs(X1)).astype(int)
        j = np.rint(_prepare_kernel_inputs(X2)).astype(int)
        sum_indices = (i[:, None] + j[None, :]) / 2.0
        diff_indices = np.abs(i[:, None] - j[None, :])
        K = (self.params['alpha'] ** sum_indices) * (self.params['rho'] ** diff_indices)
        return self.params['beta'] * K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 0.999),   # alpha
            (1e-3, 1e3),     # beta
            (-0.999, 0.999), # rho
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['alpha'], self.params['beta'], self.params['rho']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['alpha'] = params[0]
        self.params['beta'] = params[1]
        self.params['rho'] = params[2]

    @property
    def n_params(self) -> int:
        return 3

class DIKernel(Kernel):
    """Diagonal-independent kernel."""

    def __init__(self, beta: float = 1.0, alpha: float = 0.9):
        super().__init__(beta=beta, alpha=alpha)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        i = np.rint(_prepare_kernel_inputs(X1)).astype(int)
        j = np.rint(_prepare_kernel_inputs(X2)).astype(int)
        K = np.zeros((i.size, j.size), dtype=float)
        diag_mask = i[:, None] == j[None, :]
        if np.any(diag_mask):
            row_idx, col_idx = np.where(diag_mask)
            values = self.params['beta'] * (self.params['alpha'] ** i[row_idx])
            K[row_idx, col_idx] = values
        return K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e3),   # beta
            (1e-3, 0.999), # alpha
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['beta'], self.params['alpha']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['beta'] = params[0]
        self.params['alpha'] = params[1]

    @property
    def n_params(self) -> int:
        return 2

class FirstOrderStableSplineKernel(Kernel):
    """First-order stable spline kernel."""

    def __init__(self, beta: float = 1.0, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        s = _prepare_kernel_inputs(X1)
        t = _prepare_kernel_inputs(X2)
        min_grid = np.minimum.outer(s, t)
        K = np.exp(-self.params['beta'] * min_grid)
        return self.params['variance'] * K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e2),   # beta
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['beta'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['beta'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2

class SecondOrderStableSplineKernel(Kernel):
    """Second-order stable spline kernel."""

    def __init__(self, beta: float = 1.0, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        s = _prepare_kernel_inputs(X1)
        t = _prepare_kernel_inputs(X2)
        beta = self.params['beta']
        sum_grid = s[:, None] + t[None, :]
        max_grid = np.maximum.outer(s, t)
        first_term = 0.5 * np.exp(-beta * (sum_grid + max_grid))
        second_term = (1.0 / 6.0) * np.exp(-3.0 * beta * max_grid)
        K = first_term - second_term
        return self.params['variance'] * K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e2),   # beta
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['beta'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['beta'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2

class HighFrequencyStableSplineKernel(Kernel):
    """High-frequency stable spline kernel."""

    def __init__(self, beta: float = 1.0, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        s = _prepare_kernel_inputs(X1)
        t = _prepare_kernel_inputs(X2)
        s_idx = np.rint(s).astype(int)
        t_idx = np.rint(t).astype(int)
        sign = np.power(-1.0, s_idx[:, None] + t_idx[None, :])
        max_term = np.maximum(np.exp(-self.params['beta'] * s[:, None]),
                              np.exp(-self.params['beta'] * t[None, :]))
        K = sign * max_term
        return self.params['variance'] * K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e2),   # beta
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['beta'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['beta'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2

class StableSplineKernel(Kernel):
    """Non-stationary stable spline kernel with exponential warping."""

    def __init__(self, beta: float = 0.5, variance: float = 1.0):
        super().__init__(beta=beta, variance=variance)

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        x1 = _prepare_kernel_inputs(X1)
        x2 = _prepare_kernel_inputs(X2)
        beta = self.params['beta']
        exp_x1 = np.exp(-beta * x1)
        exp_x2 = np.exp(-beta * x2)
        r = np.minimum.outer(exp_x1, exp_x2)
        R = np.maximum.outer(exp_x1, exp_x2)
        K = 0.5 * r**2 * (R - r / 3.0)
        return self.params['variance'] * K

    def _get_default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (1e-3, 1e2),   # beta
            (1e-3, 1e3),   # variance
        ]

    def get_params(self) -> np.ndarray:
        return np.array([self.params['beta'], self.params['variance']])

    def set_params(self, params: np.ndarray) -> None:
        self.params['beta'] = params[0]
        self.params['variance'] = params[1]

    @property
    def n_params(self) -> int:
        return 2

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


@dataclass
class SystemIDConfig:
    """Configuration for classical/ML system identification methods."""
    method_type: str = 'gp'  # 'gp', 'nls', 'ls', 'iwls', 'tls', 'ml', 'log', 'lpm', 'lrmp', 'rf', 'gbr', 'svm'
    n_numerator: int = 2
    n_denominator: int = 2
    method_params: Dict = field(default_factory=dict)


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
        'matern12': lambda **kw: MaternKernel(nu=0.5, **{k: v for k, v in kw.items() if k != 'nu'}),
        'matern32': lambda **kw: MaternKernel(nu=1.5, **{k: v for k, v in kw.items() if k != 'nu'}),
        'matern52': lambda **kw: MaternKernel(nu=2.5, **{k: v for k, v in kw.items() if k != 'nu'}),
        'rq': RationalQuadraticKernel,
        'exp': ExponentialKernel,
        'tc': TCKernel,
        'dc': DCKernel,
        'di': DIKernel,
        'ss1': FirstOrderStableSplineKernel,
        'ss2': SecondOrderStableSplineKernel,
        'sshf': HighFrequencyStableSplineKernel,
        'stable_spline': StableSplineKernel,
    }

    if kernel_type not in kernel_map:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available: {list(kernel_map.keys())}")

    creator = kernel_map[kernel_type]
    if callable(creator) and not isinstance(creator, type):
        return creator(**kwargs)
    else:
        return creator(**kwargs)


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


def run_frequency_response(mat_files: List[str], output_dir: Path, n_files: int = 1, time_duration: Optional[float] = None, nd: int = 100) -> Path:
    """Run frequency_response.py and return path to output CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        'src/frequency_response.py',
        *mat_files,
        '--n-files', str(n_files),
        '--out-dir', str(output_dir),
        '--out-prefix', 'unified',
        '--nd', str(nd)
    ]

    if time_duration is not None and n_files == 1:
        cmd.extend(['--time-duration', str(time_duration)])

    print(f"Running frequency_response.py with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running frequency_response.py:\n{result.stderr}")
        raise RuntimeError(f"frequency_response.py failed with code {result.returncode}")

    frf_csv = output_dir / 'unified_frf.csv'
    if not frf_csv.exists():
        raise RuntimeError(f"Expected output file not found: {frf_csv}")

    return frf_csv


def run_fourier_transform(mat_files: List[str], output_dir: Path, n_files: int = 1, time_duration: Optional[float] = None, nd: int = 100) -> Path:
    """Run fourier_transform.py and return path to output CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        'src/fourier_transform.py',
        *mat_files,
        '--out-dir', str(output_dir),
        '--out-prefix', 'unified',
        '--nd', str(nd)
    ]

    if n_files is not None:
        cmd.extend(['--n-files', str(n_files)])

    if time_duration is not None:
        cmd.extend(['--time-duration', str(time_duration)])

    print(f"Running fourier_transform.py with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running fourier_transform.py:\n{result.stderr}")
        raise RuntimeError(f"fourier_transform.py failed with code {result.returncode}")

    fft_csv = output_dir / 'unified_fft.csv'
    if not fft_csv.exists():
        raise RuntimeError(f"Expected output file not found: {fft_csv}")

    return fft_csv


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
    """Run the complete frequency response -> regression pipeline (GP or other methods)."""
    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get frequency domain data
    if config.use_existing:
        print(f"Using existing frequency data from: {config.use_existing}")
        freq_csv = Path(config.use_existing)
    else:
        # Choose frequency analysis method
        if hasattr(config, 'freq_method') and config.freq_method == 'fourier':
            print("Running Fourier transform analysis...")
            freq_csv = run_fourier_transform(config.mat_files, output_dir, config.n_files, config.time_duration, config.nd)
        else:
            print("Running frequency response function (FRF) analysis...")
            freq_csv = run_frequency_response(config.mat_files, output_dir, config.n_files, config.time_duration, config.nd)

    # Step 2: Load frequency domain data
    print("Loading frequency domain data...")
    frf_df = load_frf_data(freq_csv)

    omega = frf_df['omega_rad_s'].values
    G_complex = frf_df['ReG'].values + 1j * frf_df['ImG'].values

    # Check if using non-GP method
    if hasattr(config, 'is_gp') and not config.is_gp:
        # Use classical or ML methods
        print(f"\nUsing {config.method} method...")
        results, G_pred, estimator = run_unified_system_identification(
            omega, G_complex, method=config.method, config=config,
            output_dir=output_dir / config.method,
            return_predictor=True
        )

        # Save main results
        results['config'] = {
            'method': config.method,
            'normalize': config.normalize,
            'n_files': config.n_files,
            'time_duration': config.time_duration
        }

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"{config.method.upper()} regression complete. Results saved to {output_dir}")

        # Save smoothed FRF for downstream use
        smoothed_df = pd.DataFrame({
            'omega_rad_s': omega,
            'freq_Hz': omega / (2 * np.pi),
            'ReG': np.real(G_pred),
            'ImG': np.imag(G_pred),
            'absG': np.abs(G_pred),
            'phase_rad': np.angle(G_pred)
        })
        smoothed_csv = output_dir / f'{config.method}_smoothed_frf.csv'
        smoothed_df.to_csv(smoothed_csv, index=False)
        print(f"{config.method.upper()}-smoothed FRF saved to {smoothed_csv}")

        # Continue to FIR extraction if requested
        if config.extract_fir:
            print("\n" + "="*70)
            print(f"STEP: Extracting FIR model from {config.method.upper()} predictions")
            print("="*70)

            # Get validation MAT file if specified
            validation_mat = None
            if config.fir_validation_mat:
                validation_mat = Path(config.fir_validation_mat)
                if not validation_mat.exists():
                    print(f"Warning: Validation MAT file not found: {validation_mat}")
                    validation_mat = None

            # Create a prediction function for the estimator
            def estimator_predict_at_omega(omega_new):
                """Predict using the fitted estimator at new frequencies."""
                return estimator.predict(omega_new)

            # Use GP-direct method for FIR extraction
            try:
                # Try to use the fixed version first
                from gp_to_fir_direct_fixed import gp_to_fir_direct_pipeline as gp_to_fir_direct_pipeline_fixed
                print(f"Using GP-based FIR extraction for {config.method.upper()} model...")
                fir_output_dir = output_dir / f'fir_{config.method}'
                fir_results = gp_to_fir_direct_pipeline_fixed(
                    omega=omega,
                    G=G_pred,
                    gp_predict_func=estimator_predict_at_omega,
                    mat_file=validation_mat,
                    output_dir=fir_output_dir,
                    N_fft=None,
                    fir_length=config.fir_length
                )

                # Add FIR results to overall results
                results['fir_extraction'] = fir_results

                # Update saved results
                with open(output_dir / 'results.json', 'w') as f:
                    json.dump(results, f, indent=2)

                print(f"FIR extraction complete. Results saved to {fir_output_dir}")

            except ImportError:
                print("Warning: gp_to_fir_direct_fixed module is not available")
            except Exception as e:
                print(f"Error during FIR extraction: {str(e)}")

        return

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
            X_new = omega_new.reshape(-1, 1).copy()

            # Handle zero/near-zero frequencies for log mode
            if config.log_frequency:
                # Find minimum trained frequency
                omega_min = np.min(omega[omega > 0]) if np.any(omega > 0) else 1e-3

                # Replace zero/near-zero frequencies with minimum
                near_zero_mask = X_new.ravel() <= omega_min * 0.1
                if np.any(near_zero_mask):
                    X_new[near_zero_mask] = omega_min

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

        # Store reference to scalers for GP prediction function (if using normalization)
        if config.normalize and config.gp_mode == 'separate':
            gp_predict_at_omega.X_scaler = X_scaler
            gp_predict_at_omega.y_real_scaler = y_real_scaler
            gp_predict_at_omega.y_imag_scaler = y_imag_scaler

        # Use GP-direct method
        try:
            # Try to use the fixed version first
            from gp_to_fir_direct_fixed import gp_to_fir_direct_pipeline as gp_to_fir_direct_pipeline_fixed
            print("Using *fixed* GP-based FIR extraction (ω→Ω mapping).")
            fir_output_dir = output_dir / 'fir_gp'
            fir_results = gp_to_fir_direct_pipeline_fixed(
                omega=omega,
                G=G_smoothed,
                gp_predict_func=gp_predict_at_omega,
                mat_file=validation_mat,          # REQUIRED to infer Ts
                output_dir=fir_output_dir,
                N_fft=None,
                fir_length=config.fir_length
            )

            # Add FIR results to overall results
            results['fir_extraction'] = fir_results

            # Update saved results
            with open(output_dir / 'gp_results.json', 'w') as f:
                json.dump(results, f, indent=2)

            print(f"FIR extraction complete. Results saved to {fir_output_dir}")

        except ImportError:
            # Fall back to regular version
            try:
                from gp_to_fir_direct import gp_to_fir_direct_pipeline
                print("Using standard GP-based interpolation for FIR extraction...")
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

            except ImportError:
                print("Warning: Neither gp_to_fir_direct_fixed nor gp_to_fir_direct modules are available")
            except Exception as e:
                print(f"Error during FIR extraction: {str(e)}")

        except Exception as e:
            print(f"Error during FIR extraction: {str(e)}")


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
    parser.add_argument('--time-duration', type=float, default=None,
                      help='Time duration in seconds to use from each file (only works with --n-files 1)')
    parser.add_argument('--nd', type=int, default=100,
                      help='Number of frequency points (N_d) for frequency response analysis (default: 100)')
    parser.add_argument('--freq-method', type=str, default='frf',
                      choices=['frf', 'fourier'],
                      help='Frequency analysis method: frf (frequency response function) or fourier (FFT-based) (default: frf)')

    # Method selection
    parser.add_argument('--method', type=str, default='gp',
                      choices=['gp'] + ['nls', 'ls', 'iwls', 'tls', 'ml', 'log', 'lpm', 'lrmp'] +
                              ['rf', 'gbr', 'svm'],
                      help='System identification method (default: gp)')

    # GP options
    parser.add_argument('--kernel', type=str, default='rbf',
                      choices=['rbf', 'matern', 'matern12', 'matern32', 'matern52', 'rq', 'exp', 'tc', 'dc', 'di',
                               'ss1', 'ss2', 'sshf', 'stable_spline'],
                      help='Kernel type for GP method (default: rbf)')
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

    # Classical/ML method options
    parser.add_argument('--n-numerator', type=int, default=2,
                      help='Numerator order for classical methods (default: 2)')
    parser.add_argument('--n-denominator', type=int, default=4,
                      help='Denominator order for classical methods (default: 4)')

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

    if args.time_duration is not None:
        if args.time_duration <= 0:
            parser.error("--time-duration must be positive")
        if args.n_files != 1:
            parser.error("--time-duration only works with --n-files 1")

    if args.nd <= 0:
        parser.error("--nd must be positive")

    # Add method info to args
    args.is_gp = args.method == 'gp'

    # Run pipeline
    run_gp_pipeline(args)


def run_unified_system_identification(omega: np.ndarray, G_complex: np.ndarray,
                                    method: str = 'gp', config: Optional[argparse.Namespace] = None,
                                    output_dir: Path = None, return_predictor: bool = False) -> Union[Dict, Tuple[Dict, np.ndarray, object]]:
    """
    Run system identification using GP, classical, or ML methods.

    Args:
        omega: Angular frequencies [rad/s]
        G_complex: Complex frequency response
        method: Method type ('gp', 'nls', 'ls', 'iwls', 'tls', 'ml', 'log', 'lpm', 'lrmp', 'rf', 'gbr', 'svm')
        config: Configuration namespace
        output_dir: Output directory

    Returns:
        Dictionary with results
    """
    from system_identification_methods import create_estimator

    results = {}

    if method == 'gp':
        # Use existing GP pipeline (already implemented)
        return {}

    elif method in ['lpm', 'lrmp']:
        # Special handling for local methods - need U and Y data
        # For now, assume U=1 and Y=G
        U = np.ones_like(G_complex)
        Y = G_complex

        if method == 'lpm':
            estimator = create_estimator('lpm', order=2, half_window=5)
            estimator.fit(omega, Y, U, estimate_transient=True)
        else:  # lrmp
            # Default prior poles for mechanical systems
            prior_poles = [0.9 + 0.1j, 0.9 - 0.1j, 0.8, 0.7]
            estimator = create_estimator('lrmp', prior_poles=prior_poles,
                                       order=5, half_window=10)
            estimator.fit(omega, Y, U, Ts=0.01)  # Assume Ts

        G_pred = estimator.predict(omega)

    else:
        # Classical and ML methods
        if method in ['rf', 'gbr', 'svm']:
            # ML methods
            ml_params = {}
            if method == 'rf':
                ml_params = {'n_estimators': 100, 'max_depth': 10}
            elif method == 'gbr':
                ml_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
            elif method == 'svm':
                ml_params = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}

            estimator = create_estimator(method, normalize=True, **ml_params)
            estimator.fit(omega, G_complex)
            G_pred = estimator.predict(omega)

        else:
            # Classical frequency-domain methods
            n_num = 2 if not hasattr(config, 'n_numerator') else config.n_numerator
            n_den = 4 if not hasattr(config, 'n_denominator') else config.n_denominator

            estimator = create_estimator(method, n_numerator=n_num, n_denominator=n_den)

            if method == 'ml':
                # ML needs additional parameters
                estimator.fit(omega, G_complex,
                            X_measured=np.ones_like(G_complex),
                            Y_measured=G_complex)
            else:
                estimator.fit(omega, G_complex)

            G_pred = estimator.predict(omega)

    # Calculate metrics
    residuals = G_complex - G_pred
    rmse = np.sqrt(np.mean(np.abs(residuals)**2))

    # Separate real and imaginary metrics
    rmse_real = np.sqrt(np.mean((np.real(G_complex) - np.real(G_pred))**2))
    rmse_imag = np.sqrt(np.mean((np.imag(G_complex) - np.imag(G_pred))**2))

    r2_real = 1 - np.sum((np.real(G_complex) - np.real(G_pred))**2) / \
                  np.sum((np.real(G_complex) - np.mean(np.real(G_complex)))**2)
    r2_imag = 1 - np.sum((np.imag(G_complex) - np.imag(G_pred))**2) / \
                  np.sum((np.imag(G_complex) - np.mean(np.imag(G_complex)))**2)

    results = {
        'method': method,
        'rmse': float(rmse),
        'rmse_real': float(rmse_real),
        'rmse_imag': float(rmse_imag),
        'r2_real': float(r2_real),
        'r2_imag': float(r2_imag),
    }

    # Plot results if output_dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_df = pd.DataFrame({
            'omega_rad_s': omega,
            'ReG_true': np.real(G_complex),
            'ImG_true': np.imag(G_complex),
            'ReG_pred': np.real(G_pred),
            'ImG_pred': np.imag(G_pred),
            'absG_true': np.abs(G_complex),
            'absG_pred': np.abs(G_pred),
            'phase_true': np.angle(G_complex),
            'phase_pred': np.angle(G_pred),
        })
        pred_df.to_csv(output_dir / f'{method}_predictions.csv', index=False)

        # Simple comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.loglog(omega, np.abs(G_complex), 'k.', label='Measured', alpha=0.6)
        ax1.loglog(omega, np.abs(G_pred), 'r-', label=f'{method.upper()} fit', linewidth=2)
        ax1.set_xlabel(r'$\omega$ [rad/s]')
        ax1.set_ylabel('|G(jω)|')
        ax1.set_title(f'Magnitude - {method.upper()} (RMSE={rmse:.3e})')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)

        ax2.semilogx(omega, np.unwrap(np.angle(G_complex)), 'k.', label='Measured', alpha=0.6)
        ax2.semilogx(omega, np.unwrap(np.angle(G_pred)), 'r-', label=f'{method.upper()} fit', linewidth=2)
        ax2.set_xlabel(r'$\omega$ [rad/s]')
        ax2.set_ylabel('Phase [rad]')
        ax2.set_title(f'Phase - {method.upper()}')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{method}_bode.png', dpi=300)
        plt.close()

        # Save results
        with open(output_dir / f'{method}_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    if return_predictor:
        return results, G_pred, estimator
    return results


def save_results_to_csv(result_entry: Dict, output_base_dir: Path, timestamp: str):
    """
    Save a single test result to multiple CSV files incrementally.

    Args:
        result_entry: Dictionary containing test results
        output_base_dir: Base output directory
        timestamp: Timestamp string for the test run
    """
    import csv

    base_path = Path(output_base_dir) / timestamp
    base_path.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    fieldnames = [
        'test_name', 'kernel', 'n_files', 'time_duration', 'nd',
        'gp_rmse_real', 'gp_rmse_imag', 'gp_r2_real', 'gp_r2_imag',
        'fir_rmse', 'fir_r2', 'fir_fit_percent',
        'status', 'error_message'
    ]

    # 1. Save to overall results CSV
    overall_csv = base_path / 'overall_results.csv'
    file_exists = overall_csv.exists()

    with open(overall_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_entry)

    # 2. Save to method-specific CSV
    method_name = result_entry.get('test_name', '').split('_nd')[0]  # Extract method name
    if method_name:
        method_csv = base_path / f'results_by_method_{method_name}.csv'
        file_exists = method_csv.exists()

        with open(method_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_entry)

    # 3. Save to nd-specific CSV
    nd_value = result_entry.get('nd')
    if nd_value is not None:
        nd_csv = base_path / f'results_by_nd_{nd_value}.csv'
        file_exists = nd_csv.exists()

        with open(nd_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_entry)

    print(f"  📝 Results saved to CSVs (overall, method: {method_name}, nd: {nd_value})")


def run_comprehensive_test(mat_files: List[str], output_base_dir: str = 'test_output',
                          fir_validation_mat: Optional[str] = None, nd_values: List[int] = None,
                          freq_method: str = 'frf'):
    """
    Run comprehensive tests with different kernels, time intervals, file counts, and nd values.
    Save overall RMSE results in CSV format incrementally after each test.

    Args:
        mat_files: List of MAT files to test
        output_base_dir: Base directory for output
        fir_validation_mat: MAT file for FIR validation
        nd_values: List of nd values to test (default: [10, 30, 50, 100])
        freq_method: Frequency analysis method ('frf' or 'fourier')
    """
    import csv
    from datetime import datetime

    # Test configurations
    # GP kernels
    kernels = ['rbf', 'matern', 'matern12', 'matern32', 'matern52', 'rq', 'exp', 'tc', 'dc', 'di',
               'ss1', 'ss2', 'sshf', 'stable_spline']
    # Classical and ML methods
    classical_methods = ['nls', 'ls', 'iwls', 'tls', 'ml', 'log', 'lpm']
    ml_methods = ['rf', 'gbr', 'svm']

    # Combine all methods
    all_methods = ['gp_' + k for k in kernels] + classical_methods + ml_methods
    # all_methods = classical_methods + ml_methods

    # Set default nd values if not provided
    if nd_values is None:
        nd_values = [10, 30, 50, 100]

    time_durations = [10.0, 30.0, 60.0, 120.0, 300.0, 600.0, None]  # seconds, None means use all data
    n_files_list = [1, 2, 5, 10]  # None means use all files

    # Sort MAT files to ensure consistent order across all tests
    mat_files = sorted(mat_files)

    # Results storage
    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print("Starting Comprehensive System Identification Testing Suite")
    print(f"Timestamp: {timestamp}")
    print(f"Total MAT files available: {len(mat_files)}")
    print(f"MAT files (sorted for consistency):")
    for i, f in enumerate(mat_files, 1):
        print(f"  [{i}] {f}")
    # print(f"GP Kernels: {', '.join(kernels)}")
    print(f"Classical methods: {', '.join(classical_methods)}")
    print(f"ML methods: {', '.join(ml_methods)}")
    print(f"Time durations: {time_durations}")
    print(f"File counts: {n_files_list}")
    print(f"Frequency points (nd): {nd_values}")
    print("=" * 80)

    total_tests = 0

    # For each method
    for method in all_methods:
        print(f"\n{'=' * 60}")
        print(f"Testing method: {method}")
        print(f"{'=' * 60}")

        # Determine if this is a GP method
        is_gp = method.startswith('gp_')
        if is_gp:
            kernel = method[3:]  # Remove 'gp_' prefix
        else:
            kernel = None

        # For each nd value
        for nd in nd_values:
            # For each number of files
            for n_files in n_files_list:
                if n_files is not None and n_files > len(mat_files):
                    continue  # Skip if requesting more files than available

                actual_n_files = n_files if n_files is not None else len(mat_files)

                # For n_files = 1, test different time durations
                if n_files == 1:
                    for time_duration in time_durations:
                        if time_duration is None:
                            time_str = "full"
                        else:
                            time_str = f"{time_duration}s"

                        test_name = f"{method}_nd{nd}_1file_{time_str}"
                        output_dir = Path(output_base_dir) / timestamp / test_name

                        print(f"\nTest: {test_name}")
                        print(f"  Method: {method}")
                        print(f"  Files: 1 -> Using: {mat_files[0]}")
                        print(f"  Duration: {time_str}")
                        print(f"  nd: {nd}")

                        try:
                            # Create argparse-like namespace
                            config = argparse.Namespace(
                                mat_files=mat_files[:1],
                                use_existing=None,
                                n_files=1,
                                time_duration=time_duration,
                                kernel=kernel if is_gp else 'rbf',  # Default kernel for GP
                                nu=2.5 if kernel == 'matern' else None,
                                gp_mode='separate',
                                noise_variance=1e-6,
                                normalize=True,
                                log_frequency=True,
                                optimize=True,
                                n_restarts=3,
                                out_dir=str(output_dir),
                                extract_fir=True,
                                fir_length=1024,
                                fir_validation_mat=fir_validation_mat,
                                method=method,  # Add method type
                                is_gp=is_gp,  # Flag for GP vs other methods
                                nd=nd,  # Number of frequency points
                                freq_method=freq_method  # Frequency analysis method
                            )

                            # Run the pipeline
                            run_gp_pipeline(config)

                            # Extract results
                            if is_gp:
                                results_file = output_dir / 'gp_results.json'
                            else:
                                results_file = output_dir / 'results.json'

                            if results_file.exists():
                                with open(results_file, 'r') as f:
                                    results = json.load(f)

                                # Extract metrics
                                if is_gp:
                                    rmse_real = results.get('real', {}).get('rmse', None)
                                    rmse_imag = results.get('imag', {}).get('rmse', None)
                                    r2_real = results.get('real', {}).get('r2', None)
                                    r2_imag = results.get('imag', {}).get('r2', None)
                                else:
                                    # Non-GP methods store metrics directly
                                    rmse_real = results.get('rmse_real', None)
                                    rmse_imag = results.get('rmse_imag', None)
                                    r2_real = results.get('r2_real', None)
                                    r2_imag = results.get('r2_imag', None)

                                # FIR validation results if available
                                fir_rmse = None
                                fir_r2 = None
                                fir_fit = None
                                if 'fir_extraction' in results:
                                    fir_rmse = results['fir_extraction'].get('rmse', None)
                                    fir_r2 = results['fir_extraction'].get('r2', None)
                                    fir_fit = results['fir_extraction'].get('fit_percent', None)

                                # Store result
                                result_entry = {
                                    'test_name': test_name,
                                    'kernel': kernel,
                                    'n_files': 1,
                                    'time_duration': time_duration,
                                    'nd': nd,
                                    'gp_rmse_real': rmse_real,
                                    'gp_rmse_imag': rmse_imag,
                                    'gp_r2_real': r2_real,
                                    'gp_r2_imag': r2_imag,
                                    'fir_rmse': fir_rmse,
                                    'fir_r2': fir_r2,
                                    'fir_fit_percent': fir_fit,
                                    'status': 'success',
                                    'error_message': ''
                                }
                                all_results.append(result_entry)

                                # Save to CSV files immediately
                                save_results_to_csv(result_entry, Path(output_base_dir), timestamp)

                                print(f"  ✓ Success - GP RMSE Real: {rmse_real:.3e}, Imag: {rmse_imag:.3e}")
                                if fir_rmse:
                                    print(f"            FIR RMSE: {fir_rmse:.3e}, FIT: {fir_fit:.1f}%")
                            else:
                                print(f"  ✗ Results file not found")
                                result_entry = {
                                    'test_name': test_name,
                                    'kernel': kernel,
                                    'n_files': 1,
                                    'time_duration': time_duration,
                                    'nd': nd,
                                    'status': 'no_results_file',
                                    'error_message': 'Results file not found'
                                }
                                all_results.append(result_entry)

                                # Save error to CSV files immediately
                                save_results_to_csv(result_entry, Path(output_base_dir), timestamp)

                        except Exception as e:
                            print(f"  ✗ Error: {str(e)}")
                            result_entry = {
                                'test_name': test_name,
                                'kernel': kernel,
                                'n_files': 1,
                                'time_duration': time_duration,
                                'nd': nd,
                                'status': 'error',
                                'error_message': str(e)
                            }
                            all_results.append(result_entry)

                            # Save error to CSV files immediately
                            save_results_to_csv(result_entry, Path(output_base_dir), timestamp)

                        total_tests += 1

                # For n_files > 1, use all time from each file
                else:
                    test_name = f"{method}_nd{nd}_{actual_n_files}files"
                    output_dir = Path(output_base_dir) / timestamp / test_name

                    # List of files to use
                    files_to_use = mat_files[:actual_n_files]

                    print(f"\nTest: {test_name}")
                    print(f"  Method: {method}")
                    print(f"  Files: {actual_n_files} -> Using:")
                    for i, f in enumerate(files_to_use, 1):
                        print(f"    [{i}] {f}")
                    print(f"  nd: {nd}")

                    try:
                        # Create argparse-like namespace
                        config = argparse.Namespace(
                            mat_files=mat_files[:actual_n_files] if n_files is not None else mat_files,
                            use_existing=None,
                            n_files=n_files if n_files is not None else len(mat_files),
                            time_duration=None,
                            kernel=kernel if is_gp else 'rbf',
                            nu=2.5 if kernel == 'matern' else None,
                            gp_mode='separate',
                            noise_variance=1e-6,
                            normalize=True,
                            log_frequency=True,
                            optimize=True,
                            n_restarts=3,
                            out_dir=str(output_dir),
                            extract_fir=True,
                            fir_length=1024,
                            fir_validation_mat=fir_validation_mat,
                            method=method,
                            is_gp=is_gp,
                            nd=nd,
                            freq_method=freq_method
                        )

                        # Run the pipeline
                        run_gp_pipeline(config)

                        # Extract results
                        if is_gp:
                            results_file = output_dir / 'gp_results.json'
                        else:
                            results_file = output_dir / 'results.json'

                        if results_file.exists():
                            with open(results_file, 'r') as f:
                                results = json.load(f)

                            # Extract metrics
                            if is_gp:
                                rmse_real = results.get('real', {}).get('rmse', None)
                                rmse_imag = results.get('imag', {}).get('rmse', None)
                                r2_real = results.get('real', {}).get('r2', None)
                                r2_imag = results.get('imag', {}).get('r2', None)
                            else:
                                rmse_real = results.get('rmse_real', None)
                                rmse_imag = results.get('rmse_imag', None)
                                r2_real = results.get('r2_real', None)
                                r2_imag = results.get('r2_imag', None)

                            # FIR validation results if available
                            fir_rmse = None
                            fir_r2 = None
                            fir_fit = None
                            if 'fir_extraction' in results:
                                fir_rmse = results['fir_extraction'].get('rmse', None)
                                fir_r2 = results['fir_extraction'].get('r2', None)
                                fir_fit = results['fir_extraction'].get('fit_percent', None)

                            # Store result
                            result_entry = {
                                'test_name': test_name,
                                'kernel': kernel,
                                'n_files': actual_n_files,
                                'time_duration': None,
                                'nd': nd,
                                'gp_rmse_real': rmse_real,
                                'gp_rmse_imag': rmse_imag,
                                'gp_r2_real': r2_real,
                                'gp_r2_imag': r2_imag,
                                'fir_rmse': fir_rmse,
                                'fir_r2': fir_r2,
                                'fir_fit_percent': fir_fit,
                                'status': 'success',
                                'error_message': ''
                            }
                            all_results.append(result_entry)

                            # Save to CSV files immediately
                            save_results_to_csv(result_entry, Path(output_base_dir), timestamp)

                            print(f"  ✓ Success - GP RMSE Real: {rmse_real:.3e}, Imag: {rmse_imag:.3e}")
                            if fir_rmse:
                                print(f"            FIR RMSE: {fir_rmse:.3e}, FIT: {fir_fit:.1f}%")
                        else:
                            print(f"  ✗ Results file not found")
                            result_entry = {
                                'test_name': test_name,
                                'kernel': kernel,
                                'n_files': actual_n_files,
                                'time_duration': None,
                                'nd': nd,
                                'status': 'no_results_file',
                                'error_message': 'Results file not found'
                            }
                            all_results.append(result_entry)

                            # Save error to CSV files immediately
                            save_results_to_csv(result_entry, Path(output_base_dir), timestamp)

                    except Exception as e:
                        print(f"  ✗ Error: {str(e)}")
                        result_entry = {
                            'test_name': test_name,
                            'kernel': kernel,
                            'n_files': actual_n_files,
                            'time_duration': None,
                            'nd': nd,
                            'status': 'error',
                            'error_message': str(e)
                        }
                        all_results.append(result_entry)

                        # Save error to CSV files immediately
                        save_results_to_csv(result_entry, Path(output_base_dir), timestamp)

                        total_tests += 1

    # CSV files are saved incrementally after each test (no need for final save)
    csv_file = Path(output_base_dir) / timestamp / 'overall_results.csv'

    print("\n" + "=" * 80)
    print(f"Comprehensive testing complete!")
    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {sum(1 for r in all_results if r.get('status') == 'success')}")
    print(f"Failed tests: {sum(1 for r in all_results if r.get('status') != 'success')}")
    print(f"\nResults saved incrementally to:")
    print(f"  - Overall: {csv_file}")
    print(f"  - By method: {csv_file.parent}/results_by_method_*.csv")
    print(f"  - By nd: {csv_file.parent}/results_by_nd_*.csv")
    print("=" * 80)

    # Generate summary report
    summary_file = Path(output_base_dir) / timestamp / 'summary_report.txt'
    with open(summary_file, 'w') as f:
        f.write("GP and FIR Model Testing Summary Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Successful Tests: {sum(1 for r in all_results if r.get('status') == 'success')}\n")
        f.write(f"Failed Tests: {sum(1 for r in all_results if r.get('status') != 'success')}\n")
        f.write("\n")

        # Find best performing kernel based on FIR RMSE
        best_results = []
        for result in all_results:
            if result.get('status') == 'success' and result.get('fir_rmse') is not None:
                best_results.append((result['kernel'], result['fir_rmse'], result['test_name']))

        if best_results:
            best_results.sort(key=lambda x: x[1])
            f.write("Top 5 Best Performing Configurations (by FIR RMSE):\n")
            f.write("-" * 60 + "\n")
            for i, (kernel, rmse, test_name) in enumerate(best_results[:5]):
                f.write(f"{i+1}. {test_name}: RMSE = {rmse:.3e}\n")
            f.write("\n")

        # Method-wise summary
        method_stats = {}
        for result in all_results:
            if result.get('status') == 'success':
                method = result.get('kernel', result.get('method', 'unknown'))
                metric = result.get('fir_rmse')
                if metric is None:
                    # Use GP RMSE if FIR not available
                    if result.get('gp_rmse_real') is not None:
                        metric = (result['gp_rmse_real'] + result['gp_rmse_imag']) / 2

                if metric is not None:
                    if method not in method_stats:
                        method_stats[method] = []
                    method_stats[method].append(metric)

        if method_stats:
            f.write("Method Performance Summary (Average RMSE):\n")
            f.write("-" * 60 + "\n")
            method_avg = []
            for method, rmse_list in method_stats.items():
                avg_rmse = np.mean(rmse_list)
                std_rmse = np.std(rmse_list) if len(rmse_list) > 1 else 0
                method_avg.append((method, avg_rmse, std_rmse, len(rmse_list)))

            method_avg.sort(key=lambda x: x[1])
            for method, avg, std, count in method_avg:
                f.write(f"{method}: {avg:.3e} ± {std:.3e} (n={count})\n")

    print(f"Summary report saved to: {summary_file}")

    return csv_file


if __name__ == "__main__":
    # Check if running in test mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test-mode':
        # Run comprehensive test
        print("Running in comprehensive test mode...")

        # Find all MAT files
        mat_pattern = 'input/*.mat'
        mat_files = glob.glob(mat_pattern)
        if not mat_files:
            print(f"Error: No MAT files found matching pattern: {mat_pattern}")
            sys.exit(1)

        # Sort MAT files to ensure consistent order across all test runs
        mat_files = sorted(mat_files)
        print(f"Found {len(mat_files)} MAT files (sorted):")
        for i, f in enumerate(mat_files, 1):
            print(f"  [{i}] {f}")
        print()

        # Find validation MAT file
        validation_mat = None
        for f in mat_files:
            if 'test' in f.lower():
                validation_mat = f
                break

        if not validation_mat:
            # Use first file as validation if no test file found
            validation_mat = mat_files[0]
            print(f"Warning: No test file found, using {validation_mat} for validation")
        else:
            print(f"Using validation file: {validation_mat}")
        print()

        # Run comprehensive test with BOTH frequency methods
        print("=" * 80)
        print("PHASE 1: Testing with FRF (Log-scale frequency analysis)")
        print("=" * 80)
        print()

        # Phase 1: FRF (log-scale) method
        run_comprehensive_test(
            mat_files,
            output_base_dir='test_output_frf',
            fir_validation_mat=validation_mat,
            freq_method='frf'
        )

        print("\n" + "=" * 80)
        print("PHASE 1 COMPLETED: FRF tests finished")
        print("=" * 80)
        print()

        # Phase 2: Fourier (normal-scale) method
        print("=" * 80)
        print("PHASE 2: Testing with Fourier Transform (Normal-scale frequency analysis)")
        print("=" * 80)
        print()

        run_comprehensive_test(
            mat_files,
            output_base_dir='test_output_fourier',
            fir_validation_mat=validation_mat,
            freq_method='fourier'
        )

        print("\n" + "=" * 80)
        print("PHASE 2 COMPLETED: Fourier tests finished")
        print("=" * 80)
        print()

        print("=" * 80)
        print("ALL COMPREHENSIVE TESTS COMPLETED!")
        print("=" * 80)
        print()
        print("Results saved to:")
        print("  - FRF method (log-scale):    test_output_frf/")
        print("  - Fourier method (normal):   test_output_fourier/")
        print("=" * 80)
    else:
        # Run normal pipeline
        main()