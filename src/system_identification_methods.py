#!/usr/bin/env python3
"""
system_identification_methods.py

Implementation of classical frequency-domain system identification methods
and machine learning approaches for comparison with GP methods.

Methods included:
- Classical: NLS, LS, IWLS, TLS, ML, LOG, LPM, LRMP
- Machine Learning: Random Forest, Gradient Boosting, SVM
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union, List, Callable
from scipy.optimize import minimize, least_squares
from scipy.linalg import svd, lstsq
from scipy.special import factorial
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings


# =====================================================
# Base Classes and Utilities
# =====================================================

class FrequencyDomainEstimator:
    """Base class for frequency-domain system identification methods."""

    def __init__(self, n_numerator: int = 2, n_denominator: int = 2):
        self.n_numerator = n_numerator
        self.n_denominator = n_denominator
        self.params = None
        self.omega = None
        self.H_measured = None

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        """Fit the model to frequency response data."""
        raise NotImplementedError

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Predict frequency response at given frequencies."""
        if self.params is None:
            raise ValueError("Model not fitted yet")
        return self._compute_transfer_function(omega, self.params)

    def _compute_transfer_function(self, omega: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute H(jω) = N(jω)/D(jω) for given parameters."""
        s = 1j * omega

        # Split parameters
        alpha = params[:self.n_numerator + 1]
        beta = params[self.n_numerator + 1:]

        # Compute numerator and denominator
        N = np.polyval(alpha[::-1], s)
        D = np.polyval(beta[::-1], s)

        return N / D


# =====================================================
# Classical System Identification Methods
# =====================================================

class NonlinearLeastSquares(FrequencyDomainEstimator):
    """Nonlinear Least Squares (NLS) method."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        # Initial guess
        p0 = np.ones(self.n_numerator + self.n_denominator + 2)
        p0[self.n_numerator + 1] = 1.0  # β₀ = 1

        # Cost function
        def cost(p):
            H_model = self._compute_transfer_function(omega, p)
            return np.abs(H_measured - H_model)

        # Optimize
        result = least_squares(cost, p0, method='lm')
        self.params = result.x

        return self


class LinearLeastSquares(FrequencyDomainEstimator):
    """Linear Least Squares (LS) method - Levy's method."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        s = 1j * omega
        n_points = len(omega)

        # Build regression matrix
        # H(s)D(s) = N(s) => H(s)(β₀ + β₁s + ... + βₐsᵈ) = α₀ + α₁s + ... + αₙsⁿ

        # Left side: H(s) * [1, s, s², ..., sᵈ]
        A_left = []
        for k in range(self.n_denominator + 1):
            A_left.append(H_measured * s**k)

        # Right side: [1, s, s², ..., sⁿ]
        A_right = []
        for k in range(self.n_numerator + 1):
            A_right.append(s**k)

        # Stack: left side negative (move to right)
        A = np.column_stack(A_right + [-col for col in A_left[1:]])  # Skip β₀

        # Target: H(s) * β₀ (assuming β₀ = 1)
        b = H_measured

        # Solve least squares (complex)
        x, _, _, _ = lstsq(A, b)

        # Extract parameters - take real part as we expect real coefficients
        alpha = np.real(x[:self.n_numerator + 1])
        beta = np.ones(self.n_denominator + 1)
        beta[1:] = np.real(x[self.n_numerator + 1:])

        self.params = np.concatenate([alpha, beta])

        return self


class IterativelyWeightedLS(FrequencyDomainEstimator):
    """Iteratively Weighted Linear Least Squares (IWLS) - Sanathanan-Koerner."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, max_iter: int = 10, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        # Initialize with LS solution
        ls_estimator = LinearLeastSquares(self.n_numerator, self.n_denominator)
        ls_estimator.fit(omega, H_measured)
        params = ls_estimator.params.copy()

        s = 1j * omega

        for iteration in range(max_iter):
            # Compute current denominator for weights
            beta = params[self.n_numerator + 1:]
            D = np.polyval(beta[::-1], s)
            weights = 1.0 / np.abs(D)

            # Weighted regression
            A_right = []
            for k in range(self.n_numerator + 1):
                A_right.append(weights * s**k)

            A_left = []
            for k in range(1, self.n_denominator + 1):
                A_left.append(-weights * H_measured * s**k)

            A = np.column_stack(A_right + A_left)
            b = weights * H_measured

            # Solve
            x, _, _, _ = lstsq(A, b)

            # Update parameters - take real part
            alpha = np.real(x[:self.n_numerator + 1])
            beta = np.ones(self.n_denominator + 1)
            beta[1:] = np.real(x[self.n_numerator + 1:])

            params = np.concatenate([alpha, beta])

        self.params = params
        return self


class TotalLeastSquares(FrequencyDomainEstimator):
    """Total Least Squares (TLS) with Euclidean norm constraint."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        s = 1j * omega

        # Build augmented matrix [A | b]
        A_right = []
        for k in range(self.n_numerator + 1):
            A_right.append(s**k)

        A_left = []
        for k in range(self.n_denominator + 1):
            A_left.append(-H_measured * s**k)

        # Augmented matrix
        C = np.column_stack(A_right + A_left)

        # SVD of augmented matrix
        U, S, Vt = svd(C, full_matrices=False)

        # Solution is last column of V (last row of Vt)
        x = Vt[-1, :]

        # Normalize to have unit denominator constant
        x = x / x[self.n_numerator + 1]

        # Extract parameters
        alpha = x[:self.n_numerator + 1]
        beta = x[self.n_numerator + 1:]

        self.params = np.concatenate([alpha, beta])

        return self


class MaximumLikelihood(FrequencyDomainEstimator):
    """Maximum Likelihood (ML) estimator for complex Gaussian noise."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray,
            X_measured: Optional[np.ndarray] = None,
            Y_measured: Optional[np.ndarray] = None,
            sigma_x: Optional[np.ndarray] = None,
            sigma_y: Optional[np.ndarray] = None,
            rho: Optional[np.ndarray] = None,
            **kwargs):

        self.omega = omega
        self.H_measured = H_measured

        # If X, Y not provided, assume H = Y/X with X=1
        if X_measured is None:
            X_measured = np.ones_like(H_measured)
        if Y_measured is None:
            Y_measured = H_measured

        # Default noise statistics
        if sigma_x is None:
            sigma_x = np.ones(len(omega)) * 0.01
        if sigma_y is None:
            sigma_y = np.ones(len(omega)) * 0.01
        if rho is None:
            rho = np.zeros(len(omega))

        s = 1j * omega

        def ml_cost(params):
            alpha = params[:self.n_numerator + 1]
            beta = params[self.n_numerator + 1:]

            N = np.polyval(alpha[::-1], s)
            D = np.polyval(beta[::-1], s)

            # Error: N*X - D*Y
            E = N * X_measured - D * Y_measured

            # Denominator of ML cost
            denom = (sigma_x**2 * np.abs(N)**2 +
                    sigma_y**2 * np.abs(D)**2 -
                    2 * np.real(rho * D * np.conj(N)))

            # Avoid division by zero
            denom = np.maximum(denom, 1e-10)

            # ML cost
            cost = np.sum(np.abs(E)**2 / denom)

            return cost

        # Initial guess
        p0 = np.ones(self.n_numerator + self.n_denominator + 2)
        p0[self.n_numerator + 1] = 1.0

        # Optimize
        result = minimize(ml_cost, p0, method='L-BFGS-B')
        self.params = result.x

        return self


class LogarithmicLeastSquares(FrequencyDomainEstimator):
    """Logarithmic Least Squares (LOG) method."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        # Take logarithm
        log_H = np.log(H_measured)

        def log_cost(params):
            H_model = self._compute_transfer_function(omega, params)
            # Avoid log of zero/negative
            H_model_safe = np.maximum(np.abs(H_model), 1e-10)
            log_H_model = np.log(H_model_safe)

            # Complex logarithm difference
            return np.abs(log_H_model - log_H)**2

        # Initial guess
        p0 = np.ones(self.n_numerator + self.n_denominator + 2)
        p0[self.n_numerator + 1] = 1.0

        # Optimize
        result = least_squares(log_cost, p0, method='lm')
        self.params = result.x

        return self


# =====================================================
# Local Polynomial/Rational Methods
# =====================================================

class LocalPolynomialMethod:
    """Local Polynomial Method (LPM) for nonparametric FRF estimation."""

    def __init__(self, order: int = 2, half_window: int = 5):
        self.order = order
        self.half_window = half_window
        self.omega = None
        self.G_estimate = None

    def fit(self, omega: np.ndarray, Y: np.ndarray, U: np.ndarray,
            estimate_transient: bool = True):
        """
        Fit LPM to frequency-domain data.

        Args:
            omega: Angular frequencies
            Y: Output spectrum
            U: Input spectrum
            estimate_transient: Whether to estimate transient terms
        """
        self.omega = omega
        n_freq = len(omega)
        self.G_estimate = np.zeros_like(Y)

        R = self.order
        n = self.half_window

        for k in range(n_freq):
            # Define local window
            k_min = max(0, k - n)
            k_max = min(n_freq - 1, k + n)
            indices = np.arange(k_min, k_max + 1)

            # Local indices relative to center
            r = indices - k

            # Build regression matrix
            if estimate_transient:
                # Model: Y = G*U + T with polynomial approximations
                # G = g0 + g1*r + g2*r^2 + ...
                # T = t0 + t1*r + t2*r^2 + ...

                A = []
                # G terms
                for p in range(R + 1):
                    A.append(U[indices] * r**p)
                # T terms
                for p in range(R + 1):
                    A.append(r**p)

                A = np.column_stack(A)
                b = Y[indices]

                # Solve
                theta, _, _, _ = lstsq(A, b)

                # Extract G at center (r=0)
                self.G_estimate[k] = theta[0]

            else:
                # Simple case: Y = G*U
                A = []
                for p in range(R + 1):
                    A.append(U[indices] * r**p)

                A = np.column_stack(A)
                b = Y[indices]

                theta, _, _, _ = lstsq(A, b)
                self.G_estimate[k] = theta[0]

        return self

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Interpolate to new frequencies."""
        from scipy.interpolate import interp1d

        # Interpolate real and imaginary parts
        interp_real = interp1d(self.omega, np.real(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')
        interp_imag = interp1d(self.omega, np.imag(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')

        return interp_real(omega) + 1j * interp_imag(omega)


class LocalRationalMethodPrior:
    """Local Rational Method with Prior poles (LRMP)."""

    def __init__(self, prior_poles: List[complex], order: int = 5, half_window: int = 10):
        self.prior_poles = prior_poles
        self.order = order
        self.half_window = half_window
        self.omega = None
        self.G_estimate = None

    def _compute_obf_basis(self, z: np.ndarray, poles: List[complex]) -> np.ndarray:
        """Compute Orthonormal Basis Functions (Takenaka-Malmquist)."""
        n_basis = len(poles)
        n_points = len(z)
        B = np.zeros((n_points, n_basis), dtype=complex)

        for b in range(n_basis):
            zeta_b = poles[b]

            # First factor
            B[:, b] = z * np.sqrt(1 - np.abs(zeta_b)**2) / (z - zeta_b)

            # Product of previous factors
            for i in range(b):
                zeta_i = poles[i]
                B[:, b] *= (1 - np.conj(zeta_i) * z) / (z - zeta_i)

        return B

    def fit(self, omega: np.ndarray, Y: np.ndarray, U: np.ndarray, Ts: float = 1.0):
        """Fit LRMP using orthonormal rational basis."""
        self.omega = omega
        n_freq = len(omega)
        self.G_estimate = np.zeros_like(Y)

        # Convert to z-domain
        z = np.exp(1j * omega * Ts)

        n = self.half_window

        for k in range(n_freq):
            # Define local window
            k_min = max(0, k - n)
            k_max = min(n_freq - 1, k + n)
            indices = np.arange(k_min, k_max + 1)

            # Compute basis functions at local points
            z_local = z[indices]
            B = self._compute_obf_basis(z_local, self.prior_poles)

            # Build regression matrix for Y = sum(theta_b^G * B_b * U) + sum(theta_b^T * B_b)
            A = []

            # G terms
            for b in range(B.shape[1]):
                A.append(B[:, b] * U[indices])

            # T terms (transient)
            for b in range(B.shape[1]):
                A.append(B[:, b])

            A = np.column_stack(A)
            b_vec = Y[indices]

            # Solve
            theta, _, _, _ = lstsq(A, b_vec)

            # Reconstruct G at center point
            B_center = self._compute_obf_basis(np.array([z[k]]), self.prior_poles)
            self.G_estimate[k] = np.sum(theta[:len(self.prior_poles)] * B_center[0, :])

        return self

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Interpolate to new frequencies."""
        from scipy.interpolate import interp1d

        interp_real = interp1d(self.omega, np.real(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')
        interp_imag = interp1d(self.omega, np.imag(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')

        return interp_real(omega) + 1j * interp_imag(omega)


# =====================================================
# Machine Learning Methods
# =====================================================

class MachineLearningRegressor:
    """Base class for ML-based frequency response estimation."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.X_scaler = StandardScaler() if normalize else None
        self.y_real_scaler = StandardScaler() if normalize else None
        self.y_imag_scaler = StandardScaler() if normalize else None
        self.model_real = None
        self.model_imag = None

    def fit(self, omega: np.ndarray, H_measured: np.ndarray):
        """Fit ML model to frequency response data."""
        # Prepare features (can be extended with more features)
        X = omega.reshape(-1, 1)

        # Separate real and imaginary parts
        y_real = np.real(H_measured)
        y_imag = np.imag(H_measured)

        # Normalize if requested
        if self.normalize:
            X = self.X_scaler.fit_transform(X)
            y_real = self.y_real_scaler.fit_transform(y_real.reshape(-1, 1)).ravel()
            y_imag = self.y_imag_scaler.fit_transform(y_imag.reshape(-1, 1)).ravel()

        # Fit models
        self.model_real.fit(X, y_real)
        self.model_imag.fit(X, y_imag)

        return self

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Predict frequency response at new frequencies."""
        X = omega.reshape(-1, 1)

        if self.normalize:
            X = self.X_scaler.transform(X)

        y_real_pred = self.model_real.predict(X)
        y_imag_pred = self.model_imag.predict(X)

        if self.normalize:
            y_real_pred = self.y_real_scaler.inverse_transform(y_real_pred.reshape(-1, 1)).ravel()
            y_imag_pred = self.y_imag_scaler.inverse_transform(y_imag_pred.reshape(-1, 1)).ravel()

        return y_real_pred + 1j * y_imag_pred


class RandomForestFrequencyResponse(MachineLearningRegressor):
    """Random Forest Regression for frequency response estimation."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 normalize: bool = True, **rf_params):
        super().__init__(normalize)
        self.model_real = RandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               **rf_params)
        self.model_imag = RandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               **rf_params)


class GradientBoostingFrequencyResponse(MachineLearningRegressor):
    """Gradient Boosting Regression for frequency response estimation."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, normalize: bool = True, **gb_params):
        super().__init__(normalize)
        self.model_real = GradientBoostingRegressor(n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    max_depth=max_depth,
                                                    **gb_params)
        self.model_imag = GradientBoostingRegressor(n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    max_depth=max_depth,
                                                    **gb_params)


class SVMFrequencyResponse(MachineLearningRegressor):
    """Support Vector Machine Regression for frequency response estimation."""

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 normalize: bool = True, **svm_params):
        super().__init__(normalize)
        self.model_real = SVR(kernel=kernel, C=C, gamma=gamma, **svm_params)
        self.model_imag = SVR(kernel=kernel, C=C, gamma=gamma, **svm_params)


# =====================================================
# Unified Interface
# =====================================================

def create_estimator(method: str, **kwargs) -> Union[FrequencyDomainEstimator,
                                                     MachineLearningRegressor,
                                                     LocalPolynomialMethod,
                                                     LocalRationalMethodPrior]:
    """Factory function to create estimators by name."""

    estimator_map = {
        # Classical methods
        'nls': NonlinearLeastSquares,
        'ls': LinearLeastSquares,
        'iwls': IterativelyWeightedLS,
        'tls': TotalLeastSquares,
        'ml': MaximumLikelihood,
        'log': LogarithmicLeastSquares,

        # Local methods
        'lpm': LocalPolynomialMethod,
        'lrmp': LocalRationalMethodPrior,

        # Machine learning
        'rf': RandomForestFrequencyResponse,
        'gbr': GradientBoostingFrequencyResponse,
        'svm': SVMFrequencyResponse,
    }

    if method not in estimator_map:
        raise ValueError(f"Unknown method: {method}. Available: {list(estimator_map.keys())}")

    return estimator_map[method](**kwargs)


# =====================================================
# Example Usage and Testing
# =====================================================

if __name__ == "__main__":
    # Generate example data
    omega = np.logspace(-1, 2, 100)
    s = 1j * omega

    # True system: H(s) = 1 / (s^2 + 0.1*s + 1)
    H_true = 1.0 / (s**2 + 0.1*s + 1)

    # Add noise
    noise = 0.01 * (np.random.randn(len(omega)) + 1j * np.random.randn(len(omega)))
    H_measured = H_true + noise

    # Test different methods
    methods = ['nls', 'ls', 'iwls', 'tls', 'log', 'rf', 'gbr', 'svm']

    for method in methods:
        print(f"\nTesting {method.upper()}...")

        if method in ['rf', 'gbr', 'svm']:
            estimator = create_estimator(method, normalize=True)
        elif method in ['lpm', 'lrmp']:
            continue  # Skip for now, need U and Y data
        else:
            estimator = create_estimator(method, n_numerator=0, n_denominator=2)

        try:
            estimator.fit(omega, H_measured)
            H_pred = estimator.predict(omega)

            # Compute error
            rmse = np.sqrt(np.mean(np.abs(H_measured - H_pred)**2))
            print(f"  RMSE: {rmse:.6f}")

        except Exception as e:
            print(f"  Error: {e}")