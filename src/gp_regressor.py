"""
Gaussian Process Regressor for frequency domain system identification.
Handles complex frequency response data by applying GP to real and imaginary parts.
"""

import numpy as np
from typing import Optional, Union
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

from gp_kernels import Kernel, create_kernel


class GaussianProcessRegressor:
    """
    Standard Gaussian Process Regressor with exact inference.
    Based on the implementation from pure_gp.py.
    """

    def __init__(
        self,
        kernel: Kernel,
        log_sigma_n: float = np.log(1e-3),
        mu: float = 0.0,
        jitter: float = 1e-10,
        optimize: bool = True,
        maxiter: int = 200
    ):
        self.kernel = kernel
        self.log_sigma_n = float(log_sigma_n)
        self.mu = float(mu)
        self.jitter = float(jitter)
        self.optimize = optimize
        self.maxiter = maxiter
        self.X_train = None
        self.y_train = None
        self._L = None
        self._alpha = None

    def _Ky(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix with noise."""
        K = self.kernel.K(X, None)
        sn2 = np.exp(2.0 * self.log_sigma_n)
        return K + (sn2 + self.jitter) * np.eye(X.shape[0])

    def _pack(self) -> np.ndarray:
        """Pack hyperparameters into a vector."""
        return np.concatenate([self.kernel.get_theta(), np.array([self.log_sigma_n, self.mu])])

    def _unpack(self, theta_all: np.ndarray) -> None:
        """Unpack hyperparameters from a vector."""
        kdim = len(self.kernel.get_theta())
        self.kernel.set_theta(theta_all[:kdim])
        self.log_sigma_n = float(theta_all[kdim])
        self.mu = float(theta_all[kdim + 1])

    def _nll_and_grad(self, theta_all: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Compute negative log likelihood and gradient."""
        self._unpack(theta_all)
        n = X.shape[0]
        one = np.ones(n)
        Ky = self._Ky(X)
        yc = y - self.mu * one

        try:
            L = cholesky(Ky, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            Ky = Ky + 1e-8 * np.eye(n)
            L = cholesky(Ky, lower=True, check_finite=False)

        alpha = solve_triangular(L.T, solve_triangular(L, yc, lower=True, check_finite=False), check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * yc.dot(alpha) + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi)

        # Gradient computation
        W = solve_triangular(L, np.eye(n), lower=True, check_finite=False)
        Ky_inv = W.T @ W
        A = np.outer(alpha, alpha) - Ky_inv
        dKs = self.kernel.grad_K_theta(X)
        g_kernel = np.array([0.5 * np.sum(A * dK) for dK in dKs], dtype=float)
        sn2 = np.exp(2.0 * self.log_sigma_n)
        g_logsn = 0.5 * np.sum(A * ((2.0 * sn2) * np.eye(n)))
        g_mu = -one.dot(alpha)
        grad = np.concatenate([g_kernel, np.array([g_logsn, g_mu])])

        return float(nll), grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP model to training data."""
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        assert X.shape[0] == y.shape[0]

        if self.optimize:
            theta0 = self._pack()

            def fval(t):
                return self._nll_and_grad(t, X, y)[0]

            def gval(t):
                return self._nll_and_grad(t, X, y)[1]

            res = minimize(fval, theta0, jac=gval, method="L-BFGS-B",
                         options={"maxiter": self.maxiter, "ftol": 1e-9})
            self._unpack(res.x)

        Ky = self._Ky(X)
        L = cholesky(Ky, lower=True, check_finite=False)
        one = np.ones(X.shape[0])
        yc = y - self.mu * one
        alpha = solve_triangular(L.T, solve_triangular(L, yc, lower=True, check_finite=False), check_finite=False)

        self.X_train = X
        self.y_train = y
        self._L = L
        self._alpha = alpha
        return self

    def predict(self, Xstar: np.ndarray, return_std: bool = True):
        """Predict at new points."""
        assert self.X_train is not None, "Call fit() before predict()"
        Xs = np.atleast_2d(Xstar)
        Kxs = self.kernel.K(self.X_train, Xs)
        mu_star = self.mu + Kxs.T @ self._alpha

        if not return_std:
            return mu_star, None

        v = solve_triangular(self._L, Kxs, lower=True, check_finite=False)
        Kss = self.kernel.K(Xs, None)
        cov = Kss - v.T @ v
        cov = 0.5 * (cov + cov.T)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        return mu_star, std


class GPFrequencyRegressor:
    """
    Gaussian Process Regressor for complex frequency response data.
    Applies GP to real and imaginary parts separately.
    """

    def __init__(
        self,
        kernel: Union[str, Kernel] = "matern32",
        noise: float = 1e-3,
        optimize: bool = True,
        maxiter: int = 200,
        jitter: float = 1e-10,
        standardize: bool = True
    ):
        """
        Initialize GP frequency regressor.

        Parameters:
        -----------
        kernel : Union[str, Kernel]
            Kernel name or kernel instance
        noise : float
            Observation noise level
        optimize : bool
            Whether to optimize hyperparameters
        maxiter : int
            Maximum iterations for optimization
        jitter : float
            Numerical stability parameter
        standardize : bool
            Whether to standardize input features
        """
        self.kernel_spec = kernel
        self.noise = float(noise)
        self.optimize = optimize
        self.maxiter = maxiter
        self.jitter = jitter
        self.standardize = standardize

        self._gpr_real = None
        self._gpr_imag = None
        self._freq_mean = None
        self._freq_std = None

    def _create_kernel(self) -> Kernel:
        """Create kernel instance."""
        if isinstance(self.kernel_spec, str):
            return create_kernel(self.kernel_spec)
        else:
            return self.kernel_spec

    def _standardize_frequencies(self, frequencies: np.ndarray) -> np.ndarray:
        """Standardize frequency values to zero mean and unit variance."""
        if self.standardize:
            if self._freq_mean is None:
                # Use log scale for frequencies
                log_freq = np.log10(frequencies + 1e-10)
                self._freq_mean = np.mean(log_freq)
                self._freq_std = np.std(log_freq) + 1e-10
                return (log_freq - self._freq_mean) / self._freq_std
            else:
                log_freq = np.log10(frequencies + 1e-10)
                return (log_freq - self._freq_mean) / self._freq_std
        else:
            return np.log10(frequencies + 1e-10).reshape(-1, 1)

    def fit(self, frequencies: np.ndarray, freq_response: np.ndarray):
        """
        Fit GP model to frequency response data.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency points [Hz]
        freq_response : np.ndarray
            Complex frequency response G(jω)
        """
        # Prepare input features
        X = self._standardize_frequencies(frequencies).reshape(-1, 1)

        # Separate real and imaginary parts
        y_real = np.real(freq_response).astype(float).reshape(-1)
        y_imag = np.imag(freq_response).astype(float).reshape(-1)

        # Create and fit GP models for real and imaginary parts
        kernel_real = self._create_kernel()
        kernel_imag = self._create_kernel()

        self._gpr_real = GaussianProcessRegressor(
            kernel=kernel_real,
            log_sigma_n=np.log(self.noise),
            mu=0.0,
            jitter=self.jitter,
            optimize=self.optimize,
            maxiter=self.maxiter
        )

        self._gpr_imag = GaussianProcessRegressor(
            kernel=kernel_imag,
            log_sigma_n=np.log(self.noise),
            mu=0.0,
            jitter=self.jitter,
            optimize=self.optimize,
            maxiter=self.maxiter
        )

        print("Fitting GP for real part...")
        self._gpr_real.fit(X, y_real)

        print("Fitting GP for imaginary part...")
        self._gpr_imag.fit(X, y_imag)

        return self

    def predict(
        self,
        frequencies: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Predict frequency response at new frequency points.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency points for prediction [Hz]
        return_std : bool
            Whether to return prediction uncertainty

        Returns:
        --------
        freq_response : np.ndarray
            Complex frequency response predictions
        std : Optional[np.ndarray]
            Prediction standard deviations (if return_std=True)
        """
        assert self._gpr_real is not None, "Call fit() before predict()"

        # Prepare input features
        X = self._standardize_frequencies(frequencies).reshape(-1, 1)

        # Predict real and imaginary parts
        mu_real, std_real = self._gpr_real.predict(X, return_std=True)
        mu_imag, std_imag = self._gpr_imag.predict(X, return_std=True)

        # Combine into complex response
        freq_response = mu_real + 1j * mu_imag

        if return_std:
            # Combined uncertainty (approximate)
            std_combined = np.sqrt(std_real**2 + std_imag**2)
            return freq_response, std_combined
        else:
            return freq_response

    def get_hyperparameters(self) -> dict:
        """Get optimized hyperparameters."""
        if self._gpr_real is None:
            return {}

        return {
            "real": {
                "kernel_params": self._gpr_real.kernel.get_theta().tolist(),
                "noise": float(np.exp(self._gpr_real.log_sigma_n)),
                "mean": self._gpr_real.mu
            },
            "imag": {
                "kernel_params": self._gpr_imag.kernel.get_theta().tolist(),
                "noise": float(np.exp(self._gpr_imag.log_sigma_n)),
                "mean": self._gpr_imag.mu
            }
        }


class MultiOutputGPRegressor:
    """
    Multi-output GP regressor that can handle magnitude and phase representation.
    Alternative to complex representation for frequency response.
    """

    def __init__(
        self,
        kernel: Union[str, Kernel] = "matern32",
        noise: float = 1e-3,
        optimize: bool = True,
        maxiter: int = 200
    ):
        self.kernel_spec = kernel
        self.noise = noise
        self.optimize = optimize
        self.maxiter = maxiter

        self._gpr_mag = None
        self._gpr_phase = None

    def fit_magnitude_phase(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray
    ):
        """
        Fit GP models to magnitude and phase data.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency points [Hz]
        magnitude : np.ndarray
            Magnitude of frequency response
        phase : np.ndarray
            Phase of frequency response [radians]
        """
        # Use log-frequency as input
        X = np.log10(frequencies + 1e-10).reshape(-1, 1)

        # Log-transform magnitude for better GP modeling
        log_mag = np.log(magnitude + 1e-10)

        # Create kernels
        if isinstance(self.kernel_spec, str):
            kernel_mag = create_kernel(self.kernel_spec)
            kernel_phase = create_kernel(self.kernel_spec)
        else:
            kernel_mag = self.kernel_spec
            kernel_phase = self.kernel_spec

        # Fit magnitude GP
        self._gpr_mag = GaussianProcessRegressor(
            kernel=kernel_mag,
            log_sigma_n=np.log(self.noise),
            optimize=self.optimize,
            maxiter=self.maxiter
        )
        self._gpr_mag.fit(X, log_mag)

        # Fit phase GP
        self._gpr_phase = GaussianProcessRegressor(
            kernel=kernel_phase,
            log_sigma_n=np.log(self.noise),
            optimize=self.optimize,
            maxiter=self.maxiter
        )
        self._gpr_phase.fit(X, phase)

        return self

    def predict_magnitude_phase(
        self,
        frequencies: np.ndarray,
        return_std: bool = False
    ):
        """Predict magnitude and phase at new frequencies."""
        X = np.log10(frequencies + 1e-10).reshape(-1, 1)

        log_mag_pred, log_mag_std = self._gpr_mag.predict(X, return_std=True)
        phase_pred, phase_std = self._gpr_phase.predict(X, return_std=True)

        # Transform back from log scale
        mag_pred = np.exp(log_mag_pred)

        if return_std:
            # Propagate uncertainty through exp transformation
            mag_std = mag_pred * log_mag_std
            return mag_pred, phase_pred, mag_std, phase_std
        else:
            return mag_pred, phase_pred


if __name__ == "__main__":
    # Test the GP frequency regressor
    import matplotlib.pyplot as plt

    # Generate synthetic frequency response data
    np.random.seed(42)
    freq_train = np.logspace(-1, 2, 20)  # 0.1 to 100 Hz

    # Synthetic transfer function: second-order system
    omega_n = 10.0  # Natural frequency
    zeta = 0.3  # Damping ratio
    s = 1j * 2 * np.pi * freq_train
    G_true = omega_n**2 / (s**2 + 2*zeta*omega_n*s + omega_n**2)

    # Add noise
    noise_level = 0.05
    G_noisy = G_true + noise_level * (np.random.randn(len(freq_train)) +
                                      1j * np.random.randn(len(freq_train)))

    # Fit GP model
    gp = GPFrequencyRegressor(kernel="matern32", noise=noise_level, optimize=True)
    gp.fit(freq_train, G_noisy)

    # Predict on dense grid
    freq_test = np.logspace(-1, 2, 200)
    G_pred, G_std = gp.predict(freq_test, return_std=True)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Magnitude plot
    ax1.loglog(freq_train, np.abs(G_noisy), 'ko', label='Noisy data')
    ax1.loglog(freq_test, np.abs(G_pred), 'b-', label='GP prediction')
    ax1.fill_between(freq_test, np.abs(G_pred) - 2*G_std, np.abs(G_pred) + 2*G_std,
                     alpha=0.3, color='b', label='±2σ')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Phase plot
    ax2.semilogx(freq_train, np.angle(G_noisy) * 180/np.pi, 'ko', label='Noisy data')
    ax2.semilogx(freq_test, np.angle(G_pred) * 180/np.pi, 'b-', label='GP prediction')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gp_frequency_test.png", dpi=150)
    plt.close()

    print("Test completed. Results saved to gp_frequency_test.png")
    print("Hyperparameters:", gp.get_hyperparameters())