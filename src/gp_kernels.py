"""
Gaussian Process kernel implementations.
Implements all kernels described in Method.tex for system identification.
"""

import numpy as np
import math
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod


def _sqdist(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of X and X2.
    """
    X = np.atleast_2d(X)
    if X2 is None:
        X2 = X
    else:
        X2 = np.atleast_2d(X2)
    X_norm = np.sum(X**2, axis=1)[:, None]
    X2_norm = np.sum(X2**2, axis=1)[None, :]
    D = X_norm + X2_norm - 2 * X @ X2.T
    np.maximum(D, 0.0, out=D)
    return D


class Kernel(ABC):
    """Abstract base class for all kernels."""

    @abstractmethod
    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix between X and X2."""
        pass

    @abstractmethod
    def get_theta(self) -> np.ndarray:
        """Get hyperparameters as a vector."""
        pass

    @abstractmethod
    def set_theta(self, theta: np.ndarray) -> None:
        """Set hyperparameters from a vector."""
        pass

    @abstractmethod
    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        """Compute gradient of kernel matrix w.r.t. hyperparameters."""
        pass

    @abstractmethod
    def hyperparameter_names(self) -> List[str]:
        """Get names of hyperparameters."""
        pass


class RBFKernel(Kernel):
    """
    RBF (Squared Exponential) kernel.
    K(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
    """

    def __init__(self, ell: float = 1.0, sigma_f: float = 1.0):
        assert ell > 0 and sigma_f > 0
        self.log_ell = math.log(ell)
        self.log_sigma_f = math.log(sigma_f)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X / ell, None if X2 is None else X2 / ell)
        return sf2 * np.exp(-0.5 * D2)

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_ell, self.log_sigma_f], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_ell, self.log_sigma_f = float(theta[0]), float(theta[1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X / ell)
        K = sf2 * np.exp(-0.5 * D2)
        grad_log_ell = K * D2
        grad_log_sigma_f = 2.0 * K
        return [grad_log_ell, grad_log_sigma_f]

    def hyperparameter_names(self) -> List[str]:
        return ["log_ell", "log_sigma_f"]


class ExponentialKernel(Kernel):
    """
    Exponential (BIBO-stable) kernel.
    k(x, x') = H(x)H(x')exp(-ω(x + x')), ω > 0
    where H is the Heaviside function.
    """

    def __init__(self, omega: float = 1.0):
        assert omega > 0
        self.log_omega = math.log(omega)

    @staticmethod
    def _as_1d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X
        if X.ndim == 2 and X.shape[1] == 1:
            return X[:, 0]
        raise ValueError("ExponentialKernel expects 1D inputs")

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        t = self._as_1d(X)
        s = t if X2 is None else self._as_1d(X2)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        Hs = np.heaviside(s, 0.0)[None, :]
        T = t[:, None]
        S = s[None, :]
        K = Ht * Hs * np.exp(-omega * (T + S))
        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_omega], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_omega = float(theta[0])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        t = self._as_1d(X)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        T = t[:, None]
        Hs = Ht.T
        S = T.T
        K = Ht * Hs * np.exp(-omega * (T + S))
        g_log_omega = -omega * (T + S) * K
        return [g_log_omega]

    def hyperparameter_names(self) -> List[str]:
        return ["log_omega"]


class TCKernel(Kernel):
    """
    Tuned-Correlated (TC) kernel.
    k(x, x') = H(x)H(x')exp(-ω max(x, x')), ω > 0
    """

    def __init__(self, omega: float = 1.0):
        assert omega > 0
        self.log_omega = math.log(omega)

    @staticmethod
    def _as_1d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X
        if X.ndim == 2 and X.shape[1] == 1:
            return X[:, 0]
        raise ValueError("TCKernel expects 1D inputs")

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        t = self._as_1d(X)
        s = t if X2 is None else self._as_1d(X2)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        Hs = np.heaviside(s, 0.0)[None, :]
        T = t[:, None]
        S = s[None, :]
        M = np.maximum(T, S)
        K = Ht * Hs * np.exp(-omega * M)
        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_omega], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_omega = float(theta[0])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        t = self._as_1d(X)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        T = t[:, None]
        Hs = Ht.T
        S = T.T
        M = np.maximum(T, S)
        K = Ht * Hs * np.exp(-omega * M)
        g_log_omega = -omega * M * K
        return [g_log_omega]

    def hyperparameter_names(self) -> List[str]:
        return ["log_omega"]


class DCKernel(Kernel):
    """
    Diagonal/Correlated (DC) kernel.
    k(i, j) = β α^((i+j)/2) ρ^|i-j|
    with 0 < α < 1, β > 0, |ρ| < 1
    """

    def __init__(self, alpha: float = 0.9, beta: float = 1.0, rho: float = 0.8):
        assert 0 < alpha < 1
        assert beta > 0
        assert abs(rho) < 1
        # Transform to unconstrained space
        self.log_beta = math.log(beta)
        self.alpha_logit = math.log(alpha / (1 - alpha))
        self.rho_atanh = np.arctanh(rho)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        # For DC kernel, X should be integer indices
        X = np.asarray(X).astype(int)
        if X2 is None:
            X2 = X
        else:
            X2 = np.asarray(X2).astype(int)

        # Get parameters
        beta = math.exp(self.log_beta)
        alpha = 1.0 / (1.0 + math.exp(-self.alpha_logit))
        rho = math.tanh(self.rho_atanh)

        # Compute kernel
        i = X.reshape(-1, 1)
        j = X2.reshape(1, -1)
        K = beta * (alpha ** ((i + j) / 2.0)) * (np.abs(rho) ** np.abs(i - j))

        # Handle sign of rho for negative differences
        if rho < 0:
            K *= ((-1.0) ** np.abs(i - j))

        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_beta, self.alpha_logit, self.rho_atanh], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_beta = float(theta[0])
        self.alpha_logit = float(theta[1])
        self.rho_atanh = float(theta[2])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        X = np.asarray(X).astype(int)
        beta = math.exp(self.log_beta)
        alpha = 1.0 / (1.0 + math.exp(-self.alpha_logit))
        rho = math.tanh(self.rho_atanh)

        i = X.reshape(-1, 1)
        j = X.reshape(1, -1)
        K_base = beta * (alpha ** ((i + j) / 2.0)) * (np.abs(rho) ** np.abs(i - j))
        if rho < 0:
            K_base *= ((-1.0) ** np.abs(i - j))

        # Gradient w.r.t. log_beta
        g_log_beta = K_base

        # Gradient w.r.t. alpha_logit
        dalpha_dlogit = alpha * (1 - alpha)
        g_alpha = K_base * ((i + j) / 2.0) * np.log(alpha + 1e-10) / (alpha + 1e-10)
        g_alpha_logit = g_alpha * dalpha_dlogit

        # Gradient w.r.t. rho_atanh
        drho_datanh = 1.0 - rho**2
        diff = np.abs(i - j)
        if abs(rho) > 1e-10:
            g_rho = K_base * diff * np.sign(rho) / (np.abs(rho) + 1e-10)
        else:
            g_rho = np.zeros_like(K_base)
        g_rho_atanh = g_rho * drho_datanh

        return [g_log_beta, g_alpha_logit, g_rho_atanh]

    def hyperparameter_names(self) -> List[str]:
        return ["log_beta", "alpha_logit", "rho_atanh"]


class DIKernel(Kernel):
    """
    Diagonal/Independent (DI) kernel.
    k(i, j) = β α^i if i == j, else 0
    with 0 < α < 1, β > 0
    """

    def __init__(self, alpha: float = 0.9, beta: float = 1.0):
        assert 0 < alpha < 1
        assert beta > 0
        self.log_beta = math.log(beta)
        self.alpha_logit = math.log(alpha / (1 - alpha))

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X).astype(int).reshape(-1)
        if X2 is None:
            X2 = X
        else:
            X2 = np.asarray(X2).astype(int).reshape(-1)

        beta = math.exp(self.log_beta)
        alpha = 1.0 / (1.0 + math.exp(-self.alpha_logit))

        K = np.zeros((len(X), len(X2)))
        for i in range(len(X)):
            for j in range(len(X2)):
                if X[i] == X2[j]:
                    K[i, j] = beta * (alpha ** X[i])

        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_beta, self.alpha_logit], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_beta = float(theta[0])
        self.alpha_logit = float(theta[1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        X = np.asarray(X).astype(int).reshape(-1)
        beta = math.exp(self.log_beta)
        alpha = 1.0 / (1.0 + math.exp(-self.alpha_logit))

        K = self.K(X)

        # Gradient w.r.t. log_beta
        g_log_beta = K

        # Gradient w.r.t. alpha_logit
        dalpha_dlogit = alpha * (1 - alpha)
        g_alpha = np.zeros_like(K)
        for i in range(len(X)):
            if X[i] > 0:
                g_alpha[i, i] = K[i, i] * X[i] / (alpha + 1e-10)
        g_alpha_logit = g_alpha * dalpha_dlogit

        return [g_log_beta, g_alpha_logit]

    def hyperparameter_names(self) -> List[str]:
        return ["log_beta", "alpha_logit"]


class StableSplineKernel(Kernel):
    """
    First-order stable spline kernel.
    K₁(s,t;β) = max(e^(-βs), e^(-βt)) = e^(-β min(s,t))
    """

    def __init__(self, beta: float = 1.0):
        assert beta > 0
        self.log_beta = math.log(beta)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.atleast_2d(X)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        if X2 is None:
            X2 = X
        else:
            X2 = np.atleast_2d(X2)
            if X2.ndim == 2 and X2.shape[1] == 1:
                X2 = X2.ravel()

        beta = math.exp(self.log_beta)
        S = X[:, None]
        T = X2[None, :]
        min_ST = np.minimum(S, T)
        K = np.exp(-beta * min_ST)
        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_beta], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_beta = float(theta[0])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        K = self.K(X)
        beta = math.exp(self.log_beta)
        X = X.ravel()
        S = X[:, None]
        T = X[None, :]
        min_ST = np.minimum(S, T)
        g_log_beta = -beta * min_ST * K
        return [g_log_beta]

    def hyperparameter_names(self) -> List[str]:
        return ["log_beta"]


class SecondOrderStableSplineKernel(Kernel):
    """
    Second-order stable spline kernel.
    K₂(s,t;β) = (1/2)e^(-β(s+t+max{s,t})) - (1/6)e^(-3β max{s,t})
    """

    def __init__(self, beta: float = 1.0):
        assert beta > 0
        self.log_beta = math.log(beta)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.atleast_2d(X)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        if X2 is None:
            X2 = X
        else:
            X2 = np.atleast_2d(X2)
            if X2.ndim == 2 and X2.shape[1] == 1:
                X2 = X2.ravel()

        beta = math.exp(self.log_beta)
        S = X[:, None]
        T = X2[None, :]
        max_ST = np.maximum(S, T)

        K = 0.5 * np.exp(-beta * (S + T + max_ST)) - (1.0/6.0) * np.exp(-3 * beta * max_ST)
        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_beta], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_beta = float(theta[0])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        X = X.ravel()
        beta = math.exp(self.log_beta)
        S = X[:, None]
        T = X[None, :]
        max_ST = np.maximum(S, T)

        term1 = 0.5 * np.exp(-beta * (S + T + max_ST))
        term2 = (1.0/6.0) * np.exp(-3 * beta * max_ST)

        g_term1 = -beta * (S + T + max_ST) * term1
        g_term2 = -3 * beta * max_ST * term2

        g_log_beta = g_term1 - g_term2
        return [g_log_beta]

    def hyperparameter_names(self) -> List[str]:
        return ["log_beta"]


class HighFreqStableSplineKernel(Kernel):
    """
    High frequency stable spline kernel.
    K_HF(s,t;β) = (-1)^(s+t) max(e^(-βs), e^(-βt))
    """

    def __init__(self, beta: float = 1.0):
        assert beta > 0
        self.log_beta = math.log(beta)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X).astype(int).ravel()
        if X2 is None:
            X2 = X
        else:
            X2 = np.asarray(X2).astype(int).ravel()

        beta = math.exp(self.log_beta)
        S = X[:, None]
        T = X2[None, :]
        sign = (-1.0) ** (S + T)
        max_exp = np.maximum(np.exp(-beta * S), np.exp(-beta * T))
        K = sign * max_exp
        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_beta], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_beta = float(theta[0])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        X = np.asarray(X).astype(int).ravel()
        beta = math.exp(self.log_beta)
        S = X[:, None]
        T = X[None, :]
        sign = (-1.0) ** (S + T)

        exp_S = np.exp(-beta * S)
        exp_T = np.exp(-beta * T)

        # Gradient of max function
        g_log_beta = np.zeros((len(X), len(X)))
        mask_S = exp_S >= exp_T
        mask_T = ~mask_S

        g_log_beta[mask_S] = -beta * S[mask_S] * exp_S[mask_S] * sign[mask_S]
        g_log_beta[mask_T] = -beta * T[mask_T] * exp_T[mask_T] * sign[mask_T]

        return [g_log_beta]

    def hyperparameter_names(self) -> List[str]:
        return ["log_beta"]


class MaternKernel(Kernel):
    """
    Matérn kernel for ν in {0.5, 1.5, 2.5}.
    K = σ_f² * f_ν(r/ℓ)
    """

    def __init__(self, ell: float = 1.0, sigma_f: float = 1.0, nu: float = 1.5):
        assert ell > 0 and sigma_f > 0
        assert nu in (0.5, 1.5, 2.5), "nu must be 0.5, 1.5, or 2.5"
        self.log_ell = math.log(ell)
        self.log_sigma_f = math.log(sigma_f)
        self.nu = float(nu)

    def _form(self, r: np.ndarray, ell: float, sf2: float) -> np.ndarray:
        a = np.sqrt(2.0 * self.nu) * r / ell
        if self.nu == 0.5:
            return sf2 * np.exp(-a)
        elif self.nu == 1.5:
            return sf2 * (1.0 + a) * np.exp(-a)
        elif self.nu == 2.5:
            return sf2 * (1.0 + a + a*a/3.0) * np.exp(-a)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X, X2)
        r = np.sqrt(D2 + 1e-32)
        return self._form(r, ell, sf2)

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_ell, self.log_sigma_f], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_ell, self.log_sigma_f = float(theta[0]), float(theta[1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X, None)
        r = np.sqrt(D2 + 1e-32)
        a = np.sqrt(2.0 * self.nu) * r / ell

        if self.nu == 0.5:
            K = sf2 * np.exp(-a)
            g_log_ell = K * a
        elif self.nu == 1.5:
            K = sf2 * (1.0 + a) * np.exp(-a)
            g_log_ell = K * (a*a) / (1.0 + a + 1e-10)
        elif self.nu == 2.5:
            K = sf2 * (1.0 + a + a*a/3.0) * np.exp(-a)
            denom = (1.0 + a + a*a/3.0 + 1e-10)
            g_log_ell = K * (a*a + a*a*a) / (3.0 * denom)

        g_log_sigma_f = 2.0 * K
        return [g_log_ell, g_log_sigma_f]

    def hyperparameter_names(self) -> List[str]:
        return ["log_ell", "log_sigma_f"]


class StableSplineKernelComplex(Kernel):
    """
    Complex Stable Spline kernel from pure_fir_model.py.
    Uses Wiener process basis with exponential time transformation.
    """

    def __init__(self, beta: float = 0.5, sigma_f: float = 1.0):
        assert beta > 0
        assert sigma_f > 0
        self.log_beta = math.log(beta)
        self.log_sigma_f = math.log(sigma_f)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.atleast_2d(X)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        if X2 is None:
            X2 = X
        else:
            X2 = np.atleast_2d(X2)
            if X2.ndim == 2 and X2.shape[1] == 1:
                X2 = X2.ravel()

        beta = math.exp(self.log_beta)
        sf2 = math.exp(2.0 * self.log_sigma_f)

        # Time transformation
        tau_X = np.exp(-beta * X)
        tau_X2 = np.exp(-beta * X2)

        # Compute kernel using Wiener process basis
        K = np.zeros((len(X), len(X2)))
        for i, tau_i in enumerate(tau_X):
            for j, tau_j in enumerate(tau_X2):
                r = min(tau_i, tau_j)
                R = max(tau_i, tau_j)
                K[i, j] = sf2 * 0.5 * r**2 * (R - r/3.0)

        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_beta, self.log_sigma_f], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_beta = float(theta[0])
        self.log_sigma_f = float(theta[1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        X = X.ravel()
        beta = math.exp(self.log_beta)
        sf2 = math.exp(2.0 * self.log_sigma_f)

        # Time transformation
        tau = np.exp(-beta * X)

        # Kernel matrix
        K = self.K(X)

        # Gradient w.r.t. log_beta
        g_log_beta = np.zeros_like(K)
        for i in range(len(X)):
            for j in range(len(X)):
                tau_i, tau_j = tau[i], tau[j]
                if tau_i <= tau_j:
                    r, R = tau_i, tau_j
                    dr_dbeta = -beta * X[i] * r
                    dR_dbeta = -beta * X[j] * R
                    dK_dr = sf2 * (r * R - r**2 / 3.0)
                    dK_dR = sf2 * 0.5 * r**2
                    g_log_beta[i, j] = dK_dr * dr_dbeta + dK_dR * dR_dbeta
                else:
                    r, R = tau_j, tau_i
                    dr_dbeta = -beta * X[j] * r
                    dR_dbeta = -beta * X[i] * R
                    dK_dr = sf2 * (r * R - r**2 / 3.0)
                    dK_dR = sf2 * 0.5 * r**2
                    g_log_beta[i, j] = dK_dr * dr_dbeta + dK_dR * dR_dbeta

        # Gradient w.r.t. log_sigma_f
        g_log_sigma_f = 2.0 * K

        return [g_log_beta, g_log_sigma_f]

    def hyperparameter_names(self) -> List[str]:
        return ["log_beta", "log_sigma_f"]


# Kernel factory function
def create_kernel(kernel_name: str) -> Kernel:
    """
    Create a kernel instance by name.

    Parameters:
    -----------
    kernel_name : str
        Name of the kernel

    Returns:
    --------
    kernel : Kernel
        Kernel instance
    """
    kernel_map = {
        "rbf": RBFKernel,
        "exp": ExponentialKernel,
        "exponential": ExponentialKernel,
        "tc": TCKernel,
        "tuned_correlated": TCKernel,
        "dc": DCKernel,
        "di": DIKernel,
        "ss": StableSplineKernel,
        "ss1": StableSplineKernel,
        "stable_spline": StableSplineKernel,
        "ss2": SecondOrderStableSplineKernel,
        "hf": HighFreqStableSplineKernel,
        "high_freq": HighFreqStableSplineKernel,
        "matern12": lambda: MaternKernel(nu=0.5),
        "matern32": lambda: MaternKernel(nu=1.5),
        "matern52": lambda: MaternKernel(nu=2.5),
        "stable_spline_complex": StableSplineKernelComplex,
    }

    name = kernel_name.lower()
    if name not in kernel_map:
        raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(kernel_map.keys())}")

    kernel_class = kernel_map[name]
    if callable(kernel_class) and not isinstance(kernel_class, type):
        return kernel_class()  # For lambda functions
    else:
        return kernel_class()  # For regular classes