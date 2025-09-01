
# pure_gpr.py
# A "pure" Gaussian Process Regression implementation from scratch (exact inference).
# Dependencies: numpy, scipy, matplotlib (for the optional demo).
import numpy as np
import math
from typing import Optional, Tuple, List

from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def _sqdist(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pairwise squared Euclidean distances between rows of X and X2.
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


class Kernel:
    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError
    def get_theta(self) -> np.ndarray:
        raise NotImplementedError
    def set_theta(self, theta: np.ndarray) -> None:
        raise NotImplementedError
    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError
    def hyperparameter_names(self) -> List[str]:
        raise NotImplementedError


class RBFKernel(Kernel):
    """
    Isotropic RBF (squared exponential):
    K = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / ell^2)
    theta = [log_ell, log_sigma_f]
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
        grad_log_ell = K * (D2)
        grad_log_sigma_f = 2.0 * K
        return [grad_log_ell, grad_log_sigma_f]

    def hyperparameter_names(self) -> List[str]:
        return ["log_ell", "log_sigma_f"]


class ARDRBFLinearBiasKernel(Kernel):
    """
    ARD RBF + linear + bias:
    K = v0 * exp(-0.5 * sum_l w_l (x_l - x'_l)^2) + a0 + a1 * <x, x'>
    theta = [log_v0, log_w_1..d, log_a0, log_a1]
    """
    def __init__(self, v0: float = 1.0, w: Optional[np.ndarray] = None, a0: float = 1e-6, a1: float = 1e-6):
        self.log_v0 = math.log(max(v0, 1e-12))
        if w is None:
            self.log_w = None
        else:
            w = np.asarray(w, dtype=float)
            assert np.all(w >= 0)
            self.log_w = np.log(np.maximum(w, 1e-12))
        self.log_a0 = math.log(max(a0, 1e-12))
        self.log_a1 = math.log(max(a1, 1e-12))

    def _ensure_log_w(self, X: np.ndarray):
        if self.log_w is None:
            self.log_w = np.zeros(X.shape[1], dtype=float)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.atleast_2d(X)
        self._ensure_log_w(X)
        v0 = math.exp(self.log_v0)
        w = np.exp(self.log_w)
        a0 = math.exp(self.log_a0)
        a1 = math.exp(self.log_a1)
        if X2 is None:
            X2 = X
        Wsqrt = np.sqrt(w + 1e-32)
        Xw = X * Wsqrt
        X2w = X2 * Wsqrt
        D2 = _sqdist(Xw, X2w)
        K_rbf = v0 * np.exp(-0.5 * D2)
        K_lin = a1 * (X @ X2.T)
        K_bias = a0 * np.ones((X.shape[0], X2.shape[0]))
        return K_rbf + K_bias + K_lin

    def get_theta(self) -> np.ndarray:
        return np.concatenate([np.array([self.log_v0]), np.asarray(self.log_w), np.array([self.log_a0, self.log_a1])])

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_v0 = float(theta[0])
        d = len(theta) - 3
        self.log_w = np.array(theta[1:1+d], dtype=float)
        self.log_a0 = float(theta[-2])
        self.log_a1 = float(theta[-1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        X = np.atleast_2d(X)
        self._ensure_log_w(X)
        v0 = math.exp(self.log_v0)
        w = np.exp(self.log_w)
        a0 = math.exp(self.log_a0)
        a1 = math.exp(self.log_a1)
        Wsqrt = np.sqrt(w + 1e-32)
        Xw = X * Wsqrt
        D2 = _sqdist(Xw, None)
        K_rbf = v0 * np.exp(-0.5 * D2)
        g_log_v0 = K_rbf.copy()
        grads_w = []
        for l in range(X.shape[1]):
            dx_l = X[:, [l]] - X[:, [l]].T
            g = -0.5 * w[l] * (dx_l**2) * K_rbf
            grads_w.append(g)
        g_log_a0 = a0 * np.ones_like(K_rbf)
        g_log_a1 = a1 * (X @ X.T)
        return [g_log_v0, *grads_w, g_log_a0, g_log_a1]

    def hyperparameter_names(self) -> List[str]:
        names = ["log_v0"]
        if self.log_w is None:
            names += [f"log_w[{i}]" for i in range(0)]
        else:
            names += [f"log_w[{i}]" for i in range(len(self.log_w))]
        names += ["log_a0", "log_a1"]
        return names


class MaternKernel(Kernel):
    """
    Matérn kernel for nu in {0.5, 1.5, 2.5}
    K = sigma_f^2 * f_nu(r/ell)
    theta = [log_ell, log_sigma_f]
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
        else:
            raise NotImplementedError

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
        # Kernel value
        if self.nu == 0.5:
            K = sf2 * np.exp(-a)
            g_log_ell = K * a
        elif self.nu == 1.5:
            K = sf2 * (1.0 + a) * np.exp(-a)
            g_log_ell = K * (a*a) / (1.0 + a)
        elif self.nu == 2.5:
            K = sf2 * (1.0 + a + a*a/3.0) * np.exp(-a)
            denom = (1.0 + a + a*a/3.0)
            g_log_ell = K * (a*a + a*a*a) / (3.0 * denom)
        else:
            raise NotImplementedError
        g_log_sigma_f = 2.0 * K
        return [g_log_ell, g_log_sigma_f]

    def hyperparameter_names(self) -> List[str]:
        return ["log_ell", "log_sigma_f"]


class ExpStableKernel(Kernel):
    """
    Exponential BIBO-stable kernel on nonnegative time:
      k(t1, t2) = H(t1) H(t2) exp(-omega * (t1 + t2)),  omega > 0

    We parameterize omega via log_omega for positivity.
    X is expected to be 1D inputs (shape (n,1) or (n,)).
    """
    def __init__(self, omega: float = 1.0):
        assert omega > 0, "omega must be > 0"
        self.log_omega = math.log(float(omega))

    @staticmethod
    def _as_1d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X
        if X.ndim == 2 and X.shape[1] == 1:
            return X[:, 0]
        raise ValueError("ExpStableKernel expects 1D inputs (n,) or (n,1)")

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
        # For gradient, we need training K(X,X)
        Hs = Ht.T
        S = T.T
        K = Ht * Hs * np.exp(-omega * (T + S))
        # dK/d(log_omega) = (dK/domega) * domega/dlog_omega = (- (T+S) * K) * omega
        g_log_omega = -omega * (T + S) * K
        return [g_log_omega]

    def hyperparameter_names(self) -> List[str]:
        return ["log_omega"]


class TCKernel(Kernel):
    """
    Tuned-Correlated (TC) kernel on nonnegative time:
      k(t1, t2) = H(t1) H(t2) exp(-omega * max(t1, t2)),  omega > 0

    Parameterized by log_omega for positivity.
    X is expected to be 1D inputs (shape (n,1) or (n,)).
    """
    def __init__(self, omega: float = 1.0):
        assert omega > 0, "omega must be > 0"
        self.log_omega = math.log(float(omega))

    @staticmethod
    def _as_1d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X
        if X.ndim == 2 and X.shape[1] == 1:
            return X[:, 0]
        raise ValueError("TCKernel expects 1D inputs (n,) or (n,1)")

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
        # dK/d(log_omega) = (- M * K) * omega
        g_log_omega = -omega * M * K
        return [g_log_omega]

    def hyperparameter_names(self) -> List[str]:
        return ["log_omega"]

class GaussianProcessRegressorPure:
    """
    Exact Gaussian Process Regression (O(n^3)) with:
      - Kernel (RBF, Matérn, or ARD RBF + linear + bias)
      - Constant mean mu
      - Homoskedastic noise sigma_n^2
    """
    def __init__(self, kernel: Kernel, log_sigma_n: float = math.log(1e-1), mu: float = 0.0, jitter: float = 1e-10):
        self.kernel = kernel
        self.log_sigma_n = float(log_sigma_n)
        self.mu = float(mu)
        self.jitter = float(jitter)
        self.X_train = None
        self.y_train = None
        self._L = None
        self._alpha = None

    def _Ky(self, X: np.ndarray) -> np.ndarray:
        K = self.kernel.K(X, None)
        sn2 = math.exp(2.0 * self.log_sigma_n)
        return K + (sn2 + self.jitter) * np.eye(X.shape[0])

    def _pack(self) -> np.ndarray:
        return np.concatenate([self.kernel.get_theta(), np.array([self.log_sigma_n, self.mu])])

    def _unpack(self, theta_all: np.ndarray) -> None:
        kdim = len(self.kernel.get_theta())
        self.kernel.set_theta(theta_all[:kdim])
        self.log_sigma_n = float(theta_all[kdim])
        self.mu = float(theta_all[kdim + 1])

    def _nll_and_grad(self, theta_all: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
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
        nll = 0.5 * yc.dot(alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)
        W = solve_triangular(L, np.eye(n), lower=True, check_finite=False)
        Ky_inv = W.T @ W
        A = np.outer(alpha, alpha) - Ky_inv
        dKs = self.kernel.grad_K_theta(X)
        g_kernel = np.array([0.5 * np.sum(A * dK) for dK in dKs], dtype=float)
        sn2 = math.exp(2.0 * self.log_sigma_n)
        g_logsn = 0.5 * np.sum(A * ((2.0 * sn2) * np.eye(n)))
        g_mu = -one.dot(alpha)  # gradient of NLL
        grad = np.concatenate([g_kernel, np.array([g_logsn, g_mu])])
        return float(nll), grad

    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True, maxiter: int = 200, verbose: bool = False):
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        assert X.shape[0] == y.shape[0]
        if optimize:
            theta0 = self._pack()
            def fval(t):
                return self._nll_and_grad(t, X, y)[0]
            def gval(t):
                return self._nll_and_grad(t, X, y)[1]
            res = minimize(fval, theta0, jac=gval, method="L-BFGS-B", options={"maxiter": maxiter, "disp": verbose})
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

    def predict(self, Xstar: np.ndarray, return_std: bool = True, return_cov: bool = False):
        assert self.X_train is not None
        Xs = np.atleast_2d(Xstar)
        Kxs = self.kernel.K(self.X_train, Xs)
        mu_star = self.mu + Kxs.T @ self._alpha
        if not (return_std or return_cov):
            return mu_star, None
        v = solve_triangular(self._L, Kxs, lower=True, check_finite=False)
        Kss = self.kernel.K(Xs, None)
        cov = Kss - v.T @ v
        cov = 0.5 * (cov + cov.T)
        if return_cov:
            return mu_star, cov
        else:
            std = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            return mu_star, std

    def hyperparameter_names(self) -> List[str]:
        return [*self.kernel.hyperparameter_names(), "log_sigma_n", "mu"]

    def get_hyperparameter_vector(self) -> np.ndarray:
        return self._pack()

    def set_hyperparameter_vector(self, theta_all: np.ndarray) -> None:
        self._unpack(theta_all)


def _demo_1d(savepath_png: str = "gpr_demo.png") -> None:
    rng = np.random.default_rng(0)
    X = np.linspace(-5.0, 5.0, 25)[:, None]
    f_true = np.sin(X).ravel() + 0.3 * X.ravel()
    y = f_true + 0.15 * rng.standard_normal(X.shape[0])
    kernel = MaternKernel(ell=1.2, sigma_f=1.0, nu=1.5)
    gp = GaussianProcessRegressorPure(kernel=kernel, log_sigma_n=np.log(0.15), mu=0.0, jitter=1e-10)
    gp.fit(X, y, optimize=True, maxiter=150, verbose=False)
    Xs = np.linspace(-6.0, 6.0, 400)[:, None]
    m, s = gp.predict(Xs, return_std=True)
    plt.figure(figsize=(7, 4))
    plt.plot(X, y, "o", label="observations")
    plt.plot(Xs, m, label="predictive mean")
    plt.fill_between(Xs.ravel(), m - 1.96 * s, m + 1.96 * s, alpha=0.2, label="95% CI")
    plt.title("Pure Gaussian Process Regression (Matérn 3/2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath_png, dpi=150)
    plt.close()


if __name__ == "__main__":
    _demo_1d("gpr_demo.png")
    print("Demo saved to gpr_demo.png")


# =============================
# Convenience API for pipeline
# =============================

class ComplexGPPure:
    """
    Convenience wrapper that fits two independent pure GPs on
    the real and imaginary parts of a complex target G = Re + j Im.

    - X provided as 1D or 2D (n, d); we internally ensure shape (n, d)
    - Kernel options kept simple for pipeline usage
    """
    def __init__(
        self,
        kernel: str = "matern32",   # 'matern12' | 'matern32' | 'matern52' | 'rbf'
        noise: float = 1e-2,
        optimize: bool = True,
        maxiter: int = 200,
        jitter: float = 1e-10,
    ) -> None:
        self.kernel_name = kernel
        self.noise = float(noise)
        self.optimize = bool(optimize)
        self.maxiter = int(maxiter)
        self.jitter = float(jitter)
        self._gpr_r: Optional[GaussianProcessRegressorPure] = None
        self._gpr_i: Optional[GaussianProcessRegressorPure] = None

    def _make_kernel(self) -> Kernel:
        name = self.kernel_name.lower()
        if name in ("matern12", "matern-1/2", "matern0.5"):
            return MaternKernel(ell=1.0, sigma_f=1.0, nu=0.5)
        if name in ("matern32", "matern-3/2", "matern1.5"):
            return MaternKernel(ell=1.0, sigma_f=1.0, nu=1.5)
        if name in ("matern52", "matern-5/2", "matern2.5"):
            return MaternKernel(ell=1.0, sigma_f=1.0, nu=2.5)
        if name in ("rbf", "se", "squared_exponential"):
            return RBFKernel(ell=1.0, sigma_f=1.0)
        if name in ("exp", "exponential", "exp_stable", "stable_exp"):
            return ExpStableKernel(omega=1.0)
        if name in ("tc", "tuned_correlated", "tuned-correlated"):
            return TCKernel(omega=1.0)
        # default
        return MaternKernel(ell=1.0, sigma_f=1.0, nu=1.5)

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def fit(self, X_train: np.ndarray, G_train: np.ndarray) -> "ComplexGPPure":
        X = self._ensure_2d(X_train)
        y_r = np.asarray(np.real(G_train), dtype=float).reshape(-1)
        y_i = np.asarray(np.imag(G_train), dtype=float).reshape(-1)
        kern_r = self._make_kernel()
        kern_i = self._make_kernel()
        self._gpr_r = GaussianProcessRegressorPure(kernel=kern_r, log_sigma_n=np.log(self.noise), mu=0.0, jitter=self.jitter)
        self._gpr_i = GaussianProcessRegressorPure(kernel=kern_i, log_sigma_n=np.log(self.noise), mu=0.0, jitter=self.jitter)
        self._gpr_r.fit(X, y_r, optimize=self.optimize, maxiter=self.maxiter, verbose=False)
        self._gpr_i.fit(X, y_i, optimize=self.optimize, maxiter=self.maxiter, verbose=False)
        return self

    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        assert self._gpr_r is not None and self._gpr_i is not None, "Call fit() before predict()"
        Xe = self._ensure_2d(X_eval)
        mr, _ = self._gpr_r.predict(Xe, return_std=True)
        mi, _ = self._gpr_i.predict(Xe, return_std=True)
        return mr + 1j * mi


def fit_predict_complex_gp(
    X_train: np.ndarray,
    G_train: np.ndarray,
    X_eval: np.ndarray,
    kernel: str = "matern32",
    noise: float = 1e-2,
    optimize: bool = True,
    maxiter: int = 200,
) -> np.ndarray:
    """
    One-shot helper to fit pure GP on complex G and predict at X_eval.

    - Fits independent pure GPs on real/imag parts of G
    - Returns complex predictions aligned with X_eval
    """
    model = ComplexGPPure(kernel=kernel, noise=noise, optimize=optimize, maxiter=maxiter)
    model.fit(X_train, G_train)
    return model.predict(X_eval)
