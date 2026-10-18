import numpy as np
from common.YieldCurve import market_discount, libor_forward

# This class sets all parameters used for the LIBOR market model
class LiborMarketModel:
    # Input Data:
    # T0: first exercise Date
    # TN: Maturity of the Underlying Swap
    # tau: Tenor of Swap
    # sigma: (constant) lognormal volatilities of the LIBOR rates, array of length N, sigma_i for L_i
    # rho: correlation structure of the Brownian Motions, i.e. d[W_i,W_j] = rho_{ij}dt
    # L0: LIBOR rates at time t=0, i.e. (L0)_i = L(0, T_{i-1}, T_i), i = 1, ..., N, with T_i = T0 + i * tau
    # (N = (TN - T0) / tau = number of forward rates)
    def __init__(self, T0, TN, tau, sigma, rho, L0):
        self.T0, self.TN, self.tau = T0, TN, tau
        self.sigma = np.asarray(sigma, dtype=float)
        self.rho = np.asarray(rho, dtype=float)
        self.L0 = np.asarray(L0, dtype=float)
        self.x0 = np.log(self.L0)
        self.n_forwards = len(self.L0)
        self.rho_chol = np.linalg.cholesky(self.rho)
        self.upper_rho = np.triu(self.rho, 1)
        self.cov = np.diag(self.sigma) @ self.rho @ np.diag(self.sigma)
        eigenvalues, Q = np.linalg.eigh(self.cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues, self.Q = eigenvalues[idx], Q[:, idx]

    # constructor with the parameters of this thesis:
    # T_i = T0, T0 + tau, ..., TN - tau: reset dates of the N = (TN - T0) / tau forward rates
    # sigma_i = 0.2 + 0.1 * exp(-0.1 * T_i): decreasing volatility term structure
    # rho_ij = exp(-beta_corr * |i - j|): exponentially decaying correlation
    # L0: initial forward rates from the market curve (YieldCurve.libor_forward)
    # Return: LiborMarketModel(T0, TN, tau, sigma, rho, L0)
    @classmethod
    def from_setup(cls, T0, TN, tau, beta_corr=0.075):
        n = int(round((TN - T0) / tau))
        T_reset = T0 + tau * np.arange(n)
        sigma = 0.2 + 0.1 * np.exp(-0.1 * T_reset)
        rho = np.exp(-beta_corr * np.abs(np.subtract.outer(np.arange(n), np.arange(n))))
        L0 = libor_forward(T_reset, T_reset + tau, tau)
        return cls(T0, TN, tau, sigma, rho, L0)

    # calculates the given value into basispoints multiplied with the zero bond P(0,T_N), 
    # i.e. value_{bps} = value * 10000 * P(0,T_N)
    def to_bps(self, value):
        return value * market_discount(self.TN) * 10000
