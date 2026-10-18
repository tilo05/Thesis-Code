import numpy as np
from common.lmm_functions import drift


# this function calculates one euler step
def euler_step(model, L, h, Z):
    dW = Z @ model.rho_chol.T
    L += h * L * drift(model, L) + np.sqrt(h) * (L * model.sigma * dW)
    return L

# this function calculates one milstein step
def milstein_step(model, L, h, Z):
    dW = Z @ model.rho_chol.T
    return (L + h * L * drift(model, L) + np.sqrt(h) * (L * model.sigma * dW)
            + 0.5 * (model.sigma ** 2 * L) * (h * (dW ** 2 - 1)))


# simulates the LIBOR rates up to time T with given random numbers 
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# T: final time of the simulation (exercise date)
# M: number of time steps over [0, T], step size h = T / M
# Z: independent standard normal random numbers, array of shape (n_paths, M, n_forwards)
# scheme: Used scheme ("euler" or "milstein")
# Return:
# L: LIBOR rates at time T, array of shape (n_paths, n_forwards)
def simulate_terminal(model, T, M, Z, scheme="euler"):
    n_paths = Z.shape[0]
    h = T / M
    L = np.tile(model.L0, (n_paths, 1))
    for m in range(M):
        Zm = Z[:, m]
        if scheme == "euler": 
            L = euler_step(model, L, h, Zm)
        elif scheme == "milstein":
            L = milstein_step(model, L, h, Zm)
        else:
            raise RuntimeError(f"scheme not implemented {scheme}")
    return L

# simulate the LIBOR rates up to the last date in dates (last exercise date)
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# dates: array of the dates (all exercise dates T_0,...,T_{n-1}) at which the rates are stored (must lie on the time grid)
# M: number of time steps over [0, dates[-1]]
# n_paths: number of simulated paths
# scheme: Used scheme ("euler" or "milstein")
# rng: random number generator (None: new generator without seed, not reproducible)
# Return:
# values: LIBOR rates at the dates, array of shape (n_paths, len(dates), n_forwards)
def simulate_at_dates(model, dates, M, n_paths, scheme="euler", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    h = dates[-1] / M
    values = np.zeros((n_paths, len(dates), model.n_forwards))
    index = 0
    L = np.tile(model.L0, (n_paths, 1))
    for m in range(M):
        if scheme == "euler":
            L = euler_step(model, L, h, rng.standard_normal((n_paths, model.n_forwards)))
        elif scheme == "milstein":
            L = milstein_step(model, L, h, rng.standard_normal((n_paths, model.n_forwards)))
        else:
            raise RuntimeError(f"scheme not implemented {scheme}")
        if index < len(dates) and abs((m + 1) * h - dates[index]) < 1e-8:
            values[:, index, :] = L
            index += 1
    if index != len(dates):
        raise ValueError("exercise dates not in this tenor structure")
    return values
