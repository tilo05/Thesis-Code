import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # project root

import numpy as np
import time
from common.YieldCurve import swap_rate
from common.lmm import LiborMarketModel
from common.lmm_functions import annuity_swaprate, swaption_payoff
from MonteCarlo.paths import simulate_at_dates

# LSMC to price a Bermudan Swaption
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# K: Strike
# exercise_dates: array of the exercise dates
# phi: phi=1 for a Payer Swaption (payoff max(S-K, 0)), phi=-1 for a Receiver Swaption (payoff max(K-S, 0))
# N: number of simulated paths
# steps_per_period: number of time steps per tenor period tau
# scheme: Used scheme (Milstein or Euler)
# seed: seed for the random number generator (None: not reproducible)
# Regression (continuation value, only in-the-money paths): basis 1, S, S^2, A, A*S
# with swap rate S and annuity A of the remaining swap
# Return:
# mean: calculated LSMC Price
# var: sample variance of the pathwise values
# vmin: lower bound of the 95% confidence interval, mean - 1.96 * sqrt(var / N)
# vmax: upper bound of the 95% confidence interval, mean + 1.96 * sqrt(var / N)
def bermuda_swaption(model, K, exercise_dates, phi, N=10000000, steps_per_period=100, scheme="milstein", seed=None):
    rng = np.random.default_rng(seed)
    M = int(round(exercise_dates[-1] / model.tau)) * steps_per_period # total number of time steps over [0, Tn]

    values = simulate_at_dates(model, exercise_dates, M, N, scheme=scheme, rng=rng)
    n_ex = len(exercise_dates)
    payoffs = np.zeros((N, n_ex))
    for i in range(n_ex):
        payoff = swaption_payoff(values[:, i, i:], K, model.tau, phi)
        payoffs[:, i] = payoff
    V = payoffs[:, -1]
    for t in range(n_ex - 2, -1, -1):
        intrinsic = payoffs[:, t]
        itm = intrinsic > 0 
        X = values[itm, t, t:] # only In the Money paths
        Y = V[itm] # realised (discounted) future cash flows, regressed to get the continuation value
        ann, S, _ = annuity_swaprate(X, model.tau)
        # linear regression
        A = np.column_stack([
            np.ones(len(S)),
            S,
            S**2,
            ann,
            ann*S
        ])
        coeff = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation = A @ coeff
        continuation_full = np.zeros_like(V)
        continuation_full[itm] = continuation
        continuation_full[~itm] = V[~itm]
        V = np.where(intrinsic > continuation_full, intrinsic, V)
    
    mean = np.mean(V)
    var = np.var(V)
    se = np.sqrt(var / N)
    # 95% CI
    vmin = mean - 1.96 * se
    vmax = mean + 1.96 * se
    return mean, var, vmin, vmax

if __name__ == '__main__':
    # Test Data
    iters = 1000 # number of paths simulations
    first_exercises = [1, 2, 2, 2, 5]
    last_exercises = [3, 5, 5, 10, 10]
    maturitys = [5, 7, 10, 15, 15]
    tenor = 1

    for i in range(len(maturitys)):
        first_exercise = first_exercises[i]
        last_exercise = last_exercises[i]
        maturity = maturitys[i]

        exercise_dates = np.arange(first_exercise, last_exercise + tenor, tenor)
        K = swap_rate(first_exercise, maturity, tenor)
        model = LiborMarketModel.from_setup(first_exercise, maturity, tenor)

        time_start = time.time()
        price_call, var_call, vmin_call, vmax_call = bermuda_swaption(model, K, exercise_dates, 1, iters)
        mid_price = model.to_bps((vmax_call + vmin_call) / 2)
        time_end = time.time()

        print("")
        print("Price for T0:", first_exercise, ", T_(n-1): ", last_exercise, " T_N: ", maturity)
        print("LSMC price call (bps): ", model.to_bps(price_call))
        print("95% CI: [", model.to_bps(vmin_call), model.to_bps(vmax_call), "]")
        print("mid price: ", mid_price)
        print("Execution time:", time_end - time_start, "seconds")
        print("")
