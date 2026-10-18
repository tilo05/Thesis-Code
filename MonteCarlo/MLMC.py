import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # root directory

import numpy as np
import time
from common.YieldCurve import swap_rate
from common.lmm import LiborMarketModel
from common.lmm_functions import swaption_payoff
from MonteCarlo.paths import simulate_terminal



# MLMC estimator for a European Swaption
# Input Data: 
# model: the specific LIBOR market model [lmm.py]
# K: Strike
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# L_max: maximum Level for MLMC
# N0: number of paths on the finest level L_max; level l uses N0 * 2**(L_max-l) paths
# M0: number of time steps on level 0 over [0, exercise]; level l uses M0 * 2**l steps
# scheme: Used scheme (Milstein or Euler)
# seed: seed for the random number generator (None: not reproducible)
# max_elements: maximal number of random numbers per batch (memory, 5e7 ~ 400 MB)
# Return:
# price: calculated MLMC Price
# estimator_var: variance of the MLMC estimator, sum_l Var(Y_l) / N_l
# vmin: lower bound of the 95% confidence interval, price - 1.96 * sqrt(estimator_var)
# vmax: upper bound of the 95% confidence interval, price + 1.96 * sqrt(estimator_var)
def mlmc_price(model, K, phi=1, L_max=7, N0=23000, M0=20, scheme="milstein", seed=None, max_elements=5 * 10**7):
    exercise = model.T0
    rng = np.random.default_rng(seed)
    means = np.zeros(L_max+1)
    variances = np.zeros(L_max+1)
    Ns = np.zeros(L_max+1, dtype=int)

    for l in range(L_max+1):
        M_fine = M0 * 2**l # number of time steps
        Nl = N0 * 2**(L_max-l) # number of paths
        Ns[l] = Nl
        # simulate in batches, so that Z_fine has at most max_elements entries (memory)
        batch = max(1, max_elements // (M_fine * model.n_forwards))
        s1 = s2 = 0.0 # sum and sum of squares of the samples
        for start in range(0, Nl, batch):
            n_batch = min(batch, Nl - start)
            Z_fine = rng.standard_normal((n_batch, M_fine, model.n_forwards))
            Lf = simulate_terminal(model, exercise, M_fine, Z=Z_fine, scheme=scheme)
            Pf = swaption_payoff(Lf, K, model.tau, phi)
            if l == 0:
                samples = Pf
            else:
                M_coarse = int(M_fine/2)
                Z_coarse = (Z_fine[:,0::2,:] + Z_fine[:,1::2,:]) / np.sqrt(2)
                Lc = simulate_terminal(model, exercise, M_coarse, Z=Z_coarse, scheme=scheme)
                Pc = swaption_payoff(Lc, K, model.tau, phi)
                samples = Pf - Pc
            s1 += np.sum(samples)
            s2 += np.sum(samples**2)

        means[l] = s1 / Nl
        variances[l] = s2 / Nl - means[l]**2
    price = np.sum(means)
    # variance of the MLMC estimator
    estimator_var = np.sum(variances / Ns)
    # standard error
    se = np.sqrt(estimator_var)
    # 95%-CI
    vmin = price - 1.96 * se
    vmax = price + 1.96 * se
    return price, estimator_var, vmin, vmax


if __name__ == '__main__':
    # Test data
    T0, S, tau = 1, 6, 1
    model = LiborMarketModel.from_setup(T0, S, tau)
    K = swap_rate(T0, S, tau)
    time_start = time.time()
    price, var, vmin, vmax = mlmc_price(model, K, phi=1, L_max=6, N0=5000)
    time_end = time.time()
    print("MLMC price (bps): ", model.to_bps(price))
    print("95% CI: [", model.to_bps(vmin), model.to_bps(vmax), "]")
    print("Execution time:", time_end - time_start, "seconds")
