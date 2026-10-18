import numpy as np

# Helper functions of the LIBOR market model


# calculates the drift function in the LIBOR market model under the Terminal measure
def drift(model, L):
    a = model.tau * L / (1 + model.tau * L) * model.sigma
    return - model.sigma * (a @ model.upper_rho.T)


# calculates the annuity and the swap rate at the reset date t of the first forward in L
def annuity_swaprate(L, tau):
    disc = 1.0 / (1.0 + tau * L)
    P = np.empty((L.shape[0], L.shape[1] + 1))
    P[:, 0] = 1.0
    P[:, 1:] = np.cumprod(disc, axis=1)
    annuity = tau * np.sum(P[:, 1:], axis=1)
    swap_rate = (1.0 - P[:, -1]) / annuity
    return annuity, swap_rate, P[:, -1]


# Swaption Payoff divided by the Numeraire P(t,TN): A * max(phi * (S - K), 0) / P(t, TN)
def swaption_payoff(L, K, tau, phi=1):
    annuity, swap_rate, P_end = annuity_swaprate(L, tau)
    return annuity * np.maximum(phi * (swap_rate - K), 0) / P_end


# smooth swaption payoff (to smooth the kink of the max(z, 0) function)
# for the convergence analysis, uses the softplus function
# eps: smoothing width in units of the swap rate (5e-4 = 5 bp)
def swaption_payoff_smooth(L, K, tau, phi=1, eps=5e-4):
    annuity, swap_rate, P_end = annuity_swaprate(L, tau)
    return annuity * eps * np.logaddexp(0, phi * (swap_rate - K) / eps) / P_end
