import numpy as np

# example of a continuously compounded spot rate r(0, T) at time 0 for maturity T,
# increasing from 1.5% (T = 0) to 2.7% (T -> infinity)
def spot(T):
    return 0.015 + 0.012 * (1 - np.exp(-0.15 * T))

# zero bond price at time 0
def market_discount(T):
    return np.exp(- spot(T) * T)

# LIBOR forward rate at time 0 for the accrual period [T, S] with tau = S - T:
def libor_forward(T, S, tau):
    return 1 / tau * (market_discount(T) / market_discount(S) - 1)

# fair swap rate at time 0 of a swap starting in T (= T0) and ending in S (= TN)
# with payment dates T + tenor, ..., S:
# S(0) = (P(0, T) - P(0, S)) / A(0),  A(0) = sum_i tenor * P(0, T_i)  (annuity)
def swap_rate(T, S, tenor):
    return (market_discount(T) - market_discount(S))/(np.sum(tenor * market_discount(np.arange(T + tenor, S + tenor, tenor))))
