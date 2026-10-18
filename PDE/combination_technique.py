import numpy as np
from math import comb
from concurrent.futures import ProcessPoolExecutor

# generates an equidistant one-dimensional grid
# with 2**level+1 grid points on [a,b]
def grid_1d(level, a, b):
    return np.linspace(a, b, 2**level + 1)

# generates (recursively) all multi indices l = (l_1, ..., l_d) with l_i >= 1 and |l| = l_1 + ... + l_d = k
def multi_indices_fixed_sum(d, k):
    if d == 1:
        if k >= 1:
            yield (k,)
        return
    for i in range(1, k - (d - 1) + 1):
        for rest in multi_indices_fixed_sum(d - 1, k - i):
            yield (i,) + rest

# all multi indices l needed for the combination technique of level n
# Input Data:
# n: level of the combination technique
# d: dimension
# Return:
# list of all multi indices l (tuples of length d) with l_i >= 1 and n <= |l| <= n + d - 1
def combination_indices(n, d):
    return [l for k in range(n, n + d) for l in multi_indices_fixed_sum(d, k)]

# combines the full grid solutions u_l to the sparse grid solution of level n:
# u_n = sum_{q=0}^{d-1} (-1)^q * binom(d-1, q) * sum_{|l| = n+d-1-q} u_l
# Input Data:
# sol: dictionary of all full grid solutions on the anisotropic grids l
# n: level of the combination technique
# d: dimension
# Return:
# total: combined sparse grid solution u_n
def combination_eval(sol, n, d):
    total = 0.0
    for l, u in sol.items():
        level = sum(l)
        if n <= level <= n + d - 1:
            weight = (-1) ** (n + d - level - 1) * comb(d - 1, level - n)
            total += weight * u
    return total

# combination technique: solves the problem on all anisotropic full grids l with
# n <= |l| <= n+d-1 parallised 
# Input Data:
# solve_level: function l -> u_l, solves the problem on the full grid with levels l = (l_1, ..., l_d)
# n: level of the combination technique
# d: dimension
# max_workers: maximal number of parallel processes (None: number of processors)
# Return:
# combined sparse grid solution u_n
def combination_technique(solve_level, n, d, max_workers=None):
    indices = combination_indices(n, d) # indices that satisfy n<=|l|<=n+d-1, all indices that are required for the combination technique 
    # parallelized
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        values = list(executor.map(solve_level, indices))
    solutions = dict(zip(indices, values))
    combination_technique = combination_eval(solutions, n, d)
    return combination_technique
