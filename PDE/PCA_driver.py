import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # root directory

import numpy as np
import time
from functools import partial
from PDE.PDESolverFullGrid_PCA import swaption_subset, bermuda_swaption_subset, get_solution_at_y0, check_components
from PDE.combination_technique import grid_1d, combination_technique

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# bounds for the grid in y direction: y_j in [-m sqrt(lambda_j T), m sqrt(lambda_j T)]
# for the PCA directions component_indices (0-indexed), m: number of standard deviations
def y_bounds(model, component_indices, m, T):
    component_indices = check_components(model, component_indices)
    eigenvalues = model.eigenvalues
    span = m * np.sqrt(eigenvalues[component_indices] * T)
    return -span, span

# builds the y-grid and the dy values of the sub grid l (2**l_j intervals in direction j)
def level_grid(l, y_min, y_max):
    y_grid = [grid_1d(l[i], y_min[i], y_max[i]) for i in range(len(l))]
    dy = [g[1] - g[0] for g in y_grid]
    return y_grid, dy

# solves the European swaption PDE on the sub grid l, Return: value at y = 0 (terminal measure)
def european_level(l, model, K, component_indices, y_min, y_max, dt, n_steps, theta, phi, eps=0.0):
    y_grid, dy = level_grid(l, y_min, y_max)
    V = swaption_subset(model, K, y_grid, dy, component_indices, dt, n_steps, theta, phi=phi, eps=eps)
    return get_solution_at_y0(V, y_grid)

# solves the Bermudan swaption PDE on the sub grid l, Return: value at y = 0 (terminal measure)
def bermudan_level(l, model, K, component_indices, y_min, y_max, n_steps, theta, numExerciseDates, phi):
    y_grid, dy = level_grid(l, y_min, y_max)
    V = bermuda_swaption_subset(model, K, y_grid, dy, component_indices, n_steps, theta,
                                model.T0, model.TN, numExerciseDates, phi=phi)
    return get_solution_at_y0(V, y_grid)

# components: number k (the k largest eigenvalues) or list of PCA directions (0-indexed)
def component_list(components):
    return list(range(components)) if np.isscalar(components) else list(components)


# ===============================
# FULL GRID (if number of components \in\{1,2\})
# ===============================
# European swaption (exercise in model.T0) on one full grid with 2**n intervals in every direction
# components: number k (the k largest eigenvalues) or list of PCA directions (0-indexed)
# n_steps: time steps on [0, T0] (graded with q), m: number of standard deviations of the grid
# Return: price in bps
def full_european(model, K, components, n, theta=0.5, n_steps=50, m=4, q=3, phi=1, eps=0.0):
    component_indices = component_list(components)
    dt = np.diff(model.T0 * (np.arange(n_steps + 1) / n_steps) ** q)
    y_min, y_max = y_bounds(model, component_indices, m, model.T0)
    value = european_level((n,) * len(component_indices), model, K, component_indices, y_min, y_max,
                           dt, n_steps, theta, phi, eps)
    return model.to_bps(value)

# Bermudan swaption (exercise in T0, T0 + tau, ..., numExerciseDates dates, swap until model.TN)
# on one full grid with 2**n intervals in every direction
# components: number k (the k largest eigenvalues) or list of PCA directions (0-indexed)
# n_steps: time steps per exercise period, m: number of standard deviations of the grid
# Return: price in bps
def full_bermudan(model, K, components, n, numExerciseDates, theta=0.5, n_steps=50, m=4, phi=1):
    component_indices = component_list(components)
    last_exercise = model.T0 + (numExerciseDates - 1) * model.tau
    y_min, y_max = y_bounds(model, component_indices, m, last_exercise)
    value = bermudan_level((n,) * len(component_indices), model, K, component_indices, y_min, y_max,
                           n_steps, theta, numExerciseDates, phi)
    return model.to_bps(value)


# ===============================
# SPARSE GRID (COMBINATION TECHNIQUE) (if number of components >=3)
# ===============================
# European swaption (exercise in model.T0) with the combination technique on level n
# components: number k (the k largest eigenvalues) or list of PCA directions (0-indexed)
# n_steps: time steps on [0, T0] (graded with q), m: number of standard deviations of the grid
# Return: price in bps
def sparse_european(model, K, components, n, theta=0.5, n_steps=50, m=4, q=3, phi=1, eps=0.0):
    component_indices = component_list(components)
    tvec = model.T0 * (np.arange(n_steps + 1) / n_steps) ** q
    dt = np.diff(tvec)
    y_min, y_max = y_bounds(model, component_indices, m, model.T0)
    solve = partial(european_level, model=model, K=K, component_indices=component_indices, y_min=y_min,
                    y_max=y_max, dt=dt, n_steps=n_steps, theta=theta, phi=phi, eps=eps)
    value = combination_technique(solve, n, len(component_indices))
    return model.to_bps(value)

# Bermudan swaption (exercise in T0, T0 + tau, ..., numExerciseDates dates, swap until model.TN)
# with the combination technique on level n
# components: number k (the k largest eigenvalues) or list of PCA directions (0-indexed)
# n_steps: time steps per exercise period, m: number of standard deviations of the grid
def sparse_bermudan(model, K, components, n, numExerciseDates, theta=0.5, n_steps=50, m=4, phi=1):
    component_indices = component_list(components)
    last_exercise = model.T0 + (numExerciseDates - 1) * model.tau
    y_min, y_max = y_bounds(model, component_indices, m, last_exercise)
    solve = partial(bermudan_level, model=model, K=K, component_indices=component_indices, y_min=y_min,
                    y_max=y_max, n_steps=n_steps, theta=theta, numExerciseDates=numExerciseDates, phi=phi)
    value = combination_technique(solve, n, len(component_indices))
    return model.to_bps(value)

# calls price_fn for the levels n = level_min, ..., level_max
# price_fn: function n -> price in bps, e.g. partial(sparse_european, model, K, components) with all
# other parameters fixed
# Return: level_array, prices in bps
def sparse_levels(price_fn, level_min, level_max):
    level_array = np.arange(level_min, level_max + 1)
    values = []
    for n in level_array:
        time1 = time.time()
        values.append(price_fn(n=int(n)))
        print(f"Level {n}: {values[-1]} bps, Execution Time: {time.time() - time1:.2f} s")
    return level_array, values


if __name__ == "__main__":
    from common.lmm import LiborMarketModel
    from common.YieldCurve import swap_rate
    
    # test data
    exercise = 2.0
    maturity = 15.0
    exerciseDates = 9
    tau = 1  # tenor
    numComponents = 3  # number of components for PCA

    model = LiborMarketModel.from_setup(exercise, maturity, tau)
    K = swap_rate(exercise, maturity, tau)

    theta = 0.5
    m = 4  # number of standard deviations
    n_steps = 50
    level_min = 9
    level_max = 9  # cost grows with numComponents

    price_fn = partial(sparse_bermudan, model=model, K=K, components=numComponents,
                       numExerciseDates=exerciseDates, theta=theta, n_steps=n_steps, m=m, phi=1)
    level_array, solutionVec = sparse_levels(price_fn, level_min, level_max)
