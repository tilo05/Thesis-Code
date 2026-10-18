import os
import sys
import inspect
from functools import partial
import time
import numpy as np
from contourpy import contour_generator
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root directory
sys.path.insert(0, ROOT_DIR)

from common.YieldCurve import swap_rate
from common.lmm import LiborMarketModel
from PDE.PDESolverFullGrid_PCA import swaption, get_solution_at_y0, check_components, swaption_payoff_grid
from PDE.PDESolverFullGrid_PCA import bermuda_swaption as bermuda_swaption_pde
from PDE.PCA_driver import sparse_european, sparse_bermudan, sparse_levels, full_european, full_bermudan
from MonteCarlo.MLMC import mlmc_price
from MonteCarlo.LSMC import bermuda_swaption

DATA_DIR = os.path.join(ROOT_DIR, "data")
EUROPEAN_PDE_HEADER = ["components", "level", "T", "S", "cumVar", "price", "relerror", "solver", "time", "error"]
BERMUDAN_PDE_HEADER = ["components", "level", "T0", "Tn", "TN", "cumVar", "price", "relerror", "solver", "time", "error"]
LEVEL_ANALYSIS_HEADER = ["components", "level", "T0", "Tn", "S", "cumVar", "price", "relerror", "time"]


# builds the grid in y direction
def build_y_grid(m, numComponents, eigenvalues, points, T):
    y0 = np.zeros(numComponents) 
    y_grid = []
    for j in range(numComponents):
        span = m * np.sqrt(eigenvalues[j] * T)
        y_grid.append(np.linspace(y0[j] - span, y0[j] + span, points[j] + 1))
    dy = [g[1] - g[0] for g in y_grid]
    return y_grid, dy

# self convergence without a reference solution: plots the differences e_n = |u_n - u_{n+1}|
# and calculates the orders p_n = log2(e_n / e_{n+1})
# Input Data:
# level_array: levels n of the solutions
# values: solutions u_n belonging to level_array
# label: title of the plot
# dat_path: path without ending, writes <dat_path>.dat with the columns level value error order
#           (None: no file)
# Return:
# diffs: differences e_n, belonging to the levels level_array[:-1]
# orders: orders p_n (nan, if e_n or e_{n+1} is zero; the last entry is always nan)
def plot_convergence(level_array, values, label="Convergence", dat_path=None):
    level_array = np.asarray(level_array, dtype=float)
    values = np.asarray(values, dtype=float)
    diffs = np.abs(np.diff(values))  # e_n = |u_n - u_{n+1}|
    diff_levels = level_array[:-1]
    orders = np.full(len(diffs), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        orders[:-1] = np.log2(diffs[:-1] / diffs[1:])
    orders[~np.isfinite(orders)] = np.nan
    print(f"[{label}] orders p_n:", orders[:-1])

    plt.figure(figsize=(6, 4))
    plt.loglog(2 ** diff_levels, diffs, 'o-', markersize=7)
    plt.xscale("log", base=2)
    plt.xlabel("$2^n$")
    plt.ylabel(r"$|u_n - u_{n+1}|$")
    plt.title(label)
    plt.grid(True, which='both')
    plt.show()

    if dat_path is not None:
        os.makedirs(os.path.dirname(dat_path), exist_ok=True)
        data_file = f"{dat_path}.dat"
        with open(data_file, "w") as f:
            f.write("level value error order\n")
            for n, u, e, p in zip(diff_levels, values, diffs, orders):
                f.write(f"{n:.6g} {u:.10g} {e:.10g} {p:.6g}\n")
        print(f"[{label}] data written to {data_file}")
    return diffs, orders

# calculates the eoc for the full grid solver
# direction = 0: in y direction; direction=1: t direction
def eoc_fullgrid(direction, model, K, theta, numComponents, m=4, q=3, phi=1):
    name = "x" if direction == 0 else "t"
    levels = [2, 3, 4, 5, 6, 7, 8, 9] if direction == 0 else [3, 4, 5, 6, 7, 8, 9]
    n_fixed = 500 if direction == 0 else 2 ** 7
    levels = np.array(levels)

    exercise = model.T0
    eigenvalues, Q = model.eigenvalues, model.Q
    sol = []
    for level in levels:
        if direction == 0:
            n_points, n_steps = 2 ** int(level), n_fixed
        else:
            n_points, n_steps = n_fixed, 2 ** int(level)
        N_points = [n_points] * numComponents
        tvec = exercise * (np.arange(n_steps + 1) / n_steps) ** q
        dt = np.diff(tvec)
        y_grid, dy = build_y_grid(m, numComponents, eigenvalues, N_points, exercise)
        V = swaption(model, K, y_grid, dy, numComponents, dt, n_steps, theta, phi=phi)
        value = model.to_bps(get_solution_at_y0(V, y_grid))
        print(f"Level {level}: N_points = {N_points}, n_steps = {n_steps}, value = {value}")
        sol.append(value)

    diffs, orders = plot_convergence(levels, sol, label=f"EOC-{name}-direction (fullgrid)", dat_path=os.path.join(DATA_DIR, f"eoc_{name}_fullgrid"))
    return diffs, orders

# calculates the eoc for the sparse grid solution
def eoc_sparsegrid(model, K, theta, numComponents=None, m=4, q=3, phi=1, eps=5e-4):
    n_steps = 500
    numComponents = model.n_forwards
    levels = [3, 4, 5, 6, 7, 8, 9, 10]
    price_fn = partial(sparse_european, model=model, K=K, components=numComponents, theta=theta,
                       n_steps=n_steps, m=m, q=q, phi=phi, eps=eps)
    level_array, sol = sparse_levels(price_fn, min(levels), max(levels))
    name = f"eoc_x_sparsegrid_T{model.T0:g}S{model.TN:g}_k{numComponents}" + (f"_eps{eps * 1e4:g}bp" if eps > 0 else "")
    diffs, orders = plot_convergence(level_array, sol,
                            label=f"EOC-x-direction (Sparse Grid, ({model.T0:g},{model.TN:g}), k={numComponents})",
                            dat_path=os.path.join(DATA_DIR, name))
    return diffs, orders

# PDE price of a European swaption (T0, S) on a full grid:
# Input Data:
# T0: exercise date (first reset date of the swap)
# S: maturity of the swap
# tau: tenor
# numComponents: number of retained PCA components
# theta: parameter of the Douglas scheme (0.5: Crank-Nicolson type)
# n_points: number of grid intervals per direction
# n_steps: number of time steps over [0, T0]
# m: domain size in standard deviations, y_j in [-m sqrt(lambda_j T0), m sqrt(lambda_j T0)]
# q: grading exponent of the time steps
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# Return:
# row for EUROPEAN_PDE_HEADER
def european_price(T0, S, tau, numComponents, theta=0.5, n_points=2**9, n_steps=200, m=4, q=3, phi=1):
    model = LiborMarketModel.from_setup(T0, S, tau)
    K = swap_rate(T0, S, tau) # fair swap
    check_components(model, range(numComponents))
    eigenvalues, Q = model.eigenvalues, model.Q
    N_points = [n_points] * numComponents
    tvec = T0 * (np.arange(n_steps + 1) / n_steps) ** q
    dt = np.diff(tvec)
    y_grid, dy = build_y_grid(m, numComponents, eigenvalues, N_points, T0)
    time_start = time.time() # measure the execution time
    V = swaption(model, K, y_grid, dy, numComponents, dt, n_steps, theta, phi=phi)
    time_end = time.time() # measure the execution time
    price = model.to_bps(get_solution_at_y0(V, y_grid))
    execution_time = time_end - time_start
    print(f"European PDE (T0, S) = ({T0}, {S}), {numComponents} components: {price} bps")
    return [numComponents, np.log2(n_points), T0, S, cum_var(model, numComponents), price,
            np.nan, "full", execution_time, np.nan]

# PDE price of a European swaption (T0, S) on a sparse grid:
# Input Data:
# T0: exercise date (first reset date of the swap)
# S: maturity of the swap
# tau: tenor
# numComponents: number of retained PCA components
# n: level of the combination technique
# theta: parameter of the Douglas scheme (0.5: Crank-Nicolson type)
# n_steps: number of time steps over [0, T0]
# m: domain size in standard deviations
# q: grading exponent of the time steps
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# Return:
# row for EUROPEAN_PDE_HEADER
def european_price_sparse(T0, S, tau, numComponents, n=9, theta=0.5, n_steps=200, m=4, q=3, phi=1):
    model = LiborMarketModel.from_setup(T0, S, tau)
    K = swap_rate(T0, S, tau)
    time_start = time.time()
    price = sparse_european(model, K, numComponents, n, theta=theta, n_steps=n_steps, m=m, q=q, phi=phi)
    time_end = time.time()
    execution_time = time_end - time_start
    print(f"European PDE sparse (T0, S) = ({T0}, {S}), {numComponents} components, level {n}: {price} bps")
    return [numComponents, n, T0, S, cum_var(model, numComponents), price,
            np.nan, "sparse", execution_time, np.nan]

# share of the variance explained by the numComponents largest eigenvalues (cumulative explained variance)
def cum_var(model, numComponents):
    eigenvalues = model.eigenvalues
    return np.sum(eigenvalues[:numComponents]) / np.sum(eigenvalues)

# loads the reference mid prices from the given filename (if it exists)
def load_reference_mid(filename, key_cols):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"No reference found.")
        return {}
    data = np.atleast_1d(np.genfromtxt(path, names=True))
    # swap end "TN" or "S", (European or Bermudan)
    aliases = {"TN": "S", "S": "TN"}
    cols = [c if c in data.dtype.names else aliases.get(c, c) for c in key_cols]
    missing = [c for c in cols if c not in data.dtype.names]
    if missing:
        raise ValueError(f"{path}: column {missing} missing")
    return {tuple(float(r[c]) for c in cols): float(r["mid"]) for r in data}

# calculates the reference error error=|price - mid| and relative error=error / mid
def add_reference_error(rows, header, ref_mid, key_cols):
    key_idx = [header.index(c) for c in key_cols]
    i_price, i_rel, i_err = header.index("price"), header.index("relerror"), header.index("error")
    for row in rows:
        mid = ref_mid.get(tuple(float(row[i]) for i in key_idx))
        if mid is not None:
            row[i_err] = abs(row[i_price] - mid) # absolute error
            row[i_rel] = row[i_err] / mid # relative error
    return rows

# keeps only the numbers of components k <= "number of forwards". the others are skipped
# Input Data:
# T0, TN: first exercise date and maturity of the swap
# tau: tenor
# components: numbers of PCA components
# Return:
# list of the valid numbers of components
def valid_components(T0, TN, tau, components):
    n_forwards = int(round((TN - T0) / tau))
    skipped = [k for k in components if k > n_forwards]
    if skipped:
        print(f"Setup ({T0}, {TN}) has only {n_forwards} Forwards, skip component {skipped}")
    return [k for k in components if k <= n_forwards]

# calculates the PDE price (full grid) of a Bermudan swaption 
# Input Data:
# T0, Tn, TN: first and last exercise date, maturity of the swap
# tau: tenor
# numComponents: number of retained PCA components
# theta: parameter of the Douglas scheme
# n_points: number of grid intervals per direction
# n_steps: number of time steps between two exercise dates
# m: domain size in standard deviations
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# Return:
# row for BERMUDAN_PDE_HEADER
def bermudan_price(T0, Tn, TN, tau, numComponents, theta=0.5, n_points=2**9, n_steps=50, m=4, phi=1):
    model = LiborMarketModel.from_setup(T0, TN, tau)
    K = swap_rate(T0, TN, tau)
    numExerciseDates = int(round((Tn - T0) / tau)) + 1
    check_components(model, range(numComponents))
    eigenvalues, Q = model.eigenvalues, model.Q
    N_points = [n_points] * numComponents
    y_grid, dy = build_y_grid(m, numComponents, eigenvalues, N_points, Tn)
    time_start = time.time()
    V = bermuda_swaption_pde(model, K, y_grid, dy, numComponents, n_steps, theta,
                             T0, TN, numExerciseDates, phi=phi)
    time_end = time.time()
    price = model.to_bps(get_solution_at_y0(V, y_grid))
    execution_time = time_end - time_start
    print(f"Bermudan PDE (T0, Tn, TN) = ({T0}, {Tn}, {TN}), {numComponents} components: {price} bps")
    return [numComponents, np.log2(n_points), T0, Tn, TN, cum_var(model, numComponents), price,
            np.nan, "full", execution_time, np.nan]

# calculates the PDE price with sparse grid of a Bermudan swaption
# Input Data: as in bermudan_price
# Return:
# row for BERMUDAN_PDE_HEADER
def bermudan_price_sparse(T0, Tn, TN, tau, numComponents, n=9, theta=0.5, n_steps=50, m=4, phi=1):
    model = LiborMarketModel.from_setup(T0, TN, tau)
    K = swap_rate(T0, TN, tau)
    numExerciseDates = int(round((Tn - T0) / tau)) + 1
    time_start = time.time()
    price = sparse_bermudan(model, K, numComponents, n, numExerciseDates, theta=theta,
                            n_steps=n_steps, m=m, phi=phi)
    time_end = time.time()
    execution_time = time_end - time_start
    print(f"Bermudan PDE sparse (T0, Tn, TN) = ({T0}, {Tn}, {TN}), {numComponents} components, level {n}: {price} bps")
    return [numComponents, n, T0, Tn, TN, cum_var(model, numComponents), price,
            np.nan, "sparse", execution_time, np.nan]

# sparse grid prices of the Bermudan swaption (T0, Tn, TN)
# Input Data:
# T0, Tn, TN: first and last exercise date, maturity of the swap
# tau: tenor
# levels: levels n of the sparse grid
# components: numbers of PCA components
# theta, n_steps, m, phi: as in bermudan_price
# filename: name of the output file in DATA_DIR
# Return:
# rows for LEVEL_ANALYSIS_HEADER
def level_analysis(T0=2, Tn=10, TN=15, tau=1, levels=range(6, 11), components=(3, 4, 5), theta=0.5,
                   n_steps=50, m=4, phi=1, filename="level_analysis.dat"):
    ref_mid = load_reference_mid("Bermudan_LSMC.dat", ["T0", "Tn", "TN"]).get((float(T0), float(Tn), float(TN)))
    rows = []
    for k in valid_components(T0, TN, tau, components):
        for n in levels:
            r = bermudan_price_sparse(T0, Tn, TN, tau, k, n=n, theta=theta, n_steps=n_steps, m=m, phi=phi)
            price, ex_time = r[6], r[9] # price and execution time
            relerror = abs(price - ref_mid) / ref_mid if ref_mid is not None else np.nan
            rows.append([k, n, T0, Tn, TN, r[5], price, relerror, ex_time])
            write_dat(filename, LEVEL_ANALYSIS_HEADER, rows)
    return rows

# PDE price (bps) with the PCA directions component_indices 
# Input Data:
# product: "european" or "bermudan"
# model: the specific LIBOR market model [lmm.py]
# K: strike
# component_indices: retained directions, 0-indexed
# numExerciseDates: number of exercise dates (only bermudan)
# method: "full" or "sparse"
# n: level of the full/sparse grid
# theta: parameter of the Douglas scheme
# m: domain size in standard deviations
# q: grading exponent of the time steps (european)
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# Return:
# price in bps
def price_components(product, model, K, component_indices, numExerciseDates=None, method="sparse",
                     n=9, theta=0.5, m=4, q=3, phi=1):
    if product == "european":
        solve = sparse_european if method == "sparse" else full_european
        return solve(model, K, component_indices, n, theta=theta, n_steps=200, m=m, q=q, phi=phi)
    if product == "bermudan":
        solve = sparse_bermudan if method == "sparse" else full_bermudan
        return solve(model, K, component_indices, n, numExerciseDates, theta=theta, n_steps=50, m=m, phi=phi)
    raise ValueError("product must be 'european' or 'bermudan'")

# sensitivity based validation of the PCA truncation 
# u_hat = u^(n) + sum_{j=n+1}^{N} (u^(n,j) - u^(n)), u^(n,j)
# Input Data:
# product: "european" or "bermudan"
# setup: (T0, TN) for european, (T0, Tn, TN) for bermudan
# n: number of components of the base solution u^(n), 1 <= n < N
# tau: tenor
# level: level of the full resp. sparse grid (2**level intervals per direction)
# theta: parameter of the Douglas scheme
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# Return:
# dictionary with u_n, u_hat, the reference price (MLMC resp. LSMC mid) and the corrections [j, u_nj, u_nj - u_n];
# writes sensitivity_<tag>_n<n>.dat and sensitivity_<tag>_n<n>_summary.dat
def sensitivity_check(product, setup, n, tau=1, level=9, theta=0.5, phi=1):
    if product == "european":
        T0, TN = setup
        numExerciseDates, tag = None, f"eu_T{T0:g}S{TN:g}"
        ref_mid = load_reference_mid("European_MLMC.dat", ["T0", "S"]).get((float(T0), float(TN)))
    else:
        T0, Tn, TN = setup
        numExerciseDates, tag = int(round((Tn - T0) / tau)) + 1, f"berm_T{T0:g}Tn{Tn:g}S{TN:g}"
        ref_mid = load_reference_mid("Bermudan_LSMC.dat", ["T0", "Tn", "TN"]).get((float(T0), float(Tn), float(TN)))
    model = LiborMarketModel.from_setup(T0, TN, tau)
    K = swap_rate(T0, TN, tau)
    N = model.n_forwards
    if not 1 <= n < N:
        raise ValueError(f"n has to be between 1 <= n < {N}")
    
    method = "sparse" if n + 1 >= 3 else "full"
    price = partial(price_components, product, model, K, numExerciseDates=numExerciseDates, method=method,
                    n=level, theta=theta, phi=phi)

    time_start = time.time()
    u_n = price(range(n))
    print(f"[{tag}] u^({n}) = {u_n:.4f} bps ({method})")
    u_hat = u_n
    rows = []
    for j in range(n + 1, N + 1):
        u_nj = price(list(range(n)) + [j - 1])
        rows.append([j, u_nj, u_nj - u_n])
        u_hat += u_nj - u_n
        print(f"[{tag}] u^({n},{j}) = {u_nj:.4f} bps; correction {u_nj - u_n:+.4f}")
    time_end = time.time()
    execution_time = time_end - time_start

    rel_un = abs(u_n - ref_mid) / ref_mid if ref_mid is not None else np.nan
    rel_hat = abs(u_hat - ref_mid) / ref_mid if ref_mid is not None else np.nan
    ref = ref_mid if ref_mid is not None else np.nan
    print(f"[{tag}] u^({n}) = {u_n:.4f}, u_hat = {u_hat:.4f}, reference = {ref:.4f} bps; "
          f"rel. error {rel_un:.4%} -> {rel_hat:.4%}")
    write_dat(f"sensitivity_{tag}_n{n}.dat", ["j", "u_nj", "diff"], rows)
    write_dat(f"sensitivity_{tag}_n{n}_summary.dat",
              ["n", "u_n", "u_hat", "ref", "relerror_un", "relerror_hat", "solver", "level", "time"],
              [[n, u_n, u_hat, ref, rel_un, rel_hat, method, level, execution_time]])
    return {"u_n": u_n, "u_hat": u_hat, "ref": ref, "corrections": rows}


# LSMC reference price of a Bermudan swaption
# Input Data:
# T0, Tn, TN: first and last exercise date, maturity of the swap
# tau: tenor
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# N: number of simulated paths
# steps_per_period: number of time steps per tenor period
# scheme: Used scheme ("euler" or "milstein")
# seed: seed for the random number generator (None: not reproducible)
# Return:
# (T0, Tn, TN, paths, mid, min, max, execution time in s), prices in bps
def bermuda_mc(T0, Tn, TN, tau, phi=1, N=10000000, steps_per_period=100, scheme="milstein", seed=None):
    model = LiborMarketModel.from_setup(T0, TN, tau)
    K = swap_rate(T0, TN, tau)
    exercise_dates = np.arange(T0, Tn + tau, tau)
    time_start = time.time()
    price, var, vmin, vmax = bermuda_swaption(model, K, exercise_dates, phi, N=N,
                                              steps_per_period=steps_per_period, scheme=scheme, seed=seed)
    time_end= time.time()
    execution_time = time_end - time_start

    mid, lo, hi = model.to_bps((vmin + vmax) / 2), model.to_bps(vmin), model.to_bps(vmax)
    print(f"Bermuda LSMC (T0, Tn, TN) = ({T0}, {Tn}, {TN}): mid {mid} bps, 95%-CI [{lo}, {hi}], paths {N}, {execution_time:.1f} s")
    return T0, Tn, TN, N, mid, lo, hi, execution_time

# writes the date to the given filename with the given headers
def write_dat(filename, header, rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in rows:
            f.write(" ".join(v if isinstance(v, str) else f"{v:.10g}" for v in row) + "\n")
    print(f"data wrote to {path}")

# MLMC reference price of a European swaption
def european_mc(T0, S, tau, phi=1, scheme="milstein", seed=None):
    model = LiborMarketModel.from_setup(T0, S, tau)
    K = swap_rate(T0, S, tau)
    time_start = time.time()
    price, var, vmin, vmax = mlmc_price(model, K, phi, scheme=scheme, seed=seed)
    time_end = time.time()
    execution_time = time_end - time_start

    # number of paths over all level: sum_l N0 * 2**(L_max-l)
    defaults = inspect.signature(mlmc_price).parameters
    L_max = defaults["L_max"].default
    N0 = defaults["N0"].default
    n_paths = N0 * (2 ** (L_max + 1) - 1)

    mid, lo, hi = model.to_bps((vmin + vmax) / 2), model.to_bps(vmin), model.to_bps(vmax)
    print(f"European MLMC (T0, S) = ({T0}, {S}): mid {mid} bps, 95%-CI [{lo}, {hi}], paths {n_paths}, {execution_time:.1f} s")
    return T0, S, n_paths, mid, lo, hi, execution_time

# exercise boundaries of a Bermudan swaption with 2 PCA components (full grid) at all exercise dates.
# Input Data:
# T0, Tn, TN: first and last exercise date, maturity of the swap
# tau: tenor
# level: 2**level grid intervals per direction
# n_steps: number of time steps between two exercise dates
# theta: parameter of the Douglas scheme
# m: domain size in standard deviations
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# n_out: approximate number of points of the boundary written to the file (between n_out and 2 * n_out)
# Return:
# price in bps; writes exercise_boundary_T{T0}Tn{Tn}S{TN}_k2_t{T}.dat (columns y1 y2) for every
# exercise date T
def exercise_boundary(T0=1, Tn=3, TN=5, tau=1, level=9, n_steps=50, theta=0.5, m=4, phi=1, n_out=120):
    numComponents = 2 # boundary as a line in the (y1, y2) plane
    model = LiborMarketModel.from_setup(T0, TN, tau)
    K = swap_rate(T0, TN, tau)
    Q_k = model.Q[:, :numComponents]
    y_grid, dy = build_y_grid(m, numComponents, model.eigenvalues, [2**level] * numComponents, Tn)
    numExerciseDates = int(round((Tn - T0) / tau)) + 1
    V, continuation = bermuda_swaption_pde(model, K, y_grid, dy, numComponents, n_steps, theta, T0, TN,
                                           numExerciseDates, phi=phi, return_continuation=True)
    price = model.to_bps(get_solution_at_y0(V, y_grid))
    print(f"Bermudan PDE (T0, Tn, TN) = ({T0}, {Tn}, {TN}), {numComponents} components: {price:.4f} bps")

    for T, C in sorted(continuation.items()):
        n_libors = int(round((TN - T) / tau)) # forwards of the remaining swap
        swap = (swaption_payoff_grid(model, y_grid, Q_k, K, phi, n_libors)
                - swaption_payoff_grid(model, y_grid, Q_k, K, -phi, n_libors))
        lines = contour_generator(y_grid[0], y_grid[1], (swap - C).T, line_type="Separate").lines(0.0)
        boundary = max(lines, key=len) # longest line of D = 0
        boundary = boundary[np.argsort(boundary[:, 1])] # sorted by y2
        boundary = boundary[::max(1, len(boundary) // n_out)]
        write_dat(f"exercise_boundary_T{T0:g}Tn{Tn:g}S{TN:g}_k2_t{T:g}.dat", ["y1", "y2"], boundary)
    return price

if __name__ == '__main__':
    EUROPEAN_SETUP = [(1, 6), (5, 15)]
    BERMUDAN_SETUP = [(1, 3, 5), (2, 5, 10), (2, 10, 15), (5, 10, 15)]
    q = 3
    theta = 0.5
    tau = 1

    ##############################
    # European Swaption (T,S) = (1,3) (eoc) (full grid)
    ##############################
    T0, TN = 1, 3
    model = LiborMarketModel.from_setup(T0, TN, tau)
    K = swap_rate(T0, TN, tau)
    components = model.n_forwards  # all components

    direction = 1  # 0 = x-direction, 1 = t-direction
    eoc_fullgrid(direction, model, K, theta, components)

    ##############################
    # European Swaption (T,S) \in\{(1,4), (1,5)\} (eoc) (sparse grid)
    ##############################
    for T0, TN in [(1, 4), (1, 5)]:
        model = LiborMarketModel.from_setup(T0, TN, tau)
        K = swap_rate(T0, TN, tau) 
        eoc_sparsegrid(model, K, theta)

    ##############################
    # MLMC reference prices
    ##############################
    rows = [european_mc(T0, TN, tau, phi=1) for T0, TN in EUROPEAN_SETUP]
    write_dat("European_MLMC.dat", ["T0", "S", "paths", "mid", "min", "max", "time"], rows)

    ##############################
    # PDE prices European Swaptions
    ##############################
    # 1-2 components
    rows = [european_price(T0, TN, tau, k, theta=theta, phi=1)
            for T0, TN in EUROPEAN_SETUP for k in valid_components(T0, TN, tau, (1, 2))]
    # 3-5 components
    rows += [european_price_sparse(T0, TN, tau, k, n=9, theta=theta, phi=1)
             for T0, TN in EUROPEAN_SETUP for k in valid_components(T0, TN, tau, (3, 4, 5))]
    
    rows.sort(key=lambda r: (r[2], r[3], r[0]))

    add_reference_error(rows, EUROPEAN_PDE_HEADER, load_reference_mid("European_MLMC.dat", ["T0", "S"]), ["T", "S"]) # add reference errors (if they exist)
    write_dat("European_PDE.dat", EUROPEAN_PDE_HEADER, rows)

    ##############################
    # LSMC reference prices for Bermudan Swaptions
    ##############################
    rows = [bermuda_mc(T0, Tn, TN, tau, phi=1) for T0, Tn, TN in BERMUDAN_SETUP]
    write_dat("Bermudan_LSMC.dat", ["T0", "Tn", "TN", "paths", "mid", "min", "max", "time"], rows)

    ##############################
    # PDE prices Bermudan Swaptions
    ##############################
    rows = [bermudan_price(T0, Tn, TN, tau, k, theta=theta, n_steps=50, phi=1)
            for T0, Tn, TN in BERMUDAN_SETUP for k in valid_components(T0, TN, tau, (1, 2))]
    rows += [bermudan_price_sparse(T0, Tn, TN, tau, k, n=9, theta=theta, n_steps=50, phi=1)
             for T0, Tn, TN in BERMUDAN_SETUP for k in valid_components(T0, TN, tau, (3, 4, 5))]
    
    rows.sort(key=lambda r: (r[2], r[3], r[4], r[0]))  
    add_reference_error(rows, BERMUDAN_PDE_HEADER, load_reference_mid("Bermudan_LSMC.dat", ["T0", "Tn", "TN"]),
                        ["T0", "Tn", "TN"])
    write_dat("Bermudan_PDE.dat", BERMUDAN_PDE_HEADER, rows)

    ##############################
    # sensitivity based validation of the truncation
    ##############################
    sensitivity_check("european", (5, 15), n=1, tau=tau, theta=theta)
    sensitivity_check("bermudan", (2, 10, 15), n=1, tau=tau, theta=theta)

    ##############################
    # Level-Analysis of Bermuda-Swaption (2,10,15): Level 6-10, 3-5 components
    ##############################
    level_analysis(T0=2, Tn=10, TN=15, tau=tau, levels=range(6, 11), components=(3, 4, 5), theta=theta)

    ##############################
    # exercise boundary Bermudan swaption (1,3,5), 2 components
    ##############################
    exercise_boundary(T0=1, Tn=3, TN=5, tau=tau, theta=theta)
