import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # root directory

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from common.solver import tridag_batch
from common.lmm_functions import drift, swaption_payoff, swaption_payoff_smooth
from common.lmm import LiborMarketModel
from common.YieldCurve import swap_rate

# PDE solver (full grid) for the PCA-reduced LIBOR market model in the coordinates y = Q_k^T (x - x0),
# x = log L.


# calculates the LIBOR rates L = exp(x0 + Q_k y) at all grid points
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# y_grid: list of one-dimensional grids of the k retained PCA directions
# Q_k: retained eigenvectors (columns), array of shape (N, k)
# Return:
# L: LIBOR rates at all grid points
# shape: shape of the grid (len(y_grid[0]), ..., len(y_grid[k-1]))
def forwards_on_grid(model, y_grid, Q_k):
    mesh_y = np.meshgrid(*y_grid, indexing="ij")
    shape = mesh_y[0].shape
    y_flat = np.stack([m.ravel() for m in mesh_y], axis=1)
    x_flat = model.x0 + y_flat @ Q_k.T 
    return np.exp(x_flat), shape

# calculates the drift of the PDE in y coordinates
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# y_grid: list of the one-dimensional grids of the k retained PCA directions
# Q_k: retained eigenvectors (columns), array of shape (N, k)
# Return:
# list of length k, entry j: drift b_j at all grid points (array with the shape of the grid)
def compute_drift_y(model, y_grid, Q_k):
    L_flat, shape = forwards_on_grid(model, y_grid, Q_k) # flattened LIBOR rates
    # full original drift of x = log L: mu_i(x) - 0.5 sigma_i^2
    term = drift(model, L_flat) - 0.5 * model.sigma ** 2
    b_flat = term @ Q_k # PCA projection
    return [b_flat[:, idx].reshape(shape) for idx in range(Q_k.shape[1])]

# builds the equidistant grids of the retained PCA directions on [-m*sqrt(lambda_j*T), m*sqrt(lambda_j*T)]
# Input Data:
# m: domain size in standard deviations
# eigvals_full: all eigenvalues of the covariance matrix, sorted descending
# component_indices: retained directions (beginning with index 0!)
# points: number of grid intervals per direction
# T: time used for the domain size (exercise date, for Bermudans the last exercise date)
# Return:
# y_grid: list of the one-dimensional grids
# dy: list of the step sizes of the grids
def build_y_grid_subset(m, eigvals_full, component_indices, points, T):
    component_indices = list(component_indices)
    numComponents = len(component_indices)
    eigvals_k = eigvals_full[component_indices]
    y0 = np.zeros(numComponents)
    y_grid = []
    for j in range(numComponents):
        span = m * np.sqrt(eigvals_k[j] * T)
        y_grid.append(np.linspace(y0[j] - span, y0[j] + span, points[j] + 1))
    dy = [g[1] - g[0] for g in y_grid]
    return y_grid, dy

# first (explicit) step of the Douglas scheme
# Input Data:
# Y: solution at the current time step
# y_grid: list of the one-dimensional grids
# drift: list of the drifts b_i at all grid points [compute_drift_y]
# lam: eigenvalues lambda_i of the retained directions
# dt: time step size
# dy: list of the step sizes of the grids
# payoff: payoff at all grid points used for the dirichlet boundary conditions
# Return:
# V_new: new solution after one explicit euler step
def first_euler_step(Y, y_grid, drift, lam, dt, dy, payoff):
    N = len(y_grid)
    # new solution
    V_new = Y.copy()
    V_ext = Y.copy()
    # inner domain
    inner = tuple(slice(1, -1) for _ in range(N))
    for dim in range(N):  # all forwards, boundary
        # left boundary
        idx = [slice(None)] * N
        idx[dim] = 0
        V_ext[tuple(idx)] = payoff[tuple(idx)]
        # right boundary
        idx[dim] = -1
        V_ext[tuple(idx)] = payoff[tuple(idx)]


    # derivatives for all N directions
    for i in range(N):
        dVdyi = np.zeros_like(Y)
        d2Vdyi2 = np.zeros_like(Y)

        idx_p = list(inner)
        idx_m = list(inner)

        idx_p[i] = slice(2, None)
        idx_m[i] = slice(None, -2)

        # central second derivative
        d2Vdyi2[inner] = ( V_ext[tuple(idx_p)] - 2 * V_ext[inner] + V_ext[tuple(idx_m)]) / (dy[i] ** 2)

        drift_vec = drift[i]

        # central first derivative
        dVdyi[inner] = (V_ext[tuple(idx_p)] - V_ext[tuple(idx_m)]) / (2 * dy[i])
        V_new[inner] += dt * (drift_vec[inner] * dVdyi[inner] + 0.5 * lam[i] * d2Vdyi2[inner])

    # set the boundary conditions
    for dim in range(N):
        # left boundary
        idx = [slice(None)] * N
        idx[dim] = 0
        V_new[tuple(idx)] = payoff[tuple(idx)]
        # right boundary
        idx[dim] = -1
        V_new[tuple(idx)] = payoff[tuple(idx)]
    return V_new

# implicit correction step of the Douglas scheme in direction j for all grid lines at once
# Input Data:
# Y: Y_{j-1}, array of shape (Nj, M), column = one grid line in direction j
# curr_sol: solution at the current time step, same shape
# Nj: number of grid points in direction j
# drift_j: drift b_j on the grid lines, same shape
# lam: eigenvalues lambda of the retained directions
# dt: time step size
# dy: list of the step sizes of the grids
# j: direction
# theta: parameter of the Douglas scheme
# Return:
# Y: solution after a implicit step (direction j)
def implicit_step(Y, curr_sol, Nj, drift_j, lam, dt, dy, j, theta):
    gamma = 0.5 * lam[j]
    # drift for all lines at once
    drift_val = drift_j
    # coefficients for the linear system
    avec = -drift_val / (2 * dy[j]) + gamma / (dy[j] ** 2)
    bvec = -2 * gamma / (dy[j] ** 2) * np.ones_like(drift_val)
    cvec = drift_val / (2 * dy[j]) + gamma / (dy[j] ** 2)
    
    # right hand side of the linear system
    rhs = Y - theta * dt * tridag_matvec(avec, bvec, cvec, curr_sol)
    # only inner grid points relevant for the linear system
    inner = slice(1, Nj-1)
    # solve the linear system. Everything at once
    Y[inner] = tridag_batch(
        -theta * dt * avec[inner],
        1 - theta * dt * bvec[inner],
        -theta * dt * cvec[inner],
        rhs[inner])
    return Y

# applies the tridiagonal matrix tridiag(a, b, c) to all grid lines at once 
# Input Data:
# a, b, c: lower, main and upper diagonal, arrays of shape (N0, M)
# V: grid lines, array of shape (N0, M)
# Return:
# sol: (A V) at the inner points 1, ..., N0-2, zero at the boundary, array of shape (N0, M)
def tridag_matvec(a, b, c, V):
    N0, M = V.shape
    N = N0 - 1

    sol = np.zeros_like(V)
    # inner points
    sol[2:N - 1, :] = (a[2:N - 1, :] * V[1:N - 2, :] + b[2:N - 1, :] * V[2:N - 1, :] + c[2:N - 1, :] * V[3:N, :])
    # left boundary
    sol[1, :] = (b[1, :] * V[1, :] + c[1, :] * V[2, :])
    # right boundary
    sol[N - 1, :] = (a[N - 1, :] * V[N - 2, :] + b[N - 1, :] * V[N - 1, :])
    return sol

# move the solution forward from t1 to t2
# Input Data:
# t1, t2: first and last time step (steps t1, ..., t2 with step sizes dt[t1-1], ..., dt[t2-1])
# y_grid: list of the one-dimensional grids
# dt: array of the time step sizes
# dy: list of the step sizes of the grids
# V: solution at the beginning (terminal payoff resp. value after the last exercise date)
# drift: list of the drifts b_j at all grid points [compute_drift_y]
# lam: eigenvalues lambda of the retained directions
# theta: parameter of the Douglas scheme
# payoff: payoff at all grid points, used as Dirichlet boundary condition
# Return:
# curr_sol: solution after the time step t2
def forward_grid_douglas(t1, t2, y_grid, dt, dy, V, drift, lam, theta, payoff):
    N_forwards = len(y_grid) # number of variables
    curr_sol = V.copy()
    for t_step in range(int(t1), int(t2 + 1), +1):
        Y = first_euler_step(curr_sol, y_grid, drift, lam, dt[t_step - 1], dy, payoff)  # first euler step
        for j in range(N_forwards):  # j-th variable, perform the implicit step
            axis = j
            # rotate the grid. j-th dimension at position 0 to make the notation easier...
            curr_sol_rot = np.moveaxis(curr_sol, axis, 0)
            Y_rot = np.moveaxis(Y, axis, 0)  
            mu_j_rot = np.moveaxis(drift[j], axis, 0) 

            shape = curr_sol_rot.shape # length of all grids
            N0 = shape[0] # length of the j-th grid
            M = int(np.prod(shape[1:])) # number of lines

            # flat everything
            curr_sol_rot_flat = curr_sol_rot.reshape(N0, M) # all lines as a matrix
            Y_rot_flat = Y_rot.reshape(N0, M) # for linear system
            mu_j_flat = mu_j_rot.reshape(N0, M) # for linear system
            # perform the implicit step
            Y_rot_flat = implicit_step(Y_rot_flat, curr_sol_rot_flat, N0, mu_j_flat, lam, dt[t_step - 1], dy, j, theta)
            # re-rotate the grid
            Y_new = Y_rot_flat.reshape(shape)
            Y = np.moveaxis(Y_new, 0, axis)
        curr_sol = Y # update solution
        # set the boundary conditions at the end of each step
        for j in range(N_forwards):
            idx_low = (slice(None),) * j + (0,) + (slice(None),) * (curr_sol.ndim - j - 1)
            idx_high = (slice(None),) * j + (-1,) + (slice(None),) * (curr_sol.ndim - j - 1)
            curr_sol[idx_low] = payoff[idx_low]
            curr_sol[idx_high] = payoff[idx_high]
    return curr_sol


# checks that all requested PCA directions exist
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# component_indices: retained directions, 0-indexed
# Return:
# component_indices as list (ValueError, if a direction does not exist)
def check_components(model, component_indices):
    component_indices = list(component_indices)
    if max(component_indices) >= model.n_forwards:
        raise ValueError(f"The model has only {model.n_forwards} Forwards. {max(component_indices)} exceeds this number")
    return component_indices


# swaption payoff normalized by the numeraire P(t, TN) at all grid points
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# y_grid: list of the one-dimensional grids
# Q_k: retained eigenvectors (columns)
# K: strike
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# relevant_libors: only the last relevant_libors forwards form the swap (Bermudan, remaining swap
#                  at an exercise date); None: all forwards
# eps: smoothing width of the payoff [swaption_payoff_smooth], only for convergence studies (0: original payoff)
# Return:
# payoff at all grid points
def swaption_payoff_grid(model, y_grid, Q_k, K, phi, relevant_libors=None, eps=0.0):
    L_flat, shape = forwards_on_grid(model, y_grid, Q_k)
    if relevant_libors is not None:
        L_flat = L_flat[:, -relevant_libors:]
    if eps > 0:  # smooth payoff
        return swaption_payoff_smooth(L_flat, K, model.tau, phi, eps).reshape(shape)
    return swaption_payoff(L_flat, K, model.tau, phi).reshape(shape)


# PDE price of a European swaption with exercise date model.T0 for an arbitrary selection of
# retained PCA directions 
# Input Data:
# model: the specific LIBOR market model [lmm.py]
# K: strike
# y_grid, dy: grids and step sizes of the retained directions [build_y_grid_subset]
# component_indices: retained directions
# dt: array of the time step sizes
# N: number of time steps
# theta: parameter of the Douglas scheme
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# eps: smoothing width of the payoff (0: original payoff)
# Return:
# V: solution at time 0 at all grid points
def swaption_subset(model, K, y_grid, dy, component_indices, dt, N, theta, phi=1, eps=0.0):
    eigvals_full, Q_full = model.eigenvalues, model.Q
    component_indices = check_components(model, component_indices) # check if component_indices are valid
    Q_k = Q_full[:, component_indices] 
    eigvals_k = eigvals_full[component_indices]

    V = swaption_payoff_grid(model, y_grid, Q_k, K, phi, eps=eps) # calculate the swaption payoff with the retained components
    drift = compute_drift_y(model, y_grid, Q_k)
    payoff = V.copy()
    V = forward_grid_douglas(1, N, y_grid, dt, dy, V, drift, eigvals_k, theta, payoff) # solve the PDE from time step 1 to N
    return V


# PDE price of a European swaption with the numComponents leading PCA directions
# shortcut for swaption_subset with component_indices = 0, ..., numComponents-1), i.e. all components
def swaption(model, K, y_grid, dy, numComponents, dt, N, theta, phi=1, eps=0.0):
    return swaption_subset(model, K, y_grid, dy, list(range(numComponents)), dt, N, theta, phi, eps)


# value of the solution at the origin y = 0 (i.e. x = x0, todays LIBOR rates), multilinear interpolation
# Input Data:
# V: solution at all grid points
# y_grid: list of the one-dimensional grids
# Return:
# V(0) as float
def get_solution_at_y0(V, y_grid):
    interp = RegularGridInterpolator(y_grid, V)
    return float(interp(np.zeros(len(y_grid)))[0])


# PDE price of a Bermudan swaption for retained PCA directions.
# model: the specific LIBOR market model [lmm.py]
# K: strike
# y_grid, dy: grids and step sizes of the retained directions [build_y_grid_subset]
# component_indices: retained directions, 0-indexed into model.eigenvalues / model.Q
# N: number of time steps per period
# theta: parameter of the Douglas scheme
# exercise: first exercise date T0
# maturity: maturity TN of the underlying swap
# numExerciseDates: number of exercise dates T0, T0 + tau, ..., T0 + (numExerciseDates-1) * tau
# phi: phi=1 for a Payer Swaption, phi=-1 for a Receiver Swaption
# return_continuation: if True, additionally return the continuation values
# Return:
# V: solution at time 0 at all grid points, divided by P(0, TN) [model.to_bps, get_solution_at_y0]
# continuation (only if return_continuation): dictionary {exercise date: continuation value C at all
#               grid points}, C = 0 at the last exercise date
def bermuda_swaption_subset(model, K, y_grid, dy, component_indices, N, theta,
                             exercise, maturity, numExerciseDates, phi=1, return_continuation=False):
    q = 3
    tau = model.tau
    exerciseDates = np.arange(exercise, exercise + numExerciseDates * tau, tau)
    relevant_libors = int((maturity - exerciseDates[-1]) / tau)
    lastExercise = exerciseDates[-1]
    exerciseDates = exerciseDates[-1] - exerciseDates
    exerciseDates = exerciseDates[::-1]
    exerciseDates = np.append(exerciseDates, lastExercise)

    eigvals_full, Q_full = model.eigenvalues, model.Q
    component_indices = check_components(model, component_indices)
    Q_k = Q_full[:, component_indices]
    eigvals_k = eigvals_full[component_indices]

    V = swaption_payoff_grid(model, y_grid, Q_k, K, phi, relevant_libors)
    terminal_payoff = V.copy() # dirichlet boundary
    drift = compute_drift_y(model, y_grid, Q_k)
    continuation = {lastExercise: np.zeros_like(V)}
    for t in range(0, len(exerciseDates) - 1, 1):
        t1 = exerciseDates[t]
        t2 = exerciseDates[t + 1]
        DeltaT = t2 - t1
        dtau_local = DeltaT * (np.arange(N + 1) / N) ** q # build the time grid between two exercise dates
        dt_local = np.diff(dtau_local)
        V = forward_grid_douglas(1, N, y_grid, dt_local, dy, V, drift, eigvals_k, theta, terminal_payoff)
        if t < len(exerciseDates) - 2:
            continuation[lastExercise - t2] = V.copy() # continuation value at the exercise date lastExercise - t2
            payoff = swaption_payoff_grid(model, y_grid, Q_k, K, phi, relevant_libors + 1) # new payoff at this time step
            V = np.maximum(V, payoff) # check for early exercise
            relevant_libors += 1
    if return_continuation:
        return V, continuation
    return V


# PDE price of a Bermudan swaption with the numComponents leading PCA directions
# (shortcut for swaption_subset with component_indices = 0, ..., numComponents-1)
def bermuda_swaption(model, K, y_grid, dy, numComponents, N, theta,
                     exercise, maturity, numExerciseDates, phi=1, return_continuation=False):
    return bermuda_swaption_subset(model, K, y_grid, dy, list(range(numComponents)), N, theta,
                                   exercise, maturity, numExerciseDates, phi, return_continuation)

if __name__ == "__main__":
    # test data
    tau, theta, m, n_points, numComponents = 1.0, 0.5, 4, 2**9, 2

    # European swaption (T0, TN) = (1, 6), 200 time steps graded with q = 3
    exercise, maturity = 1.0, 6.0
    model = LiborMarketModel.from_setup(exercise, maturity, tau)
    K = swap_rate(exercise, maturity, tau)
    n_steps, q = 200, 3
    dt = np.diff(exercise * (np.arange(n_steps + 1) / n_steps) ** q)
    y_grid, dy = build_y_grid_subset(m, model.eigenvalues, range(numComponents),
                                     [n_points] * numComponents, exercise)
    V = swaption(model, K, y_grid, dy, numComponents, dt, n_steps, theta, phi=1)
    print(f"European (T0, TN) = ({exercise:g}, {maturity:g}), {numComponents} components: "
          f"{model.to_bps(get_solution_at_y0(V, y_grid)):.4f} bps")


    # Bermudan swaption (T0, Tn, TN) = (1, 3, 5), 50 time steps per period
    exercise, last_exercise, maturity = 1.0, 3.0, 5.0
    model = LiborMarketModel.from_setup(exercise, maturity, tau)
    K = swap_rate(exercise, maturity, tau)
    numExerciseDates = int(round((last_exercise - exercise) / tau)) + 1
    y_grid, dy = build_y_grid_subset(m, model.eigenvalues, range(numComponents),
                                     [n_points] * numComponents, last_exercise)
    V = bermuda_swaption(model, K, y_grid, dy, numComponents, 50, theta,
                         exercise, maturity, numExerciseDates, phi=1)
    print(f"Bermudan (T0, Tn, TN) = ({exercise:g}, {last_exercise:g}, {maturity:g}), {numComponents} components: "
          f"{model.to_bps(get_solution_at_y0(V, y_grid)):.4f} bps")
