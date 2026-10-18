import numpy as np

# implementation of the Thomas algorithm, implementation from: 
# Numerical Recipes: The Art of Scientific Computing; Press, William H. and Teukolsky, Saul A. and Vetterling, William T. and Flannery, Brian P.
# 
# Input:
# a: lower diagonal
# b: main diagonal
# c: upper diagonal
# u: right hand side of the linear system
# A=b c 0 ...
#  a b c ...
#  ... ... ...
#       0 a b
#
# Return: 
# solution of the linear system
def tridagnew(a, b, c, u):
    N = len(a) - 1
    cp = np.empty(N+1)
    up = np.empty(N+1)

    cp[0] = c[0] / b[0]
    up[0] = u[0] / b[0]

    for i in range(1, N+1):
        gam = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / gam
        up[i] = (u[i] - a[i] * up[i-1]) / gam

    u[N] = up[N]
    for i in range(N-1, -1, -1):
        u[i] = up[i] - cp[i] * u[i+1]
    return u


# Thomas algorithm for M tridiagonal systems at once (vectorised version of tridagnew)
# Input: a, b, c, d of shape (N, M), column m = diagonals / right hand side of system m
# Return: solution x of shape (N, M), x[:, m] solves system m
def tridag_batch(a, b, c, d):
    N, M = d.shape

    cp = np.zeros((N, M))
    dp = np.zeros((N, M))

    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, N):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom

    # Backward substitution
    x = np.zeros((N, M))
    x[-1] = dp[-1]

    for i in range(N-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x