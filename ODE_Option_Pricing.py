import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def BS_AmPerpPut_ODE(S_max, N, r, sigma, K):
    g = lambda S: np.maximum(K - S, 0)

    S_grid = np.linspace(0, S_max, N + 1)
    v_grid = np.zeros_like(S_grid)

    # Define a 2-dimensional system of 1st-order ODEs corresponding to the given 2nd-order ODE
    fun = lambda x, v: np.array([v[1], 2 * r / (sigma ** 2 * x ** 2) * (v[0] - x * v[1])])
    x_star = 2 * K * r / (2 * r + sigma ** 2)

    # For x <= x_star, the option value v(x) is given by the payoff
    v_grid[S_grid <= x_star] = g(S_grid[S_grid <= x_star])

    # For x > x_star, the option value v(x) is given by the solution of the ODE
    result = solve_ivp(fun=fun, t_span=(x_star, S_max), y0=[g(x_star), -1], t_eval=S_grid[S_grid > x_star])
    v_grid[S_grid > x_star] = result.y[0]

    return S_grid, v_grid


S_max = 200
N = 200
r = 0.05
sigma = np.sqrt(0.4)
K = 100

S_grid, v_grid = BS_AmPerpPut_ODE(S_max, N, r, sigma, K)

plt.figure(figsize=(8, 6))
plt.plot(S_grid, v_grid)
plt.axvline(2 * K * r / (2 * r + sigma ** 2), linestyle='--', color='red', alpha=0.5)
plt.grid(alpha=0.4)
plt.legend(['Perpetual American Put', 'Optimal Exercise Barrier'])
plt.xlabel('S')
plt.ylabel('V')
plt.show()
