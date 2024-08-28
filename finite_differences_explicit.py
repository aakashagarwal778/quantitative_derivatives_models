import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
def BS_EuCall_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K):
    ### setting the parameters needed for the recursion
    q = 2 * r / sigma ** 2
    delta_x = (b - a) / m
    delta_t = sigma ** 2 * T / (2 * nu_max)
    fidi_lambda = delta_t / delta_x ** 2
    lambda_tilde = (1 - 2 * fidi_lambda)

    ### range of underlying transformed stock prices
    x = np.arange(a, b + delta_x, delta_x)

    ### allocating memory
    w = np.zeros((m + 1, nu_max + 1))

    ### initial values equivalent to transformed payoff at maturity
    w[:, 0] = np.maximum(0, np.exp(x / 2 * (q + 1)) - np.exp(x / 2 * (q - 1)))

    ### loop over columns of matrix/time
    for i in range(1, nu_max + 1):

        ### loop over rows/underlying (transformed) stock price
        ### note that we do not change the top and bottom row which are equal to zero all the time (simplified boundary conditions)
        for j in range(1, m):
            w[j, i] = fidi_lambda * w[j - 1, i - 1] + lambda_tilde * w[j, i - 1] + fidi_lambda * w[j + 1, i - 1]

    ### retransfoming underlying stock prices
    S = K * np.exp(x[1:-1])

    ### transforming the solution of (5.1) into option prices
    V = K * w[1:-1, nu_max] * np.exp(-x[1:-1] / 2 * (q - 1) - sigma ** 2 / 2 * T * ((q - 1) ** 2 / 4 + q))
    return [S, V]


### test parameter
r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100 # this is the number of space steps
nu_max = 2000 # this is the number of time steps
T = 1
K = 100

[S, V] = BS_EuCall_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K)


### BS-Formula
def EuCall_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    Call = S_t * scipy.stats.norm.cdf(d_1) - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return Call


V_BS = np.zeros(len(S))
### applying the BS-Formula to each underlying stock price
### note that we do set the stock prices only indirectly through the parameters a,b and m
for i in range(0, len(S)):
    V_BS[i] = EuCall_BlackScholes(0, S[i], r, sigma, T, K)

plt.plot(S, V, label='Price with finite difference scheme')
plt.plot(S, V_BS, label='Price with BS-Formula')
plt.xlabel('Stock Price')
plt.ylabel('Option Price')
plt.legend()
plt.show()
