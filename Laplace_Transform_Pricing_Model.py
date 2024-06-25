import math
import cmath
import scipy.integrate as integrate
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


# computes the price of a european call option in the Black-Scholes model via the Laplace transform approach
def BS_EuCall_Laplace(S0, r, sigma, T, K, R):
    # Laplace transform of the function f(x) = (e^x - K)^+
    def f_tilde(z):
        return cmath.exp((1 - z) * math.log(K)) / (z * (z - 1))

    # characteristic function of log(S(T)) in the Black-Scholes model
    def chi(u):
        return cmath.exp(
            complex(0, 1) * u * (math.log(S0) + r * T) - (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * math.pow(
                sigma, 2) / 2 * T)

    # Integrand for the Laplace transform method
    def integrand(u):
        return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    # option price
    V0 = integrate.quad(integrand, 0, 50)
    return V0


# computes the price of a call in the Black-Scholes model
def EuCall_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(d_1)
    C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return C, phi


# test parameters
S0 = range(50, 151, 1)
r = 0.03
sigma = 0.2
T = 1
K = 110
R = 1.1

print(
    'Price of European call by use of Laplace transform approach: ' + str(BS_EuCall_Laplace(S0[50], r, sigma, T, K, R)[0]))
print('Price of European call by use of the BS-formula: ' + str(EuCall_BlackScholes(0, S0[50], r, sigma, T, K)[0]))

V0 = np.empty(101, dtype=float)
for i in range(0, len(S0)):
    V0[i] = BS_EuCall_Laplace(S0[i], r, sigma, T, K, R)[0]

plt.plot(S0, V0, 'red', label='Price of the european call')
plt.legend()
plt.xlabel('Initial stock price S0')
plt.ylabel('Initial option price V0')
plt.show()