import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Computes the price of a call in the Black-Scholes model
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d_1 = (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d_2 = d_1 - sigma * np.sqrt(T - t)
    phi = ss.norm.cdf(d_1)
    C = S_t * phi - K * np.exp(-r * (T - t)) * ss.norm.cdf(d_2)
    return C

# Computes the implied volatility
def ImpVolBS(V0, S0, r, T, K):
    implied_vols = np.zeros(K.shape)  # Initialize an array to store the implied volatilities
    for i, k in enumerate(K):  # Iterate over each strike price
        def function(sigma):
            squared_error = (V0 - BlackScholes_EuCall(0, S0, r, sigma, T, k)) ** 2
            return squared_error
        initial_guess = 0.3
        res = minimize(function, initial_guess, bounds=((0, None),), method='Powell')
        implied_vols[i] = res.x  # Store the implied volatility for the current strike price
    return implied_vols


# Heston characteristic function
def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
    phi = np.cosh(0.5 * d * T)
    psi = np.sinh(0.5 * d * T) / d
    first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi))**(2 * kappa / sig_tild ** 2)
    second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
    return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

# Computes the price of a call in the Heston model using fast Fourier transform 
def Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N):
    K = np.atleast_1d(K)
    f_tilde_0 = lambda u: 1 / (u * (u - 1))
    chi_0 = lambda u: heston_char(u, S0=S0, r=r, gam0=gam0, kappa=kappa, lamb=lamb, sig_tild=sig_tild, T=T)
    g = lambda u: f_tilde_0(R + 1j * u) * chi_0(u - 1j * R)

    kappa_1 = np.log(K[0])
    M = np.minimum(2 * np.pi * (N - 1) / (np.log(K[-1]) - kappa_1), 500)
    Delta = M / N
    n = np.arange(1, N + 1)
    kappa_m = np.linspace(kappa_1, kappa_1 + 2 * np.pi * (N - 1) / M, N)

    x = g((n - 0.5) * Delta) * Delta * np.exp(-1j * (n - 1) * Delta * kappa_1)
    x_hat = np.fft.fft(x)

    V_kappa_m = np.exp(-r * T + (1 - R) * kappa_m) / np.pi * np.real(x_hat * np.exp(-0.5 * Delta * kappa_m * 1j))
    return interp1d(kappa_m, V_kappa_m)(np.log(K))

# Computes the implied volatility
def ImpVol(V0, S0, r, T, K):
    def function(sigma):
        squared_error = (V0 - BlackScholes_EuCall(0, S0, r, sigma, T, K)) ** 2
        return squared_error
    initial_guess = 0.3
    res = minimize(function, initial_guess, bounds=((0, None),), method='Powell')
    return res.x[0]

# Function to compute the implied volatility for the Heston model
def ImpVol_Heston(S0, r, gam0, kappa, lamb, sig_tild, T, K, R):
    V0 = Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N=2**15)
    implVol = np.empty(len(K), dtype=float)
    for i in range(len(K)):
        implVol[i] = ImpVol(V0[i], S0, r, T, K[i])
    return implVol, V0

# Parameters 
S0 = 100
r = 0.05
gam0 = 0.3 ** 2
kappa = 0.3 ** 2
lamb = 2.5
sig_tild = 0.2
T = 1
K = np.linspace(80, 180, 101)
R = 1.5
V0 = 10

# Compute implied volatility
sigma = ImpVolBS(V0, 100, r, T, K)
print(f'Implied volatility (Black-Scholes): {sigma}')

# Compute implied volatility and option prices
implVolheston, V0 = ImpVol_Heston(S0, r, gam0, kappa, lamb, sig_tild, T, K, R)

# Plot both on the same graph
plt.figure(figsize=(8, 6))

# Plot implied volatility
plt.plot(K, implVolheston, label='Implied Volatility')
plt.xlabel('Strike Price')
plt.ylabel('Value')
plt.grid(alpha=0.4)
plt.legend()
plt.title('Implied Volatility in the Heston Model')
plt.show()
