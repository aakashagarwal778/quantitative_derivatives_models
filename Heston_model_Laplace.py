import math
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

# parameters
S0 = range(50, 151, 1)
r = 0.05
gam0 = math.pow(0.3, 2) 
kappa = math.pow(0.3, 2)
lamb = 2.5
sig_tild = 0.2
T = 1
K = 100
p = 1
R = 1.5

def Heston_PCall_Laplace(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, p):

    # Laplace transform of the power call option (S(T)^p − K)+ 
    def f_tilde(z):
        return (p*(K**(1-z/p))) / (z * (z - p))

    # Characteristic function 
    def chi(u):
        d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
        phi = np.cosh(0.5 * d * T)
        psi = np.sinh(0.5 * d * T) / d
        first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi))**(2 * kappa / sig_tild ** 2)
        second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
        return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

    # integrand of the integral 
    def integrand(u):
        return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    # integration to obtain the option price
    V0 = integrate.quad(integrand, 0, 50)
    return V0

# Compute the price of the power call for different initial stock prices
V0 = [Heston_PCall_Laplace(S0_i, r, gam0, kappa, lamb, sig_tild, T, K, R, p) for S0_i in S0]
heston = [V0_i[0] for V0_i in V0] # Extract the price of the power call
print(heston)

# Plot
plt.plot(S0, heston)
plt.title('Power call option prices in the Heston model')
plt.xlabel('Initial stock price [S0]')
plt.ylabel('Initial option price [V0]')
plt.grid()
plt.show()