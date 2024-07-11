from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    ## Compute characteristic function of log-stock price in the Heston model, cf. equation (4.8) with t = 0
    d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
    phi = np.cosh(0.5 * d * T)
    psi = np.sinh(0.5 * d * T) / d
    first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi))**(2 * kappa / sig_tild ** 2)
    second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
    return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

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

# parameters
S0 = 100 
r = 0.05
gam0 = 0.3 ** 2
kappa = 0.3 ** 2
lamb = 2.5
sig_tild = 0.2
T = 1
K = np.arange(80, 181)
R = 1.1
N = 2 ** 15

V = Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N)

plt.figure(figsize=(8, 6))
plt.plot(K, V)
plt.grid(alpha=0.4)
plt.ylabel('Option Price')
plt.xlabel('Strike Price')
plt.title('Call option prices in the Heston model')
plt.show()
