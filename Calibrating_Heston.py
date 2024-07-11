import numpy as np
from Heston_FFT import Heston_EuCall
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = np.genfromtxt('./option_prices_sp500.csv', delimiter=',', skip_header=1)
it = 1


def callback(params):
    """ Simple callback function to be submitted to the scipy.optimize.minimize routine """
    global it
    if it == 1:
        print('Starting optimization...')
    print('It: {:4d},  '.format(it) + ''.join([name + ' = {:.4e}, '.format(param) for name, param in zip(['gamma_0', 'kappa', 'lambda', 'sigma_tilde'], params)]) + 'MSE = {:.4f}'.format(min_func(params)))
    it += 1


def min_func(params):
    """ Objective function to be minimized """
    S0 = data[0, 2]  # S&P500 level from the data set
    r = data[0, 3]  # Interest rate from the data set
    gam0, kappa, lamb, sig_tild = params
    strikes = data[:, 0]  # Strikes from the data set
    options_data = data[:, 1]  # Option prices from the data set
    options_model = Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T=1, K=strikes, R=1.5, N=2 ** 15)
    return np.square(options_model - options_data).mean()


# As initial guess, we use the parameter values from C-Exercise 26
initial_params = np.array([0.3 ** 2, 0.3 ** 2, 2.5, 0.2])

# We specify bounds for the parameters to avoid negativity or blow-up
bounds = [[1e-8, 2], [1e-8, 2], [1e-8, 10], [1e-8, 1]]

res = minimize(min_func, initial_params, bounds=bounds, callback=callback, method='L-BFGS-B')
gam0, kappa, lamb, sig_tild = res.x
print('Final estimate: gamma_0 = {:.4e}, kappa = {:.4e}, lambda = {:.4e}, sigma_tilde = {:.4e}'.format(gam0, kappa, lamb, sig_tild))

# Compute the model prices using the fitted parameters
fitted_model_prices = Heston_EuCall(S0=data[0, 2], r=data[0, 3], gam0=gam0, kappa=kappa, lamb=lamb, sig_tild=sig_tild, T=1, K=data[:, 0], R=1.5, N=2 ** 15)

plt.figure(figsize=(8, 6))
plt.plot(data[:, 0], data[:, 1])
plt.plot(data[:, 0], fitted_model_prices)
plt.xlabel('Strike $K$')
plt.ylabel('Option Prices $V(0)$')
plt.legend(['Options Data', 'Fitted Prices'])
plt.grid(alpha=0.4)
plt.show()