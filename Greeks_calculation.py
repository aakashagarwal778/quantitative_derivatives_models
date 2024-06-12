import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# test parameters
r = 0.05
sigma = 0.3
T = 1
S0 = range(60, 141, 1)
eps = 0.001

# function to compute the price of an european option in the Black-Scholes model using numerical integration
def BS_Price_Int(S0, r, sigma, T, g):
    # Pricing by integration using sqrt(T) * x instead of WQ(t) with W ~ N(0, dt) as in eq. (3.21)
    def integrand(x):
       V = 1 / math.sqrt(2 * math.pi) * g(S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * x)) * math.exp(-r * T) * math.exp(-1 / 2 * math.pow(x, 2)) # eq. (3.21)
       return V
    
    # integrate the integrand from -inf to inf
    integral = integrate.quad(integrand, -np.inf, np.inf)
    return integral[0]

# payoff function for a call option
def g(x):
    return max(x - 110, 0)
    
# function to compute the greeks of an european option in the Black-Scholes model using numerical differentiation
def BS_Greeks_num(r, sigma, S0, T, g, eps):
    # Evaluation price function for initial parameters and by epsilon augmented parameters
    initial_price = BS_Price_Int(S0, r, sigma, T, g)
    delta_up = BS_Price_Int((1 + eps) * S0, r, sigma, T, g)
    vega_up = BS_Price_Int(S0, r, (1 + eps) * sigma, T, g)
    delta_down = BS_Price_Int((1 - eps) * S0, r, sigma, T, g)

    # Compute the greeks
    Delta = (delta_up - initial_price) / (eps * S0)
    vega = (vega_up - initial_price) / (eps * sigma)
    gamma = (delta_up - 2 * initial_price + delta_down) / (math.pow(eps * S0, 2))
    return Delta, vega, gamma

# Allocate vectors for Greeks
Delta = []
vega = []
gamma = []

# Compute Greeks for each S0 value
for S in S0:
    result = BS_Greeks_num(r, sigma, S, T, g, eps)
    Delta.append(result[0])
    vega.append(result[1])
    gamma.append(result[2])

# Define the number of subplots
num_subplots = 3

# Create a figure and subplots
fig, axs = plt.subplots(num_subplots)

# Define data and labels for each subplot
plot_data = [Delta, vega, gamma]
plot_labels = ['Delta', 'Vega', 'Gamma']

# Iterate over subplots and plot data
for i, ax in enumerate(axs):
    ax.plot(S0, plot_data[i])
    ax.set_xlabel('S0')
    ax.set_ylabel(plot_labels[i])
    ax.set_title(plot_labels[i])

plt.tight_layout()
plt.show()