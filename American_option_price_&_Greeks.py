import math
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.05  # risk-free rate
sigma = 0.3  # volatility
T = 1  # time to maturity in years
S0 = range(60, 141, 1)  # range of initial stock prices
K = 110  # strike price
N = 100  # number of time steps
eps = 0.0001  # small change for finite difference

# Function to compute the binomial tree for an American option
def binomial_tree_american(S0, K, r, sigma, T, N, option_type='call'):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            asset_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)
    
    # Initialize option values at maturity
    option_values = np.zeros((N + 1, N + 1))
    if option_type == 'call':
        option_values[:, N] = np.maximum(0, asset_prices[:, N] - K)
    elif option_type == 'put':
        option_values[:, N] = np.maximum(0, K - asset_prices[:, N])
    
    # Step back through the tree to calculate option price
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            exercise = 0
            if option_type == 'call':
                exercise = max(0, asset_prices[j, i] - K)
            elif option_type == 'put':
                exercise = max(0, K - asset_prices[j, i])
            hold = (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1]) * math.exp(-r * dt)
            option_values[j, i] = max(exercise, hold)
    
    return option_values[0, 0]

# Allocate vector for American option prices
american_option_prices = []

# Compute American option price for each S0 value
for S in S0:
    price = binomial_tree_american(S, K, r, sigma, T, N, option_type='call')
    american_option_prices.append(price)
'''
# Plot the American option prices
plt.plot(S0, american_option_prices, label='American Call Option Price')
plt.xlabel('S0')
plt.ylabel('Option Price')
plt.title('American Call Option Price vs Initial Stock Price (S0)')
plt.legend()
plt.grid(True)
plt.show()
'''
# Function to compute the Greeks using finite differences
def binomial_tree_greeks(S0, K, r, sigma, T, N, eps, option_type='call'):
    price = binomial_tree_american(S0, K, r, sigma, T, N, option_type)
    price_up_S0 = binomial_tree_american(S0 * (1 + eps), K, r, sigma, T, N, option_type)
    price_down_S0 = binomial_tree_american(S0 * (1 - eps), K, r, sigma, T, N, option_type)
    price_up_sigma = binomial_tree_american(S0, K, r, sigma * (1 + eps), T, N, option_type)
    price_up_r = binomial_tree_american(S0, K, r + eps, sigma, T, N, option_type)
    price_down_T = binomial_tree_american(S0, K, r, sigma, T - eps, N, option_type)
    
    Delta = (price_up_S0 - price) / (eps * S0)
    Gamma = (price_up_S0 - 2 * price + price_down_S0) / ((eps * S0) ** 2)
    Vega = (price_up_sigma - price) / (eps * sigma)
    Rho = (price_up_r - price) / eps
    Theta = (price_down_T - price) / (-eps)
    
    return Delta, Gamma, Vega, Rho, Theta

# Allocate vectors for Greeks
Delta = []
Gamma = []
Vega = []
Rho = []
Theta = []

# Compute Greeks for each S0 value
for S in S0:
    delta, gamma, vega, rho, theta = binomial_tree_greeks(S, K, r, sigma, T, N, eps, option_type='call')
    Delta.append(delta)
    Gamma.append(gamma)
    Vega.append(vega)
    Rho.append(rho)
    Theta.append(theta)

#print the Greeks
print('Delta:', Delta)
print('Gamma:', Gamma)
print('Vega:', Vega)
print('Rho:', Rho)
print('Theta:', Theta)

# Define the number of subplots
num_subplots = 5

# Create a figure and subplots
fig, axs = plt.subplots(num_subplots, figsize=(10, 15))

# Define data and labels for each subplot
plot_data = [Delta, Gamma, Vega, Rho, Theta]
plot_labels = ['Delta', 'Gamma', 'Vega', 'Rho', 'Theta']

# Iterate over subplots and plot data
for i, ax in enumerate(axs):
    ax.plot(S0, plot_data[i])
    ax.set_xlabel('S0')
    ax.set_ylabel(plot_labels[i])
    ax.set_title(plot_labels[i])

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=1)
plt.show()
