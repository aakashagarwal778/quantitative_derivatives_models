## Aakash Agarwal, Anderson Sasu | Group: 29 | Exercise: 07
import math
import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt


# Define function to calculate European option price using Monte Carlo simulation
def Eu_Option_BS_MC(S0, r, sigma, T, K, M, f):
    X = np.random.normal(0, 1, M) # Generate M samples from a standard normal distribution
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * X) # Compute stock price at maturity
    payoff_fun = f(ST, K) # Placeholder for payoff function
    
    # Compute discounted expected payoff
    V0 = np.exp(-r * T) * np.mean(payoff_fun)
    
    # Compute standard error of the Monte Carlo estimate
    se = np.std(payoff_fun) / np.sqrt(M)
    
    # Compute asymptotic 95%-confidence interval
    z = 1.96  # 95% confidence level
    c1 = V0 - z * se # Lower bound
    c2 = V0 + z * se # Upper bound
    
    return V0, c1, c2 # Return option price and confidence interval bounds

# Define payoff function for call option
def call_payoff(ST, K): 
    payoff = np.maximum(ST - K, 0)
    return payoff

# Define parameters
S0 = 110 # Initial stock price
r = 0.04 # Risk-free rate
sigma = 0.2 # Volatility
T = 1 # Time to maturity
K = 100 # Strike price
M = 10000 # Number of samples

# Calculate option price using Monte Carlo simulation
option_price_mc = Eu_Option_BS_MC(S0, r, sigma, T, K, M, call_payoff) 

# Compute European call option price using Black-Scholes formula
def black_scholes_call_price(S0, r, sigma, T, K, t): 
    d1 = (math.log(S0 / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    call_price = S0 * scipy.stats.norm.cdf(d1) - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d2)
    return call_price

exact_option_price = black_scholes_call_price(S0, r, sigma, T, K, t=0)

# Print results
print("European Call Option Price using Monte Carlo:", option_price_mc[0])
print("95% confidence interval using Monte Carlo:", option_price_mc[1],',', option_price_mc[2])
print("European Call Option Price using Black-Scholes formula:", exact_option_price)


# Define function to visualize stock price scenarios
def visualize_stock_price_scenarios(S0, r, sigma, T, M, num_scenarios):
    X = np.random.normal(0, 1, (num_scenarios, M)) # Generate M samples from a standard normal distribution for each scenario
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * X) # Compute stock price at maturity for each scenario

    plt.figure(figsize=(10, 6))
    for i in range(num_scenarios):
        alpha_value = 0.8 - 0.005 * i  # Adjust alpha for each scenario
        plt.plot(np.arange(1, M + 1), ST[i], label=f'Scenario {i+1}', alpha=alpha_value)

    plt.title('Stock Price Scenarios')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# Visualize 20 future random stock price scenarios
num_scenarios = 20
visualize_stock_price_scenarios(S0, r, sigma, T, M, num_scenarios)