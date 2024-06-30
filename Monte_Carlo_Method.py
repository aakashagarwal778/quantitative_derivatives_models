import numpy as np
import matplotlib.pyplot as plt

# Define parameters
S0 = 500  # Initial stock price
r = 0.04  # Risk-free rate
sigma = 0.2  # Volatility
T = 1  # Time to maturity
K = 100  # Strike price
N = 252  # Number of time steps

# Define payoff functions for call and put options
def call_payoff(ST, K): 
    return np.maximum(ST - K, 0)

def put_payoff(ST, K):
    return np.maximum(K - ST, 0)

# Function to simulate stock price paths using Geometric Brownian Motion (GBM)
def simulate_gbm_paths(S0, r, sigma, T, N, num_paths): 
    if not all(isinstance(arg, (int, float)) for arg in [S0, r, sigma, T]):
        raise ValueError("Invalid parameter type. Parameters S0, r, sigma, and T must be numeric.")
        
    dt = T / N
    paths = np.zeros((num_paths, N + 1))
    paths[:, 0] = S0
    for t in range(1, N + 1):
        Z = np.random.normal(0, 1, num_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

# Function to calculate option prices using Monte Carlo simulation along each path
def MC_option_prices(paths, r, T, K, option_type='call'):
    if not all(isinstance(arg, (int, float)) for arg in [r, T, K]):
        raise ValueError("Invalid parameter type. Parameters r, T, and K must be numeric.")
    
    num_paths, N = paths.shape
    option_prices = np.zeros((num_paths, N))
    dt = T / (N - 1)
    for i in range(num_paths):
        for t in range(N):
            ST = paths[i, t]
            if option_type == 'call':
                payoff = call_payoff(ST, K)
            elif option_type == 'put':
                payoff = put_payoff(ST, K)
            else:
                raise ValueError("Unknown option type. Use 'call' or 'put'.")
            option_prices[i, t] = np.exp(-r * (t * dt)) * payoff
    return option_prices

# Function to visualize option price simulations
def plot_simulation(paths, option_prices):
    plt.figure(figsize=(10, 6))
    for i in range(len(paths)):
        plt.plot(option_prices[i], alpha=0.7) 
    plt.title('Option Price Simulations')
    plt.xlabel('Time Steps')
    plt.ylabel('Option Price')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main simulation process
def main():
    try:
        # Simulate stock price paths using GBM
        num_paths = 500
        paths = simulate_gbm_paths(S0, r, sigma, T, N, num_paths)

        # Calculate option prices for both call and put options using Monte Carlo simulation
        option_prices_call = MC_option_prices(paths, r, T, K, option_type='call')

        # Visualize option price simulations for calls
        plot_simulation(paths, option_prices_call)

    except ValueError as ve:
        print(f"Error: {ve}")

if __name__ == "__main__":
    main()
