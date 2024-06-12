## Aakash Agarwal, Anderson Sasu | Group: 29 | Exercise: 06
import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

#Parameters
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
M = range(10, 501)  # Range of time steps
K = 120

#Part(a)
#Define the function to calculate option prices using the CRR model
def CRR_AmEu_Put(S_0, r, sigma, T, M, K, EU):
    delta_t = T / M
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + sigma ** 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    q = (math.exp(r * delta_t) - d) / (u - d)

    #Create matrix S to store stock prices
    S = np.zeros((M + 1, M + 1))
    for i in range(M + 1):
        for j in range(i + 1):
            S[j, i] = S_0 * (u ** j) * (d ** (i - j))  # Stock price at each node

    #Create option value matrix
    V = np.empty((M + 1, M + 1))

    #Calculate option value at maturity
    V[:, M] = np.maximum(K - S[:, M], 0)

    #Define European and American put option pricing functions
    def european_put(Z): #where Z is the time step  
        Eu_val = math.exp(-r * delta_t) * (q * V[1:Z + 1, Z] + (1 - q) * V[0:Z, Z]) # Option value at each node
        return Eu_val

    def american_put(Z):
        Am_val = np.maximum(K - S[0:Z, Z - 1], math.exp(-r * delta_t) * (q * V[1:Z + 1, Z] + (1 - q) * V[0:Z, Z])) #Intrinsic value vs. option value
        return Am_val

    #Choose between European and American option pricing based on 'EU' parameter
    if EU == 1:
        option_function = european_put 
    elif EU == 0:
        option_function = american_put 
    else:
        raise ValueError("EU parameter must be either 0 or 1.")  

    # Recursion to calculate option prices at each step
    for Z in range(M, 0, -1):
        if Z == 1:
            V[0, 0] = option_function(Z)
        else:
            V[0:Z, Z - 1] = option_function(Z)

    return V[0, 0]

#Part (b)
# Define the function to calculate put option prices using the Black-Scholes model
def BlackScholes_EuPut(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    C = K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(-d_2) - S_t * scipy.stats.norm.cdf(-d_1)
    return C

#Part (c)
# Calculate European put option prices using CRR model for different time steps
EuPut_Prices = [CRR_AmEu_Put(S_0, r, sigma, T, Z, K, EU=1) for Z in M]
print('European Put Option Prices:', EuPut_Prices)

# Calculate Black-Scholes price
BlackScholes_Price = BlackScholes_EuPut(0, S_0, r, sigma, T, K)
print('Black-Scholes Price:', BlackScholes_Price)

# Plotting the results
plt.plot(M, EuPut_Prices, color='red', label='Binomial Model Price')
plt.axhline(BlackScholes_Price, color='green', label='Black-Scholes Price', linewidth=2)
plt.xlabel('Number of Steps')
plt.ylabel('Price')
plt.title('Comparison of CRR Model and Black-Scholes Prices for European Put Option')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the price of the American put option at M =500 for the test parameters
AmPut_Price = CRR_AmEu_Put(S_0, r, sigma, T, 500, K, EU=0)
print('American put option value: ', AmPut_Price)