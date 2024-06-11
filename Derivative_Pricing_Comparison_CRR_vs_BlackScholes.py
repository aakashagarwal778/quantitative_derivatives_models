import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#test
#Computes the stock price matrix in the CRR model 
def CRR_stock(S_0, r, sigma, T, M):

    delta_t = T / M # Time step
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + sigma**2) * delta_t)) # Beta value for the CRR model 
    u = beta + math.sqrt(math.pow(beta,2) - 1) # Up factor
    d = 1/u
    #d = beta - math.sqrt(math.pow(beta,2) - 1) # Down factor

    #Generating an empty matrix for the stock prices
    S = np.empty((M+1, M+1)) 

    #Computing and inputting stock prices into the matrix 'S'
    for i in range(M+1): # i represents the time step
        for j in range(i+1): # j represents the number of up movements
            S[j, i] = S_0 * (u ** j) * (d ** (i - j)) # Stock price at time i and j up movements

    return S


#Computes the price of a European call option using the CRR model
def CRR_EuCall(S_0, r, sigma, T, M, K): 

    delta_t = T / M
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + sigma**2) * delta_t))
    u = beta + math.sqrt(math.pow(beta,2) - 1)
    d = beta - math.sqrt(math.pow(beta,2) - 1)
    q = (math.exp(r * delta_t) - d) / (u - d) # Risk Neutral Probability 

    S = CRR_stock(S_0, r, sigma, T, M) # Stock price matrix

    #Computing option prices at maturity
    V = np.maximum(0, S - K) 

    #Backward induction to compute option prices at t=0
    for i in range(M-1, -1, -1): 
        for j in range(i+1): 
            V[j, i] = math.exp(-r * delta_t) * (q * V[j, i+1] + (1 - q) * V[j+1, i+1]) 

    return V[0, 0]


#Computes the price of a European call option using the Black-Scholes model
def BlackScholes_EuCall(t, S_t, r, sigma, T, K_values):

    d1 = (math.log(S_t / K_values) + (r + (sigma ** 2) / 2) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    V_0 = S_t * norm.cdf(d1) - K_values * math.exp(-r * (T - t)) * norm.cdf(d2)
    return V_0

#Parameters
S_0 = 100
r = 0.03
sigma = 0.3
T = 1
M = 100
K_values = range(70, 201)


Stock_price = CRR_stock(S_0, r, sigma, T, M)
print('Stock Prices:', Stock_price)

# Computing option prices using CRR model
CRR_prices = [CRR_EuCall(S_0, r, sigma, T, M, K) for K in K_values]

# Computing option prices using Black-Scholes model
BS_prices = [BlackScholes_EuCall(0, S_0, r, sigma, T, K) for K in K_values]

print('CRR Prices:', CRR_prices)
print('BS Prices:', BS_prices)

#Convert the lists of prices to NumPy arrays
CRR_prices_np = np.array(CRR_prices)
BS_prices_np = np.array(BS_prices)

#Calculate errors
errors_np = np.abs(CRR_prices_np - BS_prices_np)
print('Errors:', errors_np)

# plot the absolute error of the approximation against the real Black-Scholes price
plt.figure()
plt.clf()
plt.plot(K_values, errors_np, 'blue', label='Absolute error for original conditions')
plt.xlabel('Strike price')
plt.ylabel('Deviation from real BS-price')
plt.legend()
plt.show()