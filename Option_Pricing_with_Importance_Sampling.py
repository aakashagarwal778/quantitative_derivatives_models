import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#parameter
S0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 220
N = 10000
alpha = 0.95

# Define function to calculate European option price using Monte Carlo simulation with importance sampling 
def BS_EuCall_MC_IS(S0, r, sigma, T, K, N, mu, alpha):
    # Define payoff function for call option
    def call_payoff(ST, K): 
        payoff = np.maximum(ST - K, 0)
        return payoff
    
    # generate N random variables with N(mu, 1)-distribution
    Y = np.random.normal(mu, 1, N) # in order to make less likely events more likely  
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Y) # Compute stock price at maturity
    Payoff_fun = call_payoff(ST, K) # Placeholder for payoff function

    # Compute option price with importance sampling
    ImpAdj = np.exp(-r*T- Y * mu + (mu ** 2) / 2) # Compute importance adjustment factor (Gaussian Density of r.v. Y)
    Adj_payoff = ImpAdj * Payoff_fun # Compute adjusted payoff
    Expected_payoff = np.mean(Adj_payoff) # compute empirical mean of the adjusted payoff
    IS_call_price = np.exp(-r * T) * Expected_payoff # Compute discounted expected payoff of the option

    # Calculate the confidence interval bounds and the radius of the confidence interval
    std_error = np.std(Adj_payoff)/np.sqrt(N) # Standard error 
    z = norm.ppf(1 - (1 - alpha)) # 95% confidence level (z-score)
    E = z * std_error #Radius of the confidence interval (Margin of error i.e., (C +/- E))
    CIl = IS_call_price - E # Lower bound
    CIr = IS_call_price + E # Upper bound
    return IS_call_price, CIl, CIr, E

# since density of Y i.e, N(mu, 1) is concentrated around mu, we choose mu = d 
d = (math.log(K / S0) - ((r - (1 / 2) * math.pow(sigma, 2)) * T)) / (sigma * math.sqrt(T)) 

# Compute European call option price using Black-Scholes formula
def black_scholes_call_price(S_t, r, sigma, T, K, t): 
    d1 = (math.log(S_t / K) + (r + (sigma ** 2) / 2) * (T - t)) / (sigma * math.sqrt(T - t)) 
    d2 = d1 - sigma * math.sqrt(T - t)
    BS_call_price = S_t * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)
    return BS_call_price

#comparison of call option price with and without importance sampling

IS_call_price, CIl, CIr, E = BS_EuCall_MC_IS(S0, r, sigma, T, K, N, d, alpha) #option price with importance sampling for mu = d
print("Option Price with Importance Sampling: ", str(IS_call_price))
print("Confidence Interval Lower Bound: ",str(CIl))
print("Confidence Interval Upper Bound: ", str(CIr))
print("Margin of Error: ", str(E))

BS_call_price = black_scholes_call_price(S0, r, sigma, T, K, t=0) #option price with Black-Scholes formula
print("Option Price with Black-Scholes Formula: ", str(BS_call_price))

#experiment with different mu values to see how the option price changes
mu = np.linspace(0, d, 500) # mu values to experiment with   
IS_price_list = [] 
mu_val = [] 

for x in mu: # loop through mu values
    IS_call_price, c1l, c2r, E = BS_EuCall_MC_IS(S0, r, sigma, T, K, N, x, alpha)
    IS_price_list.append(IS_call_price) # add option price to list
    mu_val.append(x) # add respective mu value to list

# Plotting the option price with importance sampling for different mu values and the Black-Scholes price with confidence interval
plt.plot(mu_val, IS_price_list, label='Monte Carlo with Importance Sampling')
plt.axhline(y=BS_call_price, color='r', linestyle='--', label='Black-Scholes Price')
plt.xlabel('Mu values')
plt.ylabel('Option Prices')
plt.title('Comparison of Option Prices with Importance Sampling and Black-Scholes Price')
plt.legend()
plt.grid(True)
plt.show()





'''
logic behind the code:

We want to model the BS stock price formula with a Y ~ N(mu,1) instead of with a X ~ N(0,1) in order to make less likely events happen more often. 
Therefore, we are using the importance sampling in order to minimize the variance so that we won't need as many samples in order to converge to the true parameter. 
'''