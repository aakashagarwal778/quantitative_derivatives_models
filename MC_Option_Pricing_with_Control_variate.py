import math
import numpy as np
from scipy.stats import norm

# Parameters
S0 = 100
r = 0.05 #risk-free rate
sigma = 0.3
T = 1
K = 110
M = 100000

# Compute European call option price using Black-Scholes formula
def black_scholes_call_price(S_t, r, sigma, T, K, t): 
    d1 = (math.log(S_t / K) + (r + (sigma ** 2) / 2) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    call_price_BS = S_t * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)
    return call_price_BS

#function for computing the price of a self-quanto call in the BS-model using Monte-Carlo with control variate
def BS_EuOption_MC_CV(S0, r, sigma, T, K, M):

    X = np.random.normal(0,1, M) #generate M samples from a standard normal distribution
    ST = S0 * np.exp((r-0.5*math.pow(sigma,2))*T + sigma*math.sqrt(T)*X) #stock price at maturity
    f_x1 = ST*np.maximum(ST-K,0) #self-quanto call payoff (f(x))
    y = np.maximum(ST-K, 0) #European call payoff (y)
    E_y = black_scholes_call_price(S0, r, sigma, T, K, 0)  #closed form black-scholes price of the call option (E(y): expected value of the call payoff)

    # Compute beta for control variates via least squares
    covariance = np.cov(f_x1, y) #covariance matrix of the self-quanto call payoff and the european call payoff
    variance = np.var(y) #variance of the european call payoff
    beta = covariance[0][1]/variance #variance reduction factor of control variates for monte carlo simulation      

    #Compute the price of the self-quanto call option using control variates (replacing y with beta*y)
    adj_payoff = ST * np.maximum((ST-K),0)- beta*y #adjusted self-quanto call payoff using scaled european call payoff to reduce variance
    p = np.mean(adj_payoff) + beta*E_y #Expected payoff of the self-quanto call option using control variates 
    CV_call_price = math.exp(-r*T) * p #discounted expected payoff of the option
    return CV_call_price, beta

# European option price function using Monte Carlo simulation from C-Exercise 07
def Eu_Option_MC(S0, r, sigma, T, K, M, f):
    X = np.random.normal(0, 1, M) # Generate M samples from a standard normal distribution
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * X) # Compute stock price at maturity
    payoff_fun = f(ST, K) # Placeholder for payoff function
    MC_call_price = np.exp(-r * T) * np.mean(payoff_fun) # Compute discounted expected payoff of the option    
    
    #calculate the confidence interval bounds and the radius of the confidence interval
    z = 1.96  # 95% confidence level
    std_error = np.std(payoff_fun) / np.sqrt(M) #Standard error
    E = z * std_error #Radius of the confidence interval (Margin of error i.e., (C +/- E))
    c1 = MC_call_price - E # Lower bound
    c2 = MC_call_price + E # Upper bound
    
    return MC_call_price, c1, c2, E

# define payoff function of self-quanto call option for Eu_Option_MC function
def f(ST, K): 
    self_quanto_payoff = np.maximum(ST - K, 0)*ST
    return self_quanto_payoff

# Comparing European call option prices with and without control variates
CV_call_price, beta = BS_EuOption_MC_CV(S0, r, sigma, T, K, M)
print('Call option price with control variates method:', str(CV_call_price))
print("Beta (variance reduction factor):", str(beta))

MC_call_price, c1, c2, E = Eu_Option_MC(S0, r, sigma, T, K, M, f)
print('Call option price using monte carlo simulation :', str(MC_call_price))
print('Lower Bound:', str(c1))
print('Upper Bound:', str(c2))
print('Margin of Error:', str(E))





'''
logic behind the code:

The control variate approach is applicable if we know some Y ≈ f(X) such that E(Y) can be computed explicitly.
E(F(X)) = E(f(X)-Y)+E(Y)

The idea is that we want to get the expected value of our payoff f(X) which is in our case the self-quanto call 
and we already know Y which is the payoff of a normal call and which is similar (in terms of correlation) to f(X). 
So we are doing a MC simulation to calculate a beta. Then we specify the payoff of our self-quanto call 
which if f(X) and the payoff of our European call Y and the expected value E(Y) (which is our closed form BS formula here). 
Then we calculate beta and replace Y with beta*Y and then calculate the price of the self-quanto call.
E(F(X)) = E(f(X)-beta*Y) + beta*E(Y)
'''