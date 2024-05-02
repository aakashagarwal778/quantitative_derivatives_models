import math
import numpy as np
import matplotlib.pyplot as plt

#Computing the time series of log returns for a given time series data
def log_returns(data): 
    l = np.diff(np.log(data)) 
    return l 

# Importing the CSV file
data = np.genfromtxt("time_series_dax_2024.csv", delimiter=";", usecols=4, skip_header=1)
data = np.flip(data) # Reverse the order of the data to start from the earliest date

# Computing the log return of the imported data
logReturns = log_returns(data)

# Plotting log Returns of time series data
plt.plot(logReturns, color='blue')  
plt.xlabel("Period") 
plt.ylabel("Log Returns")
plt.grid(True, linestyle='--', alpha=0.5) 
plt.show()


# Computing and display the empirical mean and standard deviation of log returns
emp_mean = np.mean(logReturns) # Empirical mean
emp_std_dev = np.sqrt(np.var(logReturns)) # Empirical standard deviation
annualized_mean = emp_mean * 250 # Annualized empirical mean
annualized_std_dev = emp_std_dev * np.sqrt(250) # Annualized empirical standard deviation
print('Annualized Empirical Mean: ', annualized_mean) 
print('Annualized Empirical Standard Deviation: ', annualized_std_dev)


# Simulating normally distributed log returns with empirical mean and standard deviation
sim_log_returns = np.random.normal(emp_mean, emp_std_dev, len(logReturns))
print (np.mean(sim_log_returns))

#Plotting both lgReturns and Sim_lgReturns
plt.plot(logReturns, color='blue')
plt.plot(sim_log_returns, color='yellow', alpha = 0.7)
plt.title("log Returns vs Simulated Log Returns")
plt.xlabel('Periods')
plt.ylabel("log Returns and Sim log Returns")
plt.legend(['logReturns', 'sim_log_returns'])
plt.grid(True, linestyle='--', alpha=0.5) 
plt.show()

##(d)
''' 
The graph shows that while the log returns have outliers, the simulated log returns cluster around the mean. 
Additionally, the computed log returns fluctuate significantly over time, indicating non-stationarity, 
whereas the simulated log returns appear more stable, suggesting stationarity.
'''
