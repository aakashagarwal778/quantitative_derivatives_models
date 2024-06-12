import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
a = 0
b = 1 
mu = 0.5
sigma = 1
N = 10000

def density(x, a, b, mu, sigma): #density of the truncated normal distribution
    if a <= x and x <= b:
        normal_pdf = norm.pdf(x, mu, sigma) #pdf of the normal distribution at x 
        normal_cdf = norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma) #difference of the cdf at the "bounds" of the interval
        TruncPdf = normal_pdf / normal_cdf #pdf of the truncated normal distribution at x
        return TruncPdf
    else:
        return 0

# Acceptance-Rejection method for generating samples from a truncated normal distribution
def Sample_TruncNormal_AR(a, b, mu, sigma, N): # "a and b" are the bounds of the interval i.e., [a, b]
    f= density(1, a, b, mu, sigma) # Compute the pdf of the truncated normal distribution at x = 1
    g = 1/(b - a) # Define the pdf of the uniform distribution on the interval [a, b]
    c = f / g # Compute the constant C for the acceptance-rejection method

    # Initialize an empty list to store the generated samples
    samples_list = []

    # Generate samples one by one
    for i in range(N):

        # Initialize the flag to check if the sample is accepted
        sample_accepted = False 
        
        # Generate samples until the sample is accepted
        while not sample_accepted:
            U = np.random.uniform(a, b, 2) # Generate two uniform random numbers 
            Y = U[0] # Set Y to the first random number
            #condition for acceptance
            if U[1] <= density(Y, a, b, mu, sigma) / (c * g):
                sample_accepted = True

        # Append the accepted sample to the list if it is accepted 
        samples_list.append(Y)

    TruncSamples = np.array(samples_list) # Convert the list of samples to a numpy array to plot the histogram
    
    return TruncSamples

# plot histogram and pdf of the truncated normal distribution
TruncSamples = Sample_TruncNormal_AR(a, b, mu, sigma, N)
plt.hist(TruncSamples, density=True, label='Histogram of Truncated Normal Samples')
x = np.linspace(a, b, N) # Generate N points between a and b to plot the pdf
pdf = [density(i, a, b, mu, sigma) for i in x] # Compute the pdf of the truncated normal distribution at each point
plt.plot(x, pdf, 'r', label='Truncated Normal PDF') 
plt.show()





'''
logic behind the code:

Goal: Simulate X with pdf f 
How: Simulate Y with pdf g with f(x) <= c*g(x) for all x, where c is a constant 
    and accept Y if U <= f(Y)/(c*g(Y), where U is a uniform random variable, otherwise repeat!
    (The condition quantifies how much more probable it is for a sample to belong to 
    the desired distribution (truncated normal in this case) compared to the known distribution (uniform))
'''