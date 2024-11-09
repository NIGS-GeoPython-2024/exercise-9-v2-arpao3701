"""Functions used in Exercise 8 of Geol 297 GDA"""

# Import any modules needed in your functions here

import math
import numpy as np

# Define your new functions below
def mean (x, N):
    
    """
    This function computes for the mean for a given array of values x with size of N.

    Parameters:
    x (float): Values for which the mean will be calculated
    N (float): Number of x values

    Returns:
    float: Mean of the array of values x

    """

    return np.sum(x) / N

def stddev (x, mu, N):

    """
    This function computes for the standard deviation for a given array of values x with size of N

    Parameters:
    x (float): Values for which standard deviation will be calculated
    mu (float): Mean of the array of values
    N (float): Number of x values

    Returns:
    float: Standard Deviation of the array of values x

    """

    a = 1 / (N-1)
    b = np.sum((x-mu)**2)
    return np.sqrt(a*b)

def stderr (sigma, N):

    """
    This function computes for the standard error of mean.

    Parameters:
    sigma (float): Standard deviation of the array of values
    N (float): Number of x values

    """

    return sigma / (np.sqrt(N))
    
def gaussian (x, mu, sigma):

    """
    This function computes for the Gaussian value at x at a given mu (mean) and sigma (standard deviation).

    Parameters:
    x (float): The value for which the normal distribution is calculated
    mu (float): The mean of the normal distribution.
    sigma (float): The standard deviation of the normal distribution.

    Returns:
    Gaussian value (float) in the normal distribution
    
    """
    n = 1 / (sigma * np.sqrt(2 * np.pi))
    e = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return n * e

"""Functions used in Exercise 9 of Geol 197 GDAM"""
def linregress(x, y):
    
    """
    This function computers for the slope (B) and intercept (A) of the linear regression line of two datasets. Requires two datasets.
    """
    
    delta = (len(x) * (x ** 2).sum()) - (x.sum() ** 2)
    A = (((x ** 2).sum() * y.sum()) - (x.sum() * (x * y).sum())) / delta
    B = ((len(x)*(x * y).sum()) - (x.sum() * y.sum())) / delta

    return A, B

def pearson (x,y):
    
    """
    This function computes for Pearson's R to determine whether two variables can be correlated with each other.
    """

    #sets-up variables for the computation before summing up values
    num = 0
    den_x = 0
    den_y = 0
    den = 0

    for i in range(len(x)):
        num += ((x[i] - x.mean())*(y[i] - y.mean()))
        den_x += (x[i] - x.mean()) ** 2
        den_y += (y[i] - y.mean()) ** 2
        den = np.sqrt((den_x) * (den_y))

    r = num / den

    return r

def chi_squared (y, e, sigma):

    """
    This function computes for the chi squared value to check the goodness of fit of the regression line relative to the dataset.

    Parameters:
    y (float): dataset obtained
    e (float): expected value based on the calculated regression line
    sigma(float): standard deviation

    Returns:
    chi square (float): how good the model fits the data available
    """

    #variable set-up
    lhs = 0
    rhs = 0
    num = 0
    den = 0

    for i in range(len(y)):
        num = (y[i]-e[i]) ** 2
        den = (sigma[i]) ** 2
        rhs += num / den
        lhs = 1 / len(y)
    
    chi_square = lhs * rhs

    return chi_square
    