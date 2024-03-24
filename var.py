
## Calculate periodic returns of the stocks in the portfolio
# import pandas as pd
from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf
import numpy as np
import datetime as dt
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Create our portfolio of equities
tickers = ['AAA', 'FB', 'C', 'DIS']

# Set the investment weights (I arbitrarily picked for example)
weights = np.array([.25, .3, .15, .3])

# Set an initial investment level
initial_investment = 1000000

# Download closing prices
data = pdr.get_data_yahoo(tickers, start="2018-01-01", end=dt.date.today()-dt.timedelta(days=2))['Close']

# From the closing prices, calculate periodic returns
returns = data.pct_change()

returns.tail()

## Create a covariance matrix based on the returns
# Generate Var-Cov matrix
cov_matrix = returns.cov()

## Calculate the portfolio mean and standard deviation
# Calculate mean returns for each stock
avg_rets = returns.mean()

# Calculate mean returns for portfolio overall,
# using dot product to
# normalize individual means against investment weights
# https://en.wikipedia.org/wiki/Dot_product#:~:targetText=In%20mathematics%2C%20the%20dot%20product,and%20returns%20a%20single%20number.
port_mean = avg_rets.dot(weights)

# Calculate portfolio standard deviation
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))

# Calculate mean of investment
mean_investment = (1 + port_mean) * initial_investment

# Calculate standard deviation of investmnet
stdev_investment = initial_investment * port_stdev

## Calculate the inverse of the normal cumulative distribution (PPF) with a specified confidence interval, standard deviation, and mean
# Select our confidence interval (I'll choose 95% here)
conf_level1 = 0.05

# Using SciPy ppf method to generate values for the
# inverse cumulative distribution function to a normal distribution
# Plugging in the mean, standard deviation of our portfolio
# as calculated above
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
from scipy.stats import norm
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)

## Estimate the value at risk (VaR) for the portfolio by subtracting the initial investment from the calculation in previous step
#Finally, we can calculate the VaR at our confidence interval
var_1d1 = initial_investment - cutoff1

## Value at risk over n-day time period
# Calculate n Day VaR
var_array = []
num_days = int(15)
for x in range(1, num_days+1):
    var_array.append(np.round(var_1d1 * np.sqrt(x),2))
    print(str(x) + " day VaR @ 95% confidence: " + str(np.round(var_1d1 * np.sqrt(x),2)))

# Build plot
plt.xlabel("Day #")
plt.ylabel("Max portfolio loss (USD)")
plt.title("Max portfolio loss (VaR) over 15-day period")
plt.plot(var_array, "r")
plt.show()

## (Extra) Checking distributions of our equities against normal distribution
# Repeat for each equity in portfolio
returns['FB'].hist(bins=10, density=True, histtype="bar", alpha=0.5)
x = np.linspace(port_mean - 3*port_stdev, port_mean+3*port_stdev,100)
plt.plot(x, norm.pdf(x, port_mean, port_stdev), "r")
plt.title("FB returns (binned) vs. normal distribution")
plt.show()
input('Enter any key')

