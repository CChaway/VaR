## Comments
# IBM = pdr.DataReader("IBM", data_source = "yahoo", start = datetime(2018, 1, 1), end = datetime(2022, 12, 31))
# IBM = pdr.get_data_yahoo("IBM", start="2018-01-01", end=dt.date.today().strftime('%Y-%m-%d'))
#
#
# Use csv file / Eikon API / ICE API
# Calculate exposure
# Trailing mean and cov matrix (60 days)
# Back testing with same weights and chart the result



###
### Portfolio VaR
###

## Calculate periodic returns of the stocks in the portfolio
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()
# import fix_yahoo_finance as yf
import numpy as np
import datetime as dt
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Upload prices from csv file

raw = pd.read_csv('Data.csv', sep=';', decimal=',', index_col=0, parse_dates=True)
symbol = 'ETHc1'
symbol = 'LCOc1'
data = pd.DataFrame(raw[symbol])
data.rename(columns={'LCOc1': 'price'}, inplace=True)

# Prices from Eikon through API
#
# import eikon as ek
# ek.set_app_key('880850d5de1b459fbf721dd399169c952eb876d2')
# rics = ['SPY', 'AAPL.O', 'AMZN.O']
# data = ek.get_timeseries(rics, fields='CLOSE', start_date='2018-02-12', end_date='2018-02-28', interval='daily')

# Prices from ICE through API

# import icepython as ice
# symbols = ['%Brn 1!-ICE','%BRN 2!-ICE','%BRN 3!-ICE']
# fields = 'Last'
# data = ice.get_timeseries(symbols, fields, granularity = 'D', start_date='2020-12-01', end_date='2020-12-31',)
# df = pd.DataFrame(list(data))
# df = df.set_index(0)

# Download prices from yahoo finance

# Create our portfolio of equities
tickers = ['AAPL', 'IBM', 'XOM', 'MSFT']

# Set an initial investment level
initial_investment = 1000000

# Set the investment weights (I arbitrarily picked for example)
weights = np.array([.25, .3, .15, .3])

# Download closing prices
data = pdr.get_data_yahoo(tickers, start="2018-01-01", end=dt.date.today().strftime('%Y-%m-%d'))['Close']

# Positions for MtM
positions = np.array([10, -5, 3, 0])
MtM = positions * data.iloc[-1]
T_MtM = sum(MtM)
weights = MtM / sum(MtM)

# From the closing prices, calculate periodic returns
returns = data.pct_change()

returns.tail()

## Create a covariance matrix based on the returns
# Generate Var-Cov matrix
cov_matrix = []
avg_rets = []
port_mean = []
port_stdev = []
mean_investment = []
stdev_investment = []
var_1d1 = []

for i in range(0, len(returns)-60+1):
    cov_matrix.append(returns.iloc[i:60+i].cov())

    ## Calculate the portfolio mean and standard deviation
    # Calculate mean returns for each stock
    avg_rets.append(returns.iloc[i:60+i].mean())

    # Calculate mean returns for portfolio overall,
    # using dot product to
    # normalize individual means against investment weights
    # https://en.wikipedia.org/wiki/Dot_product#:~:targetText=In%20mathematics%2C%20the%20dot%20product,and%20returns%20a%20single%20number.
    port_mean.append(avg_rets[-1].dot(weights))

    # Calculate portfolio standard deviation
    port_stdev.append(np.sqrt(weights.T.dot(cov_matrix[-1]).dot(weights)))

    # Calculate mean of investment
    mean_investment.append((1 + port_mean[-1]) * T_MtM)

    # Calculate standard deviation of investment
    stdev_investment.append(T_MtM * port_stdev[-1])

    ## Calculate the inverse of the normal cumulative distribution (PPF) with a specified confidence interval, standard deviation, and mean
    # Select our confidence interval (I'll choose 95% here)
    conf_level1 = 0.05

    # Using SciPy ppf method to generate values for the
    # inverse cumulative distribution function to a normal distribution
    # Plugging in the mean, standard deviation of our portfolio
    # as calculated above
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    from scipy.stats import norm
    cutoff1 = norm.ppf(conf_level1, mean_investment[-1], stdev_investment[-1])

    ## Estimate the value at risk (VaR) for the portfolio by subtracting the initial investment from the calculation in previous step
    #Finally, we can calculate the VaR at our confidence interval
    var_1d1.append(T_MtM - cutoff1)

# Build plot
plt.xlabel("Day")
plt.ylabel("VaR (USD)")
plt.title("Trailing VaR")
plt.plot(var_1d1, "r")
plt.show()

## Value at risk over n-day time period
# Calculate n Day VaR
var_array = []
num_days = int(15)
for x in range(1, num_days+1):
    var_array.append(np.round(var_1d1[-1] * np.sqrt(x),2))
    print(str(x) + " day VaR @ 95% confidence: " + str(np.round(var_1d1[-1] * np.sqrt(x),2)))

# Build plot
plt.xlabel("Day #")
plt.ylabel("Max portfolio loss (USD)")
plt.title("Max portfolio loss (VaR) over 15-day period")
plt.plot(var_array, "r")
plt.show()

## (Extra) Checking distributions of our equities against normal distribution
# Repeat for each equity in portfolio
returns['XOM'].hist(bins=10, density=True, histtype="bar", alpha=0.5)
x = np.linspace(port_mean[-1] - 3*port_stdev[-1], port_mean[-1]+3*port_stdev[-1],100)
plt.plot(x, norm.pdf(x, port_mean[-1], port_stdev[-1]), "r")
plt.title("XOM returns (binned) vs. normal distribution")
plt.show()
print('a')

