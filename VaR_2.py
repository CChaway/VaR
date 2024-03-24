
###
### Market Risk and VaR Estimation Methodologies with Python
###

from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()
# import fix_yahoo_finance as yf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import scipy as sci
import scipy.stats as scs
from datetime import datetime
import matplotlib as mpl
from matplotlib import rcParams
import pandas as pd


rcParams['figure.figsize'] = (12, 8)
rcParams['figure.dpi'] = 150
plt.style.use('ggplot')

from sys import version
print('Reproductibility conditions for the VaR estimation '.center(80, '-'))
print('Python version:     ' + version)
print('Numpy version:      ' + np.__version__)
print('Scipy version:      ' + sci.__version__)
print('Pandas version:     ' + pd.__version__)
print('Matplotlib ver:     ' + mpl.__version__)

print('-'*80)

# Introduction: Stock market returns and volatility

IBM = pdr.get_data_yahoo("IBM", start="2018-01-01", end=dt.date.today().strftime('%Y-%m-%d'))

IBM['25d'] = IBM['Adj Close'].rolling(window=25).mean()

IBM['250d'] = IBM['Adj Close'].rolling(window=250).mean()

IBM[['Adj Close', '25d', '250d']].plot()
plt.title("Five year of IBM's price stock (from 2018 to end of 2022)",
          weight='bold');

(IBM["Adj Close"].pct_change().dropna()).plot(secondary_y=True, title='')

(2.32 * IBM["Adj Close"].pct_change().dropna().rolling(window=250).std()).plot(secondary_y=True, title='')

(-2.32 * IBM["Adj Close"].pct_change().dropna().rolling(window=250).std()).plot(secondary_y=True, title='')

# plt.title("IBM stock daily relative returns and rolling 99% interval of confidence", weight='bold');

plt.show()

IBM["Adj Close"].pct_change().describe()

def print_statistics(data):
    print('\n', " RETURN SAMPLE STATISTICS ".center(40, '-'), '\n')
    print("Mean of Daily  Log Returns {0:.4f}".format(np.mean(data)))
    print("Std  of Daily  Log Returns {0:.4f}".format(np.std(data)))
    print("Mean of Annua. Log Returns {0:.4f}".format(np.mean(data) * 252))
    print("Std  of Annua. Log Returns {0:.4f}".format(np.std(data) * np.sqrt(252)))
    print('\n', "-" * 40, '\n')
    print('Annua. volatility:         {0:.3f}%'.format(data.std() * np.sqrt(252) * 100))
    print('\n', "-" * 40, '\n')
    print("Skew of Sample Log Returns {0:.4f}".format(scs.skew(data)))
    print("Skew Normal Test p-value   {0:.4f}".format(scs.skewtest(data)[1]))
    print('\n', "-" * 40, '\n')
    print("Kurt of Sample Log Returns {0:.4f}".format(scs.kurtosis(data)))
    print("Kurt Normal Test p-value   {0:.4f}".format(scs.kurtosistest(data)[1]))
    print('\n', "-" * 40, '\n')
    print("Normal Test p-value        {0:.4f}".format(scs.normaltest(data)[1]))
    print('\n', "-" * 40, '\n')

print_statistics(IBM["Adj Close"].pct_change().dropna())

(IBM["Adj Close"]
 .pct_change()
 .hist(bins=50, density=True, histtype='stepfilled', alpha=0.5))

plt.title("Histogram of IBM stock daily returns from 2018 to 2022",
          weight='bold');

Q = IBM["Adj Close"].pct_change().dropna().to_numpy()
sci.stats.probplot(Q,
                   dist = sci.stats.norm,
                   plot = plt.figure().add_subplot(111))
plt.title("Normal QQ-plot of Google daily returns from 2018 to 2022",
          weight="bold")
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles');
plt.show()

def normality_tests(array):
    # Tests for normality distribution of given data set
    print("Skew of data set %14.3f" % scs.skew(array))
    print("Skew test p-value %14.3f" % scs.skewtest(array)[1])
    print("Kurt of data set %14.3f" % scs.kurtosis(array))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(array)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(array)[1])

returns = np.array(IBM["Adj Close"].pct_change().dropna())
normality_tests(returns.flatten())

tdf, tmean, tsigma = sci.stats.t.fit(Q)
sci.stats.probplot(Q,
                     dist = sci.stats.t,
                     sparams = (tdf, tmean, tsigma),
                     plot = plt.figure().add_subplot(111))
plt.title("Student QQ-plot of Google daily returns from 2010 to 2014",
          weight="bold")
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles');
plt.show()

# Value at Risk using the Historical Method

# start = datetime(2013, 1, 1)
# end   = datetime(2014, 12, 31)

stock = pdr.get_data_yahoo("IBM", start="2018-01-01", end=dt.date.today().strftime('%Y-%m-%d'))

returns = stock["Adj Close"].pct_change().dropna()
print('Statistis Summary of IBM relative returns')
print(returns.describe())
mean = returns.mean()
sigma = returns.std()
tdf, tmean, tsigma = sci.stats.t.fit(returns.to_numpy())
returns.hist(bins=40, density=True, histtype='stepfilled', alpha=0.5);
plt.title("Daily relative changes in IBM stock-price over 2012–2014 (%)", weight='bold');

print('Quantile 95%:',returns.quantile(0.05))
print('Quantile 99%:',returns.quantile(0.01))

# Value at Risk using the Variance-Covariance method

support = np.linspace(returns.min(), returns.max(), 100)
returns.hist(bins=40, density=True, histtype='stepfilled', alpha=0.5);
plt.plot(support, sci.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")
plt.title("Daily change in IBM over 2013–2014 (%)", weight='bold');
plt.show()

print('Quantile 95%:',sci.stats.norm.ppf(0.05, mean, sigma))
print('Quantile 99%:',sci.stats.norm.ppf(0.01, mean, sigma))

# Value at Risk using the Monte Carlo method

days = 300   # time horizon
dt = 1/float(days)
mu = returns.mean()
sigma = returns.std()
startprice = IBM['Adj Close'][-1]

def random_walk(startprice):
    price = np.zeros(days)
    shock = np.zeros(days)
    price[0] = startprice
    for i in range(1, days):
        shock[i] = np.random.normal(loc = mu * dt,
                                    scale = sigma * np.sqrt(dt))
        price[i] = max(0, price[i-1] + shock[i] * price[i-1])
    return price

for run in range(100):
    plt.plot(random_walk(startprice))
plt.xlabel("Time (days)")
plt.ylabel("Price Evolution");

runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = random_walk(startprice)[days-1]
q = np.percentile(simulations, 1)

plt.hist(simulations, density=True, bins=30, histtype='stepfilled', alpha=0.5)
plt.figtext(0.6, 0.8, u"Start price: 156 USD")
plt.figtext(0.6, 0.7, u"Mean final price: %.3f USD" % simulations.mean())
plt.figtext(0.6, 0.6, u"VaR(0.99): %.3f USD" % (10 - q,))
plt.figtext(0.15, 0.6, u"q(0.99): %.3f USD" % q)
plt.axhline(y=q, linewidth=2, color='r')
plt.title(u"Final price distribution after %s days" % days, weight='bold');
plt.ylim([min(simulations)-1,max(simulations)+1])
plt.show()


# VaR implementation in a Python Class

from scipy.stats import norm
from datetime import datetime
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sci


class VaR(object):
    __metaclass__ = ABCMeta

    def __init__(self, stock_code, confidence_level, time_horizon, method, volume=1, simulations=100000):
        try:
            self.stock_code = str(stock_code)
            self.confidence_level = float(confidence_level)
            self.time_horizon = int(time_horizon)
            self.method = str(method)
            self.volume = int(volume)
            self.simulations = int(simulations)

            if confidence_level < 0 or confidence_level > 1:
                raise ValueError('Confidence_level value not allowed.')
            if time_horizon < 0 or time_horizon > 1000:
                raise ValueError('Time_horizon value not allowed.')
            if volume <= 0:
                raise ValueError('Volumes(number of shares) has to be positive')
            if simulations <= 0 or simulations > 10000000:
                raise ValueError('Simulations value not allowed')
            methods = ['Historical', 'VarCov', 'MonteCarlo']
            if method not in methods:
                raise ValueError('Method unknown')
        except ValueError:
            print('Error passing VaR inputs')

    def getprices(self):
        prices = pdr.get_data_yahoo(self.stock_code, start="2018-01-01", end="2023-01-01")
        return prices['Adj Close']

    def getreturns(self):
        prices = self.getprices()
        return prices.pct_change().dropna()

    def getmethod(self):
        return self.method

    def __str__(self):
        return 'VaR estimation using {0} method'.format(self.getmethod())

    @abstractmethod
    def value(self):
        pass


class Historical(VaR):
    def __init__(self, stock_code, confidence_level, time_horizon=1):
        VaR.__init__(self, stock_code, confidence_level, time_horizon, 'Historical')

    @property
    def value(self):
        returns = VaR.getreturns(self)
        return returns.quantile(self.confidence_level) * self.volume


class VarCov(VaR):
    def __init__(self, stock_code, confidence_level, time_horizon=1):
        VaR.__init__(self, stock_code, confidence_level, time_horizon, 'VarCov')

    @property
    def value(self):
        returns = VaR.getreturns(self)
        return sci.stats.norm.ppf(self.confidence_level,
                                  returns.mean(),
                                  returns.std()) * self.volume

class MonteCarlo(VaR):
    def __init__(self, stock_code, confidence_level, time_horizon=1):
        VaR.__init__(self, stock_code, confidence_level, time_horizon, 'MonteCarlo')

    def random_walk(self, mu, sig, terminal, initial_price):
        brownian = np.sqrt(terminal) * np.random.randn(self.simulations, 1)
        price_terminal = initial_price * np.exp((mu - 0.5 * sig ** 2) * terminal + sig * brownian)
        return price_terminal

    @property
    def value(self):
        prices = VaR.getprices(self)
        initial_price = prices[-1]
        returns = prices.pct_change().dropna()
        mu = returns.mean()
        sig = returns.std() * np.sqrt(252)
        dt = self.time_horizon / 252.
        simulations = self.random_walk(mu, sig, dt, initial_price)
        return np.log(np.percentile(simulations, int(self.confidence_level * 100)) / float(initial_price))


stocks = ['^DJI', 'IBM', 'AAPL', 'GM', 'GOOG', 'HPQ', 'JPM', 'GS']
confidence_level = 0.05
hist = {}
vcov = {}
mc = {}
for i, stock in enumerate(stocks):
    hist[i] = Historical(stock, confidence_level)
    vcov[i] = VarCov(stock, confidence_level)
    mc[i] = MonteCarlo(stock, confidence_level)

print('\n', " Examples of VaR Estimation ".center(68, '-'))
for i in range(len(stocks)):
    print('\nStock: {2} -> {0} : {1:.4f}'.format(hist[i], hist[i].value, stocks[i]))
    print('Stock: {2} -> {0} : {1:.4f}'.format(vcov[i], vcov[i].value, stocks[i]))
    print('Stock: {2} -> {0} : {1:.4f} \n'.format(mc[i], mc[i].value, stocks[i]))
    print('-' * 68)


# VaR of a Portfolio
# Correlation between stocks

start = datetime(2019, 1, 1)
end = datetime(2023, 1, 1)
# CAC = DataReader("^FCHI", "yahoo", start, end)
# DAX = DataReader("^GDAXI", "yahoo", start, end)
# HSI = DataReader("^HSI", "yahoo", start, end)
# AORD = DataReader("^DJI", "yahoo", start, end)
CAC = pdr.get_data_yahoo("^FCHI", start, end)
DAX = pdr.get_data_yahoo("^GDAXI", start, end)
HSI = pdr.get_data_yahoo("^HSI", start, end)
AORD = pdr.get_data_yahoo("^DJI", start, end)

df = pd.DataFrame({ 'CAC': CAC["Close"].pct_change(),
                 'DAX': DAX["Close"].pct_change(),
                 'HSI': HSI["Close"].pct_change(),
                 'DJI': AORD["Close"].pct_change() })
dfna = df.dropna()

# pd.plotting.scatter_matrix(dfna, alpha=0.2, figsize=(12, 8),  diagonal='kde');

plt.plot(dfna["CAC"], dfna["DAX"], '.', alpha=0.5)
plt.xlabel("CAC40 daily return")
plt.ylabel("DAX daily return")
plt.title("CAC vs DAX daily returns, 2013–2014", weight='bold');
plt.show()

sci.stats.pearsonr(dfna["CAC"], dfna["DAX"])

plt.plot(dfna["CAC"], dfna["DJI"], '.', alpha=0.5)
plt.xlabel("CAC40 daily return")
plt.ylabel("DJI daily return")
# ensure square aspect ratio
plt.xlim(-0.15, 0.15)
plt.ylim(-0.15, 0.15)
plt.title("CAC vs DJI daily returns, 2005–2010", weight='bold');
plt.show()

sci.stats.pearsonr(dfna["CAC"], dfna["DJI"])

returns = dfna["CAC"]
# returns.hist(bins=30, density=True, histtype='stepfilled', alpha=0.5)
support = np.linspace(returns.min(), returns.max(), 100)
tdf, tmean, tsigma = sci.stats.t.fit(returns.to_numpy())
print("CAC t fit: mean={}, scale={}, df={}".format(tmean, tsigma, tdf))
test = sci.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf)
plt.plot(support, test, "r-")
plt.figtext(0.6, 0.7, "tμ = {:.3}".format(tmean))
plt.figtext(0.6, 0.65, "tσ = %3f" % tsigma)
plt.figtext(0.6, 0.6, "df = %3f" % tdf)
plt.title("Histogram of CAC40 daily returns over 2005–2010", weight='bold');
plt.show()

returns = dfna["DAX"]
returns.hist(bins=30, density=True, histtype='stepfilled', alpha=0.5)
support = np.linspace(returns.min(), returns.max(), 100)
tdf, tmean, tsigma = sci.stats.t.fit(returns.to_numpy())
print("DAX t fit: mean={}, scale={}, df={}".format(tmean, tsigma, tdf))
plt.plot(support, sci.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")
plt.figtext(0.6, 0.7, "tμ = %3f" % tmean)
plt.figtext(0.6, 0.65, "tσ = %3f" % tsigma)
plt.figtext(0.6, 0.6, "df = %3f" % tdf)
plt.title("Histogram of DAX daily returns over 2005–2010", weight='bold');
plt.show()

runs = 5000
fittedCAC = np.zeros(runs, float)
fittedDAX = np.zeros(runs, float)
for i in range(runs):
    fittedCAC[i] = sci.stats.t.rvs(loc=0.000478137351981,
                                     scale=0.00898201242824,
                                     df=2.75557323986)
    fittedDAX[i] = sci.stats.t.rvs(loc=0.000847802944791,
                                     scale=0.00878082895409,
                                     df=2.71766905436)
plt.plot(fittedCAC, fittedDAX, 'r.', alpha=0.5)
plt.title("CAC vs DAX returns (simulated, no correlation)", weight='bold');
plt.show()

print('End of code')

## Notes:
# IBM = pdr.DataReader("IBM", data_source = "yahoo", start = datetime(2018, 1, 1), end = datetime(2022, 12, 31))
# IBM = pdr.get_data_yahoo("IBM", start="2018-01-01", end=dt.date.today().strftime('%Y-%m-%d'))
# print(np.percentile.__doc__)
# ipython

