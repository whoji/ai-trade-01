# https://blog.quantinsti.com/stock-market-data-analysis-python/

import pyfolio as pf
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Get the data for stock Facebook from 2017-04-01 to 2019-04-30
data = yf.download('AAPL', start="2017-04-01", end="2019-04-30")

# Print the first five rows of the data
print(data.head())




tickers_list = ['AAPL', 'AMZN', 'MSFT', 'WMT']
# Import pandas and create a placeholder for the data
data = pd.DataFrame(columns=tickers_list)

for ticker in tickers_list:
    #ticker = 'AAPL'
    data[ticker] = yf.download(ticker, period='5y',)['Adj Close']

# Compute the returns of individula stocks and then compute the daily mean returns.
# The mean return is the daily portfolio returns with the above four stocks.
data = data.pct_change().dropna().mean(axis=1)
# Print first 5 rows of the data
data.head()

plt.figure(figsize=(10, 7))
pf.create_full_tear_sheet(data)
plt.show()
