
import configparser
import oandapy as opy
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('oanda.cfg')

oanda = opy.API(environment='practice',
    access_token=config['oanda']['access_token'])


data = oanda.get_history(instrument='EUR_USD',  # our instrument
                         start='2016-12-08',  # start data
                         end='2016-12-10',  # end date
                         granularity='M1')  # minute bars 

df = pd.DataFrame(data['candles']).set_index('time') 

df.index = pd.DatetimeIndex(df.index)

print(df.info() )

'''
we formalize the momentum strategy by telling Python to take 
the mean log return over the last 15, 30, 60, and 120 minute
bars to derive the position in the instrument.
'''
df['returns'] = np.log(df['closeAsk'] / df['closeAsk'].shift(1)) 
cols = []  # 13

for momentum in [15, 30, 60, 120]: 
    col = 'position_%s' % momentum 
    df[col] = np.sign(df['returns'].rolling(momentum).mean()) 
    cols.append(col)  # 17

strats = ['returns']  # 19

for col in cols:  # 20
    strat = 'strategy_%s' % col.split('_')[1]  # 21
    df[strat] = df[col].shift(1) * df['returns']  # 22
    strats.append(strat)  # 23

df[strats].dropna().cumsum().apply(np.exp).plot()  # 24
plt.show()