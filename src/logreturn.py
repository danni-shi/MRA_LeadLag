import datetime as dt
import autograd.numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt, dates
import os

import utils
import optimization

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

type = 'orderbook'
ticker = 'AAPL'
date = '2012-06-21'
nlevels = '10'

df = utils.get_dataset(type, ticker, date, nlevels)    

midprice = pd.DataFrame( {'Time': df.Time, 
                          'MidPrice': 0.5 * (df.AskPrice1 + df.BidPrice1)})
midprice_minute = utils.downsample(midprice, 60)
midprice_minute['LogReturn'] = np.log(midprice_minute.MidPrice).diff()
'''
fig, ax = plt.subplots(figsize = (20,8))
ax.plot(midprice_minute.Time.apply(lambda x: utils.seconds_to_time(x,date)), 
        midprice_minute.LogReturn, linewidth = 0.5, label = 'Log Returns')
plt.legend()
plt.grid()
# plt.tick_params(rotation=45)
ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.xlabel('Time')
plt.ylabel('Log Returns of 10000 stocks')
plt.suptitle('Log Returns of AAPL per minute')
plt.savefig('logreturns')
'''
# use log return series as the original signal
signal = (midprice_minute.LogReturn - midprice_minute.LogReturn.mean())/ midprice_minute.LogReturn.std()
signal = signal.iloc[1:len(signal)]

with open('logreturn.npy', 'wb') as f:
    np.save(f, signal)