import datetime as dt
import autograd.numpy as np
#import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates
import os
import time

import utils
import optimization_het

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

L = 50
K = 3
# option: sine wave, random gaussian, log returns
options = ['logreturns', 'sine', 'gaussian']

synthetic_data = 'gaussian'

if synthetic_data:
    assert options.count(synthetic_data) > 0, f'{synthetic_data} is not an option'
    
    if synthetic_data == 'logreturns':
        with open('../../data/logreturn.npy', 'rb') as f:
            data = np.load(f)
        l = min(L,len(data)//K)
        x = np.reshape(data[:K*l],(l,K), 'F')
    elif synthetic_data == 'sine':
        x = np.zeros((L,K))
        for k in range(K):
            y = np.linspace(0,(k+1)*np.pi, L)
            x[:,k] = np.sin(y)
    elif synthetic_data == 'gaussian':
        x = np.random.randn(L,K)
        
    # standardise the ground truth signal
    signal = (x-np.mean(x, axis = 0))/np.std(x, axis = 0) # the division is along axis 0
    
    # ground truth mixing probabilities

    p_true = np.ones(K)
    p_true += np.random.uniform(-0.3,0.3, K)
    # p_true = np.maximum(0.5/K, p_true)
    p_true = p_true/sum(p_true)
    # p_true = np.array([0.6,0.4])
    assert sum(p_true) - 1 < 1e-10
    
    # set parameters for generating observations
    sigma = 0.1
    max_shift= 0.2
    M = 1000
    Ms = np.round(p_true*M).astype(int)
    
    # generate shifted, noisy version of the signal
    start = time.time()
    observations, shifts, classes = utils.generate_data_het(signal, Ms,  max_shift, sigma, cyclic = False)
    print('time to generate data = ', time.time() - start)

    # save observations
    with open('../results/data.npy', 'wb') as f:
        np.save(f, observations)
        np.save(f, shifts)
else:
    path = '../../data/OPCL_20000103_20201231.csv'
    data = pd.read_csv(path, index_col=0)
    tickers = ['XLF','XLB','XLK','XLV','XLI','XLU','XLY','XLP','XLE']
    M = 200
    data = data.iloc[:L].dropna(axis = 0)
    data = np.array(data).transpose()
    observations = (data-np.mean(data, axis = 0))/np.std(data, axis = 0)
    
# optimization
L = len(observations)
np.random.seed(42)
X0 = np.random.randn(L, K)
X_est = optimization_het.optimise_manopt(observations, sigma, p_true, X0, extra_inits=0)

# align the estimate to original signal    
X_aligned, perm = utils.align_to_ref_het(X_est,signal)
print(f'relative error of signal =  {np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal):.5f}')

# plot
plt.rcParams['text.usetex'] = True
fig, axes = plt.subplots(nrows = K, ncols = 1, figsize = (15,6*K), squeeze=False)
ax = axes.flatten()
for i in range(K):
    ax[i].plot(np.arange(L),signal[:,i], label = 'true')
    ax[i].plot(np.arange(L), X_aligned[:,i], label = 'estimate',linestyle = '--')
    ax[i].grid()
    ax[i].legend()
    ax[i].set_title(f'relative error of signal =  {np.linalg.norm(X_aligned[:,i]-signal[:,i])/np.linalg.norm(signal[:,i]):.5f}; proportion in sample = {p_true[i]:.5f}')
fig.suptitle('Comparison of the Original and Estimated Signals, adjusted for shifts')
plt.savefig('../plots/estimate')

with open('../results/visual.npy', 'wb') as f:
    np.save(f, X_est)
    np.save(f, X_aligned)
    np.save(f, signal)
    np.save(f,X0)
    np.save(f, p_true)