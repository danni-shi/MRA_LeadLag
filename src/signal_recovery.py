import datetime as dt
import autograd.numpy as np
#import numpy as np
import scipy.io as spio
from matplotlib import pyplot as plt, dates
import os
import time
from tqdm import tqdm
import utils
import optimization

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# option: sine wave, random gaussian, log returns
options = ['logreturns', 'sine', 'gaussian']

def run(type, n, max_shift, sigma, L, signal = None):
    if not signal:
        if type == 'logreturns':
            with open('../../data/logreturn.npy', 'rb') as f:
                x = np.load(f)
            x = x[:L]
            signal = (x-np.mean(x))/np.std(x)
        elif type == 'sine':
            x = np.linspace(0,np.pi, L)
            x = np.sin(x)
            signal = (x-np.mean(x))/np.std(x)
        elif type == 'gaussian':
            signal = np.random.randn(L)

    
    # generate shifted, noisy version of the signal
    # start = time.time()
    observations, shifts = utils.generate_data(signal, n,  max_shift, sigma, cyclic = False)
    # print('time to generate data = ', time.time() - start)

#     # save observations
#     with open('../results/data.npy', 'wb') as f:
#         np.save(f, observations)
#         np.save(f, shifts)
# else:
#     path = '../../data/OPCL_20000103_20201231.csv'
#     data = pd.read_csv(path, index_col=0)
#     tickers = ['XLF','XLB','XLK','XLV','XLI','XLU','XLY','XLP','XLE']
#     data = data.iloc[:L].dropna(axis = 0)
#     data = np.array(data).transpose()
#     observations = (data-np.mean(data, axis = 0))/np.std(data, axis = 0)

# optimization
# with open('../results/data.npy', 'rb') as f:
#     observations = np.load(f)
#     shifts = np.load(f)
    X_est = optimization.optimise_manopt(observations, sigma, extra_inits=0)

    # align the estimate to original signal    
    X_aligned, lag = utils.align_to_ref(X_est.flatten(),signal)
    relative_error = np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal)
    return signal, X_aligned, relative_error

L = 50
n = 10000
max_shift = 0.1
k = 1

sigma_range = np.arange(0.1,2.1,0.1)
rel_err_x = []
rel_err_x_m = []
signal_dist = []
data_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/HeterogeneousMRA/data/'
for sigma in tqdm(sigma_range):
    
    observations_path = data_path + '_'.join(['observations', 
                                    'noise'+f'{sigma:.2g}', 
                                    'shift'+str(max_shift), 
                                    'class'+str(k)+'.mat'])
    results_path = data_path + '_'.join(['results', 
                                    'noise'+f'{sigma:.2g}', 
                                    'shift'+str(max_shift), 
                                    'class'+str(k)+'.mat'])
    observations_mat = spio.loadmat(observations_path)
    results_mat = spio.loadmat(results_path)
    observations = observations_mat['data']
    X_est_m = results_mat['x_est'].flatten()
    signal = results_mat['x_true'].flatten()
    rel_err_x_m.append(results_mat['rel_error_X'][0][0])
    
    # python optimization
    X_est = optimization.optimise_manopt(observations, sigma, extra_inits=0)
    # align the estimate to original signal    
    X_aligned, lag = utils.align_to_ref(X_est.flatten(),signal)
    relative_error = np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal)
    rel_err_x.append(relative_error)
    
    signal_dist.append(np.linalg.norm(X_aligned-X_est_m)/np.linalg.norm(X_est_m))

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, rel_err_x, label = 'Python')
ax.plot(sigma_range, rel_err_x_m, label = 'MATLAB',linestyle = '--')
ax.plot(sigma_range, signal_dist, label = 'relative difference')
plt.grid()
plt.legend()
plt.title('Comparison of Relative Error for Signal Estimation')
plt.savefig(f'../plots/rel_error_estimation')
with open('../results/visual.npy', 'wb') as f:
    np.save(f, X_aligned)
    np.save(f, signal)