import numpy as np
import matplotlib.pyplot as plt
import os
import utils

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open('../results/visual.npy', 'rb') as f:
    X_est = np.load(f)
    X_aligned = np.load(f)
    signal = np.load(f)
    X0 = np.load(f)
    p_true = np.load(f)
    # p_est = np.load(f)


# X_est = X_est.flatten()

# use convolution theorem https://en.wikipedia.org/wiki/Cross-correlation
# X_est_shifted, lag, ccf = utils.align_to_ref(X_est, signal, return_ccf = True)
print(f'true prob: {p_true}')
# print(f'estimated prob: {p_est}')
L, K = X_aligned.shape
plt.rcParams['text.usetex'] = True
fig, axes = plt.subplots(nrows = K, ncols = 1, figsize = (15,6*K), squeeze=False)
ax = axes.flatten()
for i in range(K):
    ax[i].plot(np.arange(L),signal[:,i], label = 'true')
    ax[i].plot(np.arange(L), X_aligned[:,i], label = 'estimate',linestyle = '--')
    ax[i].grid()
    ax[i].legend()
    # ax[i].set_title(f'true prob: {p_true[i]:.2f}; estimated prob: {p_est[i]:.2f}')
    ax[i].set_title(f'relative error of signal =  \
                    {np.linalg.norm(X_aligned[:,i]-signal[:,i])/np.linalg.norm(signal[:,i]):.5f}; proportion in sample = {p_true[i]:.5f}')
fig.suptitle('Comparison of the Original and Estimated Signals, adjusted for shifts')
plt.savefig('../plots/estimate')


# fig, ax = plt.subplots(figsize = (15,6))
# ax.stem(np.arange(len(X_est)), ccf)
# plt.xlabel('Lag, k')
# plt.ylabel(r'$corr(X[i], X_{est}[i+k])$')
# plt.title(f'Circular CCF, best lag = {lag}')
# # 95% UCL / LCL
# plt.axhline(-1.96/np.sqrt(len(ccf)), color='r', ls='--') 
# plt.axhline(1.96/np.sqrt(len(ccf)), color='r', ls='--')

# plt.savefig('../plots/ccf')



with open('../results/data.npy', 'rb') as f:
    observations = np.load(f)
    shifts = np.load(f)
# plot selected observations 
fig, axes = plt.subplots(10,5, figsize=(20,20));
ax = axes.flatten()
n = 50
assert n <= observations.shape[1]
for i in range(n):
    lag = shifts[i]
    ax[i].vlines(lag, np.min(observations[:,i]), np.max(observations[:,i]), color = 'red', ls = '-.')
    ax[i].plot(observations[:,i])
plt.savefig('../plots/observations')