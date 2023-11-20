import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utils
import random
# import real data
n = 10
L = 200
Xs = []
x = np.zeros(L)
x[:L//4] = np.sin(np.arange(L//4)*2*np.pi/(L//4))
for i in range(n):
    y = np.roll(x, shift=np.random.randint(0, L,1))
    y = y + np.random.normal(0, 0.1, L)
    Xs.append(y)
fig, axes = plt.subplots(n, 1, figsize=(5, 20), dpi=300,layout='tight')
for i in range(n):
    ax = axes[i]
    ax.plot(np.arange(L),Xs[i])
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('../plots/signals/signal_generation/shifted_signals.pdf')

# df_pvCLCL = pd.read_csv('../data/pvCLCL_clean_winsorized.csv', index_col=0)

 # select signals
# ind1, ind2 = 11, 22
# signal_length = 50
# start = 5
# signal1 = df_pvCLCL.iloc[ind1,start:start + signal_length]
# signal2 = df_pvCLCL.iloc[ind2,start:start + signal_length]
#
#
# fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300,layout='tight')
# ax.plot(np.arange(signal_length),signal1, label = 'X')
# ax.plot(np.arange(signal_length),np.roll(signal1, 5), label = 'Y')
# ax.axvline(22,linestyle='dashed',color='#1f77b4', alpha=0.7)
# ax.axvline(22+5,linestyle='dashed',color='#ff7f0e', alpha=0.7)
# ax.set_xlabel('Days')
# ax.legend()
# plt.savefig('../plots/signals/signal_generation/lag_signals')
#
# fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300,layout='tight')
# for i in range(5,11):
#     ax.plot(np.arange(signal_length), df_pvCLCL.iloc[i,start:start + signal_length], alpha=0.5, linewidth=0.7)
#
# ax.plot(np.arange(signal_length), signal1, color='#1f77b4', label='selected')
# ax.set_xlabel('Time')
# ax.legend()
# fig.savefig('../plots/signals/signal_generation/select_signals')
#
# num_copies = 2
# sigma = 0.4
# fig, axes = plt.subplots(num_copies, 1, figsize=(5, 3*num_copies), squeeze=False, sharex=True, dpi=300)
# for i, signal in enumerate([signal1]):
#     signal = (signal - signal.mean()) / signal.std()
#     for j in range(num_copies):
#         axes[j,i].plot(np.arange(signal_length), signal, linestyle='solid',label='signal')
#         axes[j,i].set_ylim(-4,3)
#         axes[j, i].legend(loc='lower center')
#     axes[-1,0].set_xlabel('Time')
# plt.savefig(f'../plots/signals/signal_generation/original')
#
# data, shifts = utils.random_shift(signal, num_copies, False, 0.15,'random')
# for i, signal in enumerate([signal1]):
#     # fig, axes = plt.subplots(num_copies, 1, figsize=(8, 5*num_copies), squeeze=False, sharex=True)
#     for j in range(num_copies):
#         axes[j,i].plot(np.arange(signal_length), data[:,j],linestyle='dashed', label='shifted')
#         axes[j,i].axvline(x=shifts[j], ymin=-1,ymax=2,color='grey',linestyle='dashed', label=f'lag={shifts[j]}')
#         axes[j,i].legend(loc='lower center')
# plt.savefig(f'../plots/signals/signal_generation/shifted')
#
# data = utils.random_noise(data, sigma)
# for i, signal in enumerate([signal1]):
#     # fig, axes = plt.subplots(num_copies, 1, figsize=(8, 5 * num_copies), squeeze=False, sharex=True)
#     for j in range(num_copies):
#         axes[j, i].plot(np.arange(signal_length), data[:, j], linestyle='dotted', label='shifted+noise')
#         axes[j,i].legend(loc='lower center',ncol=2)
#         # axes[j, i].axvline(x=shifts[j], ymin=-1, ymax=2, color='grey', linestyle='dashed')
# plt.savefig(f'../plots/signals/signal_generation/noisy')
#
#
