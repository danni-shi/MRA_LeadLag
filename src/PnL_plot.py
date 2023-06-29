import numpy as np
import pandas as pd
import main
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

round = 2
with open(f'../results/PnL/{round}.pkl', 'rb') as f:
    PnL = pickle.load(f)
# labels and formats
labels = {'pairwise': 'SPC-Pairwise',
          'sync': 'SPC-Sync',
          'spc-homo': 'SPC-IVF',
          'spc': 'SPC',
          'het': 'IVF',
          'het reassigned': 'IVF (regroup)',
          'true': 'True'}
color_labels = labels.keys()
col_values = sns.color_palette('Set2')
color_map = dict(zip(color_labels, col_values))
sigma_range = np.arange(0.5, 2.1, 0.5)
fig, axes = plt.subplots(nrows=len(sigma_range), ncols=1, figsize=(10, 5*(len(sigma_range))))
for i, sigma in enumerate(sigma_range):
    ax = axes[i]
    ax.set_title(f'Sigma = {sigma}')
    ax.set_xlabel('Days')
    ax.set_ylabel('PnL')
    for model, values in PnL['K=2'][f'sigma={sigma:.2g}'].items():
        cum_pnl = np.append(np.zeros(1),np.cumsum(values))
        ax.plot(np.arange(len(values)+1),cum_pnl, label=labels[model],color=color_map[model])
    ax.legend(loc='lower right')
plt.show()