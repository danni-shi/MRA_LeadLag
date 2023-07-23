import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import seaborn as sns
import pickle
#
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

# ----- Check these parameters are in sync with main_non_modularized.py -----#

sigma_range = np.arange(0.1, 2.1, 0.1)  # std of random gaussian noise
K_range = [1,2,3,4]

# num_rounds = 2

###--- create the folder to save plots ---###
# change folder name accroding to experiment specications
synthetic_results_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/results/synthetic'
folder_name = '10rounds'
results_save_dir = utils.save_to_folder('../plots', folder_name)

# labels and formats
labels = {'pairwise': 'S-Pairwise',
          'sync': 'S-Sync',
          'spc-homo': 'S-IVFhomo',
          'spc': 'SPC',
          'het': 'I-IVFhet',
          'het reassigned': 'IVF (regroup)',
          'true': 'True'}

color_labels = labels.keys()
col_values = sns.color_palette('Set2')
color_map = dict(zip(color_labels, col_values))

lty_map = {
            'pairwise': 'dotted',
           'sync': 'dashdot',
           'het': 'dashed',
           'spc-homo': 'solid'}


def mean_dict(dicts):
    """
    given a list of dictionaries, return the mean of the values in a dictionary
    with the same keys at every level

    """
    if isinstance(dicts[0], dict):
        # get the keys of the first dictionary
        keys = dicts[0].keys()
        # use a dict comprehension to create a new dictionary
        return {k: mean_dict([d[k] for d in dicts]) for k in keys}
    else:
        # return [sum(x) / len(x[~np.isnan(x)]) for x in zip(*dicts)]
        try:
            iterable = iter(dicts[0])
            return [np.nanmean(x,axis=0) for x in zip(*dicts)]
        except:
            return np.nanmean(dicts)


# # =================== plot the results ==================#
# # --- plots of performance of different methods for lag estimatiopn ---#
performance_dicts = []
for round in range(1, 11):
    # read saved results
    with open(f'{synthetic_results_path}/10rounds/performance/{round}.pkl', 'rb') as f:
        performance = pickle.load(f)
    performance_dicts.append(performance)


for round in range(1, 9):
    # read saved results
    with open(f'{synthetic_results_path}/K4_8rounds/performance/{round}.pkl', 'rb') as f:
        performance = pickle.load(f)
    performance_dicts[round-1].update(performance)

# # obtain the performance dictionary as the mean results of all parallel runs
performance = mean_dict(performance_dicts)
plot_list = ['accuracy', 'error_sign']
width = 4
height = 2.5
plt.rcParams['text.usetex'] = True
fig, axes = plt.subplots(nrows=len(plot_list)+1, ncols=len(K_range), figsize=(width*len(K_range), height*(len(plot_list)+1)), squeeze=False,sharex=True)
for j,k in enumerate(K_range):
    for i in range(len(plot_list)):
        for key, result_list in performance[f'K={k}'][plot_list[i]].items():
            label = labels[key]
            if (key != 'het') or (k != 1):
                axes[i,j].plot(sigma_range, result_list, label=label,
                           color=color_map[key], linewidth=2)

        if plot_list[i] in ['accuracy','ARI']:
            axes[i,j].legend(loc='lower left')
        else:
            axes[i,j].legend(loc='upper left')
        axes[i,j].grid(visible=True)
        # axes[i,j].set_facecolor('white')

    quantile = (0.05, 0.95)
    for key, errors_quantile in performance[f'K={k}']['errors_quantile'].items():
        if (key != 'het') or (k != 1):
            mean = [x[int(0.5 / 0.05)] for x in errors_quantile]

            quantile_l = [x[int(quantile[0] / 0.05)] for x in errors_quantile]
            quantile_u = [x[int(quantile[1] / 0.05)] for x in errors_quantile]
            if key == 'het':
                linestyle = 'dashed'
            else:
                linestyle = 'solid'
            axes[-1,j].plot(sigma_range, mean, label=labels[key],
                        color=color_map[key], linewidth=2, linestyle=linestyle)
            axes[-1,j].fill_between(sigma_range, quantile_u, quantile_l, color=color_map[key], alpha=0.3)
                               # label=f'{labels[key]}: {quantile[0]:.0%} to {quantile[1]:.0%}')
            # axes[-1,j].set_facecolor('white')

            axes[-1,j].grid(visible=True)
            axes[-1,j].legend(loc='upper left')
            axes[-1,j].set_xlabel(r'$\sigma \\$'f'$K={k}$')

    axes[-3,0].set_ylabel(f'Accuracy')
    axes[-2,0].set_ylabel(f'Average Directional Error')
    axes[-1,0].set_ylabel('Average Absolute Error')
    axes[-3,j].set_ylim(top=100, bottom=30)
    axes[-2,j].set_ylim(top=1.5, bottom=0)
    axes[-1,j].set_ylim(top=3,bottom=0)

    # ax[-1].set_title(
    #     f'Average Lag Error (shaded area is the {quantile[0]:.0%} to {quantile[1]:.0%} percentile of errors)')

    plt.tight_layout()
    plt.savefig(results_save_dir + f'/acc_err')

# fig, axes = plt.subplots(1, 3, figsize=(8, height), squeeze=False)
# for i, k in enumerate([2, 3, 4]):
#     ax = axes.flatten()
#     for key, result_list in performance[f'K={k}']['ARI'].items():
#         label = 'SPC' if key == 'spc' else 'IVF'
#         ax[i].plot(sigma_range, result_list, label=label,
#                    color=color_map[key], linewidth=2)
#         ax[i].grid(visible=True)
#         ax[i].legend(loc='lower left')
#         ax[i].set_ylim(top=1, bottom=0.1)
#         ax[i].set_xlabel(r'$\sigma \\$'f'$K={k}$')
# plt.tight_layout()
#
# plt.savefig(results_save_dir + f'/ARI')
#


# trading performance
#
# trading_results_list = []
# for round in range(1, 11):
#     with open(f'{synthetic_results_path}/{folder_name}/PnL/{round}.pkl', 'rb') as f:
#         trading_results = pickle.load(f)
#         trading_results_list.append(trading_results)
# trading_results = mean_dict(trading_results_list)
# PnL_sigma_range = np.arange(1.0,2.1,0.5)
# K_range = [1,3]
#
# models = ['pairwise', 'sync', 'spc-homo', 'het']
# plt.rcParams['text.usetex'] = True
#
# for m in range(1):
#     fig, axes = plt.subplots(len(PnL_sigma_range), len(K_range),
#                              figsize=(4 * len(K_range), 2.5 * len(PnL_sigma_range)),
#                              squeeze=False, sharey=True)
#     for j,k in enumerate(K_range):
#         for i, sigma in enumerate(PnL_sigma_range):
#             ax = axes[i, j]
#             if i==len(PnL_sigma_range)-1:
#                 ax.set_xlabel(f'K={k}')
#             if j==0:
#                 ax.set_ylabel(r'$PnL \\$'f'$\sigma={sigma:.2g}$')
#             for model in models:
#                 returns_dict = trading_results[f'K={k}'] \
#                     [f'sigma={sigma:.2g}'][model]['portfolio average']
#
#                 # cumsum evaluate the returns of a portfolio of a constant volume
#                 values = returns_dict['PnL']['class average']
#                 values[np.isnan(values)] = 0
#                 cum_pnl = np.cumsum(values)
#                 SR = returns_dict['annualized SR']['class average']
#                 mean_PnL = np.mean(values)  # include the first few days when no trading occurs
#                 ax.plot(np.arange(len(cum_pnl)), cum_pnl,
#                         label=f'{labels[model]}: SR {SR:.1f}',
#                         color=color_map[model],
#                         linestyle=lty_map[model])
#             ax.legend(loc='upper left')
#             ax.grid(visible=True)
#             ax.set_ylim(top=40)
#     plt.savefig(results_save_dir + f'/{m+1}')


