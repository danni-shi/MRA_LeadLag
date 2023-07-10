import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import seaborn as sns
import utils

# add 'src/' to path
sys.path.append('../src')

start = 4445; end = 5145
retrain_period = 10
signal_length = 50
file_name = f'start{start}end{end}_length{signal_length}_trade{retrain_period}'
prediction_folder = '2023-07-04-01h04min_clustering_full'
with open(f'../results/real/{prediction_folder}/PnL_real_single_weighted/' + file_name + '.pkl',
          'rb') as f:
    PnL_SR = pickle.load(f)

K_range = [1, 2, 3]
PnL_sigma_range = np.arange(0.2, 2.0, 0.4)
PnL_sigma_range = [round(PnL_sigma_range[i], 1) for i in range(len(PnL_sigma_range))]
models = ['pairwise', 'sync', 'spc-homo', 'het']

# map return types to labels on plots
RT_map = {'raw returns': 'RR-Lag',
                   'mkt excess returns': 'MR-Lag',
                   'leader excess returns': 'RR-Lag - RR-Lead',
                   'leader raw returns': 'RR-Lead',
                   'leader mkt excess returns': 'MR-Lead'
                   }

# labels and formats
labels = {'pairwise': 'SPC-Pairwise',
          'sync': 'SPC-Sync',
          'spc-homo': 'SPC-IVF',
          'het': 'IVF',}
color_labels = labels.keys()
col_values = sns.color_palette('Set2')
color_map = dict(zip(color_labels, col_values))

lty_map = {'sync': 'dotted',
           'spc-homo': 'dashdot',
           'het': 'dashed',
           'true': 'solid'}


def plot_by_model():
    fig, axes = plt.subplots(
        len(models), 1,
        figsize=(10, 5 * len(models)),
        squeeze=False, sharey=True)
    # 1 plot for each model
    for j, model in enumerate(models):
        ax = axes[j, 0]
        ax.set_title(f'model {model}')
        ax.set_xlabel('Days')
        ax.set_ylabel('PnL')
        returns_dict = PnL_SR[model]
        for return_type, values in returns_dict['PnL'].items():
            # cumsum evaluate the returns of a portfolio of a constant volume
            values[np.isnan(values)] = 0
            cum_pnl = np.cumsum(values)
            # show the average PnL in red and Bold
            if return_type == 'raw returns':
                plot_config = {'linestyle': 'solid',
                               'color': 'red',
                               'linewidth': 2}
            elif return_type == 'mkt excess returns':
                plot_config = {'linestyle': 'solid',
                               'color': 'blue',
                               'linewidth': 2}
            elif return_type == 'leader excess returns':
                plot_config = {'linestyle': 'solid',
                               'color': 'green',
                               'linewidth': 2}
            else:
                plot_config = {'linestyle': 'dashed'}
            SR = returns_dict['annualized SR'][return_type]
            mean_PnL = np.mean(values)  # include the first few days when no trading occurs
            ax.plot(np.arange(len(values)), cum_pnl, label=f'{RT_map[return_type]}: SR {SR:.2f}; PPD {1e2*mean_PnL:.4f}',
                    **plot_config)
        ax.legend(loc='lower right')
        ax.grid(visible=True)
    fig.suptitle(f'Cumulative Returns optimized over K range:{list(K_range)} and \nsigma range:{PnL_sigma_range}')
    plt.savefig(results_save_dir + '/by_models.png')

def plot_by_return_types():
    for return_type, label in RT_map.items():
        fig, ax = plt.subplots(
            1, 1,
            figsize=(10, 5),
            squeeze=True, sharey=True)
        ax.set_title(f'Cumulative {label} optimized over K range {list(K_range)} and \nsigma range {PnL_sigma_range}')
        ax.set_xlabel('Days')
        ax.set_ylabel('PPD')
        returns_dict = {metric: {model: PnL_SR[model][metric][return_type] \
                                 for model in models} \
                        for metric in ['PnL', 'annualized SR']}

        for model, values in returns_dict['PnL'].items():
            # cumsum evaluate the returns of a portfolio of a constant volume
            values[np.isnan(values)] = 0
            cum_pnl = np.cumsum(values)
            SR = returns_dict['annualized SR'][model]
            mean_PnL = np.mean(values)  # include the first few days when no trading occurs
            ax.plot(start + np.arange(len(values)), cum_pnl,
                    label=f'{model}: SR {SR:.2f}; Ave. PPD {1e2 * mean_PnL:.4f}',
                    color=color_map[model])
            ax.legend(loc='lower right')
            ax.grid(visible=True)
        plt.savefig(results_save_dir + f'/{label}')

# change folder name accroding to experiment specications
folder_name = f'{prediction_folder}/trading_OS_PnL_length50_retrain10_centred_optimized'
results_save_dir = utils.save_to_folder('../plots', folder_name)
plot_by_return_types()


