import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import datetime as dt
import pandas as pd
import seaborn as sns
import utils

# add 'src/' to path
sys.path.append('../src')

start = 5; end = 5146
retrain_period = 10
signal_length = 50
file_name = f'start{start}end{end}_length{signal_length}_trade{retrain_period}'
prediction_folder = 'full_non-negative_affinity'
with open(f'../results/real/{prediction_folder}/PnL_real_single_weighted/' + file_name + '.pkl',
          'rb') as f:
    PnL_SR = pickle.load(f)


def SR_by_year():
    first_day_cols = []
    for col_number, index_value in enumerate(df_pvCLCL.columns):
        if index_value.year != df_pvCLCL.columns[col_number - 1].year:
            first_day_cols.append(col_number)
    pd.DataFrame({'first day index': first_day_cols}, index=np.unique(df_pvCLCL.columns.year)).T


K_range = [1, 2, 3]
PnL_sigma_range = np.arange(0.2, 2.0, 0.4)
PnL_sigma_range = [round(PnL_sigma_range[i], 1) for i in range(len(PnL_sigma_range))]
models = ['pairwise', 'sync', 'spc-homo', 'het']
# dates
trading_dates = pd.read_csv('../data/pvCLCL_clean_winsorized.csv',index_col=0).columns
trading_dates = list(map(lambda x: dt.datetime.strptime(x,'X%Y%m%d'), trading_dates))
trading_dates = trading_dates[start+signal_length:end-1+signal_length+retrain_period]
# map return types to labels on plots
RT_map = {'raw returns': 'RR-Lag',
                   'mkt excess returns': 'MR-Lag',
                   'leader excess returns': 'RR-Lag - RR-Lead',
                   'leader raw returns': 'RR-Lead',
                   'leader mkt excess returns': 'MR-Lead'
                   }

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
            ax.plot(np.arange(len(values)), cum_pnl, label=f'{RT_map[return_type]}: SR {SR:.2f}; PnL {1e2*mean_PnL:.4f}',
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
        ax.set_xlabel('Date')
        ax.set_ylabel('PnL')
        returns_dict = {metric: {model: PnL_SR[model][metric][return_type] \
                                 for model in models} \
                        for metric in ['PnL', 'annualized SR']}

        for model, values in returns_dict['PnL'].items():
            # cumsum evaluate the returns of a portfolio of a constant volume
            values[np.isnan(values)] = 0
            cum_pnl = np.cumsum(values)
            SR = returns_dict['annualized SR'][model]
            mean_PnL = np.mean(values)  # include the first few days when no trading occurs
            ax.plot(trading_dates, cum_pnl,
                    label=f'{labels[model]}: SR {SR:.2f}; Ave. PnL {1e2 * mean_PnL:.4f}',
                    color=color_map[model])
            ax.legend(loc='lower right')
            ax.grid(visible=True)
        plt.savefig(results_save_dir + f'/{label}')

def plot_MR_RR():
    fig, axes = plt.subplots(
        2,1,
        figsize=(10, 10),
        squeeze=False, sharey=True)
    for i, return_type in enumerate(['raw returns', 'mkt excess returns']):
        ax = axes[i,0]
        ax.set_title(return_type)
        ax.set_xlabel('Date')
        ax.set_ylabel('PnL')
        returns_dict = {metric: {model: PnL_SR[model][metric][return_type] \
                                 for model in models} \
                        for metric in ['PnL', 'annualized SR']}

        for model, values in returns_dict['PnL'].items():
            # cumsum evaluate the returns of a portfolio of a constant volume
            values[np.isnan(values)] = 0
            cum_pnl = np.cumsum(values)
            SR = returns_dict['annualized SR'][model]
            mean_PnL = np.mean(values)  # include the first few days when no trading occurs
            ax.plot(trading_dates, cum_pnl,
                    label=f'{labels[model]}: SR {SR:.2f}; Ave. PnL {1e2 * mean_PnL:.4f}',
                    color=color_map[model])
            ax.legend(loc='lower right')
            ax.grid(visible=True)
        plt.savefig(results_save_dir + f'/MR_RR_v0.png')

# change folder name accroding to experiment specications
# folder_name = f'{prediction_folder}/trading_OS_PnL_length50_retrain10_centred_optimized'
# results_save_dir = utils.save_to_folder('../plots', folder_name)
results_save_dir = '../plots/2023-07-13-18h27min_full_non-negative_affinity/trading_OS_PnL_length50_retrain10_centred_optimized'
# plot_by_return_types()
plot_MR_RR()

