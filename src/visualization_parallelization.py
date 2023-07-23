import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import seaborn as sns
import pickle
import pandas as pd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# choose what to plot
plot_performance = False
plot_PnL = False
plot_signals = False
plot_signals_real = True


# ----- Check these parameters are in sync with main_non_modularized.py -----#

test = False

if test:
    sigma_range = np.arange(1.5, 1.6, 0.5)  # std of random gaussian noise
    K_range = [2,3,4]
else:
    sigma_range = np.arange(0.1, 2.1, 0.1)  # std of random gaussian noise
    K_range = [1,2,3,4]

num_rounds = 1

###--- create the folder to save plots ---###
# change folder name accroding to experiment specications
synthetic_results_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/results/synthetic'
folder_name = 'test'
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

lty_map = {'sync': 'dotted',
           'spc-homo': 'dashdot',
           'het': 'dashed',
           'true': 'solid'}


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
        return [sum(x) / len(x) for x in zip(*dicts)]


# =================== plot the results ==================#
# --- plots of performance of different methods for lag estimatiopn ---#
if plot_performance:
    performance_dicts = []
    for round in range(1, 1 + num_rounds):
        # read saved results
        with open(f'{synthetic_results_path}/8rounds/performance/{round}.pkl', 'rb') as f:
            performance = pickle.load(f)
        performance_dicts.append(performance)


    for round in range(1, 1 + num_rounds):
        # read saved results
        with open(f'{synthetic_results_path}/K4_8rounds/performance/{round}.pkl', 'rb') as f:
            performance = pickle.load(f)
        performance_dicts[round-1].update(performance)
    # obtain the performance dictionary as the mean results of all parallel runs
    performance = mean_dict(performance_dicts)

    for k in K_range:
        fig, axes = plt.subplots(3, 1, figsize=(4, 7.5), squeeze=False)
        plot_list = ['accuracy', 'error_sign']
        # if k != 1:
        #     fig, axes = plt.subplots(4, 1, figsize=(4, 10), squeeze=False)
        #     plot_list = ['ARI', 'accuracy', 'error_sign']
        # else:
        #     fig, axes = plt.subplots(3, 1, figsize=(4, 7.5), squeeze=False)
        #     plot_list = ['accuracy', 'error_sign']
        ax = axes.flatten()
        for i in range(len(plot_list)):
            for key, result_list in performance[f'K={k}'][plot_list[i]].items():
                if (plot_list[i] == 'ARI') and (key == 'het'):
                    label = 'IVF'
                else:
                    label = labels[key]
                if (key != 'het') or (k != 1):
                    ax[i].plot(sigma_range, result_list, label=label,
                               color=color_map[key], linewidth=2)
            ax[i].grid()
            if plot_list[i] in ['accuracy','ARI']:
                ax[i].legend(loc='lower left')
            else:
                ax[i].legend(loc='upper left')
            # ax[i].set_xlabel('std of added noise')

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
                ax[-1].plot(sigma_range, mean, label=labels[key],
                            color=color_map[key], linewidth=2, linestyle=linestyle)
                ax[-1].fill_between(sigma_range, quantile_u, quantile_l, color=color_map[key], alpha=0.3)
                                   # label=f'{labels[key]}: {quantile[0]:.0%} to {quantile[1]:.0%}')
                ax[-1].set_ylim(top=3,bottom=0)
                ax[-1].grid(visible=True)
                ax[-1].legend(loc='upper left')
                ax[-1].set_xlabel('std of added noise')

        if k != 1:
            ax[0].set_title(f'Ajusted Rand Index')
            ax[-3].set_ylim(top=1, bottom=0)
        ax[-3].set_title(f'Accuracy')
        ax[-2].set_title(f'Average Directional Error')
        ax[-3].set_ylim(top=100, bottom=30)
        ax[-2].set_ylim(top=1.5, bottom=0)
        ax[-1].set_title('Average Lag Error')
        # ax[-1].set_title(
        #     f'Average Lag Error (shaded area is the {quantile[0]:.0%} to {quantile[1]:.0%} percentile of errors)')

        plt.tight_layout()
        plt.savefig(results_save_dir + f'/acc_err_ARI_K={k}')
### --- plots of PnL of different methods from trading strategy based on lead-lag relationship ---#
if plot_PnL:
    trading_results_list = []
    for round in range(1, 1 + num_rounds):
        with open(f'{synthetic_results_path}/{folder_name}/PnL/{round}.pkl', 'rb') as f:
            trading_results = pickle.load(f)
            trading_results_list.append(trading_results)

    PnL_sigma_range = np.arange(0.5,2.1,0.5)
    K_range = [1,3]
    # for k in K_range:
    #     fig, axes = plt.subplots(len(PnL_sigma_range), num_rounds, figsize=(8*num_rounds, 5*len(PnL_sigma_range)),squeeze=False,sharey=True
    #                              )
    #     for i, sigma in enumerate(PnL_sigma_range):
    #         for j in range(num_rounds):
    #             ax = axes[i, j]
    #             ax.set_title(f'Sigma = {sigma}, round {j+1}')
    #             ax.set_xlabel('Days')
    #             ax.set_ylabel('PnL')
    #             for model, values in PnL_list[j][f'K={k}'][f'sigma={sigma:.2g}'].items():
    #                 cum_pnl = np.append(np.zeros(1), np.cumsum(values))
    #                 ax.plot(np.arange(len(values) + 1), cum_pnl, label=labels[model], color=color_map[model])
    #             ax.legend(loc='lower right')
    #
    #     plt.savefig(results_save_dir + f'/PnL_K={k}')

    models = ['pairwise', 'sync', 'spc-homo', 'het']
    for m in range(num_rounds):
        for k in K_range:
            fig, axes = plt.subplots(len(PnL_sigma_range), len(models),
                                     figsize=(8 * len(models), 5 * len(PnL_sigma_range)),
                                     squeeze=False, sharey=True)
            for i, sigma in enumerate(PnL_sigma_range):
                for j, model in enumerate(models):
                    ax = axes[i, j]
                    ax.set_title(f'Sigma = {sigma}, model {model}')
                    ax.set_xlabel('Days')
                    ax.set_ylabel('PnL')
                    returns_dict = trading_results_list[m][f'K={k}']\
                        [f'sigma={sigma:.2g}'][model]['portfolio average']
                    for group, values in returns_dict['PnL'].items():
                        # cumsum evaluate the returns of a portfolio of a constant volume
                        cum_pnl = np.cumsum(values)
                        # show the average PnL in red and Bold
                        if group == 'average':
                            plot_config ={'linestyle': 'solid',
                                          'color': 'red',
                                          'linewidth': 2}
                        else:
                            plot_config = {'linestyle': 'dashed'}
                        SR = returns_dict['annualized SR'][group]
                        mean_PnL = np.mean(values) # include the first few days when no trading occurs
                        ax.plot(np.arange(len(values)), cum_pnl, label=f'{group}: SR {SR:.3g}; mean PnL {mean_PnL:.3g}', **plot_config)
                    ax.legend(loc='lower right')
                    ax.grid(visible=True)
            plt.savefig(results_save_dir + f'/PnL_K={k}_round{m+1}')

###--- plots of signal estimates of different methods ---###
K_range = [1,2]
sigma_range = np.arange(0.5,2.1,0.5)

if plot_signals:
    for round in range(1, 1 + num_rounds):

        with open(f'../results/synthetic/10rounds/signal_estimates/{round}.pkl', 'rb') as f:
            estimates = pickle.load(f)

        # --- plots of signal estimates of different methods ---#
        results_save_dir_2 = results_save_dir + f'/signal_estimates{round}'
        os.makedirs(results_save_dir_2)

        # align the estimated signals to ground truth signals
        for k in K_range:
            os.makedirs(results_save_dir_2 + f'/K={k}')
            i = 0
            for sigma in sigma_range:
                # # calculate the estimated mixing probabilities

                estimates_i = estimates[f'K={k}'][f'sigma={sigma:.2g}']
                fig, ax = plt.subplots(k, 1, figsize=(10, 5 * k),squeeze=False)
                X_true = estimates_i['signals']['true']

                for j in range(k):
                    rel_errors_str = []
                    p_est_str = []
                    for key, X_estimates in estimates_i['signals'].items():
                        ax[j,0].plot(X_estimates[:, j],
                                   label=labels[key],
                                   color=color_map[key],
                                   linestyle=lty_map[key])

                    if key != 'true':
                        rel_err = np.linalg.norm(X_estimates[:, j] - X_true[:, j]) / np.linalg.norm(X_true[:, j])
                        rel_errors_str.append(f'{labels[key]} {rel_err:.3f}')

                    for key_p, value_p in estimates_i['probabilities'].items():
                        p_est_str.append(f'{labels[key_p]} {value_p[j]:.3g}')

                    title = 'rel. err.: ' + ';'.join(rel_errors_str)
                    if k != 1:
                        title = title + '\nmix. prob.: ' + '; '.join(p_est_str)
                    ax[j,0].set_title(title)
                    ax[j,0].grid()
                    ax[j,0].legend()

                fig.suptitle(f'Comparison of the True and Estimated Signals, K = {k}, noise = {sigma:.2g}')
                plt.tight_layout()
                plt.savefig(results_save_dir_2 + f'/K={k}/{int(i)}')
                plt.close()
                i += 1

###--- plots of signal estimates of different methods ---###
if plot_signals_real:
    K_range=[1,2,3]
    sigma_range = np.arange(0.2,2.1,0.2)
    results_path = '../results/real/full_non-negative_affinity'
    start = 1905
    end = 2646
    retrain_period = 10
    signal_length = 50
    start_indices = range(start, end, 5*retrain_period)
    df_K_sigma = pd.read_csv(results_path+'/PnL_real_single_weighted/best_k_sigma.csv', index_col=0).applymap(eval)
    for start_index in start_indices:
        end_index = start_index + signal_length
        with open(f'{results_path}/signal_estimates/start{start_index}end{end_index}.pkl', 'rb') as f:
            estimates = pickle.load(f)
        models = ['sync', 'spc-homo', 'het']
        # --- plots of signal estimates of different methods ---#
        results_save_dir_2 = results_save_dir + f'/signals/start{start_index}end{end_index}'
        os.makedirs(results_save_dir_2)
        ind = f'({start_index}, {end_index})'
        for model in models:
            k, sigma = df_K_sigma.loc[ind,model]
            estimates_i = estimates[f'K={k}'][f'sigma={sigma:.2g}']
            fig, ax = plt.subplots(k, 1, figsize=(10, 5 * k), squeeze=False)
            ax = ax.flatten()

            X_estimates = estimates_i['signals'][model]
            # K, sigma = df_K_sigma.iloc[row_num, col_num]
            for j in range(k):
                ax[j].plot(X_estimates[:, j],
                           label=labels[model],
                           color=color_map[model],
                           linestyle=lty_map[model])

                        # if key != 'true':
                        #     rel_err = np.linalg.norm(X_estimates[:, j] - X_true[:, j]) / np.linalg.norm(X_true[:, j])
                        #     rel_errors_str.append(f'{labels[key]} {rel_err:.3f}')

                p_est_str = []
                for key_p, value_p in estimates_i['probabilities'].items():
                    p_est_str.append(f'{labels[key_p]} {value_p[j]:.3g}')

                title = 'mix. prob.: ' + '; '.join(p_est_str)
                ax[j].set_title(title)
                ax[j].grid()
                ax[j].legend()

            fig.suptitle(f'Comparison of the True and Estimated Signals, K = {k}, noise = {sigma:.2g}')
            plt.savefig(results_save_dir_2 + '/' + model)
            plt.close()
def extract_quantile(q, values):
    """
    Extract quantile from a list of values of percentiles at 0, 5, 10, ..., 100.
    Args:
        q: quantiles to extract, valid inputs are [0, 0.05, 0.1,..., 1]
        values: ndarray of length 21. quantiles values at 0.05 interval

    Returns:

    """
    ind = int(q / 0.05)
    return values[ind]
