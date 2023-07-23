import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import seaborn as sns
import pickle
import scipy.io as spio

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

###--- create the folder to save plots ---###
# change folder name accroding to experiment specications
synthetic_results_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/results/synthetic'
folder_name = 'test'
results_save_dir = utils.save_to_folder('../plots', folder_name)


# labels and formats
labels = {'pairwise': 'S-Pairwise',
          'sync': 'Estimated',
          'het reassigned': 'IVF (regroup)',
          'true': 'True'}

color_labels = labels.keys()
col_values = sns.color_palette('hls')
color_map = dict(zip(color_labels, col_values))

lty_map = {'sync': 'solid',
           'spc-homo': 'dashdot',
           'het': 'dashed',
           'true': 'solid'}


data_path = '../data/data500_shift2_pvCLCL_init2_set1/1/'
max_shift = 2
rounds = [1]

###--- plots of signal estimates of different methods ---###
K_range = [1]
sigma_range = [0.5,2.0]

for round in rounds:

    with open(f'../results/synthetic/10rounds/signal_estimates/{round}.pkl', 'rb') as f:
        estimates = pickle.load(f)

    # --- plots of signal estimates of different methods ---#
    results_save_dir_2 = results_save_dir + f'/signal_estimates{round}'
    utils.create_folder_if_not_existed(results_save_dir_2)

    # align the estimated signals to ground truth signals
    for k in K_range:
        utils.create_folder_if_not_existed(results_save_dir_2 + f'/K={k}')
        i = 0
        s=-1
        fig, ax = plt.subplots(k, 2, figsize=(5 * 2, 4 * k), squeeze=False, sharey=True)
        for sigma in sigma_range:
            s+=1
            observations_path = data_path + '_'.join(['observations',
                                                      'noise' + f'{sigma:.2g}',
                                                      'shift' + str(max_shift),
                                                      'class' + str(k) + '.mat'])
            results_path = data_path + '_'.join(['results',
                                                 'noise' + f'{sigma:.2g}',
                                                 'shift' + str(max_shift),
                                                 'class' + str(k) + '.mat'])
            observations_mat = spio.loadmat(observations_path)
            results_mat = spio.loadmat(results_path)

            shifts = observations_mat['shifts'].flatten()
            index = [np.where(shifts==i)[0][0] for i in range(max_shift+1)]
            observations = observations_mat['data_train'][:,index]

            # # calculate the estimated mixing probabilities

            estimates_i = estimates[f'K={k}'][f'sigma={sigma:.2g}']

            X_true = estimates_i['signals']['true']

            for j in range(k):
                rel_errors_str = []
                p_est_str = []
                # ax[j, 0].plot(estimates_i['signals']['sync'][:, j],
                #               label=labels['sync'],
                #               color=color_map['sync'],
                #               linestyle=lty_map['sync'])
                # ax[j, 0].plot(estimates_i['signals']['true'][:,j],
                #               label=labels['true'],
                #               color=color_map['true'],
                #               linestyle=lty_map['true'])

                for key in ['true', 'sync']:
                    X_estimates = estimates_i['signals'][key]
                    ax[j,s].plot(X_estimates[:30, j],
                               label=labels[key],
                               linestyle=lty_map[key])
                for l in range(max_shift+1):
                    ax[j, s].plot(observations[:30,l],alpha=0.5,linestyle='dotted', label = f'shift {l}')

                # if key != 'true':
                #     rel_err = np.linalg.norm(X_estimates[:, j] - X_true[:, j]) / np.linalg.norm(X_true[:, j])
                #     rel_errors_str.append(f'{labels[key]} {rel_err:.3f}')

                # for key_p in ['true', 'sync']:
                #     value_p = estimates_i['probabilities'][key]
                #     p_est_str.append(f'{labels[key_p]} {value_p[j]:.3g}')

                # title = 'rel. err.: ' + ';'.join(rel_errors_str)
                title = f'Homogeneous MRA, $\sigma$ = {sigma:.2g}'
                if k != 1:
                    title = title + '\nmix. prob.: ' + '; '.join(p_est_str)
                ax[j,s].set_title(title)
                ax[j,s].set_xlabel('$t$')
                ax[j,s].legend()
                ax[j,s].set_ylim(top=7,bottom=-7)
            ax[j, 0].set_ylabel('$x(t)$')
            # fig.suptitle(f'MRA K = {k}, noise = {sigma:.2g}')
        plt.tight_layout()
        plt.savefig(results_save_dir_2 + f'/K={k}/{int(i)}')
        plt.close()
