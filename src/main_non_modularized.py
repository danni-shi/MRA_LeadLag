"""compare clustering performance of heterogeneous optimization of MRA and spectral clustering;
Compare the accuracy of lag recovery between the two methods
"""
#======== imports ===========
import numpy as np
import pickle
from tqdm import tqdm
import os
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.io as spio
# from scipy.linalg import block_diag

import utils
import alignment

test = False
# ======= initialisation ===========
# intialise parameters
if test:
    sigma_range = np.arange(1.4,2.0,0.1) # std of random gaussian noise
    K_range = [2]
    n = 500
else:
    sigma_range = np.arange(0.1,2.1,0.1) # std of random gaussian noise
    K_range = [2,3,4]
    n = 500 # number of observations we evaluate
max_shift= 0.1 # max proportion of lateral shift
assumed_max_lag = 10
# data path
data_path = '../../data/data500_logreturns_init3/'

labels = ['pairwise', 'sync', 'spc-homo', 'het']
metrics = ['error', 'error_sign', 'accuracy', 'errors_list']
performance = {}
signal_estimates = {}
classes_estimates = {}
p_estimates = {}
estimates = {}
#===== calculatations over a grid of K and sigma ===========

for k in tqdm(K_range):
    # iniitialise containers
    
    performance[f'K={k}'] = {'ARI'     : {'spc': [],
                                          'het': []}}
    for metric in metrics:
        performance[f'K={k}'][metric] = {label:[] for label in labels}
    
    estimates[f'K={k}'] = {}
    
    j = 0
    
    for sigma in tqdm(sigma_range):
        # read data produced by matlab
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
        observations = observations_mat['data'][:,:n]
        shifts = observations_mat['shifts'].flatten()[:n]
        classes_true = observations_mat['classes'].flatten()[:n] - 1
        X_est = results_mat['x_est']
        P_est = results_mat['p_est'].flatten()
        X_true = results_mat['x_true']

        if j == 0:
            print('length and size of observations: ', observations.shape)
            j += 1
        
        #--------- Clustering ----------#
        
        # baseline clustering method, obtain lag matrix from pairwise CCF
        affinity_matrix, lag_matrix = alignment.score_lag_mat(observations, max_lag = assumed_max_lag, score_fn=alignment.alignment_similarity)
        SPC = SpectralClustering(n_clusters=k,
                                affinity = 'precomputed',
                                random_state=0).fit(affinity_matrix)
        
        # compare baseline and IVF clustering
        classes_spc = SPC.labels_
        classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
        classes_spc_aligned = utils.align_classes(classes_spc,classes_true)
        classes_est_aligned = utils.align_classes(classes_est,classes_true)
        assert np.sum(classes_spc_aligned==classes_true) >= np.sum(classes_spc==classes_true)
        assert np.sum(classes_est_aligned==classes_true) >= np.sum(classes_est==classes_true)
        classes_spc = classes_spc_aligned
        classes_est = classes_est_aligned
        
        performance[f'K={k}']['ARI']['spc']\
            .append(adjusted_rand_score(classes_true, classes_spc))
        performance[f'K={k}']['ARI']['het']\
            .append(adjusted_rand_score(classes_true, classes_est))

        #----- Evaluate the lag estimation methods -----#
        
        # ground truth pairwise lag matrix
        lag_mat_true = alignment.lag_vec_to_mat(shifts)
        # error_penalty = int(observations.shape[0]/2)
        error_penalty = 0

        # SPC + pairwaise correlation-based lags
        results_pair = alignment.eval_lag_mat_het(lag_matrix, lag_mat_true, \
                                                classes_spc, classes_true,\
                                                error_penalty)

        # SPC + synchronization
        X_est_sync = alignment.get_synchronized_signals(observations, classes_spc, lag_matrix)
        results_sync = \
            alignment.eval_alignment_het(observations, lag_mat_true, \
                                         classes_spc, classes_true, \
                                        X_est_sync, penalty=error_penalty, \
                                        max_lag = assumed_max_lag)

        # SPC + homogeneous optimization
        results_spc = \
            alignment.eval_alignment_het(observations, lag_mat_true, \
                                        classes_spc, classes_true,\
                                        sigma = sigma, penalty=error_penalty, \
                                        max_lag = assumed_max_lag)
        X_est_spc = results_spc[-1] 
        results_spc = results_spc[:-1]

        # heterogeneous optimization
        results_het = \
            alignment.eval_alignment_het(observations, lag_mat_true, \
                                        classes_est, classes_true, \
                                        X_est, penalty=error_penalty, \
                                        max_lag = assumed_max_lag)

        results_list = [results_pair,results_sync,results_spc,results_het]
        
        for i in range(len(metrics)):
            for j in range(len(labels)):
                label = labels[j]
                metric = metrics[i]
                metric_result = results_list[j][i]
                performance[f'K={k}'][metric][label].append(metric_result)
        # aligned the estimated signals and mixing probabilities to the ground truth
        X_est_sync_aligned, perm = utils.align_to_ref_het(X_est_sync, X_true)
        
        X_est_spc_aligned, perm = utils.align_to_ref_het(X_est_spc, X_true)
        prob_spc = utils.mixing_prob(classes_spc, k)
        prob_spc = np.array([prob_spc[i] for i in perm])

        # reminder that the estimations from hetergeneous IVF method is alreadly aligned with the truth
        prob_het_reasigned = utils.mixing_prob(classes_est, k)    
        
        # true mxiing probabilities
        P_true = [np.mean(classes_true == c) for c in np.unique(classes_true)]
        
        # record estimations for each K and sigma
        estimates[f'K={k}'][f'sigma={sigma:.2g}'] = {
            'signals': {'true': X_true,
                        'sync': X_est_sync_aligned,
                        'spc': X_est_spc_aligned,
                        'het':X_est
                        } ,
            'classes': {'true': classes_true,
                        'spc': classes_spc,
                        'het': classes_est  
                        },
            'probabilities': {'true': P_true,
                            'spc': prob_spc,
                            'het reassigned': prob_het_reasigned,
                            'het': P_est
                             }
        }


    # store results
    # performance[f'K={k}'] = {
    #                     'ARI'     : {'spc': ARI_list_spc,
    #                                 'het': ARI_list},
    #                     'error'   : {'pairwise': error_list_pair,
    #                                  'sync': error_list_sync,
    #                                 'spc-homo': error_list_spc,
    #                                 'het': error_list_het},
    #                     'Errors'  : {'pairwise': Errors_pair,
    #                                  'sync': Errors_sync,
    #                                 'spc-homo': Errors_spc,
    #                                 'het': Errors_het},
    #                     'accuracy': {'pairwise': acc_list_pair,
    #                                  'sync': acc_list_sync,
    #                                 'spc-homo': acc_list_spc,
    #                                 'het': acc_list_het}      
    #                         }
    
# save the results
with open('../results/performance.pkl', 'wb') as f:   
        pickle.dump(performance, f)

with open('../results/estimates.pkl', 'wb') as f:   
    pickle.dump(estimates, f)

# with open('../results/clustering.pkl', 'rb') as f:   
#         result = pickle.load(f)
    
#======== plot the results ==================

# labels = {'pairwise': 'SPC',
#           'sync': 'SPC-Synchronization',
#             'spc': 'SPC-IVF',
#             'het': 'IVF',
#             'true': 'True'}
# color_labels = labels.keys()
# col_values = sns.color_palette('Set2')
# color_map = dict(zip(color_labels, col_values))

# lty_map = {'sync': 'dotted',
#             'spc': 'dashdot',
#             'het': 'dashed',
#             'true': 'solid'}

# results_save_dir = utils.save_to_folder('../plots/SPC_cluster', '')
# for k in K_range: 
#     fig, axes = plt.subplots(3,1, figsize = (15,18), squeeze=False)
#     ax = axes.flatten()
#     plot_list = ['ARI', 'error', 'accuracy']
    
#     for i in range(len(plot_list)):
#         for key, result_list in performance[f'K={k}'][plot_list[i]].items():
#             ax[i].plot(sigma_range, result_list, label = labels[key], color = color_map[key])
#         ax[i].grid()
#         ax[i].legend()
#         ax[i].set_xlabel('std of added noise')
#     ax[0].set_title(f'Ajusted Rand Index of clustering against noise level, K = {k}')
#     ax[1].set_title(f'Change of Alignment Error with Noise Level')
#     ax[2].set_title(f'Change of Alignment Accuracy with Noise Level')
#     plt.savefig(results_save_dir + f'/acc_err_ARI_K={k}')
    
    # # plot signal estimates
    # results_save_dir_2 = results_save_dir + f'/signal_estimates_K={k}'
    # os.makedirs(results_save_dir_2)
    # # plot the difference of estimated signals
    # fig, ax = plt.subplots(k, 1, figsize = (10,5*k))
    # for key, X_estimates in result[f'K={k}']['signals'].items():
    #     if key != 'true':
    #         X_estimates, perm = utils.align_to_ref_het(X_estimates, X_true)
        
    #     rel_errors = []
    #     for j in range(k):
    #         rel_err = np.linalg.norm(X_estimates[:,j]-X_true[:,j])/np.linalg.norm(X_true[:,j])
    #         rel_errors.append(rel_err)
    #         p_true = np.sum(classes_true==j)/observations.shape[1]
    #         ax[j].plot(X_estimates[:,j], label = labels[key], color = color_map[key], linestyle = lty_map[key])
    # # color_map = dict(zip(color_labels, col_values))
    # # for j in range(k):
    # #     ax[j].set_title(f'rel. err.:  hetero {rel_err_hetero:.2f}; '\
    # #                     f'spc {rel_err_spc:.2f}; '\
    # #                     f'true p: {p_true:.2f}, '\
    # #                     f'est. p: {P_est[i]:.2f}')
    #     ax[j].grid()
    #     ax[j].legend()

    #     fig.suptitle(f'Comparison of the True and Estimated Signals, K = {k}, noise = {sigma:.2g}')
    #     plt.savefig(results_save_dir_2 + f'/sigma={sigma:.2g}')
    
    



### If we only want clustering results 

# for k in K_range:
#     # iniitialise containers
#     ARI_list = []
#     ARI_list_spc = []
    
#     for sigma in tqdm(sigma_range):
#         # read data produced by matlab
#         observations_path = data_path + '_'.join(['observations', 
#                                       'noise'+f'{sigma:.2g}', 
#                                       'shift'+str(max_shift), 
#                                       'class'+str(k)+'.mat'])
#         results_path = data_path + '_'.join(['results', 
#                                       'noise'+f'{sigma:.2g}', 
#                                       'shift'+str(max_shift), 
#                                       'class'+str(k)+'.mat'])
#         observations_mat = spio.loadmat(observations_path)
#         results_mat = spio.loadmat(results_path)
#         observations = observations_mat['data']
#         shifts = observations_mat['shifts'].flatten()
#         classes_true = observations_mat['classes'].flatten() - 1
#         X_est = results_mat['x_est']
#         P_est = results_mat['p_est'].flatten()

#         # baseline clustering method
#         residuals, lags = utils.residual_lag_mat(observations)
#         delta = 1
#         affinity_matrix = np.exp(- residuals ** 2 / (2. * delta ** 2))

#         SPC = SpectralClustering(n_clusters=k,
#                                 affinity = 'precomputed',
#                                 random_state=0).fit(affinity_matrix)
#         classes_spc = SPC.labels_
#         classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
#         ARI_list_spc.append(adjusted_rand_score(classes_true, classes_spc))
#         ARI_list.append(adjusted_rand_score(classes_true, classes_est))

#     fig, axes = plt.subplots( figsize = (15,6))
#     ax = axes.flatten()
#     ax.plot(sigma_range, ARI_list, label = 'Assignment to Heterogeneous Signals')
#     ax.plot(sigma_range, ARI_list_spc, label = 'Spectral Clustering')
#     ax.grid()
#     ax.legend()
#     ax.set_title(f'Ajusted Rand Index of clustering against noise level, K = {k}')
#     plt.savefig(results_save_dir + f'/ARI_K={k}_0')
