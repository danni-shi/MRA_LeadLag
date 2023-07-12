"""
Perform the normal clustering on scaled / normalized returns
and save the clustering results and lag matrices based on pairwise lags
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import multiprocessing
import pickle
import time

import scipy.io as spio
from itertools import repeat
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

import alignment
import utils

# def cluster_SPC(observations, k, assumed_max_lag, score_fn=alignment.alignment_similarity):
#     # baseline clustering method, obtain lag matrix from pairwise CCF
#     # affinity_matrix, lag_matrix = alignment.score_lag_mat(observations, max_lag=assumed_max_lag,
#     #                                                       score_fn=score_fn)
#     # if (affinity_matrix<0).any():
#     #     print('detected negative similarity')
#     # affinity_matrix[affinity_matrix<0] = 0
#     SPC = SpectralClustering(n_clusters=k,
#                              affinity='precomputed',
#                              random_state=0).fit(affinity_matrix)
#
#     # compare baseline and IVF clustering
#     classes_spc = SPC.labels_
#     return classes_spc, lag_matrix
def cluster_SPC(affinity_matrix, k):
    # baseline clustering method, obtain lag matrix from pairwise CCF
    SPC = SpectralClustering(n_clusters=k,
                             affinity='precomputed',
                             random_state=0).fit(affinity_matrix)

    # compare baseline and IVF clustering
    classes_spc = SPC.labels_
    return classes_spc
def clustering(observations, k, assumed_max_lag, X_est, classes_true=None, score_fn=alignment.alignment_similarity):
    # --------- Clustering ----------#
    affinity_matrix, lag_matrix = alignment.score_lag_mat(observations, max_lag=assumed_max_lag,
                                                          score_fn=score_fn)
    # affinity_matrix = np.exp(affinity_matrix)
    classes_spc = cluster_SPC(affinity_matrix, k)
    classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)

    if classes_true is None:
        classes_est = utils.align_classes(classes_est, classes_spc)

        return classes_spc, classes_est, lag_matrix

    else:
        classes_spc_aligned = utils.align_classes(classes_spc, classes_true)
        classes_est_aligned = utils.align_classes(classes_est, classes_true)
        assert np.sum(classes_spc_aligned == classes_true) >= np.sum(classes_spc == classes_true)
        assert np.sum(classes_est_aligned == classes_true) >= np.sum(classes_est == classes_true)
        classes_spc = classes_spc_aligned
        classes_est = classes_est_aligned

        ARI_dict = {'spc': adjusted_rand_score(classes_true, classes_spc),
                    'het': adjusted_rand_score(classes_true, classes_est)}

        return classes_spc, classes_est, lag_matrix, ARI_dict
def clustering_real_data(K_range=[1,2,3], assumed_max_lag=5,
                  start_index=0, signal_length=50, scale_method='normalized',
                         data_path='please check data_path',
                         save_path='please check save_path'):
    """
    For time series of selected range, perform SPC clustering for each of the given number of classes in K_range
    """

    # save parameters only once
    params_save_dir = f'{save_path}/params_clustering.json'
    if not os.path.exists(params_save_dir):
        params = dict(K_range=K_range,
                      signal_length=signal_length,
                      scale_method=scale_method,
                      assumed_max_lag=assumed_max_lag,
                      data_path=data_path,
                      save_path=save_path)
        with open(params_save_dir, 'w') as json_file:
            json.dump(params, json_file)

    # read data
    end_index = start_index + signal_length
    data = pd.read_csv(data_path, index_col=0).iloc[:, start_index:end_index]
    # if start_index == 5145:
    #     data.drop('TIF', axis=0, inplace=True)
    if scale_method == 'normalized': # normalized by subtracting the mean and scale by std
        obs = utils.normalize_by_column(np.array(data.T))
    elif scale_method == 'scaled': # scale the observation by std
        obs = np.array(data.T) / np.std(np.array(data.T), axis=0)  # do not subtract the mean
    classes = {f'K{k}': {} for k in K_range} # special key name for MATLAB
    # lag_matrices = {f'K={k}': {} for k in K_range}
    affinity_matrix, lag_matrix = alignment.score_lag_mat(obs, max_lag=assumed_max_lag,
                                                          score_fn=alignment.alignment_similarity)
    # affinity_matrix = np.exp(affinity_matrix)
    print(f'start index:{start_index}\npercentage of neg. similarities: {np.sum(affinity_matrix<0)/affinity_matrix.size:.1%}') 
    affinity_matrix[affinity_matrix<0] = 0
    for k in K_range:
        # SPC clustering and obtain lag_matrix from pairwise CCF
        classes_spc = cluster_SPC(affinity_matrix, k)
        # make sure the labels are continuous starting from 0
        classes_spc = np.unique(classes_spc, return_inverse=True)[1]
        classes[f'K{k}'] = classes_spc
        # lag_matrices[f'K={k}'] = lag_matrix

    # with open(save_path + f'/classes/start{start_index}end{end_index}.pkl', 'wb') as f:
    #     pickle.dump(classes, f)
    spio.savemat(save_path + f'/classes/start{start_index}end{end_index}.mat', classes)
    with open(save_path + f'/lag_matrices_pairwise/start{start_index}end{end_index}.pkl', 'wb') as f:
        pickle.dump(lag_matrix, f)

def clustering_real_data_wrapper(inputs):
    start_index, save_path = inputs
    # print(f'Running period from day {start_index} to {start_index+50}')
    clustering_real_data(K_range=[1,2,3],
                         assumed_max_lag=2,
                         start_index=start_index,
                         signal_length=50,
                         data_path='../data/pvCLCL_clean_winsorized.csv',
                         save_path=save_path
                         )
    # progress_bar.update(1)
if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser(description="iDAD: Hidden Object Detection.")
    parser.add_argument("--start", default=5, type=int)
    parser.add_argument("--end", default=5145, type=int)
    parser.add_argument("--retrain-period", default=10, type=int)
    parser.add_argument("--folder-name", default='clustering_full', type=str)
    parser.add_argument('--use-save-path', action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-path", default='../results/real/test', type=str)
    parser.add_argument("--parallelize", action=argparse.BooleanOptionalAction)
    

    args = parser.parse_args()
    
    # set the range and retrain period of time series data
    start = args.start
    end = args.end
    retrain_period = args.retrain_period
    use_save_path = args.use_save_path
    parallelize = args.parallelize
    
    if use_save_path:
        save_path = args.save_path
        utils.create_folder_if_not_existed(save_path)
    else:
        folder_name = args.folder_name
        save_path = utils.save_to_folder('../results/real', folder_name)
    
    start_indices = range(start, end, retrain_period)
    inputs = list(zip(start_indices, repeat(save_path)))
    # create folders to store results
    utils.create_folder_if_not_existed(save_path + '/classes')
    utils.create_folder_if_not_existed(save_path + '/lag_matrices_pairwise')

    # start clustering with multiprocessing
    start_time = time.time()
    if parallelize:
        # map inputs to functions
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # use the pool to apply the worker function to each input in parallel
            _ = list(tqdm(pool.imap(clustering_real_data_wrapper, inputs),
                        total=len(start_indices)))
            # pool.starmap(clustering_real_data_wrapper, inputs)
            pool.close()
            pool.join()
    else:
        for start_index in tqdm(start_indices):
            clustering_real_data_wrapper((start_index, save_path))

    print(f'time taken to run {len(start_indices)} predictions: {time.time() - start_time:.1f}')

   # save more params, important that we run this after clustering
   #  params_save_dir = f'{save_path}/params_clustering.json'
   #  with open(params_save_dir, 'r') as json_file:
   #      params = json.load(json_file)
   #  params['start'] = start
   #  params['end'] = end
   #  params['retrain_period'] = retrain_period
   #  with open(params_save_dir, 'w') as json_file:
   #      json.dump(params, json_file)