"""compare clustering performance of heterogeneous optimization of MRA and spectral clustering;
Compare the accuracy of lag recovery between the two methods
"""
# ======== imports ===========#

import numpy as np
import pickle

import pandas as pd
from tqdm import tqdm
import os


import scipy.io as spio
import multiprocessing
import time
import json
from itertools import repeat
import utils
from clustering import cluster_SPC, clustering
import alignment
import trading


def read_data(data_path, sigma, max_shift, k, n=None):
    # read data produced by matlab
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
    observations = observations_mat['data_train'][:, :n]
    shifts = observations_mat['shifts'].flatten()[:n]
    classes_true = observations_mat['classes'].flatten()[:n] - 1
    X_est = results_mat['x_est']
    P_est = results_mat['p_est'].flatten()
    X_true = results_mat['x_true_train']

    return observations, shifts, classes_true, X_est, P_est, X_true

def eval_models(lag_matrix, shifts=None, assumed_max_lag=5, \
                models=['pairwise', 'sync', 'spc-homo', 'het'],
                observations=None,
                classes_true=None,
                classes_spc=None,
                classes_est=None,
                X_est_spc_homo=None,
                X_est=None,
                sigma=None,
                return_signals=False,
                return_lag_vec=True,
                return_lag_mat=False,
                return_PnL=False,
                **trading_kwargs):
    # if shifts if None meannign we are running real data and have no ground truth
    results_dict = {}
    signal_dict = {}
    lag_mat_dict = {}
    lag_vec_dict = {}
    PnL_dict = {}

    # ----- Evaluate the lag estimation methods -----#

    # ground truth pairwise lag matrix
    if shifts is not None:
        lag_mat_true = alignment.lag_vec_to_mat(shifts)
    # error_penalty = int(observations.shape[0]/2)
    error_penalty = 0

    if 'pairwise' in models:
        lag_mat_dict['pairwise'] = lag_matrix
        if shifts is not None:
            # SPC + pairwise correlation-based lags
            results_pair = alignment.eval_lag_mat_het(
                lag_matrix,
                lag_mat_true,
                classes_spc,
                classes_true,
                error_penalty)
            results_dict['pairwise'] = results_pair

    if 'sync' in models:
        # SPC + synchronization
        X_est_sync = alignment.get_synchronized_signals(
            observations,
            classes_spc,
            lag_matrix)

        lag_mat_sync = alignment.get_lag_matrix_het(
            observations,
            classes_spc,
            X_est_sync,
            assumed_max_lag)
        signal_dict['sync'] = X_est_sync
        lag_mat_dict['sync'] = lag_mat_sync

        if shifts is not None:
            results_sync = alignment.eval_lag_mat_het(
                lag_mat_sync,
                lag_mat_true,
                classes_spc,
                classes_true,
                error_penalty)
            results_dict['sync'] = results_sync

    if 'spc-homo' in models:
        # SPC + homogeneous optimization
        # X_est_spc_homo = alignment.latent_signal_homo(observations, classes_spc, sigma)
        lag_mat_spc_homo = alignment.get_lag_matrix_het(observations, classes_spc, X_est_spc_homo, assumed_max_lag)
        signal_dict['spc-homo'] = X_est_spc_homo
        lag_mat_dict['spc-homo'] = lag_mat_spc_homo
        if shifts is not None:
            results_spc_homo = alignment.eval_lag_mat_het(lag_mat_spc_homo, lag_mat_true, classes_spc, classes_true,
                                                          error_penalty)
            results_dict['spc-homo'] = results_spc_homo

    if 'het' in models:
        # heterogeneous optimization
        lag_mat_het = alignment.get_lag_matrix_het(observations, classes_est, X_est, assumed_max_lag)
        signal_dict['het'] = X_est
        lag_mat_dict['het'] = lag_mat_het
        if shifts is not None:
            results_het = alignment.eval_lag_mat_het(lag_mat_het,
                                                     lag_mat_true,
                                                     classes_est,
                                                     classes_true,
                                                     error_penalty)
            results_dict['het'] = results_het

    for model, lag_mat in lag_mat_dict.items():
        lag_vec_dict[model] = {alignment_type: alignment.lag_mat_to_vec(lag_mat, alignment_type) \
                               for alignment_type in ['row mean', 'SVD']
                               }
    # store results in dictionary
    return_dict = {}

    if shifts is not None:
        return_dict = {'eval': results_dict}
    if return_signals:
        return_dict['signals'] = signal_dict
    if return_lag_mat:
        return_dict['lag mat'] = lag_mat_dict
    if return_lag_vec:
        return_dict['lag vec'] = lag_vec_dict
    if return_PnL:
        for model, lag_mat in lag_mat_dict.items():
            if model == 'het':
                classes = classes_est
            else:
                classes = classes_spc
            PnL_dict[model] = trading.strategy_het(observations, lag_mat, classes, **trading_kwargs)
        return_dict['PnL'] = PnL_dict

    return return_dict


def align_all_signals(X_est_sync, X_est_spc, X_true, classes_spc, classes_est, classes_true, k, X_est, P_est):
    # aligned the estimated signals and mixing probabilities to the reference
    if X_true is None:
        X_ref = X_est
    else:
        X_ref = X_true
    X_est_sync_aligned, perm = utils.align_to_ref_het(X_est_sync, X_ref)
    X_est_spc_aligned, perm = utils.align_to_ref_het(X_est_spc, X_ref)
    prob_spc = utils.mixing_prob(classes_spc, k)
    prob_spc = np.array([prob_spc[i] for i in perm])

    # reminder that the estimations from heterogeneous IVF method is already aligned with the truth
    prob_het_reassigned = utils.mixing_prob(classes_est, k)
    # record estimations for each K and sigma
    signal_class_prob = {
        'signals': {
            'sync': X_est_sync_aligned,
            'spc-homo': X_est_spc_aligned,
            'het': X_est
        },
        'classes': {
            'spc': classes_spc,
            'het': classes_est
        },
        'probabilities': {
            'spc-homo': prob_spc,
            'het reassigned': prob_het_reassigned,
            'het': P_est
        }
    }
    if X_true is not None:
        signal_class_prob['signals']['true'] = X_true
    if classes_true is not None:
        P_true = [np.mean(classes_true == c) for c in np.unique(classes_true)]
        signal_class_prob['classes']['true'] = classes_true
        signal_class_prob['probabilities']['true'] = P_true

    return signal_class_prob


def initialise_containers(K_range, models):
    metrics = ['error', 'error_sign', 'accuracy', 'errors_quantile']

    performance = {}
    estimates = {}
    PnL = {}
    lag_matrices = {}

    for k in K_range:
        performance[f'K={k}'] = {'ARI': {'spc': [],
                                         'het': []}}
        for metric in metrics:
            performance[f'K={k}'][metric] = {model: [] for model in models}

        estimates[f'K={k}'] = {}
        lag_matrices[f'K={k}'] = {}
        PnL[f'K={k}'] = {}
    return performance, estimates


"""
model labels: ['pairwise', 'sync', 'spc-homo', 'het']
"""


# def empty_folders():
#     if os.path.exists('../results/performance'):
#         shutil.rmtree('../results/performance')
#     os.mkdir('../results/performance')
#     if os.path.exists('../results/signal_estimates'):
#         shutil.rmtree('../results/signal_estimates')
#     os.mkdir('../results/signal_estimates')
#     if os.path.exists('../results/lag_matrices'):
#         shutil.rmtree('../results/lag_matrices')
#     os.mkdir('../results/lag_matrices')
#     if os.path.exists('../results/PnL'):
#         shutil.rmtree('../results/PnL')
#     os.mkdir('../results/PnL')

def run(sigma_range=np.arange(0.1, 2.1, 0.1), K_range=None,
        n=None, test=False,
        max_shift=0.04, assumed_max_lag=5,
        models=None, data_path='please check data_path',
        save_path='../results/synthetic',
        return_signals=False, return_lag_mat=False,
        return_PnL=False, round=1):
    # default values
    if models is None:
        models = ['pairwise', 'sync', 'spc-homo', 'het']
    if K_range is None:
        K_range = [2, 3, 4]
    if test:
        sigma_range = np.arange(0.5, 2.1, 0.5)
        K_range = [2]
    metrics = ['error', 'error_sign', 'accuracy', 'errors_quantile']

    # save parameters
    params = dict(sigma_range=list(sigma_range), K_range=K_range,
                  n=n, test=test,
                  max_shift=max_shift, assumed_max_lag=assumed_max_lag,
                  models=models, data_path=data_path,
                  save_path=save_path,
                  return_signals=return_signals, return_lag_mat=return_lag_mat,
                  return_PnL=return_PnL, round=round)
    with open(f'{save_path}/params_round{round}.json', 'w') as json_file:
        json.dump(params, json_file)

    # initialise containers
    performance = {}
    estimates = {}
    PnL = {}
    lag_matrices = {}
    with tqdm(total=len(K_range) * len(sigma_range)) as pbar:
        for k in tqdm(K_range):
            performance[f'K={k}'] = {'ARI': {'spc': [],
                                             'het': []}}
            for metric in metrics:
                performance[f'K={k}'][metric] = {model: [] for model in models}

            estimates[f'K={k}'] = {}
            lag_matrices[f'K={k}'] = {}
            PnL[f'K={k}'] = {}

            for sigma in sigma_range:

                # read data produced from matlab code base
                observations, shifts, classes_true, X_est, P_est, X_true = read_data(
                    data_path=data_path + str(round) + '/',
                    sigma=sigma,
                    max_shift=max_shift,
                    k=k,
                    n=n
                )
                # args_dict = {'observations'}

                # calculate clustering and pairwise lag matrix
                classes_spc, classes_est, lag_matrix, ARI_dict = clustering(observations=observations,
                                                                            k=k,
                                                                            classes_true=classes_true,
                                                                            assumed_max_lag=assumed_max_lag,
                                                                            X_est=X_est
                                                                            )

                # evaluate model performance in lag predictions
                results = eval_models(lag_matrix=lag_matrix,
                                      shifts=shifts,
                                      assumed_max_lag=assumed_max_lag,
                                      classes_true=classes_true,
                                      models=models,
                                      observations=observations,
                                      classes_spc=classes_spc,
                                      classes_est=classes_est,
                                      X_est=X_est,
                                      sigma=sigma,
                                      return_signals=True,
                                      return_lag_mat=return_lag_mat,
                                      return_PnL=return_PnL
                                      )
                # store model performance results in dictionaries

                # clustering performance
                for label, value in ARI_dict.items():
                    performance[f'K={k}']['ARI'][label].append(value)
                # prediction performance
                results_dict = results['eval']
                for i in range(len(metrics)):
                    for model in models:
                        metric = metrics[i]
                        metric_result = results_dict[model][i]
                        performance[f'K={k}'][metric][model].append(metric_result)

                if return_signals:
                    signal_dict = results['signals']
                    # organize signal estimates, classes estimates and mixing prob estimates
                    signal_class_prob = align_all_signals(X_est_sync=signal_dict['sync'],
                                                          X_est_spc=signal_dict['spc-homo'],
                                                          X_true=X_true,
                                                          classes_spc=classes_spc,
                                                          classes_est=classes_est,
                                                          classes_true=classes_true,
                                                          k=k,
                                                          X_est=X_est,
                                                          P_est=P_est
                                                          )
                    # store the signal estimates, classes estimates and mixing prob estimates
                    estimates[f'K={k}'][f'sigma={sigma:.2g}'] = signal_class_prob

                # store the  lag matrices predicted by the models
                if return_lag_mat:
                    lag_mat_dict = results['lag mat']
                    lag_matrices[f'K={k}'][f'sigma={sigma:.2g}'] = lag_mat_dict
                if return_PnL:
                    PnL_dict = results['PnL']
                    PnL[f'K={k}'][f'sigma={sigma:.2g}'] = PnL_dict
                pbar.update(1)

    # save the results to folder
    for subfolder in ['performance', 'signal_estimates', 'lag_matrices', 'PnL']:
        utils.create_folder_if_not_existed(f'{save_path}/{subfolder}')

    with open(f'{save_path}/performance/{round}.pkl', 'wb') as f:
        pickle.dump(performance, f)

    if return_signals:
        with open(f'{save_path}/signal_estimates/{round}.pkl', 'wb') as f:
            pickle.dump(estimates, f)
    if return_lag_mat:
        with open(f'{save_path}/lag_matrices/{round}.pkl', 'wb') as f:
            pickle.dump(lag_matrices, f)
    if return_PnL:
        with open(f'{save_path}/PnL/{round}.pkl', 'wb') as f:
            pickle.dump(PnL, f)


def read_realdata_results(data_path, sigma, k):
    results_path = data_path + '_'.join(['results',
                                         'noise' + f'{sigma:.2g}',
                                         'class' + str(k) + '.mat'])
    results_mat = spio.loadmat(results_path)
    X_est = results_mat['x_est']
    P_est = results_mat['p_est'].flatten()
    X_est_homo = results_mat['x_est_homo']

    return X_est, P_est, X_est_homo


# def clustering_real_data(K_range=[1,2,3], assumed_max_lag=5,
#                   start_index=0, signal_length=50, scale_method='normalized',
#                          data_path='please check data_path',
#                          save_path='please check save_path'):
#     """
#     For time series of selected range, perform SPC clustering for each of the given number of classes in K_range
#     """
#
#     # read data
#     end_index = start_index + signal_length
#     data = pd.read_csv(data_path, index_col=0).iloc[:, start_index:end_index]
#     if scale_method == 'normalized': # normalized by subtracting the mean and scale by std
#         obs = normalize_by_column(np.array(data.T))
#     elif scale_method == 'scaled': # scale the observation by std
#         obs = np.array(data.T) / np.std(np.array(data.T), axis=0)  # do not subtract the mean
#     classes = {f'K={k}': {} for k in K_range}
#     lag_matrices = {f'K={k}': {} for k in K_range}
#     for k in K_range:
#         # SPC clustering and obtain lag_matrix from pairwise CCF
#         classes_spc, lag_matrix = cluster_SPC(obs, k, assumed_max_lag)
#         classes[f'K={k}'] = classes_spc
#         lag_matrices[f'K={k}'] = lag_matrix
#
#     with open(save_path + f'/classes/start{start_index}end{end_index}.pkl', 'wb') as f:
#         pickle.dump(classes, f)
#     with open(save_path + f'/lag_matrices_pairwise/start{start_index}end{end_index}.pkl', 'wb') as f:
#         pickle.dump(lag_matrices, f)

def run_real_data(sigma_range=np.arange(0.2, 2.1, 0.2), K_range=[1,2,3],
                  start_index=0, signal_length=50, assumed_max_lag=5,
                  scale_method = 'normalized',
                  models=['pairwise', 'sync', 'spc-homo', 'het'],
                  data_path='please check data_path',
                  estimates_path='please check estimates_path',
                  save_path='please check save_path',
                  return_signals=False,
                  return_lag_vec=True,
                  return_lag_mat=False,
                  return_PnL=False):

    # save parameters only once
    params_save_dir  = f'{save_path}/params.json'
    if not os.path.exists(params_save_dir):
        params = dict(sigma_range=list(sigma_range), K_range=K_range,
                      signal_length=signal_length,
                      assumed_max_lag=assumed_max_lag,
                      scale_method=scale_method,
                      models=models, data_path=data_path,
                      estimates_path=estimates_path,
                      save_path=save_path,
                      return_signals=return_signals,
                      return_lag_mat=return_lag_mat,
                      return_lag_vec=return_lag_vec,
                      return_PnL=return_PnL)
        with open(params_save_dir, 'w') as json_file:
            json.dump(params, json_file)


    # read data
    end_index = start_index + signal_length
    data = pd.read_csv(data_path, index_col=0).iloc[:, start_index:end_index]
    if start_index == 5145:
        data.drop('TIF', axis=0, inplace=True)
    ticker = data.index
    dates = data.columns
    if scale_method == 'normalized':  # normalized by subtracting the mean and scale by std
        obs = utils.normalize_by_column(np.array(data.T))
    elif scale_method == 'scaled':  # scale the observation by std
        obs = np.array(data.T) / np.std(np.array(data.T), axis=0)  # do not subtract the mean
    # obs_scaled = np.array(data.T) / np.std(np.array(data.T), axis=0)  # do not subtract the mean

    # initialise containers
    estimates = {}
    PnL = {}
    lag_matrices = {}
    lag_vectors = {}
    # with open(save_path + f'/classes/start{start_index}end{end_index}.pkl', 'rb') as f:
    #     classes_spc_dict = pickle.load(f)
    classes_spc_dict = spio.loadmat(save_path + f'/classes/start{start_index}end{end_index}.mat')
    with open(save_path + f'/lag_matrices_pairwise/start{start_index}end{end_index}.pkl', 'rb') as f:
        lag_matrices_dict = pickle.load(f)

    for k in tqdm(K_range):
        estimates[f'K={k}'] = {}
        lag_vectors[f'K={k}'] = {}
        lag_matrices[f'K={k}'] = {}
        PnL[f'K={k}'] = {}
        # SPC clustering and obtain lag_matrix from pairwise CCF
        # classes_spc, lag_matrix = cluster_SPC(obs_scaled, k, assumed_max_lag)
        classes_spc = classes_spc_dict[f'K{k}'].flatten()
        lag_matrix = lag_matrices_dict[f'K={k}']
        # evaluate models pairwise and sync
        results1 = eval_models(
            lag_matrix=lag_matrix,
            assumed_max_lag=assumed_max_lag,
            models=['pairwise', 'sync'],
            observations=obs,
            classes_spc=classes_spc,
            return_signals=return_signals,
            return_lag_vec=return_lag_vec,
            return_lag_mat=return_lag_mat,
            return_PnL=return_PnL
        )
        for sigma in sigma_range:

            # read data produced from matlab code base
            # X_est, P_est = read_realdata_results(
            #     data_path=estimates_path, sigma=sigma, k=k)
            X_est, P_est, X_est_homo = read_realdata_results(
                data_path=estimates_path, sigma=sigma, k=k)
            # calculate and align the clustering based on  Het-IVF signal estimates
            classes_est = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, obs)
            classes_est = utils.align_classes(classes_est, classes_spc)
            # predict lags matrices using different models
            results = eval_models(
                lag_matrix=lag_matrix,
                assumed_max_lag=assumed_max_lag,
                models=['spc-homo', 'het'],
                observations=obs,
                classes_spc=classes_spc,
                classes_est=classes_est,
                X_est_spc_homo=X_est_homo,
                X_est=X_est,
                sigma=sigma,
                return_signals=return_signals,
                return_lag_vec=return_lag_vec,
                return_lag_mat=return_lag_mat,
                return_PnL=return_PnL
            )
            for key in results.keys():
                results[key].update(results1[key])
            # store model performance results in dictionaries

            if return_signals:
                signal_dict = results['signals']
                # organize signal estimates, classes estimates and mixing prob estimates
                signal_class_prob = align_all_signals(X_est_sync=signal_dict['sync'],
                                                      X_est_spc=signal_dict['spc-homo'],
                                                      X_true=None,
                                                      classes_spc=classes_spc,
                                                      classes_est=classes_est,
                                                      classes_true=None,
                                                      k=k,
                                                      X_est=X_est,
                                                      P_est=P_est
                                                      )
                # store the signal estimates, classes estimates and mixing prob estimates
                estimates[f'K={k}'][f'sigma={sigma:.2g}'] = signal_class_prob

            # store the  lag matrices predicted by the models
            if return_lag_vec:
                lag_vec_dict = results['lag vec']
                lag_vectors[f'K={k}'][f'sigma={sigma:.2g}'] = lag_vec_dict
            if return_lag_mat:
                lag_mat_dict = results['lag mat']
                lag_matrices[f'K={k}'][f'sigma={sigma:.2g}'] = lag_mat_dict
            if return_PnL:
                PnL_dict = results['PnL']
                PnL[f'K={k}'][f'sigma={sigma:.2g}'] = PnL_dict
     # save the results to folder
    for subfolder in ['signal_estimates', 'lag_matrices', 'PnL','lag_vectors']:
        utils.create_folder_if_not_existed(f'{save_path}/{subfolder}')

    if return_signals:
        sub_dir = f'{save_path}/signal_estimates'
        with open(sub_dir + f'/start{start_index}end{end_index}.pkl', 'wb') as f:
            pickle.dump(estimates, f)
    if return_lag_vec:
        sub_dir = f'{save_path}/lag_vectors'
        with open(sub_dir + f'/start{start_index}end{end_index}.pkl', 'wb') as f:
            pickle.dump(lag_vectors, f)
    if return_lag_mat:
        sub_dir = f'{save_path}/lag_matrices'
        with open(sub_dir + f'/start{start_index}end{end_index}.pkl', 'wb') as f:
            pickle.dump(lag_matrices, f)
    if return_PnL:
        sub_dir = f'{save_path}/PnL'
        with open(sub_dir + f'/start{start_index}end{end_index}.pkl', 'wb') as f:
            pickle.dump(PnL, f)


# set main() run parameters here
def run_wrapper(round, save_path):
    # sigma_range = np.arange(0.1, 2.1, 0.1)
    # K_range = [2, 3, 4]
    sigma_range = np.arange(0.5, 2, 1)
    K_range = [2,3]
    max_shift = 2
    data_path = '../data/data500_shift2_pvCLCL_init2_set1/'
    run(sigma_range=sigma_range, K_range=K_range,
        max_shift=max_shift,data_path=data_path,
        save_path=save_path,
        test=False, return_lag_mat=True,
        return_signals=True, round=round)



def run_wrapper_real_data(inputs):
    start_index, save_path = inputs

    params_save_dir = f'{save_path}/params_clustering.json'
    with open(params_save_dir, 'r') as json_file:
        params = json.load(json_file)
    K_range = params['K_range']
    signal_length = params['signal_length']
    assumed_max_lag = params['assumed_max_lag']
    scale_method = params['scale_method']
    data_path = params['data_path']
    estimates_path = f'{save_path}/pvCLCL_results/start{start_index + 1}_end{start_index + signal_length}/'
    run_real_data(sigma_range=np.arange(0.2, 2.1, 0.2), K_range=K_range,
                  start_index=start_index, signal_length=signal_length,
                  assumed_max_lag=assumed_max_lag, scale_method=scale_method,
                  models=['pairwise', 'sync', 'spc-homo', 'het'],
                  data_path=data_path,
                  save_path=save_path,
                  estimates_path=estimates_path,
                  return_signals=True,
                  return_lag_vec=True,
                  return_lag_mat=False,
                  return_PnL=False)


if __name__ == "__main__":
    real_data = True
    if real_data:
        save_path = '../results/real/2023-07-04-01h04min_clustering_full'
        # inherit parameters from clustering experiments
        params_save_dir = f'{save_path}/params_clustering.json'
        with open(params_save_dir, 'r') as json_file:
            params = json.load(json_file)
        # start = params['start']
        # end = params['end']
        # retrain_period = params['retrain_period']
        start = 1005
        end = 4445
        retrain_period = 10


        start_indices = range(start, end, retrain_period)
        inputs = list(zip(start_indices,repeat(save_path)))
        start_time = time.time()

        # map inputs to functions
        # for start_index in start_indices:
        #     run_wrapper_real_data((start_index, save_path))
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # use the pool to apply the worker function to each input in parallel
            # pool.map(run_wrapper_real_data, start_indices)     previous pool
            _ = list(tqdm(pool.imap(run_wrapper_real_data, inputs),
                          total=len(start_indices)))
            pool.close()
            pool.join()

        print(f'time taken to run {len(start_indices)} predictions: {time.time() - start_time}')

    else:
        folder_name = 'test'
        save_path = utils.save_to_folder('../results/synthetic', folder_name)
        # remember to untick 'Run with Python console' in config
        rounds = 4
        inputs = range(1, 1 + rounds)
        start_time = time.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # use the pool to apply the worker function to each input in parallel
            pool.starmap(run_wrapper, zip(inputs, repeat(save_path)))
            pool.close()
        print(f'time taken to run {rounds} rounds: {time.time() - start_time}')

        # run single thread for debugging
        # run(max_shift=2, K_range=[2], sigma_range=np.arange(1.0, 2.0, 0.5),
        #     data_path='../data/data500_shift2_pvCLCL_init2_set1/',
        #     save_path='../results/synthetic',
        #     test=False, return_lag_mat=True,
        #     return_signals=True, round=1)

    # TODO: modify align_all_signals to process only the outputs of the selected models
    # note the current maximum shift is 4
