import random

import numpy as np
import matplotlib.pyplot as plt
import utils
from optimization import optimise_matlab
import pickle
import time
from tqdm import tqdm
import scipy.io as spio
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy import signal

    
def get_signal(type, L):
    if type == 'logreturns':
        with open('../../data/logreturn.npy', 'rb') as f:
            signal = np.load(f)
        signal = signal[L:2*L]      
    elif type == 'sine':
        x = np.linspace(0,2*np.pi, L)
        x = np.sin(x)
        signal = (x-np.mean(x))/np.std(x)
    elif type == 'gaussian':
        signal = np.random.randn(L)
    return signal  

def lag_vec_to_mat(vec):
    # note this function is invariant to addition and subtraction 
    # of the same value to every element of vec
    L = len(vec)
    vec = vec.reshape(-1,1)
    ones = np.ones((L,1))
    return vec @ ones.T - ones @ vec.T

assert np.linalg.norm(lag_vec_to_mat(np.array([0,1]))-np.array([[0,-1],[1,0]])) < 1e-10
assert np.linalg.norm(lag_vec_to_mat(np.array([1,2,3]))-np.array([[0,-1,-2],[1,0,-1],[2,1,0]])) < 1e-10

def lag_mat_het(lags, classes, return_block_mat = False):
    """arrange lags vector or lags matrix into block-diagonal form based on the given class labels. 

    Args:
        lags (np array): lags vector or matrix
        classes (np array): class labels of each observation
        return_block_mat (bool, optional): if True, return the list of matrices in block-diagonal form; else return the list. Defaults to False.

    Returns:
        _type_: _description_
    """
    lag_mat_list = []

    for c in np.unique(classes):
        if lags.ndim == 2 and lags.shape[0] == lags.shape[1]:
            sub_lags = lags[classes == c, classes == c]
        else:
            sub_lags = lag_vec_to_mat(lags[classes == c])
        lag_mat_list.append(sub_lags)
    
    if return_block_mat:
        return block_diag(*lag_mat_list)
    else:
        return lag_mat_list


def alignment_residual(x1, x2, max_lag = None, return_lag = False):
    """align the vector x1 after circularly shifting it such that it is optimally aligned with x2 in 2-norm. Calculate the 

    Args:
        x1 (np array): 
        x2 (np array): 

    Returns:
        relative_residual (float): normalized residual between the aligned vector and x2.
        lag (int): lag of best alignment
    """
    # align x1 to x2
    x1_aligned, lag = utils.align_to_ref(x1,x2, max_lag = None)
    relative_residual = np.linalg.norm(x1_aligned-x2)/np.linalg.norm(x1_aligned)/np.linalg.norm(x2)
    
    if return_lag:
        return relative_residual, lag
    else:
        return relative_residual


def alignment_similarity(x1, x2, max_lag = None, normalised = True, return_lag = False):
    """return the highest cross correlation coefficient between two vectors up to a cyclic shift.

    Args:
        x1 (np array): 
        x2 (np array):

    Returns:
        float: normalized correlation coefficient
    """
    _, lag, ccf = utils.align_to_ref(x1, x2, max_lag, return_ccf=True, normalised=normalised)
    
    if return_lag:
        return ccf[lag], lag
    else: return ccf[lag]

def alignment_similarity_linear(x1, x2, max_lag = None, normalised = True, return_lag = False):
    """return the highest linear cross correlation coefficient between two vectors.

    Args:
        x1 (np array): 
        x2 (np array):

    Returns:
        correlation: normalized correlation coefficient
        lag: lag of signal 
    """
    x1 = x1.flatten()
    x2 = x2.flatten()
    
    if normalised:
        m1 = np.mean(x1);s1 = np.std(x1)
        m2 = np.mean(x2);s2 = np.std(x2)
        x1 = (x1-m1)/s1
        x2 = (x2-m2)/s2
    ccf = signal.correlate(x1, x2, 'full')
    
    L = len(x1)
    # set default value of max lag to be the maximum possible lag
    if max_lag is None:
        max_lag = L-1
        
    if return_lag:
        # lag = len(x2) - (np.argmax(ccf[L-max_lag-1:L+max_lag])+ L-max_lag-1) - 1
        lag = max_lag - np.argmax(ccf[L-max_lag-1:L+max_lag])
        return np.max(ccf[L-max_lag-1:L+max_lag]), lag
    else: return np.max(ccf[L-max_lag-1:L+max_lag])

# for n in [8,10,12]:
#     x = np.random.normal(0,1,n)
#     for l in [-3,-2,-1,1,2,3]:
#         y = np.roll(x, l)
#         if l > 0:
#             y[0:l] = np.zeros(abs(l))
#         else:
#             y[len(y)+l:len(y)] = np.zeros(abs(l))
        
#         corr, lag = alignment_similarity_linear(x,y, return_lag=True)
#         assert lag == l, f'fn lag{lag}; true lag: {l}; n:{n}'


def score_lag_mat(observations, max_lag = None, score_fn = alignment_similarity_linear):
    """produce the similarity or residual scores and best lags of a set of observations with a given score function

    Args:
        observations (LxN np array): vectors 
        score_fn (python function, optional): score function which is used to compute the scores and lags between every pair of observations. Defaults to alignment_similarity_linear.

    Returns:
        scores: (NxN np array) ij-th entry denotes the scores between observations i and j
        lags: (NxN np array) ij-th entry denotes the best predicted lag between observations i and j
    """
    L, N = observations.shape
    scores = np.zeros((N,N))
    lags = np.zeros((N,N))
    for j in range(N):
        for i in range(j):
            score, lag = score_fn(observations[:,i], observations[:,j], max_lag=max_lag, return_lag=True)
            scores[i,j] = scores[j,i] = score
            if lag >= L//2 + 1:
                    lag -= L
            lags[i,j] = lag
            lags[j,i] = -lag
    
    return scores, lags

def circ_rolling_sum(vec, win_width):
    """

    Args:
        vec: 1-dim np array of length n
        win_width: width of rolling window

    Returns:
        1-dim np array of length n
        rolling sum of a window of win-width over vec. When the rolling window hits the tail of the vector, continue to slide as if the head of the vector connect to the tail.

    """
    vec = vec.flatten()
    assert win_width > 0
    assert win_width <= len(vec)

    vec = np.append(vec, vec[:win_width-1])
    cumsum = np.cumsum(vec)
    cumsum[win_width:] = cumsum[win_width:] - cumsum[:-win_width]

    return cumsum[win_width-1:]

def argmax_last(x):
    """

    Args:
        x: 2-dim np array

    Returns: the index of the last occurence of the maximum value in an array

    """

    x_rev = x[::-1]
    i = len(x_rev) - np.argmax(x_rev) - 1
    return i
def lag_to_ref(X, X_ref, normalised = True, start = 0, lag_range = None):
    """

    Args:
        X: LxN array
        ref: 1-dim array of length L
        normalised:

    Returns:

    """
    X_ref = X_ref.flatten()
    if X.ndim == 1:
        X.reshape(-1,1)
    assert len(X_ref) == len(X), 'Lengths of data and reference are not equal'

    # set default value for the assumed range of lags
    if not lag_range:
        lag_range = len(X_ref)
    assert lag_range <= len(X_ref)

    if normalised:
        m1 = np.mean(X, axis = 0); s1 = np.std(X, axis = 0)
        m2 = np.mean(X_ref); s2 = np.std(X_ref)
        X = (X - m1) / s1
        X_ref = (X_ref - m2) / s2

    X_ref_fft = np.fft.fft(X_ref)
    X_fft = np.fft.fft(X, axis = 0)
    ccf = np.fft.ifft(np.tile(X_ref_fft.conj().reshape(-1, 1), (1, X_fft.shape[1])) \
                      * X_fft, axis=0).real
    ccf = np.roll(ccf, shift = -start, axis = 0)
    lags = np.argmax(ccf[:lag_range], axis=0) + start

    return lags % len(X_ref)

# functions to check if the estimated lags are more wide-spread than the assumed maximum lag
def consecutive_zeros(array):
    result = 0
    streak = 0
    for ele in array:
        if ele == 0:
            streak += 1
        else:
            streak = 0
        result = max(result, streak)
    return result
def smallest_lag_range(lag_counts):
    """
    find the shortest length of subarray that does not start or end with zero in a cyclic fashion.
    Args:
        lag_counts: counts of lags. ith element is the count of lags that equals i

    Returns: int

    """
    if lag_counts.ndim > 1:
        lag_counts = lag_counts.flatten()
    extended_counts = np.append(lag_counts,lag_counts)
    max_consecutive_zeros = consecutive_zeros(extended_counts)

    return len(lag_counts) - max_consecutive_zeros

def get_lag_matrix_ref(observations, ref, max_lag = None):
    """Calculate the best lags estimates of a given set of observations with a latent reference signal. 

    Args:
        observations (np array): L x N matrix with columns consist of time series
        ref (np array): 1-dim length L reference time series signal.
        max_lag: maximum lag wrt to the ref signal
    """
    # initilizations
    L, N = observations.shape
    ref = ref.flatten()
    assert len(ref) == L

    # Calculate unconstrained lags for all observations
    shifts_est = lag_to_ref(observations, ref)
    assert np.min(shifts_est) >= 0
    assert np.max(shifts_est) < L
    # find window of lags with the highest frequency
    lag_freq = np.bincount(shifts_est, minlength=L)
    if max_lag < smallest_lag_range(lag_freq) - 1:
        lag_start = np.argmax(circ_rolling_sum(lag_freq, max_lag+1)) # window width should be max lag + 1
        recalculate = (shifts_est - lag_start) % L > max_lag
        if np.count_nonzero(recalculate) > 0:
            shifts_est[recalculate] = lag_to_ref(observations[:, recalculate],\
                                                 ref,\
                                                 start=lag_start,\
                                                 lag_range= max_lag)
        shifts_est = (shifts_est - lag_start) % L
    # ref_aligned = np.roll(ref, shift = window_start)
    # if window_end > window_start:
    #     accept_arr = (shifts_est >= window_start) * (shifts_est < window_end)
    # else:
    #     accept_arr = (shifts_est >= window_start) + (shifts_est < window_end)

    # ref_lag = round(np.mean(shifts_est))
    # ref_aligned = np.roll(ref, shift=ref_lag)
    # shifts_est -= ref_lag
    # limit the range of lag predictions

    # recalculate the lags outside predicted range
    # for i in range(N):
    #     if shifts_est > max_lag or shifts_est < 0:
    #         lag = utils.align_to_ref(observations[:,i], ref_aligned, max_lag = max_lag)[1]
    #         # ccf = utils.align_to_ref(observations[:,i], ref, return_ccf=True)[2]
    #         # lag = np.argmax(ccf[int(shifts_est[i]])
    #         shifts_est[i] = lag
    lag_mat = lag_vec_to_mat(shifts_est)
    lag_mat[abs(lag_mat) >= (L+1)//2] -= np.sign(lag_mat[abs(lag_mat) >= (L+1)//2]) * L
    if max_lag:
        assert (abs(lag_mat) <= max_lag).all()
    return lag_mat, shifts_est
#
# N = 10
# x = np.arange(20)
# l=3
# shifts = random.choices(range(10,10+5), k=N)
# X = np.array([list(np.roll(x, shift=shifts[i]) for i in range(N))])[0].T
# lags, lags_mat = get_lag_matrix_ref(X,x,max_lag=3)
# shifts_mat = lag_vec_to_mat(np.array(shifts))
# assert (lags == lag_vec_to_mat(np.array(shifts))).all()
# print('done')

# def get_lag_matrix(observations, ref = None, max_lag = None):
#     """calculate the best lags estimates of a given set of observations, with or without a latent reference signal

#     Args:
#         observations (np array): L x N matrix with columns consist of time series
#         ref (np array, optional): 1-dim length L reference time series signal. Defaults to None.
#     """
#     L, N = observations.shape
    
#     if ref is not None:
#         assert len(ref == L)
#         shifts_est = np.zeros(N)
#         for i in range(N):
#             _, lag = utils.align_to_ref(observations[:,i], ref, max_lag)
#             shifts_est[i] = lag
#         # limit the range of lag predictions
#         # shifts_est_centred = shifts_est - np.mean(shifts_est)
#         # for i in range(N):
#         #     if abs(shifts_est_centred[i]) >= max_shift * L:
#         #         ccf = utils.align_to_ref(observations[:,i], ref, return_ccf=True)[2]    
#         #         lag = np.argmax(ccf[int(shifts_est[i]])
#         lag_mat = lag_vec_to_mat(shifts_est)
#         # for i in range(N):
#         #     for j in range(N):
#         #         if abs(lag_mat[i,j]) >= L//2 + 1:
#         #             lag_mat[i,j] -= np.sign(lag_mat[i,j]) * L
#     else:
#         lag_mat = np.zeros((N,N))
#         for j in range(N):
#             for i in range(j):
#                 _, lag = utils.align_to_ref(observations[:,i], observations[:,j])
#                 if lag >= L//2 + 1:
#                     lag -= L
#                 lag_mat[i,j] = lag
#                 lag_mat[j,i] = -lag
#     return lag_mat  


def get_lag_mat_het(observations, ref = None, classes = None):
    lag_mat_list = []
    
    # for c in np.unique(classes):
    #     sub_observations = observations[classes == c]
    #     sub_ref = ref.iloc[:,int(c-1)]
    #     lag_mat_list.append(get_lag_matrix(sub_observations,sub_ref))
    
    # return block_diag(*lag_mat_list)
    pass

def total_error(array1, array2):
    """compute the differences between corresponding elements of two arrays of the same shape. NaN values are allowed and do not add to the result.
    """
    diff_array = array1 - array2
    errors = list(abs(diff_array[~np.isnan(diff_array)]).flatten())
    return errors

def accuracy(array1, array2):
    """compute the percentage of corresponding elements having the same value in two arrays of the same shape. NaN values are allowed and is counted as wrong.
    """
    diff_array = array1 - array2

    return np.sum(abs(diff_array[~np.isnan(diff_array)]) < 1e-5)/diff_array.size * 100

def eval_lag_mat(lag_mat, lag_mat_true):
    """compute the relative error and accuracy of a lag matrix wrt to a ground truth lag matrix.

    Args:
        lag_mat (nxn array): _description_
        lag_mat_true (nxn array): _description_

    Returns:
        _type_: _description_
    """
    if lag_mat_true.ndim == 1 or \
        np.count_nonzero(np.array(lag_mat_true.shape) != 1) == 1:
        lag_mat_true = lag_vec_to_mat(lag_mat_true)
    
    # skew-symmetric matrices, we only need the upper triangle
    N = len(lag_mat)
    iu1 = np.triu_indices(N,1)
    lag_mat_true_u = lag_mat_true[iu1]
    lag_mat_u = lag_mat[iu1]
    
    errors = total_error(lag_mat_u,lag_mat_true_u)
    tol_error = np.sum(errors)
    sign_errors = total_error(np.sign(lag_mat_u),np.sign(lag_mat_true_u))
    tol_error_sign = np.sum(sign_errors)
    # rel_error /= np.sum(abs(lag_mat_true_u[~np.isnan(lag_mat_true_u)]))
    
    acc = accuracy(lag_mat_u,lag_mat_true_u)
    
    return tol_error, tol_error_sign, acc, errors

def lag_mat_post_clustering(lag_mat, classes):
    """mask the i-j entry of the lag matrix if sample i,j are not in the same cluster.

    Args:
        lag_mat (_type_): _description_
        classes (_type_): _description_

    Returns:
        _type_: _description_
    """
    lag_mat_out = lag_mat.copy()
    for c in np.unique(classes):
        mask = (classes==c)[:,None] * (classes!=c)[None,:]
        lag_mat_out[mask] = np.nan
    np.fill_diagonal(lag_mat_out, np.nan)
    
    return lag_mat_out


def eval_lag_mat_het(lag_mat, lag_mat_true, classes, classes_true, penalty=0):
    """evaluate the relative error and accurcy of a lag matrix if there are more than one class of samples.

    Args:
        lag_mat (_type_): _description_
        lag_mat_true (_type_): _description_
        classes (_type_): _description_
        classes_true (_type_): _description_

    Returns:
        _type_: _description_
    """
    if lag_mat_true.ndim == 1 or \
        np.count_nonzero(np.array(lag_mat_true.shape) != 1) == 1:
        lag_mat_true = lag_vec_to_mat(lag_mat_true)
    
    # # mask th irrelavant entries with nan
    # lag_mat = lag_mat_post_clustering(lag_mat, classes)
    # lag_mat_true = lag_mat_post_clustering(lag_mat_true, classes_true)
    # # calculate scale
    # N = lag_mat_true.shape[0]
    # # number of valid lags for evaluation
    # tol_count_val = np.count_nonzero(~np.isnan(lag_mat-lag_mat_true))
    # tol_count_true = np.count_nonzero(~np.isnan(lag_mat_true))
    # count_diff = tol_count_true - tol_count_val
    # assert count_diff >= 0
    # diff_mat = lag_mat - lag_mat_true
    # # if the cluster assignment is wrong, add error = assumed max shift
    # penalty = np.nanmax(abs(lag_mat_true))
    # rel_error = np.sum(abs(diff_mat[~np.isnan(diff_mat)])) + penalty * count_diff
    # rel_error /= np.sum(abs(lag_mat_true[~np.isnan(lag_mat_true)]))
    # # ignore the diagonal entries 
    # accuracy = np.sum(abs(diff_mat[~np.isnan(diff_mat)]) < 0.1)/tol_count_true * 100
    # calculate the proportion of valid entries
    
    # initialization
    rel_error = rel_error_sign = accuracy = 0
    errors_list = []
    
    lag_mat = lag_mat_post_clustering(lag_mat, classes)
    lag_mat_true = lag_mat_post_clustering(lag_mat_true, classes_true)
    n = 0 # number of lags evaluated
    n_total = np.count_nonzero(~np.isnan(lag_mat_true))//2
    
    for c in np.unique(classes):
        # calculate true lags
        # sub_lag_mat_true = lag_mat_true[classes == c][:,classes == c]
        # sub_lag_mat = lag_mat[classes == c][:,classes == c]
        intersection = (classes == c) & (classes_true == c)
        sub_lag_mat_true = lag_mat_true[intersection][:,intersection]
        sub_lag_mat = lag_mat[intersection][:,intersection]
        n_nan= np.count_nonzero(np.isnan(sub_lag_mat))
        assert n_nan == len(sub_lag_mat), f'{n_nan} null values in predictions'
        # lag_mat_0 = get_lag_matrix(sub_observations)
        n += np.count_nonzero(~np.isnan(sub_lag_mat))//2
        # evaluate error and accuracy, weighted by cluster size
        class_error, class_error_sign, class_accuracy, class_errors = eval_lag_mat(sub_lag_mat, sub_lag_mat_true)
        
        rel_error += class_error
        rel_error_sign += class_error_sign
        weight = len(sub_lag_mat)/len(classes)
        accuracy += class_accuracy * weight
        errors_list += class_errors
    
    rel_error_sign = (rel_error_sign + (n_total-n) * 2) / n_total 
    # average error of lags
    if penalty > 0:
        rel_error = (rel_error + (n_total-n) * penalty) / n_total
        errors_list += [penalty] * (n_total - n)
    else:
        rel_error /= n 
    
    assert abs(np.mean(errors_list) - rel_error) < 1e-6, f'difference in error = {abs(np.mean(errors_list) - rel_error):.3g}'
    # store only the errors percentiles
    error_percentiles = np.percentile(errors_list, [range(0,101,5)]).flatten()
    return rel_error, rel_error_sign, accuracy, error_percentiles

# def eval_alignment(observations, shifts, sigma, X_est = None):
#     """compare the performance of lead-lag predition using intermidiate latent signal to naive pairwise prediciton

#     Args:
#         observations (np array): L x N matrix with columns consist of time series
#         shifts (np array): 1-dim array that contains the ground true lags of the observations to some unknown signal

#     Returns:
#         mean_error: error of prediction
#         accuracy: accuracy of prediction
#         mean_error_0: error of naive approach
#         accuracy_0: accuracy of naive approach

#     """
#     L, N = observations.shape
#     lag_mat_true = lag_vec_to_mat(shifts)
    
#     if X_est is None:
#         # estimate and align to signal
#         X_est = optimization.optimise_manopt(observations, sigma)
    
#     # calculate lags of observation to the aligned estimate
#     lag_mat = get_lag_matrix(observations, X_est)
#     lag_mat_0 = get_lag_matrix(observations)
    
#     # evaluate error and accuracy
#     norm = np.linalg.norm(lag_mat_true,1)
#     rel_error, accuracy = eval_lag_mat(lag_mat, lag_mat_true)
#     rel_error_0, accuracy_0 = eval_lag_mat(lag_mat_0, lag_mat_true)
    
#     return rel_error, accuracy, rel_error_0, accuracy_0, X_est

def latent_signal_homo(observations, classes,sigma):
    X_est_list = []
    for c in np.unique(classes):
        sub_observations = observations[:, classes == c]
        # estimate and align to signal
        sub_X_est, _, _ = optimise_matlab(sub_observations, sigma, 1)
        X_est_list.append(sub_X_est.reshape(-1, 1))

    X_est = np.concatenate(X_est_list, axis=1)

    return X_est

def get_lag_matrix_het(observations, classes, X_est, max_lag =None):


        # assign observations to the closest cluster centre
        if classes is None:
            assert X_est != None, 'Cannot assign classes without cluster signals'
            classes = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)

        if X_est is None:
            X_est_list = []

        N = observations.shape[1]
        # mask to nan the irrelevant entries
        lag_mat = np.zeros((N,N))
        for c in np.unique(classes):
            # estimate lags from data
            sub_observations = observations[:, classes == c]
            sub_X_est = X_est[:, c]

            sub_lag_mat = get_lag_matrix_ref(sub_observations, sub_X_est, max_lag=max_lag)[0]
            indices = np.where(np.outer(classes==c,classes==c))
            lag_mat[indices[0], indices[1]] = sub_lag_mat.flatten()

        return lag_mat
            # np.fill_diagonal(sub_lag_mat, np.nan)
            #sub_lag_mat_eval = sub_lag_mat[sub_classes_true == c][:, sub_classes_true == c]

        #     n += np.count_nonzero(~np.isnan(sub_lag_mat_eval)) // 2
        #     # evaluate error and accuracy, weighted by class size
        #     class_error, class_error_sign, class_accuracy, class_errors = eval_lag_mat(sub_lag_mat_eval,
        #                                                                                sub_lag_mat_true_eval)
        #
        #     rel_error += class_error
        #     rel_error_sign += class_error_sign
        #     weight = len(sub_lag_mat) / len(classes)
        #     accuracy += class_accuracy * weight
        #     errors_list += class_errors
        #
        # rel_error_sign = (rel_error_sign + (n_total - n) * 2) / n_total
        # # average error of lags
        # if penalty > 0:
        #     rel_error = (rel_error + (n_total - n) * penalty) / n_total
        #     errors_list += [penalty] * (n_total - n)
        # else:
        #     rel_error /= n
        # assert abs(np.mean(
        #     errors_list) - rel_error) < 1e-6, f'difference in error = {abs(np.mean(errors_list) - rel_error):.3g}'
        # # store only the errors percentiles
        # error_percentiles = np.percentile(errors_list, [range(0, 101, 5)]).flatten()
        #
        # if X_est is None:
        #     X_est = np.concatenate(X_est_list, axis=1)
        #     return rel_error, rel_error_sign, accuracy, error_percentiles, X_est
        # else:
        #     return rel_error, rel_error_sign, accuracy, error_percentiles


def eval_alignment_het(observations, lag_mat_true, classes, classes_true,  X_est, penalty = 0, max_lag = None):
    # assign observations to the closest cluster centre
    # if classes is None:
    #     assert X_est != None, 'Cannot assign classes without cluster signals'
    #     classes = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
    #
    # if X_est is None:
    #     X_est = latent_signal_homo(observations, classes,sigma)

    lag_mat = get_lag_matrix_het(observations, classes, X_est, max_lag)
    # lag_mat1 = lag_mat[classes == 0][:, classes == 0]
    # lag_mat_true1 = lag_mat_true[classes_true == 0][:, classes_true == 0]
    # lag_mat2 = lag_mat[classes == 1][:, classes == 1]
    # lag_mat_true2 = lag_mat_true[classes_true == 1][:, classes_true == 1]
    # assert (lag_mat1[np.triu_indices(len(lag_mat1),1)] == lag_mat_true1[np.triu_indices(len(lag_mat1),1)]).all()
    # assert (lag_mat2[np.triu_indices(len(lag_mat2), 1)] == lag_mat_true2[np.triu_indices(len(lag_mat2), 1)]).all()

    results = eval_lag_mat_het(lag_mat, lag_mat_true, classes, classes_true, penalty)

    return results

def eval_alignment_het0(observations, lag_mat_true, classes = None, classes_true = None,  X_est = None, sigma = None, penalty = 0, max_lag = None):
    """compare the performance of lead-lag predition using intermidiate latent signal to naive pairwise prediciton

    Args:
        observations (np array): L x N matrix with columns consist of time series
        shifts (np array): 1-dim array that contains the ground true lags of the observations to some unknown signal

    Returns:
        mean_error: error of prediction
        accuracy: accuracy of prediction
        mean_error_0: error of naive approach
        accuracy_0: accuracy of naive approach

    """
    # initialization
    rel_error = rel_error_sign = accuracy = 0
    errors_list = []
        
    # assign observations to the closest cluster centre
    if classes is None:
        assert X_est != None, 'Cannot assign classes without cluster signals'
        classes = np.apply_along_axis(lambda x: utils.assign_classes(x, X_est), 0, observations)
    
    if X_est is None:
        X_est_list = []
    
    # mask to nan the irrelevant entries
    lag_mat_true = lag_mat_post_clustering(lag_mat_true, classes_true)
    # evaluate the lag estimation for each cluster
    n = 0
    n_total = np.count_nonzero(~np.isnan(lag_mat_true))//2
    for c in np.unique(classes):
        
        # estimate lags from data
        sub_observations = observations[:,classes == c]
        if X_est is None:
            # estimate and align to signal
            sub_X_est, _, _ = optimise_matlab(sub_observations, sigma, 1)
            X_est_list.append(sub_X_est.reshape(-1,1))
        else:
            sub_X_est = X_est[:,c]
        
        # calculate true lags
        intersection = (classes == c) & (classes_true == c)
        sub_lag_mat_true_eval = lag_mat_true[intersection][:,intersection]
        sub_classes_true = classes_true[classes == c]
        # test
        sub_lag_mat_true = lag_mat_true[classes == c][:,classes == c]
        sub_lag_mat_true_eval_test = sub_lag_mat_true[sub_classes_true == c][:,sub_classes_true == c]
        assert (np.triu(sub_lag_mat_true_eval,1) == np.triu(sub_lag_mat_true_eval_test,1)).all()
        # subset lags to evaluate
        sub_lag_mat = get_lag_matrix_ref(sub_observations, sub_X_est, max_lag=max_lag)[0]
        np.fill_diagonal(sub_lag_mat, np.nan)
        sub_lag_mat_eval = sub_lag_mat[sub_classes_true == c][:,sub_classes_true == c]
        
        n += np.count_nonzero(~np.isnan(sub_lag_mat_eval))//2
        # evaluate error and accuracy, weighted by class size
        class_error, class_error_sign, class_accuracy, class_errors = eval_lag_mat(sub_lag_mat_eval,sub_lag_mat_true_eval)

        rel_error += class_error
        rel_error_sign += class_error_sign
        weight = len(sub_lag_mat)/len(classes)
        accuracy += class_accuracy * weight
        errors_list += class_errors
     
    rel_error_sign = (rel_error_sign + (n_total-n) * 2) / n_total   
    # average error of lags
    if penalty > 0:
        rel_error = (rel_error + (n_total-n) * penalty) / n_total
        errors_list += [penalty] * (n_total - n)
    else:
        rel_error /= n 
    assert abs(np.mean(errors_list) - rel_error) < 1e-6, f'difference in error = {abs(np.mean(errors_list) - rel_error):.3g}'
    # store only the errors percentiles
    error_percentiles = np.percentile(errors_list, [range(0,101,5)]).flatten()

    if X_est is None:
        X_est = np.concatenate(X_est_list,axis=1)
        return rel_error, rel_error_sign, accuracy, error_percentiles, X_est
    else:
        return  rel_error, rel_error_sign, accuracy, error_percentiles

#---- Implementation of SVD-Synchronization ----#
def reconcile_score_signs(H,r, G=None):
    L = H.shape[0]
    ones = np.ones((L,1))
     
    # G is the (true) underlying measurement graph
    if G is None:
        G = np.ones((L,L)) - np.eye(L)
   
    const_on_rows = np.outer(r,ones) 
    const_on_cols = np.outer(ones,r) 
    recompH = const_on_rows - const_on_cols
    recompH = recompH * G
    
    # difMtx{1,2} have entries in {-1,0,1}
    difMtx1 = np.sign(recompH) - np.sign(H)    
    difMtx2 = np.sign(recompH) - np.sign(H.T)
    
    # Compute number of upsets:
    upset_difMtx_1 = np.sum(abs(difMtx1))/2
    upset_difMtx_2 = np.sum(abs(difMtx2))/2
    
    if upset_difMtx_1 > upset_difMtx_2:
        r = -r 
        
    return r

def SVD_NRS(H, scale_estimator = 'median'):
    """perform SVD normalised ranking and synchronization on a pairwise score matrix H to obtain a vector of lags

    Args:
        H (_type_): _description_
    """
    L = H.shape[0]
    ones = np.ones((L,1))

    D_inv_sqrt = np.diag(np.sqrt(abs(H).sum(axis = 1))) # diagonal matrix of sqrt of column sum of abs
    H_ss = D_inv_sqrt @ H @ D_inv_sqrt
    U, S, _ = np.linalg.svd(H_ss) # S already sorted in descending order, U are orthonormal basis
    assert np.all(S[:-1] >= S[1:]), 'Singular values are not sorted in desceding order'

    u1_hat = U[:,0]; u2_hat = U[:,1]
    u1 = D_inv_sqrt @ ones
    u1 /= np.linalg.norm(u1) # normalize

    u1_bar = (U[:,:2] @ U[:,:2].T @ u1).flatten()
    # u1_bar /= np.linalg.norm(u1_bar) # normalize
    # u2_tilde = u1_hat - np.dot(u1_hat,u1_bar)*u1_bar # same as proposed method
    # u2_tilde /= np.linalg.norm(u2_tilde) # normalize
    # test
    T = np.array([np.dot(u2_hat,u1_bar),-np.dot(u1_hat,u1_bar)])
    u2_tilde_test = U[:,:2] @ T

    u2_tilde_test /= np.linalg.norm(u1_bar)

    # assert np.linalg.norm(u2_tilde_test.flatten()-u2_tilde) <1e-8 or np.linalg.norm(u2_tilde_test.flatten()+u2_tilde) <1e-8
    pi = D_inv_sqrt @ u2_tilde_test.reshape(-1,1)
    pi = reconcile_score_signs(H, pi)
    S = lag_vec_to_mat(pi)

    # median
    if scale_estimator == 'median':

        offset = np.divide(H, (S+1e-9), out=np.zeros(H.shape, dtype=float), where=np.eye(H.shape[0])==0)
        tau = np.median(offset[np.where(~np.eye(S.shape[0],dtype=bool))])
        if tau == 0:
            tau = np.sum(abs(np.triu(H,k=1)))/np.sum(abs(np.triu(S,k=1)))

    if scale_estimator == 'regression':
        tau = np.sum(abs(np.triu(H,k=1)))/np.sum(abs(np.triu(S,k=1)))

    r = tau * pi - tau * np.dot(ones.flatten(), pi.flatten()) * ones / L
    r_test = tau * pi
    # test
    r_test = r_test - np.mean(r_test)
    assert np.linalg.norm(r_test.flatten()-r.flatten()) <1e-8 or np.linalg.norm(r_test.flatten()+r.flatten()) <1e-8

    return pi.flatten(), r.flatten(), tau

def shift(X, shifts, cyclic = False):
    """shifts a set of time series by a given set of lags

    Args:
        X (LxN array): each column contains a time series
        shifts (len N array): i-th entry denote the lag to the i th column of X
        cyclic (bool, optional): whether the shift is cyclic. Defaults to False.
    """
    L, N = X.shape
    data = np.zeros(X.shape)
    
    for i in range(N):
        k = shifts[i]
        y = np.roll(X[:,i],k)
        if not cyclic:
            # y[:k] = np.random.normal(0, 1, size = k)
            if k < 0:
                y[L+k:L] = np.zeros(-k)
            else:
                y[:k] = np.zeros(k)
        data[:,i] = y
    return data

def synchronize(X, shifts, cyclic = False):
    """for a sample of shifted copies, with the knowledge of their lags, shifts the samples back to their original positions and compute the sample average 

    Args:
        X (LxN array): each column contains a time series
        shifts (len N array): i-th entry denote the lag to the i th column of X
        cyclic (bool, optional): whether the shift is cyclic. Defaults to False.

    Returns:
        _type_: _description_
    """
    X_shifted = shift(X, -shifts, cyclic = cyclic)
    
    return X_shifted.mean(axis = 1)

def get_synchronized_signals(observations, classes, lag_matrix, max_lag = None):
    # initialize
    L = observations.shape[0]
    K = len(np.unique(classes))
    X_est = np.zeros((L,K))
    
    if not max_lag:
        max_lag = L-2
        
    # synchronize the samples in each class
    for c in np.unique(classes):
        # compute the synchronized lags
        sub_lag_matrix = lag_matrix[classes == c][:,classes == c]
        if (sub_lag_matrix==0).all():
            X_est[:,c] = np.mean(observations[:,classes == c],axis = 1)
        else:
            pi, r, _ = SVD_NRS(sub_lag_matrix)
            r_rounded = np.array(np.round(r), dtype=int)
            # r_rounded -= min(r_rounded) # make the relative lags start from zero
            # compute the cluster average X
            sub_observations = observations.T[classes == c][abs(r_rounded) <= (max_lag+2)//2].T
            X_est[:,c] = synchronize(sub_observations, r_rounded[abs(r_rounded) <= max_lag])

    return X_est

def align_plot():
    # intialise parameters for generating observations
    L = 50 # length of signal
    N = 500 # number of copies
    sigma = 1 # std of random gaussian noise
    max_shift= 0.1 # max proportion of lateral shift

    # intialise parameter for experiments
    n = 10 # number of points
    sigma_range = np.linspace(0.1,3,n) # range of noise level
    options = ['logreturns', 'sine', 'gaussian'] # types of signals

    count = 0
    result = {}
    for i in range(len(options)):
        type = options[i]
        # iniitialise containers
        result[type] = {}
        error_list = np.zeros(n)
        acc_list = np.zeros(n)
        error_list_0 = np.zeros(n)
        acc_list_0 = np.zeros(n)
        
        
        # generate signal
        signal = get_signal(type, L)
        
        for j in range(n):
            sigma = sigma_range[j]
            # generate shifted, noisy version of the signal
            observations, shifts = utils.generate_data(signal, N, max_shift, sigma, cyclic = False)
            mean_error, accuracy, mean_error_0, accuracy_0, X_est = eval_alignment(observations, shifts, sigma)
            X_aligned, lag = utils.align_to_ref(X_est, signal)
            print('relative error = ', np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal))
            error_list[j] = mean_error
            acc_list[j] = accuracy
            error_list_0[j] = mean_error_0
            acc_list_0[j] = accuracy_0
            count += 1
            print(f'{count}/{n*len(options)} steps completed')
        result[type]['accuracy'] = {'intermediate': acc_list,
                                    'pairwise': acc_list_0}        
        result[type]['error'] = {'intermediate': error_list,
                                    'pairwise': error_list_0}        
        
        fig, ax = plt.subplots(figsize = (15,6))
        ax.plot(sigma_range, error_list, label = 'with intermediate')
        ax.plot(sigma_range, error_list_0, label = 'pairwise')
        plt.grid()
        plt.legend()
        plt.title(f'Change of Alignment Error with Noise Level ({type} signal)')
        plt.savefig(f'../plots/align_error_{type}')

        fig, ax = plt.subplots(figsize = (15,6))
        ax.plot(sigma_range, acc_list, label = 'with intermediate')
        ax.plot(sigma_range, acc_list_0, label = 'pairwise')
        plt.grid()
        plt.legend()
        plt.title(f'Change of Alignment Accuracy with Noise Level ({type} signal)')
        plt.savefig(f'../plots/align_acc_{type}')

    with open('../results/alignment.pkl', 'wb') as f:   
        pickle.dump(result, f)


"""
### plot error and accuracy with data and results from MATLAB

# intialise parameters
sigma_range = np.arange(0.1,2.1,0.1) # std of random gaussian noise
max_shift= 0.1 # max proportion of lateral shift
options = ['gaussian'] # types of signals
K_range = [1]
# n = 2000 # number of observations we evaluate

# data path
data_path = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/HeterogeneousMRA/data/'
count = 0
result = {}
type = options[0]
# iniitialise containers
result[type] = {}
error_list = []
acc_list = []
error_list_0 = []
acc_list_0 = []
# class_acc_list = []
k = 1  
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
    observations = observations_mat['data']
    shifts = observations_mat['shifts'].flatten()
    classes = observations_mat['classes'].flatten()
    X_est = results_mat['x_est']
    
    mean_error, accuracy, mean_error_0, accuracy_0, X_est = \
        eval_alignment(observations, shifts, sigma, X_est)
        
    
    # X_aligned, lag = utils.align_to_ref(X_est, signal)
    # print('relative error = ', np.linalg.norm(X_aligned-signal)/np.linalg.norm(signal))
    error_list.append(mean_error)
    acc_list.append(accuracy)
    error_list_0.append(mean_error_0)
    acc_list_0.append(accuracy_0)
    # class_acc_list.append(class_accuracy)
    
result[type]['accuracy'] = {'intermediate': acc_list,
                            'pairwise': acc_list_0}        
result[type]['error'] = {'intermediate': error_list,
                            'pairwise': error_list_0}        
# result[type]['class accuracy'] = class_acc_list

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, error_list, label = 'with intermediate')
ax.plot(sigma_range, error_list_0, label = 'pairwise')
plt.grid()
plt.legend()
plt.title(f'Change of Alignment Error with Noise Level ({type} signal)')
plt.savefig(f'../plots/align_error_{type}_K=1')

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, acc_list, label = 'with intermediate')
ax.plot(sigma_range, acc_list_0, label = 'pairwise')
plt.grid()
plt.legend()
plt.title(f'Change of Alignment Accuracy with Noise Level ({type} signal)')
plt.savefig(f'../plots/align_acc_{type}_K=1')
    
    # fig, ax = plt.subplots(figsize = (15,6))
    # ax.plot(sigma_range, class_acc_list)
    # plt.grid()
    # plt.title(f'Change of Class Assignment Accuracy with Noise Level ({type} signal)')
    # plt.savefig(f'../plots/class_acc_{type}_K={k}_0')

with open('../results/alignment_homo.pkl', 'wb') as f:   
    pickle.dump(result, f)


    
# L = 50 # length of signal
# N = 500 # number of copies
# sigma = 1 # std of random gaussian noise
# max_shift= 0.1
# signal = get_signal('sine', L)

# with open('../results/data.npy', 'rb') as f:
#     observations = np.load(f)
#     shifts = np.load(f)
    
# with open('../results/visual.npy', 'rb') as f:
#     X_est = np.load(f)
#     X_aligned = np.load(f)

# # calculate lags of observation to the aligned estimate
# lag_mat = get_lag_matrix(observations, X_est)
# lag_mat_0 = get_lag_matrix(observations)
# lag_mat_true = lag_vec_to_mat(shifts)

# # evaluate error and accuracy
# mean_error = np.linalg.norm(lag_mat - lag_mat_true)/np.linalg.norm(lag_mat_true)
# accuracy = np.mean(abs(lag_mat - lag_mat_true) < 0.1) * 100
# mean_error_0 = np.linalg.norm(lag_mat_0 - lag_mat_true)/np.linalg.norm(lag_mat_true)
# accuracy_0 = np.mean(abs(lag_mat_0 - lag_mat_true)< 0.1) * 100

# print(f'accuracy of lag predictions: experiment: {accuracy:.2f}%; benchmark: {accuracy_0:.2f}%')
# print(f'relative error of lag predictions: experiment: {mean_error:.2f}; benchmark: {mean_error_0:.2f}')



# baseline: matrix entry as the residual of aligned observations, cluster them and everage the observaton in each class to recover the latent signal , then we compute the lags of observations against the signals

# 
"""