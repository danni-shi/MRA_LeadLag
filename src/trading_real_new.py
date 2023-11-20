import numpy as np
import pandas as pd
import alignment
import warnings
import time
import scipy.io as spio
import pickle
from trading import *
from tqdm import tqdm
import datetime as dt
import os


def rolling_sum(A, n):
    """
    find the rolling sum of A (2d array) along axis 1 (column), with window size n
    """
    ret = np.cumsum(A, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:]


def estimate_volitility(returns, type, window_width):
    returns = returns.iloc[:, -window_width:]
    if type == 'std':
        return np.array(returns.std(axis=1))
    if type == 'variance':
        return np.array(returns.std(axis=1)) ** 2
    if type == 'squared returns':
        return np.mean(returns.values ** 2, axis=1)


def weights_by_inverse_vol(returns, type, window_width, winsorize_percentiles=(5, 95)):
    window_width_recalculated = min(window_width, returns.shape[1])
    vol = estimate_volitility(returns, type, window_width_recalculated)
    weights = 1 / vol
    weights = winsorize(weights, winsorize_percentiles)
    # weights /= np.sum(weights)

    return weights


def agg_signals_by_group(group_signals, lags):
    """
    return the weights of trading each group for a period of time
    """
    G, L = group_signals.shape
    assert G == len(lags) - 1
    rolled_back_signals = np.zeros((G, L))
    for i, l in enumerate(lags[:-1]):
        rolled_back_signals[i, :L - l] = group_signals[i, l:]
    # total signals received by each group j (in rows), ignore the first group
    rolled_back_signals = np.cumsum(np.sign(rolled_back_signals), axis=0)
    signals_received_each_group_day = np.zeros((G, L))
    for j, l in enumerate(lags[1:]):
        signals_received_each_group_day[j, l:] = rolled_back_signals[j, :L - l]
    return signals_received_each_group_day


def weights_lag_groups(returns, trading_start, trading_end,
                       days_advanced, lag_vector, lags,
                       watch_period=1, hold_period=1, equal_group_weight=True):
    """
    Return the portfolio weights of assets in lag group based trading,
    the total absolute volume, i.e. sum(|w_i|) equals 1.
    Return also the portfolio PnL
    """
    sub_returns = returns.iloc[:, trading_start - days_advanced:trading_end]
    Nk, L = sub_returns.shape
    sub_returns = np.array(sub_returns)
    weights = weights_by_inverse_vol(returns.iloc[:, :trading_start],
                                     'std',
                                     window_width=20)
    lags = np.sort(lags)
    num_groups = len(lags)
    # ij entry is the signal of group i on day j (last row should be zeros)
    group_signals = np.zeros((num_groups - 1, L))
    returns_by_group_by_day = np.zeros((num_groups, L))
    agg_returns_by_day = np.zeros((num_groups - 1, L))

    for i, l in enumerate(lags):
        # reweight the weights such that each lag group weights add up to 1
        if equal_group_weight:
            weights[lag_vector == l] /= np.sum(weights[lag_vector == l])
        group_returns = sub_returns[lag_vector == l]
        returns_by_group_by_day[i, :] = np.dot(group_returns.T, weights[lag_vector == l])

    # aggregated signal from watch period, ignore the last group
    group_signals[:, watch_period - 1:] = \
        rolling_sum(returns_by_group_by_day[:-1, :], watch_period)

    # aggregated returns from hold period, ignore the first group
    agg_returns_by_day[:, hold_period - 1:] = \
        rolling_sum(returns_by_group_by_day[1:, :], hold_period)  # (num_group-1)x(L)

    # signals received by each group each day (except the first group)
    signals_received = agg_signals_by_group(group_signals, lags)[:, days_advanced:]  # (num_group-1)x(L)

    group_weights = signals_received / np.sum(abs(signals_received), axis=0)
    group_weights[np.isnan(group_weights)] = 0

    assert ((np.sum(abs(group_weights), axis=0) - 1) < 1e-10).all()

    PnL = np.sum(group_weights * agg_returns_by_day[:, days_advanced:], axis=0)

    # portfolio weights of each asset
    portfolio_weights = weights_from_group_signal(weights, lag_vector, lags, group_weights)
    assert ((np.sum(abs(portfolio_weights[:, days_advanced:]),
                    axis=0) - 1) < 1e-10).all(), 'portfolio weights do not sum to 1'
    assert (abs(PnL - np.sum(sub_returns[:, days_advanced:] * portfolio_weights, axis=0)) < 1e-6).all()

    return portfolio_weights, PnL


def weights_all_classes(df_returns, trading_start, trading_end,
                        lag_vec, classes, watch_period=1,
                        hold_period=1, class_threshold=None,
                        assumed_max_lag=5
                        ):
    """

    Returns:
         the weights of assets in all classes
         the simple returns of the asset at each time step, averaged over different classes

    """

    class_labels = np.unique(classes)
    N = df_returns.shape[0]  # number of assets
    # set the threshold for minimum class size
    if not class_threshold:
        class_threshold = int(0.2 * N / len(class_labels))
    class_counts = []
    class_PnLs = []

    # initialize the portfolio weights container for each assets for a trading period
    portfolio_weights = np.zeros((N, trading_end - trading_start))

    for c in class_labels:
        # count number of samples in class c
        count = np.count_nonzero(classes == c)
        if count > class_threshold:  # ignore classes with size below a certain threshold
            sub_lag_vector = lag_vec[classes == c]
            lags, counts = np.unique(sub_lag_vector, return_counts=True)
            # ignore the lag groups whose sizes are below 10% of expected group size
            min_group_size = 0.1 * int(len(sub_lag_vector) / assumed_max_lag + 1)
            lags = lags[counts >= min_group_size]

            if len(lags) > 1:
                # relocate the lags to start from 0
                min_lag = np.min(lags)
                lags -= min_lag
                sub_lag_vector -= min_lag
                # need to pass in previous days returns to calculate current profits
                days_advanced = min(max(lags) + watch_period, trading_start)
                sub_returns = df_returns.iloc[classes == c]
                class_portfolio_weights, class_PnL = weights_lag_groups(sub_returns,
                                                                        trading_start,
                                                                        trading_end,
                                                                        days_advanced,
                                                                        sub_lag_vector,
                                                                        lags,
                                                                        watch_period,
                                                                        hold_period)
                portfolio_weights[classes == c] = class_portfolio_weights * count
                class_counts.append(count)
                class_PnLs.append(class_PnL)

    total_volume = np.sum(abs(portfolio_weights), axis=0)
    portfolio_weights[:, total_volume != 0] /= total_volume[total_volume != 0].reshape(1, -1)
    # if no trades are done due to small class/group sizes, then PnL = 0
    if not class_counts:
        portfolio_PnL = np.zeros(trading_end - trading_start)
    else:
        portfolio_PnL = np.average(class_PnLs, axis=0, weights=class_counts)

    return portfolio_weights, portfolio_PnL


def weights_from_group_signal(weights, lag_vector, lags, group_weights):
    # calculate weights of each asset
    Gn, T = group_weights.shape

    portfolio_weights = np.zeros((len(weights), T))
    # for each day t, multiply the original weights by group signals
    for t in range(T):
        for i, l in enumerate(lags):
            if i > 0:
                portfolio_weights[lag_vector == l, t] = weights[lag_vector == l] * group_weights[i - 1, t]
    # normalise the weights
    weight_sum = np.sum(abs(portfolio_weights), axis=0)
    portfolio_weights[:, weight_sum != 0] /= weight_sum[weight_sum != 0].reshape(1, -1)
    return portfolio_weights


def turnover(returns, portfolio_weights, days_advanced):
    """
    returns: N by T
    weights: length N
    """
    assert returns.shape == portfolio_weights.shape
    daily_total_returns = np.sum(portfolio_weights * returns, axis=0)
    portfolio_weights_next = portfolio_weights * (1 + returns) / (1 + daily_total_returns.reshape(1, -1))
    rebalance = abs(portfolio_weights[:, 1:] - portfolio_weights_next[:, :-1])
    assert days_advanced - 1 >= 0
    turnover_rate = np.sum(rebalance[:, days_advanced - 1:], axis=0)
    return turnover_rate


def trading_single(df_returns,
                   lag_vectors, estimates, k, sigma,
                   model, trading_period_start, trading_period_end,
                   assumed_max_lag, hedge):
    # if ((trading_period_start, trading_period_end) == (5145,5195))\
    #         or (trading_period_start, trading_period_end) == (5195,5205):
    #    df_returns = df_returns.drop('TIF', axis=0)
    # load estimates of lags
    lag_vec = lag_vectors[f'K={k}'][f'sigma={sigma:.2g}'][model]['row mean']

    if model == 'het':
        classes = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['het']
    else:
        classes = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['spc']
    weights, PnL = weights_all_classes(df_returns,
                                       trading_period_start, trading_period_end,
                                       lag_vec, classes, assumed_max_lag=assumed_max_lag)
    SPY_index = df_returns.index.get_loc('SPY')

    if hedge:
        SPY_weights = - np.sum(weights, axis=0)
        PnL = PnL + SPY_weights * weights[SPY_index, :]
        weights[SPY_index, :] += SPY_weights
        assert ((PnL - np.sum(weights *
                              np.array(df_returns.iloc[:, trading_period_start:trading_period_end]),
                              axis=0)) < 1e5).all()

    return weights, PnL


def trading_real_data(data_path, prediction_path, k, sigma,
                      model,
                      train_period_start=0,
                      train_period_end=50,
                      out_of_sample=True,
                      trading_period=10,
                      assumed_max_lag=2,
                      hedge=True):
    """
    run trading strategy on real stocks returns data.

    """
    # load lag prediction
    with open(prediction_path + f'/signal_estimates/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(prediction_path + f'/lag_vectors/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        lag_vectors = pickle.load(f)

    # load returns data
    df_returns = pd.read_csv(data_path, index_col=0)
    market_index = df_returns.loc['SPY']

    # trade on untrained data if out_of_sample is True
    if out_of_sample:
        trading_period_start = train_period_end
        trading_period_end = train_period_end + trading_period
    else:
        trading_period_start = train_period_start
        trading_period_end = train_period_end

    weights, PnL = trading_single(df_returns, lag_vectors, estimates,
                                  k, sigma, model,
                                  trading_period_start, trading_period_end,
                                  assumed_max_lag, hedge=hedge)

    return weights, PnL


def best_K_and_sigma(df_returns, prediction_path,
                     K_range, sigma_range,
                     model,
                     train_period_start=0,
                     train_period_end=50,
                     assumed_max_lag=5,
                     hedge=False):
    trading = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}

    # load lag prediction
    with open(prediction_path + f'/signal_estimates/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(prediction_path + f'/lag_vectors/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        lag_vectors = pickle.load(f)

    trading_period_start = train_period_start
    trading_period_end = train_period_end

    if model in ['pairwise', 'sync']:
        sigma_range = [sigma_range[0]]
    score = np.zeros((len(K_range), len(sigma_range)))
    fail_count = 0
    for i, k in enumerate(K_range):
        for j, sigma in enumerate(sigma_range):
            # load estimates of lags
            weights, PnL = trading_single(df_returns,
                                          lag_vectors, estimates,
                                          k, sigma, model,
                                          trading_period_start, trading_period_end,
                                          assumed_max_lag, hedge=hedge)

            if (weights == 0).all():
                # print(f'Trading Criteria not met at period {trading_period_start} to {trading_period_end} with model {model} at K {k}, sigma {sigma:.1faren}')
                score[i, j] = -np.Inf
                fail_count += 1
            else:
                result = PnL
                score[i, j] = np.sum(result)

    if fail_count == len(K_range) * len(sigma_range):
        print(
            f'Trading Criteria not met at period {trading_period_start} to {trading_period_end} with model {model}')
    ind_k, ind_s = np.unravel_index(score.argmax(), score.shape)

    return round(K_range[ind_k]), round(sigma_range[ind_s], 1)


def best_K_and_sigma_for_all(df_returns, prediction_path,
                             K_range, sigma_range,
                             models,
                             start_indices, signal_length,
                             assumed_max_lag,
                             hedge=False):
    end_indices = np.array(start_indices) + signal_length
    result = np.zeros((len(start_indices), len(models)), dtype=tuple)
    for i in range(len(start_indices)):
        train_period_start = start_indices[i]
        train_period_end = end_indices[i]
        for j in range(len(models)):
            model = models[j]
            K_sigma = best_K_and_sigma(df_returns, prediction_path, K_range, sigma_range, model,
                                       train_period_start, train_period_end, assumed_max_lag, hedge)
            result[i, j] = K_sigma

    start_end_indices = [(start_indices[t], end_indices[t]) for t in range(len(start_indices))]
    df_results = pd.DataFrame(result, index=start_end_indices, columns=models)
    return df_results


def concat_results(models, prediction_path, folder_name,
                     start, end, signal_length,
                     trading_period
                     ):
    """
    concatenate PnL simulation on out-of-sample data from multiple retrained experiments
    Args:
        train_start:
        train_end:
        retrain_period:
        return_SR:

    Returns: PnL and SR (optional)

    """

    start_indices = range(start, end, retrain_period)

    keys = ['weights', 'PnL']
    list_dict = {key:
                     {model: [] for model in models}
                 for key in keys}


    for train_start in start_indices:
        train_end = train_start + signal_length
        file_name = f'start{train_start}end{train_end}trade{trading_period}'
        with open(f'{prediction_path}/{folder_name}/{file_name}.pkl', 'rb') as f:
            trading = pickle.load(f)
        for key in keys:
            for model in models:
                value = trading[model][key]
                list_dict[key][model].append(value.reshape(-1, retrain_period))

    for key in keys:
        for model in models:
            list_dict[key][model] = np.concatenate(list_dict[key][model], axis=1)

    return list_dict


# def save_trading_results_by_date(trading_results, dates, save_path):
#     """
#     trading_results: dict containing trading results for a period of time
#     dates: list of string 'YYYY-MM-DD' corresponding to the date of the trading results
#     save_path: directory to save the daily trading results
#     """
#
#     # for key, value in trading_results.item():
#     #     if value.ndim == 1:
#     #         assert len(value) == len(dates), \
#     #             f'{key} dim is incompatible with date dimension, expected {len(dates)}'
#     #     elif value.ndim == 2:
#     #         assert value.shape[1] == len(dates), \
#     #             f'{key} dim is incompatible with date dimension, expected {len(dates)}'
#
#     for i, date in enumerate(dates):
#         results_dict = {}
#         for key, value in trading_results.item():
#             if value.ndim == 1:
#                 results_dict[key] = value[i]
#             elif value.ndim == 2:
#                 results_dict[key] = value[:, i]
#
#         with open(save_path + date + '.pkl', 'wb') as f:
#             pickle.dump(results_dict, f)


def string_to_int(string):
    # string format '(x,y)'
    l = string.split(',')
    return int(l[0][1:]), int(l[1][:-1])


# def trading_weights_by_class()

if __name__ == '__main__':

    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    # Use the relevant data and lag prediction for different experiment settings
    data_path = '../data/pvCLCL_clean_winsorized.csv'
    df_returns = pd.read_csv(data_path, index_col=0)  # data
    SPY = df_returns.loc['SPY']
    # df_returns.drop(index='SPY')
    prediction_path = '../results/real/full_non-negative_affinity'
    PnL_folder_name = 'PnL_real_single_weighted_both'
    PnL_folder_name = 'PnL_new_method'
    # range of K and sigma we run grid search on
    K_range = [1, 2, 3]
    sigma_range = np.arange(0.2, 2.1, 0.2)
    # start, ending, training data length, period of retrain
    start = 5
    end = 5146
    retrain_period = 10
    signal_length = 50
    start_indices = range(start, end, retrain_period)
    models = ['pairwise', 'sync', 'spc-homo', 'het']
    # grid search on K and sigma for all models based on in-sample performance
    df_results = best_K_and_sigma_for_all(df_returns,
                                          prediction_path,
                                          K_range, sigma_range,
                                          models,
                                          start_indices,
                                          signal_length,
                                          assumed_max_lag=2,
                                          hedge=False)

    folder_path = prediction_path + '/' + PnL_folder_name + '/'
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
    best_K_sigma_path = folder_path + 'best_k_sigma.csv'
    # df_results.to_csv(best_K_sigma_path)
    # # path_old = '/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/results/real/full_non-negative_affinity/PnL_real_single_weighted_both/'
    # # df_results1 = pd.read_csv(path_old+'best_k_sigma.csv',
    # #                           index_col=0)
    #
    df_results = pd.read_csv(best_K_sigma_path, index_col=0).applymap(eval)
    #
    # # ###--------------------- Trading by sliding window ----------------------------###
    # remove the X from the dates
    # trading_dates = [date[1:] for date in df_returns.columns]
    #
    for row_num, index in tqdm(enumerate(df_results.index)):
        train_period_start, train_period_end = string_to_int(index)
        # train_period_start, train_period_end = index
        trading_results_models = {}
        for col_num, model in enumerate(df_results.columns):
            K, sigma = df_results.iloc[row_num, col_num]

            weights, PnL = trading_real_data(data_path, prediction_path, K, sigma, model,
                                             train_period_start=train_period_start,
                                             train_period_end=train_period_start + signal_length,
                                             out_of_sample=True,
                                             trading_period=retrain_period,
                                             assumed_max_lag=2,
                                             hedge=True)

            trading_results_models[model] = {'weights': weights,
                                             'PnL': PnL}

        # with open(path_old + file_name + '.pkl', 'rb') as f:
        #     r = pickle.load(f)
        file_name = f'start{train_period_start}end{train_period_end}trade{retrain_period}'
        with open(folder_path + file_name + '.pkl', 'wb') as f:
            pickle.dump(trading_results_models, f)

    # ###---------------- Concatenate segments of trading results together ----------------------###
    concatenated_results = concat_results(
        models, prediction_path, PnL_folder_name,
        start, end,
        signal_length, retrain_period)


    file_name = f'start{start}end{end}_length{signal_length}_trade{retrain_period}'
    with open(prediction_path + '/' + PnL_folder_name + '/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(concatenated_results, f)
