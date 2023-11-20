import numpy as np
import pandas as pd
import alignment
import warnings
import time
import scipy.io as spio
import pickle
from trading import *
from tqdm import tqdm
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
        return np.array(returns.std(axis=1))**2
    if type == 'squared returns':
        return np.mean(returns.values**2, axis=1)

def weights_by_inverse_vol(returns, type, window_width, winsorize_percentiles=(5, 95)):
    window_width_recalculated = min(window_width, returns.shape[1])
    vol = estimate_volitility(returns, type, window_width_recalculated)
    weights = 1/vol
    weights = winsorize(weights,winsorize_percentiles)
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
        rolled_back_signals[i, :L-l] = group_signals[i, l:]
    # total signals received by each group j (in rows), ignore the first group
    rolled_back_signals = np.cumsum(np.sign(rolled_back_signals), axis=0)
    signals_received_each_group_day = np.zeros((G, L))
    for j, l in enumerate(lags[1:]):
        signals_received_each_group_day[j, l:] = rolled_back_signals[j, :L-l]
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
        returns_by_group_by_day[i,:] = np.dot(group_returns.T, weights[lag_vector == l])

    # aggregated signal from watch period, ignore the last group
    group_signals[:, watch_period - 1:] = \
        rolling_sum(returns_by_group_by_day[:-1, :], watch_period)

    # aggregated returns from hold period, ignore the first group
    agg_returns_by_day[:, hold_period - 1:] = \
        rolling_sum(returns_by_group_by_day[1:, :], hold_period)  # (num_group-1)x(L)

    # signals received by each group each day (except the first group)
    signals_received = agg_signals_by_group(group_signals, lags)  # (num_group-1)x(L)

    group_weights = signals_received / np.sum(abs(signals_received), axis=0)
    group_weights[np.isnan(group_weights)] = 0

    assert ((np.sum(abs(signals_received[:, days_advanced:]), axis=0) - 1) < 1e-10).all()

    PnL = np.sum(group_weights * agg_returns_by_day, axis=0)

    # portfolio weights of each asset
    portfolio_weights = weights_from_group_signal(weights, lag_vector, lags, group_weights)
    assert ((np.sum(abs(portfolio_weights[:, days_advanced:]),
                    axis=0) - 1) < 1e-10).all(), 'portfolio weights do not sum to 1'
    assert (abs(PnL - np.sum(sub_returns * portfolio_weights)) < 1e-6).all()

    return portfolio_weights[:, days_advanced:], PnL[days_advanced:]

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
    N = df_returns.shape[0] # number of assets
    # set the threshold for minimum class size
    if not class_threshold:
        class_threshold = int(0.2 * N / len(class_labels))
    class_counts = []
    class_PnLs = []
    market_index = df_returns.loc['SPY']
    # initialize the portfolio weights container for each assets for a trading period
    portfolio_weights = np.zeros((N, trading_end-trading_start))

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

    portfolio_weights /= np.sum(abs(portfolio_weights), axis=0).reshape(1,-1)
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
    portfolio_weights[:, weight_sum != 0] /= weight_sum[weight_sum != 0].reshape(1,-1)
    return portfolio_weights

def turnover(returns, portfolio_weights, days_advanced):
    """
    returns: N by T
    weights: length N
    """
    assert returns.shape == portfolio_weights.shape
    daily_total_returns = np.sum(portfolio_weights * returns, axis = 0)
    portfolio_weights_next = portfolio_weights * (1 + returns)/(1 + daily_total_returns.reshape(1, -1))
    rebalance = abs(portfolio_weights[:, 1:] - portfolio_weights_next[:, :-1])
    assert days_advanced - 1 >= 0
    turnover_rate = np.sum(rebalance[:, days_advanced - 1:], axis=0)
    return turnover_rate




def PnL_two_groups_real(returns, leaders, laggers, lag,
                        watch_period=1, hold_period=1,
                        signal_weights=None, portfolio_weights=None,
                        return_leader_pnl = False):
    """
    Use the past returns of the leaders group to devise long or short trading decisions on the laggers group.

    Args:
        returns: returns of all stocks
        leaders: index of leaders
        laggers: index of laggers
        lag: The lag between leaders and laggers laggers.

    Returns: returns of trading the laggers portfolio

    """
    N, L = returns.shape
    leader_returns = returns[leaders]
    lagger_returns = returns[laggers]
    ahead = lag - 1
    assert ahead >= 0
    portfolio_returns = np.full((L,), np.nan)
    portfolio_signals = np.zeros(L)
    portfolio_leader_pnl = np.full((L,), np.nan)

    if signal_weights is None:
        signal_weights = np.ones(len(leaders))/len(leaders)
    if portfolio_weights is None:
        portfolio_weights = np.ones(len(laggers))/len(laggers)

    for t in range(ahead + watch_period + hold_period, L + 1):
        signal = np.dot(np.sum(leader_returns[:, t-ahead-watch_period-hold_period:t-ahead-hold_period],axis=1),\
                        signal_weights)
        alpha = np.sign(signal) * np.average(np.sum(lagger_returns[:, t - hold_period:t], axis=1),
                                          axis=0, weights=portfolio_weights)
        # alpha0=np.sign(signal) * np.mean(np.sum(lagger_returns[:, t - hold_period:t], axis=1))
        # leader_alpha = np.sign(signal) * np.average(np.sum(leader_returns[:, t - hold_period:t], axis=1),
        #                                   axis=0, weights=signal_weights)
        leader_alpha = np.sign(signal) * np.mean(np.sum(leader_returns[:, t - hold_period:t], axis=1))

        portfolio_returns[t - 1] = alpha
        portfolio_signals[t - 1] = signal
        portfolio_leader_pnl[t - 1] = leader_alpha # denotes the trend following strategy pnl on leaders

    if return_leader_pnl:
        return portfolio_returns, portfolio_signals, portfolio_leader_pnl
    else:
        return portfolio_returns, portfolio_signals



def strategy_lag_groups(returns, trading_start, trading_end,
                        days_advanced, lag_vector, lags,
                        watch_period=1, hold_period=1,
                        market_index=None, hedge=True):
    """
    return the trading returns and performance statistics of pairs of lag groups in a single class
    """
    # including returns from dates before trading takes place to construct trading signals
    # pi, r, _ = alignment.SVD_NRS(lag_matrix)
    # lag_vector = np.array(np.round(r), dtype=int)
    sub_returns = returns.iloc[:, trading_start - days_advanced:trading_end]
    Nk, L = sub_returns.shape
    market_index = np.array(market_index.iloc[trading_start - days_advanced:trading_end])
    return_types = ['raw returns', 'mkt excess returns', 'leader excess returns', 'leader raw returns', 'leader mkt excess returns']
    results = {return_type: {} for return_type in return_types}
    signals = {}
    sub_returns = np.array(sub_returns)
    weights = weights_by_inverse_vol(returns.iloc[:, :trading_start],
                                     'std',
                                     window_width=20)
    lags = np.sort(lags)
    num_groups = len(lags)
    # ij entry is the signal of group i on day j (last row should be zeros)
    returns_by_group_by_day = np.zeros((num_groups, L))

    agg_signals_by_day = np.zeros((num_groups-1, L))
    agg_returns_by_day = np.zeros((num_groups - 1, L))

    # for each lag group, normalize the weights
    for i, l in enumerate(lags):
        # calculate the average returns of each group weighted by volatlity
        weights[lag_vector == l] /= np.sum(weights[lag_vector == l])
        group_returns = sub_returns[lag_vector == l]
        group_weights = weights[lag_vector == l]
        returns_by_group_by_day[i,:] = np.dot(group_returns.T, group_weights)
    # aggregated signal from watch period, ignore the last group
    agg_signals_by_day[:, watch_period-1:] = \
        rolling_sum(returns_by_group_by_day[:-1,:], watch_period)
    # aggregated returns from hold period, ignore the first group
    agg_returns_by_day[:, hold_period - 1:] = \
        rolling_sum(returns_by_group_by_day[1:, :], hold_period) # (num_group-1)x(L)
    # signals received by each group each day (except the first group)
    signals_received = agg_signals_by_group(agg_signals_by_day, lags) # (num_group-1)x(L)
    group_weights = signals_received / np.sum(abs(signals_received), axis=0)
    group_weights[np.isnan(group_weights)] = 0
    assert ((np.sum(abs(signals_received[:, days_advanced:]), axis=0) - 1) < 1e-10).all()


    PNL = np.sum(group_weights * agg_returns_by_day, axis = 0)
    net_volume = np.sum(group_weights, axis=0)
    mkt_excess_PnL = PNL - net_volume * market_index

    # portfolio weights of each asset
    portfolio_weights = weights_from_group_signal(weights, lag_vector, lags, group_weights)
    assert ((np.sum(abs(portfolio_weights[:, days_advanced:]), axis=0) - 1) < 1e-10).all(), 'portfolio weights do not sum to 1'

    # calculate turnover
    turnover_rate = turnover(sub_returns, portfolio_weights, days_advanced)
    turnover_rate_index = turnover(market_index.reshape(1,-1), net_volume.reshape(1,-1), days_advanced)
    results1 = {'raw returns': {'PnL': {'average': PNL[days_advanced:]},
                               'financial_stats':
                                   {'annualized SR': annualized_sharpe_ratio(PNL[days_advanced:]),
                                    'turnover': turnover_rate[days_advanced:]}},
               'mkt excess returns': {'PnL': {'average': mkt_excess_PnL[days_advanced:]},
                                      'financial_stats':
                                          {'annualized SR': annualized_sharpe_ratio(mkt_excess_PnL[days_advanced:]),
                                           'turnover': turnover_rate[days_advanced:] +
                                                       turnover_rate_index[days_advanced:]}}
                }

    for l1 in lags:
        for l2 in lags:
            if l1 < l2:
                leaders = np.where(lag_vector == l1)[0]
                laggers = np.where(lag_vector == l2)[0]

                leader_weights = weights[lag_vector == l1]
                lagger_weights = weights[lag_vector == l2]

                lag = l2 - l1
                pnl, signal, pnl_leader = PnL_two_groups_real(sub_returns, leaders, laggers, lag,
                                                              watch_period, hold_period,
                                                              leader_weights, lagger_weights,
                                                              return_leader_pnl=True)
                results['raw returns'][f'{l1}->{l2}'] = pnl
                signals[f'{l1}->{l2}'] = signal[days_advanced:]
                if hedge:
                    results['mkt excess returns'][f'{l1}->{l2}'] = pnl - np.sign(signal) * np.array(market_index)
                    results['leader excess returns'][f'{l1}->{l2}'] = pnl - pnl_leader
                    results['leader raw returns'][f'{l1}->{l2}'] = pnl_leader
                    results['leader mkt excess returns'][f'{l1}->{l2}'] = pnl_leader - np.sign(signal) * np.array(market_index)
    # calculate the simple average of PnL of each group pair
    for type, returns in results.items():
        for group, values in returns.items():
            returns[group] = values[days_advanced:]

        returns['average'] = np.nanmean(np.stack(list(returns.values())), axis=0) # ignore nans when calculating average
        # fill nans with 0 for every value in the results dictionary
        for values in returns.values():
            values[np.isnan(values)] = 0
        # calculate financial metrics
        results[type] = group_performance(returns, signals)

        # PnL['average'] = np.nanmean(np.stack(list(PnL.values())), axis=0)
        # PnL_excess_mkt['average'] = np.nanmean(np.stack(list(PnL_excess_mkt.values())), axis=0)
        # PnL_excess_leader['average'] = np.nanmean(np.stack(list(PnL_excess_leader.values())), axis=0)


        # for values in PnL.values():
        #     values[np.isnan(values)] = 0
        # for values in PnL_excess_mkt.values():
        #     values[np.isnan(values)] = 0
        # for values in PnL_excess_leader.values():
        #     values[np.isnan(values)] = 0

        # calculate financial metrics
        # results_dict = group_performance(PnL, signals)
        # results_excess_dict = group_performance(PnL_excess_mkt, signals)
        # results_excess_leader_dict = group_performance(PnL_excess_leader, signals)

    # return {'raw returns': results_dict,
    #         'mkt excess returns': results_excess_dict,
    #         'leader excess returns': results_excess_leader_dict}
    assert abs(results['raw returns']['financial_stats']['annualized SR']['average'] -
        results1['raw returns']['financial_stats']['annualized SR']) < 1e-10
    assert abs(results['mkt excess returns']['financial_stats']['annualized SR']['average'] -
               results1['mkt excess returns']['financial_stats']['annualized SR']) < 1e-10
    assert (abs(results['raw returns']['PnL']['average'] - results1['raw returns']['PnL']['average'])).all() < 1e-10
    assert (abs(results['mkt excess returns']['PnL']['average'] - results1['mkt excess returns']['PnL']['average'])).all() < 1e-10
    return results


def group_performance(PnL, signals):
    # results contains a dictionary for each metric which contains metrics of each group
    results_dict = {}
    results_dict['PnL'] = PnL
    fin_stats_by_group = {}
    # fin_dict = {}
    # for group, returns in results.items():
    #     signal = signals[group]
    #     fin_stats = financial_stats(returns,signal)
    #     for metric, value in fin_stats.items():
    #         metric_dict = fin_dict.get(metric,{})
    #         metric_dict[group] = value
    #     fin_dict[metric] =
    for group in signals.keys():
        lag_pair = tuple(int(a) for a in group.split('->'))
        assert lag_pair[0] < lag_pair[1]
        lag = lag_pair[1] - lag_pair[0]
        fin_stats = financial_stats(PnL[group][lag:], signals[group][lag:])
        fin_stats_by_group[group] = fin_stats
    # fin_stats_by_group = {group:financial_stats(PnL[group],signals[group]) for group in signals.keys()}
    fin_metrics = ['annualized SR', 'corr SP',
                   'corr SP p-value', 'hit ratio',
                   'long ratio', 'reg R2']
    fin_stats_by_metric = {metric: {group: fin_stats_by_group[group][metric]
                                    for group in fin_stats_by_group}
                           for metric in fin_metrics}
    fin_stats_by_metric['annualized SR']['average'] = annualized_sharpe_ratio(PnL['average'])
    results_dict['financial_stats'] = fin_stats_by_metric

    return results_dict



def performance_analysis(PnL, weights, lags, signals):
    pass
    # evaluate the turnover and sharpe ratio of a PnL
    stats_dict = {'annualized SR': {},
                  'turnover': {}}
    for key, pnl in PnL.items():
        ann_SR = annualized_sharpe_ratio(pnl)

        stats_dict['annualized SR'][key] = ann_SR
        portfolio_weights = weights_from_group_signal(weights, lags, signals)
        turnover_rate = turnover()



def strategy_het_real(df_returns, trading_start, trading_end,
                      lag_vec, classes, watch_period=1,
                      hold_period=1, class_threshold=None,
                      assumed_max_lag=5, hedge=False,
                      ):
    """
    returns: NxT np array. N is the number of instruments and T is the number of time points

    Returns: the simple returns of the asset at each time step, averaged over different classes

    """

    results_dict = {}
    class_labels = np.unique(classes)
    if not class_threshold:
        class_threshold = int(0.2 * df_returns.shape[0] / len(class_labels))
    class_counts = []
    market_index = df_returns.loc['SPY']
    for c in class_labels:
        # count number of samples in class c
        count = np.count_nonzero(classes == c)
        if count > class_threshold:  # ignore classes with size below a certain threshold
            sub_lag_vector = lag_vec[classes == c]
            lags, counts = np.unique(sub_lag_vector, return_counts=True)
            min_group_size = 0.1 * int(len(sub_lag_vector) / assumed_max_lag + 1)
            lags = lags[counts >= min_group_size]

            if len(lags) > 1:
                min_lag = np.min(lags)
                lags -= min_lag
                sub_lag_vector -= min_lag
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
                results = strategy_lag_groups(
                    sub_returns, trading_start, trading_end,
                    days_advanced, sub_lag_vector, lags,
                    watch_period, hold_period,
                    market_index=market_index, hedge=hedge)

                assert len(results['raw returns']) > 0, 'empty results'
                results_dict[f'class {c}'] = results
                class_counts.append(count)

    if hedge:
        return_types = ['raw returns', 'mkt excess returns', 'leader excess returns', 'leader raw returns', 'leader mkt excess returns']
    else:
        return_types = ['raw returns']
    # average PnL of each group across all classes, if any valid results are produced
    if len(results_dict) > 0:
        results_dict['portfolio average'] = {}
        for key in return_types:
            PnL_group_list = [{i: results[key]['PnL'][i] for i in results[key]['PnL'] \
                               if i != 'average'} for group, results in
                              results_dict.items() if group != 'portfolio average']
            PnL = class_average_returns_each_group(PnL_group_list)

            # average return weighted by class size
            pnl_average_list = [results[key]['PnL']['average'] for group, results in
                                results_dict.items() if group != 'portfolio average']
            pnl_average = np.average(pnl_average_list, axis=0, weights=class_counts)
            PnL['average'] = pnl_average
            SR = {group: annualized_sharpe_ratio(returns) for group, returns in PnL.items()}
            results_dict['portfolio average'][key] = {'PnL': PnL,
                                                      'annualized SR': SR}

    return results_dict

def trading_single(df_returns,
                   lag_vectors, estimates, k, sigma,
                   model, trading_period_start, trading_period_end,
                   assumed_max_lag, hedge,  **trading_kwargs):
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

    trading_results = strategy_het_real(
        df_returns,
        trading_period_start, trading_period_end,
        lag_vec, classes, assumed_max_lag=assumed_max_lag,
        hedge=hedge, **trading_kwargs)
    PnL1 = trading_results['portfolio average']['raw returns']['PnL']

    return trading_results
#

def trading_real_data(data_path, prediction_path, k, sigma,
                      model,
                      train_period_start=0,
                      train_period_end=50,
                      out_of_sample=True,
                      trading_period=10,
                      assumed_max_lag=2,
                      hedge=True,
                      **trading_kwargs):
    """
    run trading strategy on real stocks returns data.

    """
    # load lag prediction
    with open(prediction_path+f'/signal_estimates/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(prediction_path+f'/lag_vectors/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        lag_vectors = pickle.load(f)

    # load returns data
    df_returns = pd.read_csv(data_path, index_col=0)

    # trade on untrained data if out_of_sample is True
    if out_of_sample:
        trading_period_start = train_period_end
        trading_period_end = train_period_end + trading_period
    else:
        trading_period_start = train_period_start
        trading_period_end = train_period_end

    trading = trading_single(df_returns, lag_vectors, estimates,
                       k, sigma, model,
                       trading_period_start, trading_period_end,
                       assumed_max_lag, hedge, **trading_kwargs)
    return trading
    # if out_of_sample:
    #     file_name = f'start{train_period_start}end{train_period_end}trade{trading_period}'
    # else:
    #     file_name = f'start{train_period_start}end{train_period_end}_insample'
    #
    # with open('../results/PnL_real_single/' + file_name + '.pkl', 'wb') as f:
    #     pickle.dump(trading, f)

def best_K_and_sigma(df_returns, prediction_path,
                     K_range, sigma_range,
                     model,
                      train_period_start=0,
                      train_period_end=50,
                    assumed_max_lag=5,
                      criterion='raw returns'):
    trading = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}

    # load lag prediction
    with open(prediction_path+f'/signal_estimates/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(prediction_path+f'/lag_vectors/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
        lag_vectors = pickle.load(f)

    trading_period_start = train_period_start
    trading_period_end = train_period_end

    if model in ['pairwise', 'sync']:
        sigma_range = [sigma_range[0]]
    score = np.zeros((len(K_range),len(sigma_range)))
    fail_count = 0
    for i, k in enumerate(K_range):
        for j, sigma in enumerate(sigma_range):
            # load estimates of lags
            trading_results = trading_single(df_returns,
                                             lag_vectors, estimates,
                           k, sigma, model,
                           trading_period_start, trading_period_end,
                           assumed_max_lag, hedge=True)

            return_type, metric = criterion.split(' ')
            name_extension = {'raw': 'raw returns',
                              'mkt excess': 'mkt excess returns',
                              'leader excess': 'leader excess returns',
                              'returns': 'PnL',
                              'SR': 'annualized SR'}
            return_type = name_extension[return_type]
            metric = name_extension[metric]
            try:
                result = trading_results['portfolio average'][return_type][metric]['average']
                score[i, j] = np.sum(result)
            except:
                # print(f'Trading Criteria not met at period {trading_period_start} to {trading_period_end} with model {model} at K {k}, sigma {sigma:.1faren}')
                score[i, j] = -np.Inf
                fail_count += 1
    if fail_count == len(K_range) * len(sigma_range):
        print(
            f'Trading Criteria not met at period {trading_period_start} to {trading_period_end} with model {model}')
    ind_k, ind_s = np.unravel_index(score.argmax(), score.shape)

    return K_range[ind_k], sigma_range[ind_s]

def best_K_and_sigma_for_all(df_returns, prediction_path,
                             K_range, sigma_range,
                            models,
                          start_indices, signal_length,
                            assumed_max_lag,
                          criterion='raw returns'):

    end_indices = np.array(start_indices) + signal_length
    result = np.zeros((len(start_indices), len(models)), dtype=tuple)
    for i in range(len(start_indices)):
        train_period_start = start_indices[i]
        train_period_end = end_indices[i]
        for j in range(len(models)):
            model = models[j]
            K_sigma = best_K_and_sigma(df_returns,prediction_path,K_range,sigma_range,model,
                                        train_period_start,train_period_end,assumed_max_lag, criterion)
            result[i,j] = K_sigma

    start_end_indices = [(start_indices[t], end_indices[t]) for t in range(len(start_indices))]
    df_results = pd.DataFrame(result, index=start_end_indices, columns=models)
    return df_results

def concat_PnL_real2(model, prediction_path, folder_name,
                    start, end, signal_length,
                    trading_period,
                    return_excess=True,
                    return_SR=True):
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

    return_types = ['raw returns']
    if return_excess:
        return_types = ['raw returns', 'mkt excess returns', 'leader excess returns', 'leader raw returns',
                        'leader mkt excess returns']
    PnL_list_dict = {type: [] for type in return_types}

    for train_start in start_indices:
        train_end = train_start + signal_length
        file_name = f'start{train_start}end{train_end}trade{trading_period}'
        with open(f'{prediction_path}/{folder_name}/{file_name}.pkl', 'rb') as f:
            trading = pickle.load(f)

        for return_type in return_types:
            try:
                pnl = trading[model]['portfolio average'][return_type]['PnL']['average']
                assert len(pnl) == trading_period
            except:
                pnl = np.empty((trading_period))
                pnl[:] = np.nan

            PnL_list_dict[return_type].append(pnl)

    PnL = {return_type: np.concatenate(PnL_list_dict[return_type]) for return_type in return_types}

    if return_SR:
        SR = {type: annualized_sharpe_ratio(returns[~np.isnan(returns)]) for type, returns in PnL.items()}
        return PnL, SR
    else:
        return PnL
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
    prediction_path = '../results/real/full_non-negative_affinity'
    PnL_folder_name = 'PnL_real_single_weighted_both'
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
    # df_results = best_K_and_sigma_for_all(df_returns,
    #                                       prediction_path,
    #                          K_range, sigma_range,
    #                          models,
    #                          start_indices, signal_length,
    #                                       assumed_max_lag=2,
    #                          criterion='raw returns')
    #
    folder_path = prediction_path + '/' + PnL_folder_name + '/'
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
    best_K_sigma_path = folder_path + 'best_k_sigma.csv'
    # df_results.to_csv(best_K_sigma_path)

    df_results = pd.read_csv(best_K_sigma_path, index_col=0).applymap(eval)
    
    # ###--------------------- Trading by sliding window ----------------------------###
    for row_num, index in tqdm(enumerate(df_results.index)):
        train_period_start, train_period_end = string_to_int(index)
        #train_period_start, train_period_end = index
        trading_results_models = {}
        for col_num, model in enumerate(df_results.columns):
            K, sigma = df_results.iloc[row_num, col_num]
    
            trading = trading_real_data(data_path, prediction_path, K, sigma, model,
                              train_period_start=train_period_start,
                              train_period_end=train_period_start + signal_length,
                              out_of_sample=True,
                              trading_period=retrain_period,
                              assumed_max_lag=2,
                              hedge=True)
            trading_results_models[model] = trading
    
        file_name = f'start{train_period_start}end{train_period_end}trade{retrain_period}'
    
        with open(folder_path + file_name + '.pkl', 'wb') as f:
            pickle.dump(trading_results_models, f)

    ###---------------- Concatenate segments of trading results together ----------------------###
    PnL_concat_dict = {}
    for model in models:

        PnL, SR = concat_PnL_real2(
            model, prediction_path, PnL_folder_name,
            start, end,
            signal_length, retrain_period,
            return_excess=True)

        PnL_concat_dict[model] = \
            {'PnL': PnL,
            'annualized SR': SR}

    file_name = f'start{start}end{end}_length{signal_length}_trade{retrain_period}'
    with open(prediction_path + '/' + PnL_folder_name + '/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(PnL_concat_dict, f)


    # df_returns = pd.read_csv(data_path, index_col=0)
    # k_list = []
    # sigma_list = []
    # for model in models:
    #     k, sigma = best_K_and_sigma(df_returns, K_range,
    #                                 prediction_path, sigma_range,
    #              model,
    #               train_period_start=5,
    #               train_period_end=55,
    #               criterion='raw returns')
    #     k_list.append(k)
    #     sigma_list.append(sigma)
    # print(k_list)
    # print(sigma_list)


#
#
# def strategy_plain(returns, lag_matrix, shifts, watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2,
#                    rank='plain',
#                    hedge='no'):
#     """
#
#     Args:
#         returns:
#         lag_matrix:
#         watch_period:
#         hold_period:
#         leader_prop:
#         lagger_prop:
#         rank:
#         hedge:
#
#     Returns: the simple returns of the asset at each time step
#
#     """
#     result = []
#     signs = []
#     df = pd.DataFrame(lag_matrix)
#     L, N = returns.shape
#     returns = pd.DataFrame(returns.T)
#
#     if rank == 'plain':
#         ranking = np.mean(lag_matrix, axis=1)
#
#     elif rank == 'Synchro':
#         ranking = alignment.SVD_NRS(lag_matrix)[0]
#
#     sort_index = np.argsort(ranking)
#     lead_ind = sort_index[:int(leader_prop * N)]
#     lag_ind = sort_index[-int(lagger_prop * N):]
#
#     # calculate the average lag between the leader and lagger groups
#     lag_mat_nan = lag_matrix.copy()
#     np.fill_diagonal(lag_mat_nan, np.nan)
#     leaders = lag_mat_nan[:, lead_ind]
#     laggers = lag_mat_nan[:, lag_ind]
#     ahead = np.mean(leaders[~np.isnan(leaders)]) - np.mean(laggers[~np.isnan(laggers)])
#     ahead = max(0, round(ahead) - 1)
#     # 选取leader和lagger
#
#     # 找到leader和lagger的return
#     leader_returns = returns.iloc[lead_ind]
#     lagger_returns = returns.iloc[lag_ind]
#
#     size = len(lagger_returns.columns)
#     for i in range(watch_period, L - hold_period):
#         # this part is written based on simple returns
#         signal = np.sign(np.mean(leader_returns[leader_returns.columns[i - watch_period:i]].sum(axis=1), axis=0))
#         # hold period denotes the number of consecutive days we trade the laggers close to close
#         alpha = signal * np.mean(lagger_returns[lagger_returns.columns[ahead + i: ahead + i + hold_period]].sum(axis=1),
#                                  axis=0)
#         if hedge == 'no':
#             result.append(alpha)
#         elif hedge == 'mkt':
#             alpha2 = alpha - signal * (returns.loc['SPY'][returns.columns[i:i + hold_period]].sum(axis=0))
#             result.append(alpha2)
#         elif hedge == 'lead':
#             alpha2 = alpha - signal * np.mean(leader_returns[leader_returns.columns[i]])
#             result.append(alpha2)
#         signs.append(int(signal))
#         # print(i)
#         # print(alpha2)
#     return result, signs

# def trading_real_data_multiple(data_path, prediction_path,
#                                K_range, sigma_range,
#                       train_period_start=0,
#                       train_period_end=500,
#                       out_of_sample=True,
#                       trading_period=50,
#                       assumed_max_lag=5,
#                       hedge=False,
#                       **trading_kwargs):
#     """
#     run trading strategy on real stocks returns data.
#
#     """
#     trading = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}
#     with open(prediction_path + f'/signal_estimates/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
#         estimates = pickle.load(f)
#     with open(prediction_path + f'/lag_vectors/start{train_period_start}end{train_period_end}.pkl', 'rb') as f:
#         lag_vectors = pickle.load(f)
#
#     # load returns data
#     df_returns = pd.read_csv(data_path, index_col=0)
#
#     # # compute signals and returns based on excess returns (hedged with SPY)
#     # if hedge:
#     #     df_returns = df_returns - df_returns.loc['SPY']
#
#     # trade on untrained data if out_of_sample is True
#     if out_of_sample:
#         # returns = df_returns.iloc[:, train_period_end:train_period_end + trading_period]
#         trading_period_start = train_period_end
#         trading_period_end = train_period_end + trading_period
#     else:
#         # returns = df_returns.iloc[:, train_period_start:train_period_end]
#         trading_period_start = train_period_start
#         trading_period_end = train_period_end
#     models = ['pairwise', 'sync', 'spc-homo', 'het']
#     for k in K_range:
#         for sigma in sigma_range:
#             for model in models:
#                 trading[f'K={k}'][f'sigma={sigma:.2g}'][model] = \
#                 trading_single(df_returns, lag_vectors, estimates,
#                                k, sigma, model,
#                                trading_period_start, trading_period_end,
#                                assumed_max_lag, hedge, **trading_kwargs)
#
#     if out_of_sample:
#         file_name = f'start{train_period_start}end{train_period_end}trade{trading_period}'
#     else:
#         file_name = f'start{train_period_start}end{train_period_end}_insample'
#     if hedge:
#         file_name = file_name + 'excess'
#     with open('../results/PnL_real_excess/' + file_name + '.pkl', 'wb') as f:
#         pickle.dump(trading, f)

# def strategy_multiple_lags(returns, lag_matrix, watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2,
#                            rank='plain',
#                            hedge='no'):
#     L, N = returns.shape
#     returns = returns.T
#     # lag_ij = shift_i - shift_j
#     # positive means i lag j
#     ranking = np.mean(np.sign(lag_matrix), axis=1)  # vanilla
#     ranking1 = np.mean(lag_matrix, axis=1)  # plain
#     ranking2 = alignment.SVD_NRS(lag_matrix)[0]  # synchro
#     sort_index = np.argsort(ranking)  # ascending
#     lag_ind = sort_index[-int(lagger_prop * N):]
#     # sort_index1 = np.argsort(ranking1)  # ascending
#     # lag_ind1 = sort_index1[-int(lagger_prop * N):]
#     # sort_index2 = np.argsort(ranking2)  # ascending
#     # lag_ind2 = sort_index2[-int(lagger_prop * N):]
#     # lag_mat_nan = lag_matrix.copy()
#     # np.fill_diagonal(lag_mat_nan, np.nan)
#     # leaders = lag_mat_nan[lead_ind,:]
#     laggers = lag_matrix[lag_ind, :].astype(int)
#     # for every lagger we trade, find all the leaders and respective lags
#     leaders_list_by_laggers = [[(i, lag) for i, lag in enumerate(row) if lag > 0] for row in laggers]
#     lagger_returns = returns[lag_ind, :]
#
#     portfolio_returns = []
#     for t in range(watch_period, L - hold_period):
#         signals_by_leader = [np.sum([returns[p[0], t - p[1]] for p in l if (t - p[1] >= 0)]) \
#                              for l in leaders_list_by_laggers]
#         # weights proportional to the strength of signals, sum of absolute values equal to 1
#         weights = winsorize(signals_by_leader)
#         weights = weights / np.sum(np.abs(weights) + 1e-9)
#         # total returns of the weighted portfolio
#         alpha = np.dot(weights, np.sum(lagger_returns[:, t:t + hold_period], axis=1))
#         # alpha1 = np.average(np.sum(lagger_returns[:,t:t+hold_period],axis=1),weights=signals_by_leader)
#         # assert abs(alpha - alpha1) < 1e-8
#         if hedge == 'no':
#             portfolio_returns.append(alpha)
#         elif hedge == 'mkt':
#             alpha2 = alpha - np.sum(signals_by_leader) * (
#                 returns.loc['SPY'][returns.columns[t:t + hold_period]].sum(axis=0))
#             portfolio_returns.append(alpha2)
#     return portfolio_returns