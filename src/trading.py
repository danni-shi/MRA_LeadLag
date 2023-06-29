import numpy as np
import pandas as pd
import alignment
import warnings
import time
import scipy.io as spio
import pickle
from tqdm import tqdm

"""
这是通过为股票进行全局排序从而进行选股择的策略
t：回看窗口
modulo：更新leader的频率
matrix：相关性矩阵所在文件夹的路径的list, at different days
daily_return：每日return的dataFrame
Tlag: roll-over从而控制TVR的系数 （0-1）
leader_prop: 选取leader的百分比（默认20%）
lagger_prop：选取lagger的百分比（默认50%）
neutralize: 选择是否进行neutralization
"""


def cum_returns(returns, return_type):
    """"
    returns: list like, the returns of an asset
    return_type: string denoting the type of returns given by the list 'returns'. 'simple', 'percentage' or 'log'

    return: list like, the cumulative returns of the asset at each time step
    """
    if return_type == 'simple':
        cum_returns = np.cumsum(returns)
    if return_type == 'percentage':
        cum_returns = np.cumprod(returns)
    if return_type == 'log':
        cum_returns = np.cumprod(np.exp(returns))

    return cum_returns


## Some performance metrics
def annualized_sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns) * np.sqrt(252)


from scipy import stats


def corr_SP(returns, signals):
    res = stats.spearmanr(returns, signals)
    return res.correlation, res.pvalue


def hit_ratio(returns, signals):
    signals_nonzero = signals[signals != 0]
    returns_nonzero = returns[signals != 0]
    frac = np.mean(np.sign(returns_nonzero) == np.sign(signals_nonzero))
    return frac


def long_ratio(signals):
    signals = signals[signals != 0]
    return np.mean(np.sign(signals) == 1)


from sklearn.linear_model import LinearRegression


def regression_score(returns, signals):
    reg = LinearRegression().fit(signals.reshape(-1, 1), returns)
    return reg.score(signals.reshape(-1, 1), returns)


def financial_stats(returns, signals):
    stats_dict = {
        'annualized SR': annualized_sharpe_ratio(returns),
        'corr SP': corr_SP(returns, signals)[0],
        'corr SP p-value': corr_SP(returns, signals)[1],
        'hit ratio': hit_ratio(returns, signals),
        'long ratio': long_ratio(signals),
        'reg R2': regression_score(returns, signals)
    }
    return stats_dict


def class_average_returns_each_group(returns_dict_list):
    result = {}
    mean_result = {}
    for returns_dict in returns_dict_list:
        for group, returns in returns_dict.items():
            value = result.get(group, [])
            value.append(returns)
            result[group] = value
    for group, returns in result.items():
        mean_result[group] = np.mean(returns, axis=0)

    return mean_result


def winsorize(data, percentiles=(5, 95)):
    # Define the percentile thresholds for winsorizing
    lower_percentile = percentiles[0]
    upper_percentile = percentiles[1]

    # Calculate the threshold values
    lower_threshold = np.percentile(data, lower_percentile)
    upper_threshold = np.percentile(data, upper_percentile)

    # Winsorize the values
    winsorized_data = np.clip(data, lower_threshold, upper_threshold)

    return winsorized_data


def lag_mat_to_vec(lag_mat):
    vec = np.mean(lag_mat, axis=1)
    vec = vec - np.min(vec)
    return np.round(vec).astype(int)


def PnL_two_groups(returns, leaders, laggers, lag, watch_period=1, hold_period=1, return_leader_pnl = False):
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

    for t in range(ahead + watch_period + hold_period, L + 1):
        signal = np.sum(leader_returns[:, t - ahead - watch_period - hold_period:t - ahead - hold_period])
        alpha = np.sign(signal) * np.mean(np.sum(lagger_returns[:, t - hold_period:t], axis=1),
                                          axis=0)
        leader_alpha = np.sign(signal) * np.mean(np.sum(leader_returns[:, t - hold_period:t], axis=1),
                                          axis=0)

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
                        hedge=False):
    """
    trading strategy to work on different lead-lag group pairs in the same class

    Returns: returns, fin stats on each group pair and the simple average across the group pairs

    """
    # including returns from dates before trading takes place to construct trading signals
    # pi, r, _ = alignment.SVD_NRS(lag_matrix)
    # lag_vector = np.array(np.round(r), dtype=int)
    sub_returns = returns.iloc[:, trading_start - days_advanced:trading_end]
    PnL = {}
    signals = {}
    PnL_excess = {}

    sub_returns = np.array(sub_returns)
    for l1 in lags:
        for l2 in lags:
            if l1 < l2:
                leaders = np.where(lag_vector == l1)[0]
                laggers = np.where(lag_vector == l2)[0]
                lag = l2 - l1
                pnl, signal = PnL_two_groups(sub_returns, leaders, laggers, lag, watch_period, hold_period)
                PnL[f'{l1}->{l2}'] = pnl[days_advanced:]
                signals[f'{l1}->{l2}'] = signal[days_advanced:]
    # calculate the simple average of PnL of each group pair
    PnL['class average'] = np.nanmean(np.stack(list(PnL.values())), axis=0)
    # PnL_excess['class average'] = np.nanmean(np.stack(list(PnL_excess.values())), axis=0)

    # fill nans with 0 for every value in the results dictionary
    for values in PnL.values():
        values[np.isnan(values)] = 0
    # for values in PnL_excess.values():
    #     values[np.isnan(values)] = 0

    results_dict = group_performance(PnL, signals)
    #results_excess_dict = group_performance(PnL_excess, signals)

    return results_dict


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
    fin_stats_by_metric['annualized SR']['class average'] = annualized_sharpe_ratio(PnL['class average'])
    results_dict['financial_stats'] = fin_stats_by_metric

    return results_dict


def strategy_multiple_lags(returns, lag_matrix, watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2,
                           rank='plain',
                           hedge='no'):
    L, N = returns.shape
    returns = returns.T
    # lag_ij = shift_i - shift_j
    # positive means i lag j
    ranking = np.mean(np.sign(lag_matrix), axis=1)  # vanilla
    ranking1 = np.mean(lag_matrix, axis=1)  # plain
    ranking2 = alignment.SVD_NRS(lag_matrix)[0]  # synchro
    sort_index = np.argsort(ranking)  # ascending
    lag_ind = sort_index[-int(lagger_prop * N):]
    # sort_index1 = np.argsort(ranking1)  # ascending
    # lag_ind1 = sort_index1[-int(lagger_prop * N):]
    # sort_index2 = np.argsort(ranking2)  # ascending
    # lag_ind2 = sort_index2[-int(lagger_prop * N):]
    # lag_mat_nan = lag_matrix.copy()
    # np.fill_diagonal(lag_mat_nan, np.nan)
    # leaders = lag_mat_nan[lead_ind,:]
    laggers = lag_matrix[lag_ind, :].astype(int)
    # for every lagger we trade, find all the leaders and respective lags
    leaders_list_by_laggers = [[(i, lag) for i, lag in enumerate(row) if lag > 0] for row in laggers]
    lagger_returns = returns[lag_ind, :]

    portfolio_returns = []
    for t in range(watch_period, L - hold_period):
        signals_by_leader = [np.sum([returns[p[0], t - p[1]] for p in l if (t - p[1] >= 0)]) \
                             for l in leaders_list_by_laggers]
        # weights proportional to the strength of signals, sum of absolute values equal to 1
        weights = winsorize(signals_by_leader)
        weights = weights / np.sum(np.abs(weights) + 1e-9)
        # total returns of the weighted portfolio
        alpha = np.dot(weights, np.sum(lagger_returns[:, t:t + hold_period], axis=1))
        # alpha1 = np.average(np.sum(lagger_returns[:,t:t+hold_period],axis=1),weights=signals_by_leader)
        # assert abs(alpha - alpha1) < 1e-8
        if hedge == 'no':
            portfolio_returns.append(alpha)
        elif hedge == 'mkt':
            alpha2 = alpha - np.sum(signals_by_leader) * (
                returns.loc['SPY'][returns.columns[t:t + hold_period]].sum(axis=0))
            portfolio_returns.append(alpha2)

    return portfolio_returns


def strategy_plain(returns, lag_matrix, shifts, watch_period=1, hold_period=1, leader_prop=0.2, lagger_prop=0.2,
                   rank='plain',
                   hedge='no'):
    """

    Args:
        returns:
        lag_matrix:
        watch_period:
        hold_period:
        leader_prop:
        lagger_prop:
        rank:
        hedge:

    Returns: the simple returns of the asset at each time step

    """
    result = []
    signs = []
    df = pd.DataFrame(lag_matrix)
    L, N = returns.shape
    returns = pd.DataFrame(returns.T)

    # df = pd.read_csv(matrix[i]) # NxN matrix where each element is a pairwise lag
    # df = df.set_index('Unnamed: 0')
    # df.columns = df1.columns
    # df.index = df1.index

    # date = daily_return.columns[i]

    if rank == 'plain':
        ranking = np.mean(lag_matrix, axis=1)

    elif rank == 'Synchro':
        ranking = alignment.SVD_NRS(lag_matrix)[0]

    sort_index = np.argsort(ranking)
    lead_ind = sort_index[:int(leader_prop * N)]
    lag_ind = sort_index[-int(lagger_prop * N):]

    # calculate the average lag between the leader and lagger groups
    lag_mat_nan = lag_matrix.copy()
    np.fill_diagonal(lag_mat_nan, np.nan)
    leaders = lag_mat_nan[:, lead_ind]
    laggers = lag_mat_nan[:, lag_ind]
    ahead = np.mean(leaders[~np.isnan(leaders)]) - np.mean(laggers[~np.isnan(laggers)])
    ahead = max(0, round(ahead) - 1)
    # 选取leader和lagger

    # 找到leader和lagger的return
    leader_returns = returns.iloc[lead_ind]
    lagger_returns = returns.iloc[lag_ind]

    size = len(lagger_returns.columns)
    for i in range(watch_period, L - hold_period):
        # this part is written based on simple returns
        signal = np.sign(np.mean(leader_returns[leader_returns.columns[i - watch_period:i]].sum(axis=1), axis=0))
        # hold period denotes the number of consecutive days we trade the laggers close to close
        alpha = signal * np.mean(lagger_returns[lagger_returns.columns[ahead + i: ahead + i + hold_period]].sum(axis=1),
                                 axis=0)
        if hedge == 'no':
            result.append(alpha)
        elif hedge == 'mkt':
            alpha2 = alpha - signal * (returns.loc['SPY'][returns.columns[i:i + hold_period]].sum(axis=0))
            result.append(alpha2)
        elif hedge == 'lead':
            alpha2 = alpha - signal * np.mean(leader_returns[leader_returns.columns[i]])
            result.append(alpha2)
        signs.append(int(signal))
        # print(i)
        # print(alpha2)
    return result, signs


def strategy_het(returns, trading_start, trading_end,
                 lag_matrix, classes, watch_period=1,
                 hold_period=1, class_threshold=None,
                 assumed_max_lag=5, market_excess=False,
                 ):
    """
    returns: NxT np array. N is the number of instruments and T is the number of time points

    Returns: the simple returns of the asset at each time step, averaged over different classes

    """

    class_labels = np.unique(classes)
    if not class_threshold:
        class_threshold = int(0.2 * returns.shape[0] / len(class_labels))

    results_dict = {}
    class_counts = []
    for c in class_labels:
        # count number of samples in class c
        count = np.count_nonzero(classes == c)
        if count > class_threshold:  # ignore classes with size below a certain threshold
            sub_lag_matrix = lag_matrix[classes == c][:, classes == c]
            min_group_size = 0.3 * int(len(sub_lag_matrix) / assumed_max_lag)
            lag_vector = lag_mat_to_vec(sub_lag_matrix)
            # pi, r, _ = alignment.SVD_NRS(lag_matrix)
            # lag_vector = np.array(np.round(r), dtype=int)
            lags, counts = np.unique(lag_vector, return_counts=True)
            lags = lags[counts >= min_group_size]

            if len(lags) > 1:
                min_lag = np.min(lags)
                lags -= min_lag
                lag_vector -= min_lag
                days_advanced = 0
                sub_returns = np.array(returns)[classes == c]

                results = strategy_lag_groups(
                    sub_returns, trading_start, trading_end,
                    days_advanced, lag_vector, lags,
                    watch_period, hold_period, min_group_size)

                results_dict[f'class {c}'] = results
                class_counts.append(sub_returns.shape[1])

    # average PnL of each group across all classes, if any valid results are produced
    if len(results_dict) > 0:
        PnL_group_list = [{i: results['PnL'][i] for i in results['PnL'] if i != 'class average'} for results in
                          results_dict.values()]
        PnL = class_average_returns_each_group(PnL_group_list)

        # average return weighted by class size
        pnl_average_list = [results['PnL']['class average'] for results in results_dict.values()]
        pnl_average = np.average(pnl_average_list, axis=0, weights=class_counts)
        PnL['class average'] = pnl_average
        SR = {group: annualized_sharpe_ratio(returns) for group, returns in PnL.items()}
        results_dict['portfolio average'] = {'PnL': PnL,
                                             'annualized SR': SR}

    return results_dict


def run_trading(data_path, K_range, sigma_range, max_shift=2, round=1, out_of_sample=True, **trading_kwargs):
    trading = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}

    with open(f'../results/signal_estimates/{round}.pkl', 'rb') as f:
        estimates = pickle.load(f)
    with open(f'../results/lag_matrices/{round}.pkl', 'rb') as f:
        lag_matrices = pickle.load(f)
    n = 10
    for k in K_range:
        for sigma in sigma_range:
            observations_path = data_path + '_'.join(['observations',
                                                      'noise' + f'{sigma:.2g}',
                                                      'shift' + str(max_shift),
                                                      'class' + str(k) + '.mat'])
            # load returns
            if out_of_sample:
                dataset = 'test'
            else:
                dataset = 'train'
            observations_mat = spio.loadmat(observations_path)
            observations = observations_mat['data_' + dataset]
            shifts = observations_mat['shifts'].flatten()
            index = observations_mat['index_' + dataset].flatten()
            lag_mat_dict = lag_matrices[f'K={k}'][f'sigma={sigma:.2g}']
            classes_spc = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['spc']
            classes_est = estimates[f'K={k}'][f'sigma={sigma:.2g}']['classes']['het']

            for model, lag_mat in lag_mat_dict.items():
                if model == 'het':
                    classes = classes_est
                else:
                    classes = classes_spc
                trading[f'K={k}'][f'sigma={sigma:.2g}'][model] = strategy_het(observations.T, lag_mat,
                                                                              classes, shifts=shifts,
                                                                              **trading_kwargs)

    with open(f'../results/PnL/{round}.pkl', 'wb') as f:
        pickle.dump(trading, f)


def run_wrapper(round):
    data_path = '../../data/data500_shift2_pvCLCL_init2_set1/' + str(round) + '/'
    K_range = [2]
    sigma_range = np.arange(0.5, 2.1, 0.5)
    run_trading(data_path=data_path, K_range=K_range,
                sigma_range=sigma_range, round=round,
                out_of_sample=True)


def concat_PnL_real(K, sigma, model,
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
        return_types.append('excess returns')
    PnL_list_dict = {type: [] for type in return_types}

    for train_start in start_indices:
        train_end = train_start + signal_length
        file_name = f'start{train_start}end{train_end}trade{trading_period}excess'
        folder_name = 'PnL_real_excess'
        with open(f'../results/{folder_name}/{file_name}.pkl', 'rb') as f:
            trading = pickle.load(f)
        with open(f'../results/PnL_real/{file_name}.pkl', 'rb') as f:
            trading_test = pickle.load(f)

        for return_type in return_types:
            try:
                pnl = trading[f'K={K}'][f'sigma={sigma:.2g}'][model]['portfolio average'][return_type]['PnL']['class average']
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


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    # for testing run without parallelization
    real_data = True
    if real_data:
        data_path = '../../data/pvCLCL_clean.csv'
        K_range = [1, 2, 3]
        sigma_range = np.arange(0.2, 2.1, 0.2)
        start = 5;
        end = 500
        retrain_period = 10
        signal_length = 50
        start_indices = range(start, end, retrain_period)
        # for train_period_start in tqdm(start_indices):
        #     trading_real_data(data_path, K_range, sigma_range,
        #                       train_period_start=train_period_start,
        #                       train_period_end=train_period_start + signal_length,
        #                       out_of_sample=True,
        #                       trading_period=retrain_period,
        #                       assumed_max_lag=2,
        #                       hedge=True)
        models = ['pairwise', 'sync', 'spc-homo', 'het']
        PnL_concat_dict = {f'K={k}': {f'sigma={sigma:.2g}': {} for sigma in sigma_range} for k in K_range}
        for k in K_range:
            for sigma in sigma_range:
                for model in models:
                    # PnL, SR = concat_PnL_real(
                    #     k, sigma, model, start, end,
                    #     signal_length, retrain_period,
                    #     return_excess=False)
                    PnL, SR = concat_PnL_real(
                        k, sigma, model, start, end,
                        signal_length, retrain_period,
                        return_excess=True)

                    PnL_concat_dict[f'K={k}'][f'sigma={sigma:.2g}'][model] = \
                        {'PnL': PnL,
                        'annualized SR': SR}

        file_name = f'start{start}end{end}_length{signal_length}_trade{retrain_period}'
        with open('../results/PnL_real_excess/' + file_name + '.pkl', 'wb') as f:
            pickle.dump(PnL_concat_dict, f)
    else:
        rounds = 4
        start = time.time()
        for i in range(rounds):
            run_wrapper(round=i + 1)
        # inputs = range(1, 1+rounds)

        # with multiprocessing.Pool() as pool:
        #     # use the pool to apply the worker function to each input in parallel
        #     pool.map(run_wrapper, inputs)
        #     pool.close()
        print(f'time taken to run {rounds} rounds: {time.time() - start}')

"""

n = None
test = False
max_shift = 0.1
assumed_max_lag = 10
models = None
data_path = '../../data/data500_OPCLreturns_init3/'
return_signals = False
round = 1
sigma = 0.1
k = 2
cum_pnl = []
sigma_range = np.arange(0.1, 2.1, 0.5)
return_type = 'simple'

lags = 1

for sigma in sigma_range:
    # read data produced from matlab code base
    observations, shifts, classes_true, X_est, P_est, X_true = main.read_data(
        data_path=data_path + str(round) + '/',
        sigma=sigma,
        max_shift=max_shift,
        k=k,
        n=n
    )

    # calculate clustering and pairwise lag matrix
    classes_spc, classes_est, lag_matrix, ARI_dict = main.clustering(observations=observations,
                                                                     k=k,
                                                                     classes_true=classes_true,
                                                                     assumed_max_lag=assumed_max_lag,
                                                                     X_est=X_est
                                                                     )

    sub_observations = observations[:, shifts < 2]
    sub_lag_matrix = lag_matrix[shifts < 2][:, shifts < 2]
    sub_classes_true = classes_true[shifts < 2]
    #
    results = strategy_het(sub_observations, sub_lag_matrix, sub_classes_true)

    cum_pnl.append(results)
# cmap = {1:'green',-1:'red'}
# colors_mapped = [cmap[c] for c in signs]
# sns.barplot(x=np.arange(len(results)),y=results,palette=colors_mapped)

fig, axes = plt.subplots(len(sigma_range), 1, figsize=(10, 5 * len(sigma_range)))
n = 15
for i in range(len(sigma_range)):
    results = cum_pnl[i]
    cum_returns = np.cumsum(results)
    sns.lineplot(x=np.arange(n), y=cum_returns[:n], ax=axes[i], label=f'sigma = {sigma_range[i]:.1g}')
    axes[i].set_xlabel('day')
    axes[i].set_ylabel('cumulative return')
    axes[i].legend()
    # axes[i].set_title(f'sigma = {sigma_range[i]:.1g}')
plt.show()
"""
