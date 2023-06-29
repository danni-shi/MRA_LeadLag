from matplotlib import pyplot as plt
import utils
from optimization import create_cost_function, optimise_matlab
import scipy.io as spio
# import autograd.numpy as np
import random
import numpy as np
# import pymanopt
# import pymanopt.manifolds
# import pymanopt.optimizers
import alignment


# data path
k = 2
sigma = 0.1
max_shift= 0.1
assumed_max_lag = 10
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
    observations = observations_mat['data'][:, :n]
    shifts = observations_mat['shifts'].flatten()[:n]
    classes_true = observations_mat['classes'].flatten()[:n] - 1
    X_est = results_mat['x_est']
    P_est = results_mat['p_est'].flatten()
    X_true = results_mat['x_true']

    return observations, shifts, classes_true, X_est, P_est, X_true

#
# observations1, shifts1, classes_true1, X_est1, P_est1, X_true1 = read_data('../../data/data500_logreturns/', sigma, max_shift, k, n=None)
# observations2, shifts2, classes_true2, X_est2, P_est2, X_true2 = read_data('../../data/data500_logreturns_init3/1/',sigma, max_shift, k, n=None)
#
# lag_matrix1 = alignment.score_lag_mat(observations1,
#                                       max_lag=assumed_max_lag,
#                                     score_fn=alignment.alignment_similarity)[1]
#
# lag_matrix2 = alignment.score_lag_mat(observations2,
#                                       max_lag=assumed_max_lag,
#                                     score_fn=alignment.alignment_similarity)[1]
#
# shifts11 = shifts1[classes_true1==0]
# shifts12 = shifts1[classes_true1==1]
# X_est_sync1 = alignment.get_synchronized_signals(observations1,
#                                                 classes_true1,
#                                                 lag_matrix1)
#
# shifts21 = shifts1[classes_true2==0]
# shifts22 = shifts1[classes_true2==1]
# X_est_sync2 = alignment.get_synchronized_signals(observations2,
#                                                 classes_true2,
#                                                 lag_matrix2)


def lag_vec_to_mat(vec):
    # note this function is invariant to addition and subtraction
    # of the same value to every element of vec
    L = len(vec)
    vec = vec.reshape(-1,1)
    ones = np.ones((L,1))
    return vec @ ones.T - ones @ vec.T

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



def SVD_NRS_test(n_range, L_range, rep = 100):
    acc_array = np.zeros((len(L_range),len(n_range)))
    std_array = np.zeros((len(L_range), len(n_range)))
    for l in range(len(L_range)):
        for n in range(len(n_range)):
            acc = 0
            acc_list= []
            for i in range(rep):
                x = np.array(random.choices(range(n_range[n]), k = L_range[l]))
                H = lag_vec_to_mat(x)
                pi, r, tau = SVD_NRS(H, scale_estimator='median')
                r_rounded = np.array(np.round(r), dtype=int)
                r_rounded -= np.min(r_rounded)
                acc += np.mean(r_rounded == x)
                acc_list.append(np.mean(r_rounded == x))
            acc_array[l,n]=(acc/rep)
            std_array[l,n]=(np.std(acc_list))
    return acc_array, std_array

n_range = range(2,10)
L_range = [40,50,60,70,80]
accuracy, std = SVD_NRS_test(n_range = n_range,L_range = L_range)
# annotation = np.concatenate((accuracy.reshape(*accuracy.shape,1), std.reshape(*std.shape, 1)), axis=2)
# annotations = np.char.add(np.char.mod('%.2f', accuracy), '\n' + std.astype(str))
import seaborn as sns

ax = sns.heatmap(accuracy, annot=std, annot_kws={'va':'top','size':8}, fmt=".2f", cbar=False)
ax = sns.heatmap(accuracy, annot=True, annot_kws={'va':'bottom'}, xticklabels=n_range, yticklabels=L_range)
ax.set_xlabel('Number of discrete lags')
ax.set_ylabel('Length of time series')
ax.set_title('Accuracy and Std of SVD-NRS in Lag Recovery')
plt.show()
# sigma = 1
# X0 = np.arange(1,6).astype('float64')
# mean_est =  0.04164962198820365
# P_est = np.array([ 5.69322879, 10.34707154,  7.28589838,  7.28589838, 10.34707154])
# B_est = np.array([[ 3.86926394+0.00000000e+00j,  3.75393181-1.35167485e-16j,
#         0.76951695+8.18746113e-17j,  0.76951695-8.18746113e-17j,
#         3.75393181+1.35167485e-16j],
#     [ 3.75393181+1.35167485e-16j,  3.75393181+0.00000000e+00j,
#         1.10766601+9.11979295e+00j, -0.38176641+1.35677705e+00j,
#         1.10766601+9.11979295e+00j],
#     [ 0.76951695-8.18746113e-17j,  1.10766601-9.11979295e+00j,
#         0.76951695+0.00000000e+00j, -0.38176641+1.35677705e+00j,
#     -0.38176641+1.35677705e+00j],
#     [ 0.76951695+8.18746113e-17j, -0.38176641-1.35677705e+00j,
#     -0.38176641-1.35677705e+00j,  0.76951695+0.00000000e+00j,
#         1.10766601+9.11979295e+00j],
#     [ 3.75393181-1.35167485e-16j,  1.10766601-9.11979295e+00j,
#     -0.38176641-1.35677705e+00j,  1.10766601-9.11979295e+00j,
#         3.75393181+0.00000000e+00j]])
# with open('test.npy', 'wb') as f:
#     np.save(f, sigma)
#     np.save(f, X0)
#     np.save(f, mean_est)
#     np.save(f, P_est)
#     np.save(f, B_est)

# with open('test.npy', 'rb') as f:
#     sigma = float(np.load(f))
#     X0 = np.load(f) 
#     mean_est = float(np.load(f))
#     P_est = np.load(f) 
#     B_est = np.load(f)  
    
    
def test1():
    
    L = 5
    manifold = pymanopt.manifolds.Euclidean(L)
    cost, grad, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
    x = np.array([[1,2,3,4,5],
                [0,1,0,1,0],
                [1,1,1,1,1]])
    for i in range(x.shape[0]):
        y = cost(x[i])
        z = grad(x[i])
     


def test2(num=10, num_copies = 500, sigma = 0.1):
    max_shift = 0
    for i in range(num):
        L = random.randint(5,50)
        X = np.random.normal(0,1,L)
        X = (X-np.mean(X))/np.std(X)
        
        observations, shifts = utils.generate_data(X, num_copies,  max_shift, sigma, cyclic = True)
        
        mean_est, P_est, B_est = utils.invariants_from_data(observations)
        manifold = pymanopt.manifolds.Euclidean(L,1)
        cost, grad, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
        X = X.reshape(-1,1)
        print('singal length: ', L)
        print('cost at solution: ', cost(X)/L)
        print('grad norm at solution: ', np.linalg.norm(grad(X),2)/L)
        print('/n')
