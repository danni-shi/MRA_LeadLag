from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import utils
import random

def fn_gp_smooth(X_train,y_train, sigma):
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=5*sigma**2, n_restarts_optimizer=2
    )
    gp.fit(X_train, y_train)
    
    return gp.predict(X_train)

def interp(x_interp, y):
    y_interp = np.interp(x_interp, np.arange(L), y)
    
    return y_interp
k = 2
sigma = 1.8
data_path = '../../data/data500_logreturns/'
# data_path = '../data_n=500/'
max_shift= 0.1
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
classes_true = observations_mat['classes'].flatten() - 1
X_est = results_mat['x_est']
P_est = results_mat['p_est'].flatten()
X_true = results_mat['x_true']

L, N = observations.shape
scale  = 10
num_sample_pts = scale * (L-1) + 1
x_interp = np.linspace(0,L-1,num_sample_pts)
num_cand = 5
random.seed(42)
rand_ind = random.sample(range(N), num_cand)
#rand_ind = range(100,100+num_cand)
observations_smoothed = np.array([fn_gp_smooth(x_interp.reshape(-1,1), interp(x_interp, observations[:,i]),sigma) for i in rand_ind]).T

x = np.arange(L)
#plot all the candidate signals along with their class labels (depicted as colors)
fig0, ax = plt.subplots(num_cand,1,sharex=True, figsize=(10,3*num_cand))
for i in range(num_cand):
    ax[i].plot(x,observations[:,rand_ind[i]],linestyle = ':',label='Original')
    ax[i].plot(x,np.roll(X_true[:,classes_true[rand_ind[i]]],shifts[rand_ind[i]]),linestyle = '-', label = 'True ref signal')
    y = observations_smoothed[:,i]
    # y0 = y[range(0,len(observations_smoothed[:,i]),scale)]
    # obs_smoothed, lag = utils.align_to_ref(y0, \
    #     observations[:,rand_ind[i]])
    ax[i].plot(x_interp,y,color=[.5,.5,.5],linestyle='--',label=f'GP smoothed')
        
    ax[i].set_ylabel("X"+str(rand_ind[i])+"(t)")
    ax[i].legend()
ax[i].set_xlabel("t")   
plt.savefig(f'../plots/GP/smoothed_observation_noise{10*sigma:.2g}_0.png')