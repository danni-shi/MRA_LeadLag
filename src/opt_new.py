import autograd.numpy as np
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

import utils

def make_A(L):
    A = np.eye(L, dtype = 'complex_') 
    A[0,:] += np.ones(L)
    A[:,0] += np.ones(L)
    return A

def create_cost_function(mean_est, P_est, B_est, sigma, manifold, autograd = True):
    euclidean_gradient = euclidean_hessian = None
    
    if autograd:
        
        @pymanopt.function.autograd(manifold)
        def cost(X):
            L = len(X)
            FX = np.fft.fft(X)
            A = make_A(L)
            
            M1 = np.mean(X)
            M2 = abs(FX)**2 + L * sigma**2 * np.ones(L)
            M3 = utils.bispectrum(FX.reshape(L,1)) + mean_est * (sigma**2) * (L**2) * A
            
            M3_min_Best = M3 - B_est
            
            # compute coefficients
            a1 = L**2
            a2 = 1/(L*(2+sigma**2))
            a3 = 1/(L**2*(3+sigma**4))
            
            scale = 3 + sigma**4
            f = scale * 0.5 * \
                (a1*(M1-mean_est)**2 + \
                    a2*np.linalg.norm(M2 - P_est)**2 + \
                        a3*np.linalg.norm(M3_min_Best, 'fro')**2 
                )
                    
            return f
    else:
        print('not using autograd')
    return cost, euclidean_gradient, euclidean_hessian
    
    
    
    
def optimise_manopt(data, sigma, X0, extra_inits = 0):
    assert isinstance(extra_inits, int)
    L, N = data.shape
    mean_est, P_est, B_est = utils.invariants_from_data(data)
    

    
    manifold = pymanopt.manifolds.Euclidean(L)
    
    cost, euclidean_gradient, euclidean_hessian = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.TrustRegions(min_gradient_norm = 1e-6, max_iterations = 200)
    result = optimizer.run(problem, initial_point=X0, )
    X_est = result.point
    result_cost = result.cost
    
    if extra_inits > 0:
        for i in range(extra_inits):
            result = optimizer.run(problem)
            if result.cost < result_cost:
                result_cost = result.cost
                X_est = result.point
            
    return X_est


    # X0 = np.arange(1,6)
    # mean_est =  0.04164962198820365
    # P_est = np.array([ 5.69322879, 10.34707154,  7.28589838,  7.28589838, 10.34707154])
    # B_est = np.array([[ 3.86926394+0.00000000e+00j,  3.75393181-1.35167485e-16j,
    #      0.76951695+8.18746113e-17j,  0.76951695-8.18746113e-17j,
    #      3.75393181+1.35167485e-16j],
    #    [ 3.75393181+1.35167485e-16j,  3.75393181+0.00000000e+00j,
    #      1.10766601+9.11979295e+00j, -0.38176641+1.35677705e+00j,
    #      1.10766601+9.11979295e+00j],
    #    [ 0.76951695-8.18746113e-17j,  1.10766601-9.11979295e+00j,
    #      0.76951695+0.00000000e+00j, -0.38176641+1.35677705e+00j,
    #     -0.38176641+1.35677705e+00j],
    #    [ 0.76951695+8.18746113e-17j, -0.38176641-1.35677705e+00j,
    #     -0.38176641-1.35677705e+00j,  0.76951695+0.00000000e+00j,
    #      1.10766601+9.11979295e+00j],
    #    [ 3.75393181-1.35167485e-16j,  1.10766601-9.11979295e+00j,
    #     -0.38176641-1.35677705e+00j,  1.10766601-9.11979295e+00j,
    #      3.75393181+0.00000000e+00j]])