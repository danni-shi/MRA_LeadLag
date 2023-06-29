import autograd.numpy as np
from autograd import jacobian
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import matlab.engine

import utils

def optimise_matlab(data, sigma, K, X0=[], p0=[], opts=[], w=[], nextrainits=0):
    if opts == []:
         opts = {'maxiter': 200,
                 'tolgradnorm': 1e-7,
                 'tolcost': 1e-18,
                 'verbosity': 0}
    eng = matlab.engine.start_matlab()
    eng.addpath('/Users/caribbeanbluetin/Desktop/Research/MRA_LeadLag/MRA_MATLAB')
    # opts_struct = eng.struct(opts)
    # X_est, p_est, problem = eng.MRA_het_mixed_invariants_free_p(data, sigma, K, X0, p0, opts, w, nextrainits)
    data = data.astype(np.float64)
    K = float(K)
    X_est, p_est, problem = eng.MRA_het_mixed_invariants_free_p(data, sigma, K, [], [], opts, nargout=3)
    eng.quit()
    
    return np.array(X_est), np.array(p_est).flatten(), problem


def optimise_manopt(data, sigma, X0 = None, extra_inits = 0, verbosity = 1):
    assert isinstance(extra_inits, int)
    L, N = data.shape
    mean_est, P_est, B_est = utils.invariants_from_data(data)
    # for test only
    # with open('test.npy', 'rb') as f:
    #     sigma = float(np.load(f))
    #     X0 = np.load(f) 
    #     mean_est = float(np.load(f))
    #     P_est = np.load(f) 
    #     B_est = np.load(f)  
    # L = len(X0)
    # assert L > 1

    manifold = pymanopt.manifolds.Euclidean(L,1)
    cost, grad, hess = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
    # grad = hess = None
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad, euclidean_hessian=hess)
    optimizer = pymanopt.optimizers.TrustRegions(min_gradient_norm = 1e-7, max_iterations = 50, verbosity = verbosity)
    if X0 is not None : 
        if X0.ndim == 1:
            X0 = X0.reshape(-1,1)
    result = optimizer.run(problem, initial_point=X0)
    X_est = result.point
    result_cost = result.cost
    
    if extra_inits > 0:
        for i in range(extra_inits):
            result = optimizer.run(problem)
            if result.cost < result_cost:
                result_cost = result.cost
                X_est = result.point
            
    return X_est



def create_cost_function(mean_est, P_est, B_est, sigma, manifold):
    euclidean_gradient = euclidean_hessian = None
    
    @pymanopt.function.autograd(manifold)
    def cost(X):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        L, K = X.shape
        FX = np.fft.fft(X, axis = 0)
        A = make_A(L)
        
        # limits
        M1 = np.mean(X)
        M2 = abs(FX)**2 + L * sigma**2 * np.ones((L,K))
        M3 = mean_est * (sigma**2) * (L**2) * A
        # matmul = np.zeros((L,L,K))
        for k in range(K):
            y =FX[:,k]
            mat1 = utils.circulant(y)
            mat2 = np.outer(y, np.conjugate(y))
            M3 = M3 + mat1 * mat2
            
        M3_min_Best = M3 - B_est
        
        # compute coefficients
        a1 = L**2
        a2 = 1/(L*(2+sigma**2) )
        a3 = 1/(L**2*(3+sigma**4))
        
        scale = 3 + sigma**4
        # assert M2.shape == P_est.shape
        f = scale * 0.5 * \
            (a1*(M1-mean_est)**2 + \
                a2*np.linalg.norm(M2.flatten() - P_est)**2 + \
                    a3*np.linalg.norm(M3_min_Best, 'fro')**2 
            )
                
        return f
    
    @pymanopt.function.numpy(manifold)
    def grad(X):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        L, K = X.shape
        FX = np.fft.fft(X, axis = 0)
        A = make_A(L)
        
        # limits
        M1 = np.mean(X)
        M2 = abs(FX)**2 + L * sigma**2 * np.ones((L,K))
        M3 = mean_est * (sigma**2) * (L**2) * A
        for k in range(K):
            y =FX[:,k]
            mat1 = utils.circulant(y)
            mat2 = np.outer(y, np.conjugate(y)) 
            M3 += mat1 * mat2
            
        M3_min_Best = M3 - B_est
    
        # compute coefficients
        a1 = L**2
        a2 = 1/(L*(2+sigma**2) )
        a3 = 1/(L**2*(3+sigma**4))
        gradX = (a1/L) * (M1 - mean_est) * np.ones((L,K)) + \
            2 * L * a2 * np.fft.ifft((M2-P_est.reshape(-1,1))*FX, axis = 0) 
        grad_list = []
        test_gradX = gradX.flatten()
        for k in range(K):
            vec = gradX[:,k]
            vec = vec + a3 * DBx_adj(FX, M3_min_Best)
            grad_list.append(vec.reshape(-1,1))
        gradX = np.concatenate(grad_list, axis = 1)
        # simple test for K = 1
        test_gradX = test_gradX + a3 * DBx_adj(FX, M3_min_Best)
        assert np.linalg.norm(test_gradX - gradX.flatten()) < 1e-6
        scale = 3 + sigma**4
        gradX = scale * np.real(gradX)
        # gradX = manifold.euclidean_to_riemannian_gradient(X, gradX)
        # print('used defined grad')
        return gradX
    
    @pymanopt.function.numpy(manifold)
    def hess(X,y):
        assert X.shape == y.shape
        grad_f = lambda x: grad(x).flatten()
        H = jacobian(grad_f)(X.flatten())
        # print('used defined hess')
        return H @ y
    
    return cost, grad, hess

def make_A(L):
    A = np.eye(L, dtype = 'complex_') 
    A[0,:] += np.ones(L)
    A[:,0] += np.ones(L)
    return A
    
def DBx_adj(y, W):
    y = y.reshape(-1,1)
    L = y.shape[0]   
    H = W * utils.circulant(np.conjugate(y))
    y_prime = np.conjugate(np.transpose(y))
    H_prime = np.conjugate(np.transpose(H))
    z = L * np.fft.ifft(utils.circulantadj(W * np.conjugate((y @ y_prime))) + (H + H_prime) @ y.reshape(-1,1),axis=0)
    return z.flatten()

# @pymanopt.function.autograd(manifold)
    # def cost(X):
    #     L = len(X)
    #     # print(type(X))
    #     FX = np.fft.fft(X)
    #     A = make_A(L)
        
    #     M1 = np.mean(X)
    #     M2 = abs(FX)**2 + L * sigma**2 * np.ones(L)
    #     # x = np.array(X)[:,i]
    #     mat1 = np.array([np.roll(FX,k) for k in range(L)])
    #     mat2 = np.outer(FX, np.conjugate(FX))
    #     matmul = mat1 * mat2
    #     M3 = matmul + mean_est * (sigma**2) * (L**2) * A
            
    #     # M3 = utils.bispectrum(X.reshape(L,1))[0] + mean_est * (sigma**2) * (L**2) * A
    
    #     M3_min_Best = M3 - B_est
        
    #     # compute coefficients
    #     a1 = L**2
    #     a2 = 1/(L*(2+sigma**2) )
    #     a3 = 1/(L**2*(3+sigma**4))
        
    #     scale = 3 + sigma**4
    #     assert M2.shape == P_est.shape
    #     f = scale * 0.5 * \
    #         (a1*(M1-mean_est)**2 + \
    #             a2*np.linalg.norm(M2 - P_est,2)**2 + \
    #                 a3*np.linalg.norm(M3_min_Best, 'fro')**2 
    #         )
                
    #     return f
    
    # @pymanopt.function.numpy(manifold)
    # def cost(X):
    #     return -np.trace(X.T @ B_est @ X)

    # @pymanopt.function.numpy(manifold)
    # def euclidean_gradient(X):
    #     return -2 * B_est @ X

    # @pymanopt.function.numpy(manifold)
    # def euclidean_hessian(X, H):
    #     return -2 * B_est @ H
    

    #return cost, grad, euclidean_hessian
    


