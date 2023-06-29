import autograd.numpy as np
from autograd import jacobian
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

import utils

def optimise_manopt(data, sigma, p, X0 = None, extra_inits = 0):
    assert isinstance(extra_inits, int)
    L, N = data.shape
    K = len(p)
    mean_est, P_est, B_est = utils.invariants_from_data(data)

    manifold = pymanopt.manifolds.Euclidean(L,K)
    cost, grad, hess = create_cost_function(p, mean_est, P_est, B_est, sigma, manifold)
    # hess = None
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad, euclidean_hessian=hess)
    optimizer = pymanopt.optimizers.TrustRegions(min_gradient_norm = 1e-7, max_iterations = 100, verbosity = 2)
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


def create_cost_function(p, mean_est, P_est, B_est, sigma, manifold):
    euclidean_gradient = euclidean_hessian = None
    
    @pymanopt.function.autograd(manifold)
    def cost(X): 
        if X.ndim == 1:
            X = X.reshape(-1,1)
        L, K = X.shape
        FX = np.fft.fft(X, axis = 0)
        A = make_A(L)
        
        # limits
        M1 = np.mean(np.dot(X, p))
        M2 = np.dot(abs(FX)**2, p) + L * sigma**2 * np.ones((L))
        M3 = mean_est * (sigma**2) * (L**2) * A
        # assert isinstance(M1, float)
        assert M2.shape == P_est.shape
        assert M3.shape == B_est.shape
        
        for k in range(K):
            y =FX[:,k]
            mat1 = utils.circulant(y)
            mat2 = np.outer(y, np.conjugate(y))
            M3 = M3 + p[k] * mat1 * mat2
            
        M3_min_Best = M3 - B_est
        
        # compute coefficients
        a1 = L**2
        a2 = 1/(L*(2+sigma**2) )
        a3 = 1/(L**2*(3+sigma**4))
        
        scale = 3 + sigma**4
        # assert M2.shape == P_est.shape
        f = scale * 0.5 * \
            (a1*(M1-mean_est)**2 + \
                a2*np.linalg.norm(M2 - P_est)**2 + \
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
        M1 = np.mean(np.dot(X, p))
        M2 = np.dot(abs(FX)**2, p) + L * sigma**2 * np.ones((L))
        M3 = mean_est * (sigma**2) * (L**2) * A
        assert M1.ndim == 0
        assert M2.shape == P_est.shape
        assert M3.shape == B_est.shape
        
        matmul = []
        for k in range(K):
            y =FX[:,k]
            mat1 = utils.circulant(y)
            mat2 = np.outer(y, np.conjugate(y))
            matmul.append(mat1 * mat2)
            M3 = M3 + p[k] * mat1 * mat2
            
        M3_min_Best = M3 - B_est
    
        # compute coefficients
        a1 = L**2
        a2 = 1/(L*(2+sigma**2) )
        a3 = 1/(L**2*(3+sigma**4))
        gradX = (a1/L) * (M1 - mean_est) * np.outer(np.ones(L), p) + \
            2 * L * a2 * np.fft.ifft(np.outer((M2-P_est), p)*FX, axis = 0) 
        gradX_list = []

        for k in range(K):
            vec = gradX[:,k]
            
            vec = vec + a3 * DBx_adj(FX[:,k], M3_min_Best)
            gradX_list.append(vec.reshape(-1,1))
        
        scale = 3 + sigma**4    
        gradX = np.concatenate(gradX_list, axis = 1)
        gradX = scale * np.real(gradX)
        
        
        grad = (gradX)
        return grad
    
    @pymanopt.function.numpy(manifold)
    def hess(X, yX):
        assert X.shape == yX.shape
        gradX = lambda x: grad(x)
        HX = jacobian(gradX)(X)     
        hessian = np.tensordot(HX, yX, 2)
        # print('used defined hess')
        return hessian
    
    return cost, grad, hess

def make_A(L):
    A = np.eye(L, dtype = 'complex_') 
    A[0,:] += np.ones(L)
    A[:,0] += np.ones(L)
    return A
    
def DBx_adj(y, W):
    if y.ndim == 1:
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
    


