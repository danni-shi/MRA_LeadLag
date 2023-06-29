import autograd.numpy as np
from autograd import jacobian
# import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

import utils

def optimise_manopt(data, sigma, K, XP0 = None, extra_inits = 0):
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
    if XP0 is not None:
        assert (XP0[0].shape == (L,K)) & (XP0[1].shape == (K,)), 'Initial point dimension is wrong'
        if XP0[0].ndim == 1:
            XP0[0] = XP0[0].reshape(-1,1)
        XP0 = (XP0[0], np.sqrt(XP0[1]))
    X_manifold = pymanopt.manifolds.Euclidean(L,K)
    p_manifold = pymanopt.manifolds.Sphere(K)
    manifold = pymanopt.manifolds.Product((X_manifold,p_manifold))
    cost, grad, hess = create_cost_function(mean_est, P_est, B_est, sigma, manifold)
    # hess = None
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad, euclidean_hessian=hess)
    optimizer = pymanopt.optimizers.TrustRegions(min_gradient_norm = 1e-7, max_iterations = 50, verbosity = 2)
    result = optimizer.run(problem, initial_point=XP0)
    Xp_est = result.point
    result_cost = result.cost
    
    if extra_inits > 0:
        for i in range(extra_inits):
            result = optimizer.run(problem)
            if result.cost < result_cost:
                result_cost = result.cost
                Xp_est = result.point
            
    return Xp_est[0], Xp_est[1]**2



def create_cost_function(mean_est, P_est, B_est, sigma, manifold):
    euclidean_gradient = euclidean_hessian = None
    
    @pymanopt.function.autograd(manifold)
    def cost(X,sqrt_p): 
        p = sqrt_p**2
        if X.ndim == 1:
            X = X.reshape(-1,1)
        L, K = X.shape
        FX = np.fft.fft(X, axis = 0)
        A = make_A(L)
        
        # limits
        M1 = np.mean(np.dot(X, p))
        M2 = np.dot(abs(FX)**2, p) + L * sigma**2 * np.ones((L))
        M3 = mean_est * (sigma**2) * (L**2) * A
        # matmul = np.zeros((L,L,K))
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
    def grad(X, sqrt_p):
        p = sqrt_p**2
        if X.ndim == 1:
            X = X.reshape(-1,1)
        L, K = X.shape
        FX = np.fft.fft(X, axis = 0)
        A = make_A(L)
        
        # limits
        M1 = np.mean(np.dot(X, p))
        M2 = np.dot(abs(FX)**2, p) + L * sigma**2 * np.ones((L))
        M3 = mean_est * (sigma**2) * (L**2) * A
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
        gradp = (a1/L) * (M1 - mean_est) * np.sum(X, axis = 0) + \
            a2 * np.dot(np.transpose(abs(FX)**2), (M2-P_est))
        gradX_list = []
        gradp_list = []
        
        # test_gradX = gradX.flatten()
        for k in range(K):
            vec = gradX[:,k]
            point = gradp[k]
            
            vec = vec + a3 * DBx_adj(FX[:,k], M3_min_Best)
            point = point + a3 * np.sum(np.real(M3_min_Best * matmul[k]))
            
            gradX_list.append(vec.reshape(-1,1))
            gradp_list.append(point)
        
        scale = 3 + sigma**4    
        gradX = np.concatenate(gradX_list, axis = 1)
        gradX = scale * np.real(gradX)
        gradp = scale * np.array(gradp_list)  * 2 * sqrt_p
        # simple test for K = 1
        # test_gradX = test_gradX + a3 * DBx_adj(FX, M3_min_Best)
        # assert np.linalg.norm(test_gradX - gradX.flatten()) < 1e-6
        
        
        
        grad = (gradX, gradp)
        # gradX = manifold.euclidean_to_riemannian_gradient(X, gradX)
        # print('used defined grad')
        return grad
    
    @pymanopt.function.numpy(manifold)
    def hess(X, sqrt_p, yX, ysqrtp):
        # X = Xp[0]
        # p = Xp[1]
        assert X.shape == yX.shape
        assert sqrt_p.shape == ysqrtp.shape
        # gradXp = lambda x, p: grad(x, p)
        # grad_f = lambda x: grad(x).flatten()
        # H = jacobian(grad_f)(X.flatten())
        gradX = lambda x: grad(x, sqrt_p)[0]
        gradp = lambda p: grad(X, p)[1]
        Hp = jacobian(gradp)(sqrt_p)
        HX = jacobian(gradX)(X)
        
        hessian = (np.tensordot(HX, yX, 2), Hp @ ysqrtp)
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
    


