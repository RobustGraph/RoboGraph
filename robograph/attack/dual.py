import numpy as np

from robograph.attack.dp import exact_solver_wrapper
import scipy.optimize as optim
# from nsopy.methods.bundle import CuttingPlanesMethod, BundleMethod
from nsopy.methods.quasi_monotone import SGMDoubleSimpleAveraging, SGMTripleAveraging
from nsopy.methods.subgradient import SubgradientMethod
from nsopy.loggers import GenericMethodLogger
from robograph.attack.SPG import *

def dual_solver(A_org, XW, U, delta_l, delta_g, **params):
    """
    Dual approach (lower bound) for solving min_{A_G^{1+2+3}} F_c(A)

    param:
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets (vector)
        delta_g:        global budget (scalar)
        params:         params for optimizing Lambda
                        'nonsmooth': could be 'random' or 'subgrad'
                        1. random: choose best solution from 5 random initialization
                        2. initilize LBFGS by solution of Quasi-Monotone Methods


    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        opt_f:          optimal dual objective
    """
    
    nG = A_org.shape[0]
    XWU = XW @ U

    def objective(x):
        lamb = x.reshape(nG, nG)
        L = lamb.T - lamb
        dp_sol = exact_solver_wrapper(A_org, np.tile(XWU, (nG, 1)), np.zeros(nG), L.T, delta_l, delta_g, '1+2')
        _, opt_val, A_pert = dp_sol
        grad_on_lambda = A_pert - A_pert.T
        return -opt_val, -grad_on_lambda.flatten()
    # print(optim.check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], lamb.flatten()))

    opt_lamb, fopt = optimize(objective, nG, params['iter'], params.get('verbose'), params['nonsmooth_init'])
    L = opt_lamb.T - opt_lamb
    sol = {
        'opt_A': exact_solver_wrapper(A_org, np.tile(XWU, (nG, 1)), np.zeros(nG), L.T, delta_l, delta_g, '1+2')[-1],
        'opt_f': fopt
    }
    return sol


def dual_solver_doubleL(A_org, Q, p, delta_l, delta_g, **params):
    """
    Dual approach (lower bound) for solving min_{A_G^{1+2+3}} F_c(A)
    where the activation is ReLU, and F_c(A) has been linearized via doubleL
                
    param: 
        A_org:          original adjacency matrix
        Q:              Q
        p:              p
        delta_l:        row budgets
        delta_g:        global budgets
        params:         params for optimizing Lambda
                        'nonsmooth': could be 'random' or 'subgrad'
                        1. random: choose best solution from 5 random initialization
                        2. initilize LBFGS by solution of Quasi-Monotone Methods
                        
    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        opt_f:          optimal dual objective
    """
    
    nG = A_org.shape[0]
    
    def objective(x):
        lamb = x.reshape(nG, nG)
        L = lamb.T - lamb
        dp_sol = exact_solver_wrapper(A_org, Q, p, L.T, delta_l, delta_g, '1+2')
        _, opt_val, A_pert = dp_sol
        grad_on_lambda = A_pert - A_pert.T
        return -opt_val, -grad_on_lambda.flatten()
    # print(optim.check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], lamb.flatten()))

    opt_lamb, fopt = optimize(objective, nG, params['iter'], params.get('verbose'), params['nonsmooth_init'])
    L = opt_lamb.T - opt_lamb
    sol = {
        'opt_A': exact_solver_wrapper(A_org, Q, p, L.T, delta_l, delta_g, '1+2')[-1],
        'opt_f': fopt
    }
    return sol


def optimize(objective, var_dim, iters, verbose, init):
    def callback(x):
        print(-objective(x)[0])

    if init == 'subgrad':
        # Subgradient Method to optimize the non-smooth objective
        # method = SubgradientMethod(lambda x: (0, -objective(x)[0], -objective(x)[1]), lambda x: x, dimension=var_dim*var_dim,\
        #                                                  stepsize_0=0.003, stepsize_rule='constant', sense='max')
        method = SGMTripleAveraging(lambda x: (0, -objective(x)[0], -objective(x)[1]), lambda x: x, dimension=var_dim*var_dim,\
                                                        gamma=3, sense='max')
        logger = GenericMethodLogger(method)
        for iteration in range(100):
            method.step()
        init_lamb = logger.x_k_iterates[-1]
        fopt1 = logger.f_k_iterates[-1]

        if verbose:
            res = optim.fmin_l_bfgs_b(objective, init_lamb.flatten(), maxiter=iters, callback=callback)
            res_status = res[2]
            print('warnflag: %d,  iters: %d,  funcalls: %d' % (res_status['warnflag'], res_status['nit'], res_status['funcalls']) )
            print(res_status['task'])
        else:
            res = optim.fmin_l_bfgs_b(objective, init_lamb.flatten(), maxiter=iters, callback=None)
        xopt, fopt = res[0], -res[1]

    elif init == 'random':
        maximum = -np.inf
        for i in range(5):
            init_lamb = np.random.randn(var_dim, var_dim) * 0.01
            if verbose:
                res = optim.fmin_l_bfgs_b(objective, init_lamb.flatten(), maxiter=iters, callback=callback)
                res_status = res[2]
                print('warnflag: %d,  iters: %d,  funcalls: %d' % (res_status['warnflag'], res_status['nit'], res_status['funcalls']) )
                print(res_status['task'])
            else:
                res = optim.fmin_l_bfgs_b(objective, init_lamb.flatten(), maxiter=iters, callback=None)
            if -res[1] > maximum:
                maximum = -res[1]
                xopt, fopt = res[0], -res[1]
            # if res[2]['warnflag'] < 2: break

    lamb = xopt.reshape(var_dim, var_dim)
    return lamb, fopt