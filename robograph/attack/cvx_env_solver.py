import numpy as np
import time

import scipy.optimize as optim
from robograph.attack.convex_relaxation import ConvexRelaxation
from robograph.attack.dp import exact_solver_wrapper
from robograph.attack.greedy_attack import Greedy_Attack
from robograph.attack.frank_wolfe import polar_operator
from robograph.attack.SPG import *
from robograph.attack.utils import projection_A123
from nsopy.methods.subgradient import SubgradientMethod
from nsopy.methods.quasi_monotone import SGMDoubleSimpleAveraging, SGMTripleAveraging
from nsopy.loggers import GenericMethodLogger

# import matlab.engine
from docplex.mp.model import Model

def cvx_env_solver(A_org, XW, U, delta_l, delta_g, **params):
    """
    Solver for min_{Z\in co(A)} F_\circ(Z), where F_\circ(Z) is an convex relaxation of F(Z) via convex envelop.

    param:
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets
        delta_g:        global budgets
        params:         'algo': could be 'swapping' or 'pqn'
                        'nonsmooth_init': could be 'random' or 'subgrad'
                        1. random initialization
                        2. initilize LBFGS by solution of Quasi-Monotone Methods

    return a dict with keywords:
        opt_A:          original A
        opt_f:          optimal objective value
    """ 

    if params['algo'] == 'swapping':
        return swapping_solver(A_org, XW, U, delta_l, delta_g, **params)
    elif params['algo'] == 'pqn':
        return projected_lbfgs(A_org, XW, U, delta_l, delta_g, **params)


def swapping_solver(A_org, XW, U, delta_l, delta_g, **params):
    """
    Solving the convexified problem: 
        min_{Z\in co(A)} F_\circ(Z) = min_{Z\in co(A)} max_R min_W ...
                                    = max_R min_{Z\in co(A)} min_W ...
        where the inner min_{Z\in co(A)} can be exactly solved by CPLEX LP.
    """ 

    cvx_relaxation = ConvexRelaxation(A_org, XW, U, delta_l, delta_g, params['activation'], 'envelop', \
                                                        relu_relaxation=params.get('relu_bound'))
    if params.get('relu_bound') == 'singleL':
        doubleL_relax = ConvexRelaxation(A_org, XW, U, delta_l, delta_g, params['activation'], 'envelop', \
                                                        relu_relaxation='doubleL')
    nG = A_org.shape[0]
    XWU = XW @ U

    # Setup SPG solver for min_W
    spg_options = default_options
    spg_options.verbose = 0
    spg_options.maxIter = 20

    # Setup CPLEX solver for min_Z
    mdl = Model("LP")
    n = nG
    V = -2*A_org + 1
    ub = np.ones(A_org.shape)
    np.fill_diagonal(ub, 0)
    x_vars = mdl.continuous_var_matrix(n, n, ub= ub.ravel())
    mdl.add_constraints_( mdl.sum(V[i, j] * x_vars[i, j] + A_org[i, j] for j in range(n)) <= delta_l[i]
                                                                            for i in range(n) )
    mdl.add_constraint_( mdl.sum(V[i, j] * x_vars[i, j] + A_org[i, j] for i in range(n) for j in range(n)) <= delta_g)
    mdl.add_constraints_( x_vars[i, j] - x_vars[j, i] == 0 for i in range(n) for j in range(i+1, n) )

    tighest_value = -np.inf
    for i in range(params.get('luiter', 1)):
        iters, eps = params['iter'], 0.0
        def objective(x):
            R = x.reshape(nG, nG)
            # Given R, find optimal W
            if params['activation'] == 'relu':
                if params['relu_bound'] == 'doubleL':
                    local_budget_sol = exact_solver_wrapper(A_org, cvx_relaxation.Q, cvx_relaxation.p, -R, delta_l, delta_g, '1+2')
                    _, opt_f_W, opt_W = local_budget_sol
                    opt_f_W = np.sum(opt_f_W)
                    
                elif params['relu_bound'] == 'singleL':
                    local_budget_sol = exact_solver_wrapper(A_org, doubleL_relax.Q, doubleL_relax.p, -R, delta_l, delta_g, '1+2')
                    _, opt_f_W_1, opt_W_1 = local_budget_sol
                    opt_f_W_1 = np.sum(opt_f_W_1)
                    # cvx_relaxation.warm_start_W = opt_W_1
                    # opt_W, opt_f_W = cvx_relaxation.get_opt_W(R, spg_options)
                    opt_W = opt_W_1
                    opt_f_W = np.sum(cvx_relaxation.g_hat(opt_W)[0]/(1+np.sum(opt_W, axis=1))) - np.sum(R*opt_W)
            else:
                # local_budget_sol = exact_solver_wrapper(A_org, np.tile(XWU, (nG, 1)), np.zeros(nG), -R, delta_l, delta_g, '1')
                # _, opt_f_W, opt_W = local_budget_sol
                # opt_f_W = np.sum(opt_f_W)
                
                dp_sol = exact_solver_wrapper(A_org, np.tile(XWU, (nG, 1)), np.zeros(nG), -R, delta_l, delta_g, '1+2')
                _, opt_f_W, opt_W = dp_sol

            # Given R, solve min_{Z \in co(A1) \cap co(A2) \cap A3} tr(R.T*Z)
            opt_Z = RZ_solver(x_vars, mdl, A_org, delta_l, delta_g, R)

            fval = opt_f_W + np.sum(R*opt_Z) - eps*np.sum(R**2)/2
            grad = opt_Z - opt_W - eps*R
            return -fval, -grad.flatten()
        # print(optim.check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], init_R.flatten()))

        def callback(x):
            print(-objective(x)[0])
            
        if params['nonsmooth_init'] == 'subgrad':
            # Subgradient Method to optimize the non-smooth objective
            # method = SubgradientMethod(lambda x: (0, -objective(x)[0], -objective(x)[1]), lambda x: x, dimension=nG*nG,\
            #                                                  stepsize_0=0.01, stepsize_rule='constant', sense='max')
            method = SGMTripleAveraging(lambda x: (0, -objective(x)[0], -objective(x)[1]), lambda x: x, dimension=nG*nG,\
                                                            gamma=3, sense='max')
            logger = GenericMethodLogger(method)
            for iteration in range(50):
                method.step()
            init_R = logger.x_k_iterates[-1].reshape(nG, nG)
            fval_sub = logger.f_k_iterates[-1]
        elif params['nonsmooth_init'] == 'random':
            init_R = np.random.randn(nG, nG)*0.01


        if params['verbose']:
            res = optim.fmin_l_bfgs_b(objective, init_R.flatten(), maxiter=iters, m=20, maxls=20, callback=callback)
            res_status = res[2]
            print('warnflag: %d,  iters: %d,  funcalls: %d' % (res_status['warnflag'], res_status['nit'], res_status['funcalls']) )
            print(res_status['task'])
        else:
            res = optim.fmin_l_bfgs_b(objective, init_R.flatten(), maxiter=iters, m=20, maxls=20, callback=None)

        fval, opt_R = -res[1], res[0].reshape(nG, nG)
        # print(fval)

        # if params['relu_bound'] == 'doubleL':
        #     flag = update_lb_ub(opt_R, cvx_relaxation)
        # elif params['relu_bound'] == 'singleL':
        #     flag = update_lb_ub(opt_R, doubleL_relax, cvx_relaxation)

        # if flag and fval > tighest_value:
        #     tighest_value = fval

    sol = {
        'opt_A': A_org,
        'opt_f': fval
    }
    return sol

def update_lb_ub(opt_R, doubleL, singleL=None):
    local_budget_sol = exact_solver_wrapper(doubleL.A_org, doubleL.Q, doubleL.p, -opt_R, \
                                                doubleL.delta_l, doubleL.delta_g, '1+2')
    opt_W = local_budget_sol[-1]
    
    # update lb and ub
    before_relu = opt_W @ doubleL.XW + doubleL.XW

    # case i
    i_idx = (before_relu > doubleL.ub)
    # case ii
    ii_idx = (before_relu < doubleL.lb)
    # within [lb, ub]
    idx = ~(i_idx ^ ii_idx)
    # idx = (before_relu <= doubleL.ub) & (before_relu >= doubleL.lb)
    # case iii
    iii_idx = idx & (before_relu > 0) 
    # case iv
    iv_idx = idx & (before_relu < 0)

    if np.all(idx):
        flag = True
        # print('pre_relu in [lb, ub]!')
    else:
        flag = False
        # print('pre_relu NOT in [lb, ub]!')

    doubleL.ub[i_idx] = before_relu[i_idx]
    doubleL.lb[ii_idx] = before_relu[ii_idx]
    # doubleL.ub[iii_idx] = (before_relu[iii_idx] +  doubleL.ub[iii_idx])/2
    doubleL.ub[iii_idx] = before_relu[iii_idx]
    # doubleL.lb[iv_idx] = (before_relu[iv_idx] +  doubleL.lb[iv_idx])/2
    doubleL.lb[iv_idx] = before_relu[iv_idx]

    doubleL.S_minus = (doubleL.lb*doubleL.ub < 0) & np.tile((doubleL.U < 0), (doubleL.nG, 1))
    doubleL.I_plus = (doubleL.lb > 0)
    doubleL.I_mix = (doubleL.lb*doubleL.ub < 0)
    doubleL.Q, doubleL.p = doubleL.doubleL_lb_coefficient()

    if singleL != None:
        singleL.ub = doubleL.ub
        singleL.lb = singleL.lb

        singleL.S_minus = (singleL.lb*singleL.ub < 0) & np.tile((singleL.U < 0), (singleL.nG, 1))
        singleL.S_others = ~(singleL.S_minus)

    return flag

def projected_lbfgs(A_org, XW, U, delta_l, delta_g, **params):
    """
    Solving the convexified problem:
        min_{Z\in co(A)} F_\circ(Z) = min_{Z\in co(A)} max_R min_W ...
    """ 
    cvx_relaxation = ConvexRelaxation(A_org, XW, U, delta_l, delta_g, params['activation'], 'envelop', \
                                                        relu_relaxation=params.get('relu_bound'))
    n = A_org.shape[0]

    def objective(x):
        X = x.reshape(n, n)
        f, G = cvx_relaxation.convex_F(X)
        return f, G.flatten()
    def proj(x):
        X = x.reshape(n, n)
        projected_value = projection_A123(X, A_org, delta_l, delta_g)
        return projected_value.flatten()

    spg_options = default_options
    spg_options.curvilinear = 0
    spg_options.interp = 2
    spg_options.numdiff = 0 # 0 to use gradients, 1 for numerical diff
    spg_options.verbose = 2 if params['verbose'] else 0
    spg_options.maxIter = params['iter']

    # 1. init Z as A_org
    init_x = A_org.copy()
    # 2. init by optimal R of (39), given Z* from greedy attack.
    # greedy_attack = Greedy_Attack(A_org, XW, U, delta_l, delta_g, 'relu')
    # greedy_sol = greedy_attack.attack(A_org)
    # init_x = greedy_sol['opt_A']

    x, f = SPG(objective, proj, init_x.flatten(), spg_options)
    sol = {
        'opt_A': A_org,
        'opt_f': fval
    }
    return sol


def RZ_solver(x_vars, mdl, A_org, delta_l, delta_g, R):
    # PO on co(A^{1+2+3}) is equivalent to PO on A^{1+2+3}
    # opt_Z, _ = polar_operator(A_org, -R, delta_l, delta_g, '1+2+3')

    n = A_org.shape[0]
    mdl.minimize( mdl.sum(R[i, j] * x_vars[i, j] for i in range(n) for j in range(n)) )
    msol = mdl.solve()
    if msol:
        opt_Z = np.array([x_vars[(i, j)].solution_value for i in range(n) for j in range(n)]).reshape(n, n)
    else:
        raise Exception("cplex failed!")
    
    return opt_Z
