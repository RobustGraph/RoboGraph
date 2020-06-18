import numpy as np

from robograph.attack.utils import projection_A123
from robograph.attack.convex_relaxation import ConvexRelaxation
from robograph.attack.greedy_attack import Greedy_Attack
import scipy.optimize as optim
from robograph.attack.SPG import *


def cvx_pers_solver(A_org, XW, U, delta_l, delta_g, **params):
    """
    Solver for min_{X\in co(A)} F_\circ(X), where F_\circ(X) is an convex relaxation of F(X) via perspective function.

    param:
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets
        delta_g:        global budgets
        params:         'algo': 'pqn' or 'pg'

    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        opt_f:          optimal objective value
    """

    if params['algo'] == 'pqn':
        return projected_lbfgs(A_org, XW, U, delta_l, delta_g, **params)
    elif params['algo'] == 'nlp':
        return nlp_solver(A_org, XW, U, delta_l, delta_g, **params)
    else:
        raise NotImplementedError(
            'Algorithm `{}` is not implemented for perspective relaxation!'.format(params['algo']))


def projected_lbfgs(A_org, XW, U, delta_l, delta_g, **params):
    cvx_relaxation = ConvexRelaxation(A_org, XW, U, delta_l, delta_g, params['activation'], 'perspective',
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
    spg_options.curvilinear = 1
    spg_options.interp = 2
    spg_options.numdiff = 0     # 0 to use gradients, 1 for numerical diff
    spg_options.verbose = 2 if params['verbose'] else 0
    spg_options.maxIter = params['iter']

    # Initialization of X
    # 1. use greedy attack A_pert
    greedy_attack = Greedy_Attack(A_org, XW, U, delta_l, delta_g, params['activation'])
    greedy_sol = greedy_attack.attack(A_org)
    init_X = greedy_sol['opt_A']
    # 2. original A
    # init_X = A_org.copy()

    x, f = SPG(objective, proj, init_X.flatten(), spg_options)
    sol = {
        'opt_A': x.reshape(n, n),
        'opt_f': f
    }
    return sol


def nlp_solver(A_org, XW, U, delta_l, delta_g, **params):
    cvx_relaxation = ConvexRelaxation(A_org, XW, U, delta_l, delta_g, params['activation'], 'perspective',
                                      relu_relaxation=params.get('relu_bound'))
    n = A_org.shape[0]
    # Generating LP constraint matrix
    # 1. inequality constraint: local/global budgets
    V = -2 * A_org + 1
    A_local_budget = np.zeros((n, n**2))
    b_local_budget = np.zeros((n, 1))
    for i in range(n):
        A_local_budget[i, n * i: n * (i + 1)] = V[:, i]
        b_local_budget[i] = delta_l - np.sum(A_org[:, i])

    A_global_budget = V.reshape(1, n**2)
    b_global_budget = np.asarray(delta_g - np.sum(A_org)).reshape(1, 1)
    A = np.concatenate((A_local_budget, A_global_budget), axis=0)
    b = np.concatenate((b_local_budget, b_global_budget), axis=0)

    # 2. equality constraint: symmetry
    eq_len = int(n * (n - 1) / 2)
    Aeq = np.zeros((eq_len, n**2))
    beq = np.zeros((eq_len, 1))
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx = np.ravel_multi_index((i, j), dims=A_org.shape, order='C')
            idx1 = np.ravel_multi_index((j, i), dims=A_org.shape, order='C')
            Aeq[count, idx], Aeq[count, idx1] = 1, -1
            count += 1

    # 3. lower/upper bound
    lb = np.zeros(A_org.shape)
    ub = np.ones(A_org.shape)
    np.fill_diagonal(ub, 0)

    bounds = optim.Bounds(lb.ravel(), ub.ravel())
    A_total = np.concatenate((A, Aeq), axis=0)
    lb_total = np.concatenate((np.asarray([-np.inf] * len(b)).reshape(-1, 1), beq), axis=0)
    ub_total = np.concatenate((b, beq), axis=0)
    inequality_ctr = optim.LinearConstraint(A_total, lb_total, ub_total)
    init_x = A_org.ravel()
    res = optim.minimize(lambda x: cvx_relaxation.convex_F(x.reshape(n, n))[0], init_x, method='trust-constr',
                         jac=lambda x: cvx_relaxation.convex_F(x.reshape(n, n))[1].ravel(),
                         constraints=inequality_ctr,
                         options={'verbose': params['verbose']}, bounds=bounds)
    # print(res.message)
    opt_Z, opt_f = res.x.reshape(n, n), res.fun

    sol = {
        'opt_A': opt_Z,
        'opt_f': opt_f
    }
    return sol
