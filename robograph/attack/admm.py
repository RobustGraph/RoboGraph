import numpy as np

from robograph.attack.dp import exact_solver_wrapper
from robograph.attack.utils import calculate_Fc, calculate_doubleL_Fc


def admm_solver(A_org, XW, U, delta_l, delta_g, **params):
    """
    ADMM approach (upper bound) for solving min_{A_G^{1+2+3}} F_c(A)

    param: 
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets
        delta_g:        global budgets
        params:         params for admm optimiztion

    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        opt_f:          optimal objective value
    """

    nG = A_org.shape[0]
    XWU = XW @ U
    lamb = np.random.randn(nG, nG)*0
    B = params['init_B']
    mu = params['mu']
    iters = params['iter']
    func_vals = [0]*iters
    for i in range(iters):
        # linear term: fnorm(A-B)^2 = \sum_{ij} A_ij^2 - 2*A_ijB_ij + B_ijB_ij
        L = -lamb.T + (np.ones((nG, nG)) - 2*B.T)/(2*mu)
        dp_sol = exact_solver_wrapper(A_org, np.tile(XWU, (nG, 1)), np.zeros(nG), L.T, delta_l, delta_g, '1+2')
        _, opt_val, A_pert = dp_sol
        func_vals[i] = calculate_Fc(A_pert, XW, U)
        B = closed_form_B(lamb, A_pert, mu)
        lamb += (B-A_pert)/mu
        if params.get('verbose'):
            print(func_vals[i])

    sol = {
        'opt_A': A_pert,
        'opt_f': func_vals[-1]
    }
    return sol


def admm_solver_doubleL(A_org, Q, p, delta_l, delta_g, **params):
    """
    ADMM approach (upper bound) for solving min_{A_G^{1+2+3}} F_c(A)
    where the activation is ReLU, and F_c(A) has been linearized via doubleL (upper bound)

    param: 
        A_org:          original adjacency matrix
        Q:              Q
        p:              p
        delta_l:        row budgets
        delta_g:        global budgets
        params:         params for admm optimiztion

    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        opt_f:          optimal objective value
    """

    nG = A_org.shape[0]
    lamb = np.random.randn(nG, nG)*0
    B = params['init_B']
    mu = params['mu']
    iters = params['iter']
    func_vals = [0]*iters
    for i in range(iters):
        # linear term: fnorm(A-B)^2 = \sum_{ij} A_ij^2 - 2*A_ijB_ij + B_ijB_ij
        L = -lamb.T + (np.ones((nG, nG)) - 2*B.T)/(2*mu)
        dp_sol = exact_solver_wrapper(A_org, Q, p, L.T, delta_l, delta_g, '1+2')
        _, opt_val, A_pert = dp_sol
        func_vals[i] = calculate_doubleL_Fc(A_pert, Q, p)
        B = closed_form_B(lamb, A_pert, mu)
        lamb += (B-A_pert)/mu
        if params.get('verbose'):
            print(func_vals[i])

    sol = {
        'opt_A': A_pert,
        'opt_f': func_vals[-1]
    }
    return sol


def closed_form_B(lamb, A_pert, mu):
    # Closed-form solution of B in eq (11)

    # Continuous solution
    # BB = (-mu/2) * (lamb + lamb.T) + (1/2) * (A_pert + A_pert.T)
    # BB = np.round(BB)
    # print(np.trace(lamb.T @ BB) + np.sum((A_pert - BB)**2)/(2*mu))
    # return BB

    # Discreet solution
    nG = A_pert.shape[0]
    L = lamb + (np.ones((nG, nG)) - 2*A_pert)/(2*mu)
    L_hat = L + L.T
    B = np.zeros((nG, nG))
    for i in range(nG):
        for j in range(nG):
            if j > i and L_hat[i][j] <= 0:
                B[i][j], B[j][i] = 1, 1
    # print(np.trace(lamb.T @ B) + np.sum((A_pert - B)**2)/(2*mu))
    return B
