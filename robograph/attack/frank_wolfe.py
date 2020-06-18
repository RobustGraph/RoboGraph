import numpy as np
from robograph.attack.utils import check_budgets_and_symmetry
from robograph.attack.convex_relaxation import ConvexRelaxation
from robograph.attack.dp import po_dp_solver


def frank_wolfe(A_org, XW, U, delta_l, delta_g, **params):
    """ Frank wolfe algorithm for solving convexified problem:
         min_{X\in co(A)} F_\circ(Z)

    param:
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets
        delta_g:        global budgets
        params:         params for frank wolfe algorithm

    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        func_vals:      objective value in each iteation (list type)
    """

    cvx_relaxation = ConvexRelaxation(A_org, XW, U, delta_l, delta_g, params['activation'], params['relaxation'])
    # env_relaxation = ConvexRelaxation(A_org, XW, U, delta_l, delta_g, params['activation'], 'envelop')

    # nG = A_org.shape[0]
    # XWU = XW @ U
    X = A_org.copy()
    iters, constr = params['iter'], params['constr']
    func_vals = [0] * iters
    for t in range(1, iters + 1):
        conv_F_c, G = cvx_relaxation.convex_F(X)
        R = -G
        func_vals[t - 1] = conv_F_c
        # print('pers f: %f     env f: %f ' % (conv_F_c, env_relaxation.convex_F(X)[0]) )
        B_opt, V_opt = polar_operator(A_org, R, delta_l, delta_g, constr)
        X = B_opt * (2 / (t + 2)) + X * (t / (t + 2))

    sol = {
        'opt_A': X,
        'func_vals': func_vals,
    }
    return sol


def polar_operator(A_org, R, delta_l, delta_g, constr):
    if constr == '1+3':
        # TODO
        raise NotImplementedError('attack constr `{}` is not implemented'.format(constr))
    elif constr == '2+3':
        # TODO
        raise NotImplementedError('attack constr `{}` is not implemented'.format(constr))
    elif constr == '1+2':
        sol = po_dp_solver(A_org, R, delta_l, delta_g)
    elif constr == '1+2+3':
        sol = greedy_solver(A_org, R, delta_l, delta_g)
        assert check_budgets_and_symmetry(sol[0], A_org, delta_l, delta_g)[-1] == 'symmetric'
    else:
        raise NotImplementedError('attack constr `{}` is not implemented'.format(constr))
    return sol


def greedy_solver(A_org, R, delta_l, delta_g):
    """
    Greedy solver for polar_operator under A_G^{1+2+3}

    Complexity: nG^2*log(nG)
    """
    nG = A_org.shape[0]
    J = R * (-2 * A_org + 1)
    J_hat = (J + J.T) / 2
    num_valid_idx = nG * nG
    for i in range(nG):
        for j in range(nG):
            if j <= i or J_hat[i][j] <= 0:
                J_hat[i][j] = -float('inf')
                num_valid_idx -= 1
    J_hat = -J_hat
    indices = np.dstack(np.unravel_index(np.argsort(J_hat.ravel()), (nG, nG)))[0]

    V = np.zeros((nG, nG))
    for i in range(num_valid_idx):
        V_hat = V.copy()
        i, j = indices[i][0], indices[i][1]
        V_hat[i][j], V_hat[j][i] = 1, 1
        if satisfy_constr(V_hat, delta_l, delta_g):
            V = V_hat.copy()
    A_pert = ((2 * A_org - 1) * (-2 * V + 1) + 1) / 2
    return A_pert, V


def satisfy_constr(V, delta_l, delta_g):
    rows_budget = np.sum(V, axis=1)
    total_budget = np.sum(rows_budget)
    if total_budget > delta_g or np.any(rows_budget > delta_l):
        return False
    return True
