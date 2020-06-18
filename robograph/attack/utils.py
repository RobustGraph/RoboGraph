import numpy as np
import time
import scipy.optimize as optim
from docplex.mp.model import Model
from qpsolvers import solve_qp
from scipy.sparse import identity


def projection_coA1(A, A_org, delta_l):
    if np.all((A >= 0) & (A <= 1)) and np.all(np.abs(A-A_org).sum(1) <= delta_l):
        return A
    proj_Z = np.zeros(A.shape)
    for i in range(A.shape[0]):
        proj_Z[i] = projection_coA2(A[i], A_org[i], delta_l[i])

    eps = 1e-5
    assert np.all((proj_Z >= (-eps)) & (proj_Z <= (1+eps))) and np.all(np.abs(proj_Z-A_org).sum(1) <= (delta_l+eps))
    return proj_Z


def projection_coA2(A, A_org, delta_g):
    if delta_g == 0:
        return A_org
    if np.all((A >= 0) & (A <= 1)) and np.sum(np.abs(A-A_org)) <= delta_g:
        return A
    # Dual approach
    # start = time.time()
    # lamb, iters = 0.1, 10
    # bounds = [(0, np.inf)]
    # def objective(lamb):
    #     opt_z = A - lamb*(-2*A_org+1)
    #     opt_z[opt_z>1] = 1
    #     opt_z[opt_z<0] = 0
    #     s = np.sum(np.abs(opt_z - A_org))
    #     f = np.sum((opt_z-A)**2)/2 + lamb*( s - delta_g)
    #     grad_on_lamb = -delta_g + s
    #     return -f, -np.array([grad_on_lamb])
    # # print(optim.check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], np.asarray([lamb])))
    # def callback(x):
    #     print(objective(x)[0])

    # res = optim.fmin_l_bfgs_b(objective, lamb, bounds=bounds, maxiter=iters, callback=None)
    # opt_z = A - res[0]*(-2*A_org+1)
    # opt_z[opt_z>1] = 1
    # opt_z[opt_z<0] = 0
    # f = np.sum((opt_z-A_org)**2)/2
    # print(f, np.sum(np.abs(opt_z-A_org)))
    # print(f'bfgs cputime: {time.time() - start}')

    # Quadractic Programming (quadprog)
    start = time.time()
    n = len(A_org)
    P = np.eye(n)
    q = -A
    G = -2*A_org + 1
    h = np.array([delta_g - np.sum(A_org)])
    lb = np.zeros(n)
    ub = np.ones(n)
    opt_z_1 = solve_qp(P, q, G, h, lb=lb, ub=ub)
    # f_1 = np.sum((opt_z_1-A_org)**2)/2
    # print(f_1, np.sum(np.abs(opt_z_1-A_org)))
    # print(f'qp cputime: {time.time() - start}')

    return opt_z_1


def projection_coPi_and_affine(Ai, Ai_org, delta_l, alpha):
    if np.all((Ai >= 0) & (Ai <= 1)) and np.sum(np.abs(Ai-Ai_org)) <= delta_l \
            and np.sum(Ai) == alpha:
        return Ai
    lamb, u = 0.1, 0.1
    for i in range(10):
        opt_z = Ai - lamb*(1-2*Ai_org) - u
        opt_z[opt_z > 1] = 1
        opt_z[opt_z < 0] = 0
        grad_on_lamb = -delta_l + np.sum(np.abs(opt_z - Ai_org))
        grad_on_u = -alpha + np.sum(opt_z)
        lamb = max(0, lamb+0.01*grad_on_lamb)
        u += 0.01*grad_on_u
    #     print(lamb, u, np.sum((opt_z-Ai)**2))
    # print('final slackness: ', np.sum(opt_z), alpha, np.sum(np.abs(opt_z-Ai)), delta_l)
    # assert np.sum(np.abs(opt_z-Ai)) <= delta_l+0.1
    return opt_z


def projection_A3(A):
    return (1/2) * (A + A.T)


def projection_A123(A, A_org, delta_l, delta_g):
    n = A.shape[0]
    for i in range(20):
        A_begin = A.copy()
        A_proj = projection_A3(A_begin)
        A_proj = projection_coA1(delete_diagonal(A_proj), delete_diagonal(A_org), delta_l)
        A = fill_diagonal(A_proj)
        A_proj = projection_coA2(delete_diagonal(A).flatten(), delete_diagonal(A_org).flatten(), delta_g)
        A = fill_diagonal(A_proj.reshape(n, n-1))
        # A = projection_coA1(A_begin, A_org, delta_l)
        # A = projection_coA2(A, A_org, delta_g)

        changes = np.max(np.abs(A-A_begin))
        # print(changes)
        if changes < 1e-5:
            break
    return A


def delete_diagonal(A):
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


def fill_diagonal(A, value=0):
    n = A.shape[0]
    S = np.ones((n, n))*value
    S[:, :-1] += np.tril(A, -1)
    S[:, 1:] += np.triu(A, 0)
    return S


def check_budgets_and_symmetry(A_new, A, delta_l, delta_g, check_symmetry=True):
    """
    Check if A_new satisfies: local budgets & global budgets & symmetry
    """
    nG = A.shape[0]
    local_budgets, global_budgets = np.zeros(nG), 0
    for i in range(nG):
        if A_new[i, i]:
            print('Violate zero diagonals!')
            return False
        local = np.sum(abs(A[i]-A_new[i]))
        if local > delta_l[i]:
            print('Violate local budgets!')
            return False
        local_budgets[i] = local
        global_budgets += local
    if global_budgets > delta_g:
        print('Violate global budgets!')
        return False

    if check_symmetry:
        if not np.array_equal(A_new, A_new.T):
            print('A_pert is non-symmetric!')
            return False
    return True


def calculate_Fc(A, XW, U, activation='linear'):
    """
    Calculate F_c
    """
    nG = A.shape[0]
    total_val = 0
    if activation == 'linear':
        XWU = XW@U
        for i in range(nG):
            total_val += (A[i] @ XWU + XWU[i]) / (np.sum(A[i]) + 1)
    else:
        for i in range(nG):
            total_val += (np.maximum(A[i] @ XW + XW[i], 0) @ U) / (np.sum(A[i]) + 1)
    return total_val


def calculate_doubleL_Fc(A, Q, p):
    """
    Calculate F_c
    """
    total_val = 0
    nG = A.shape[0]
    f = np.sum((A + np.eye(nG)) * Q, axis=1) + p
    de = 1+np.sum(A, axis=1)
    total_val = np.sum(f/de)
    return total_val


def display(A_org, XW, U, delta_l, delta_g, solutions):
    print('')
    # print('original A:\n', A_org)
    print('original_Fc:   ', calculate_Fc(A_org, XW, U))

    iters = 100
    x_axis = list(range(1, iters+1))
    fig, ax = plt.subplots()

    if 'brute_sol' in solutions:
        brute_sol = solutions['brute_sol']
        brute_opt_A = brute_sol['opt_A']
        brute_opt_f = brute_sol['opt_f']
        # print(check_budgets_and_symmetry(brute_opt_A, A_org, delta_l, delta_g))
        print('brute_opt_f:   ', brute_opt_f)
        ax.plot(x_axis, [brute_opt_f]*iters, 'g-', label='brute-force')

    if 'admm_sol' in solutions:
        admm_sol = solutions['admm_sol']
        admm_opt_A = admm_sol['opt_A']
        admm_opt_f = admm_sol['opt_f']
        if check_budgets_and_symmetry(admm_opt_A, A_org, delta_l, delta_g):
            symmtric = 'symmetric'
        else:
            symmtric = 'non_symmetric'
        print('admm_opt_f:    ', admm_opt_f, '(A_admm is', symmtric, ')')
        ax.plot(x_axis, [admm_opt_f]*iters, 'r-', label='admm')

    if 'admm_g_sol' in solutions:
        admm_g_sol = solutions['admm_g_sol']
        admm_g_opt_A = admm_g_sol['opt_A']
        admm_g_opt_f = admm_g_sol['opt_f']
        # print(check_budgets_and_symmetry(admm_g_opt_A, A_org, delta_l, delta_g))
        print('admm_g_opt_f:  ', admm_g_opt_f)
        ax.plot(x_axis, [admm_g_opt_f]*iters, 'c-', label='admm_greedy')

    if 'greedy_sol' in solutions:
        greedy_sol = solutions['greedy_sol']
        greedy_opt_A = greedy_sol['opt_A']
        greedy_opt_f = greedy_sol['opt_f']
        # print('greedy opt A:\n', greedy_opt_A)
        # print(check_budgets_and_symmetry(greedy_opt_A, A_org, delta_l, delta_g))
        print('greedy_opt_f:  ', greedy_opt_f)
        ax.plot(x_axis, [greedy_opt_f]*iters, 'm--', label='greedy')

    if 'dual_sol' in solutions:
        brute_sol = solutions['dual_sol']
        dual_opt_A = dual_sol['opt_A']
        dual_opt_f = dual_sol['opt_f']
        # print(check_budgets_and_symmetry(dual_opt_A, A_org, delta_l, delta_g))
        print('dual_opt_f:    ', dual_opt_f)
        ax.plot(x_axis, [dual_opt_f]*iters, 'b-.', label='dual')

    if 'cvx_env_sol' in solutions:
        cvx_env_sol = solutions['cvx_env_sol']
        cvx_env_opt_A = cvx_env_sol['opt_A']
        cvx_env_opt_f = cvx_env_sol['opt_f']
        print('cvx_env_opt:   ', cvx_env_opt_f)
        ax.plot(x_axis, [cvx_env_opt_f] * iters, 'k-', label='cvx_env')

    if 'cvx_pers_sol' in solutions:
        cvx_pers_sol = solutions['cvx_pers_sol']
        cvx_pers_opt_A = cvx_pers_sol['opt_A']
        cvx_pers_opt_f = cvx_pers_sol['opt_f']
        print('cvx_pers_opt:  ', cvx_pers_opt_f)
        ax.plot(x_axis, [cvx_pers_opt_f] * iters, 'y-', label='cvx_pers')

    legend = ax.legend(loc='upper right', fontsize='large')
    plt.ylabel('f_val', fontsize=16)
    plt.xlabel('iter', fontsize=16)
    plt.savefig('robograph/tests/bound_linear.png')
