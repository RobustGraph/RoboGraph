import numpy as np
import time
from robograph.attack.dp import exact_solver_wrapper
import robograph.attack.utils as utils
import scipy.optimize as optim
from robograph.attack.SPG import *
from docplex.mp.model import Model
# import matlab.engine


class ConvexRelaxation(object):
    """
    Convex relaxation of F_c(A)

    param:
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets
        delta_g:        global budgets
        activation:     'linear' or 'relu'
        relaxation:     'envelop' or 'perspective'
        relu_relaxation:    'singleL' or 'doubleL'
    """

    def __init__(self, A_org, XW, U, delta_l, delta_g, activation='linear', relaxation='envelop', relu_relaxation='singleL'):
        self.A_org = A_org
        self.XW, self.U = XW, U
        self.XWU = XW @ U
        self.delta_l, self.delta_g = delta_l, delta_g
        self.activation = activation
        self.relaxation = relaxation
        self.nG = A_org.shape[0]
        self.warm_start_R = np.random.randn(self.nG, self.nG)
        self.warm_start_r = np.random.randn(self.nG)
        self.warm_start_W = A_org.copy()
        if activation == 'relu':
            self.relu_relaxation = relu_relaxation
            self.hidden_dim = len(U)
            self.lb, self.ub = self.relu_bounds()
            self.S_minus = (self.lb*self.ub < 0) & np.tile((self.U < 0), (self.nG, 1))
            if relu_relaxation == 'doubleL':
                self.I_plus = (self.lb > 0)
                self.I_mix = (self.lb*self.ub < 0)
                self.Q, self.p = self.doubleL_lb_coefficient()
            elif relu_relaxation == 'singleL':
                self.S_others = ~(self.S_minus)

    def convex_F(self, A_pert):
        if self.activation == 'relu':
            return self.convex_F_relu(A_pert)
        elif self.activation == 'linear':
            return self.convex_F_linear(A_pert)

    def convex_F_linear(self, A_pert):
        if self.relaxation == 'perspective':
            # Perspective Relaxation for linear activation
            fval, G = 0, np.zeros((self.nG, self.nG))
            M = max(0, -min(self.XWU))
            b = self.XWU + M
            for i in range(self.nG):
                x = A_pert[i, :]
                Dx = b*x
                dom = sum(x)+1
                fval += (x@Dx+b[i])/dom - M
                G[i, :] = (2*Dx*dom - (x@Dx+b[i])*np.ones(x.shape))/(dom**2)
            return fval, G

        elif self.relaxation == 'envelop':
            # Convex Envelop Relaxation for linear activation
            R = np.zeros((self.nG, self.nG))
            iters, eps = 100, 0.0

            def objective(x):
                R = x.reshape(self.nG, self.nG)
                dp_sol = exact_solver_wrapper(self.A_org, np.tile(self.XWU, (self.nG, 1)),
                                              np.zeros(self.nG), -R, self.delta_l, self.delta_g, '1')
                _, opt_val, opt_W = dp_sol
                fval = opt_val.sum() + np.sum(R*A_pert) - eps*np.sum(R**2)/2
                grad = A_pert - opt_W - eps*R
                return -fval, -grad.flatten()
            # print(optim.check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], R.flatten()))

            def callback(x):
                print(-objective(x)[0])

            res = optim.fmin_l_bfgs_b(objective, R.flatten(), maxiter=iters, m=30, callback=None)
            F, G = -res[1], res[0].reshape(self.nG, self.nG)
            return F, G

    def convex_F_relu(self, A_pert):
        # Perspective Relaxation for ReLU (double linear)
        if self.relaxation == 'perspective':
            if self.relu_relaxation == 'doubleL':
                fval, G = 0, np.zeros((self.nG, self.nG))
                for i in range(self.nG):
                    q = self.Q[i]
                    M = max(0, -min(q))
                    b = q + M
                    x = A_pert[i]
                    Dx = b*x
                    dom = np.sum(x)+1
                    fval += (x@Dx + b[i] + self.p[i])/dom - M
                    G[i] = (2*Dx*dom - (x@Dx+q[i]+b[i]+self.p[i])*np.ones(x.shape))/(dom**2)
                return fval, G
            else:
                raise Exception("Perspective Relaxation is only for 'doubleL' ReLU!")

        elif self.relaxation == 'envelop':
            # Convex Envelop Relaxation for ReLU

            # Convex envelop (38) by solving R row-wise
            spg_options = default_options
            spg_options.curvilinear = 1
            spg_options.interp = 2
            spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
            spg_options.verbose = 0
            spg_options.maxIter = 20

            F, G = 0, np.zeros((self.nG, self.nG))
            init_W = self.warm_start_W
            for i in range(self.nG):
                z = A_pert[i]

                def objective(r):
                    opt_w_i, opt_f_i = self.get_opt_W_i(i, r, init_W[i], spg_options)
                    f = opt_f_i + r@z
                    return -f, opt_w_i-z
                # print(optim.check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], np.random.randn(self.nG)))

                def callback(x):
                    print(-objective(x)[0])

                init_r_i = np.random.randn(self.nG)*0.01
                res = optim.fmin_l_bfgs_b(objective, init_r_i, maxiter=20, callback=None)
                fval, opt_r = -res[1], res[0]
                F += fval
                G[i] = opt_r

            # Convex envelop (38) by solving R as a matrix
            # R = np.zeros((self.nG, self.nG))
            # iters = 100
            # def objective(x):
            #     R = x.reshape(self.nG, self.nG)
            #     dp_sol = exact_solver_wrapper(self.A_org, self.Q.T, self.p, -R.T, self.delta_l, self.delta_g, '1')
            #     _, opt_val, opt_W = dp_sol
            #     fval = opt_val.sum() + np.sum(R*A_pert)
            #     grad = A_pert - opt_W
            #     return -fval, -grad.flatten()
            # def callback(x):
            #     print(-objective(x)[0])

            # res = optim.fmin_l_bfgs_b(objective, R.flatten(), maxiter=iters, m=20, callback=callback)
            # F, G = -res[1], res[0].reshape(self.nG, self.nG)

            return F, G

    def get_opt_W_i(self, i, r, init_w, spg_options):
        # # ------------------ Solving W_i by Projected Gradient Method ------------------
        def func(w):
            f, g = self.g_i_hat(w, i)
            de = 1+np.sum(w)
            return f/de - r @ w, (g*de-f)/de**2 - r
        # print(optim.check_grad(lambda x: func(x)[0], lambda x: func(x)[1], init_w))

        # Projected Gradient Method
        def proj(w):
            proj_w = utils.projection_coA2(np.delete(w, i), np.delete(self.A_org[i], i), self.delta_l[i])
            return np.insert(proj_w, i, 0)
        opt_w_i, opt_f_i = SPG(func, proj, init_w, spg_options)

        # # ------------------ Solving W_i as a Linear Constrained Non-convex Problem ------------------
        # lb = np.zeros(self.nG)
        # ub = np.ones(self.nG)
        # ub[i] = 0
        # bounds = optim.Bounds(lb, ub)
        # v = -2*self.A_org[i] + 1
        # linear_constraint = optim.LinearConstraint(v, -np.inf, self.delta_l[i] - np.sum(self.A_org[i]))
        # res = optim.minimize(lambda x: func(x)[0], init_w, method='trust-constr', jac=lambda x: func(x)[1],
        #                         constraints=linear_constraint, options={'verbose': 0}, bounds=bounds)
        # # print(res.message)
        # opt_w_i, opt_f_i = res.x, res.fun

        return opt_w_i, opt_f_i

    def get_opt_W(self, R, spg_options):
        # # ------------------ Solving W as a Linear Constrained Non-convex Problem ------------------
        # start = time.time()
        # init_W = self.warm_start_W
        # def func(W):
        #     mat_W = W.reshape(self.nG, self.nG)
        #     kappa, G = self.g_hat(mat_W)
        #     de = 1+np.sum(mat_W, axis=1)
        #     f = np.sum(kappa/de) - np.sum(R*mat_W)
        #     grad = (G*de.reshape(-1, 1)-kappa.reshape(-1, 1))/(de**2).reshape(-1, 1) - R
        #     return f, grad.ravel()
        # # print(optim.check_grad(lambda x: func(x)[0], lambda x: func(x)[1], self.A_org.copy().ravel()))
        # def proj(W):
        #     mat_W = W.reshape(self.nG, self.nG)
        #     proj_W = utils.projection_coA1(utils.delete_diagonal(mat_W), utils.delete_diagonal(self.A_org), self.delta_l)
        #     return utils.fill_diagonal(proj_W).ravel()
        # opt_W_1, F_1 = SPG(func, proj, init_W.ravel(), spg_options)
        # opt_W_1 = opt_W_1.reshape(self.nG, self.nG)
        # # self.warm_start_W = opt_W
        # print(f'opt on W cputime: {time.time() - start}')

        # # ------------------ Solving W Row-wise  ------------------
        start = time.time()
        init_W = self.warm_start_W
        opt_F, opt_W = 0, np.zeros((self.nG, self.nG))
        for i in range(self.nG):
            r_i = R[i]
            opt_w_i, opt_f_i = self.get_opt_W_i(i, r_i, init_W[i], spg_options)
            opt_W[i] = opt_w_i
            opt_F += opt_f_i
        # self.warm_start_W = opt_W
        # print(f'row-wise W cputime: {time.time() - start}')
        return opt_W, opt_F

    def g_hat(self, A_pert):
        if self.relu_relaxation == 'doubleL':
            f = np.sum((A_pert + np.eye(self.nG)) * self.Q, axis=1) + self.p
            grad_Z = self.Q
        else:
            A_hat_XW = A_pert @ self.XW + self.XW
            rep_U = np.tile(self.U, (self.nG, 1))

            # second term in eq (36)
            a, b = self.ub, self.ub - self.lb
            tmp1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            term2 = np.sum(rep_U * ((A_hat_XW - self.lb)*tmp1) * self.S_minus, axis=1)
            g_on_term2 = (rep_U*tmp1*self.S_minus) @ self.XW.T

            # first term in eq (36)
            before_relu = A_hat_XW
            after_relu = np.maximum(before_relu, 0)
            term1 = np.sum(rep_U * after_relu * self.S_others, axis=1)
            g_on_term1 = (rep_U*(before_relu > 0)*self.S_others) @ self.XW.T

            f = term1+term2
            grad_Z = g_on_term1+g_on_term2

        return f, grad_Z

    def g_i_hat(self, z, i):
        if self.relu_relaxation == 'doubleL':
            f = z @ self.Q[i] + self.Q[i, i] + self.p[i]
            grad_z_i = self.Q[i]
        else:
            S_i_minus = self.S_minus[i]
            A_hat_XW = z @ self.XW + self.XW[i]
            ub_i, lb_i = self.ub[i], self.lb[i]

            # second term in eq (36)
            tmp1 = ub_i[S_i_minus]/(ub_i[S_i_minus] - lb_i[S_i_minus])
            term2 = self.U[S_i_minus] @ ((A_hat_XW[S_i_minus] - lb_i[S_i_minus])*tmp1)
            g_on_term2 = self.XW[:, S_i_minus] @ (self.U[S_i_minus]*tmp1)

            # second term under vanilla relu
            # before_relu = A_hat_XW[S_i_minus]
            # after_relu = np.maximum(before_relu, 0)
            # term2_vanilla = self.U[S_i_minus] @ after_relu

            # first term in eq (36)
            S_i_others = ~(S_i_minus)
            before_relu = A_hat_XW[S_i_others]
            after_relu = np.maximum(before_relu, 0)
            term1 = self.U[S_i_others] @ after_relu
            g_on_term1 = self.XW[:, S_i_others] @ (self.U[S_i_others]*(before_relu > 0))

            f = term1+term2
            grad_z_i = g_on_term1+g_on_term2

        return f, grad_z_i

    def relu_bounds(self):
        # update lb and ub matrix
        # lb_{ij} = min_{|| A_{i:} -  A_{i:}^ori||_1 \le delta_l} A_{i:} @ (XW)_{:j} + (XW)_{i,j}
        #         = min_{1'v \le delta_l} [(2*A_{i:}^ori-1)\circ(-2v+1) + 1]/2 @ (XW)_{:j} + (XW)_{i,j}
        #         = min_{1'v \le delta_l} v.T @ [(1 - 2*A_{i:}^ori) \circ (XW)_{:j}] + A_{i:}^ori @ (XW)_{:j} + (XW)_{i,j}
        # Also need to consider delta_g
        lb = np.zeros((self.nG, self.hidden_dim))
        ub = np.zeros((self.nG, self.hidden_dim))
        for i in range(self.nG):
            A_i = self.A_org[i]
            const = A_i @ self.XW
            L = np.expand_dims((1-2*A_i), axis=1) * self.XW
            for j in range(self.hidden_dim):
                L_j = L[:, j]
                indices = np.argsort(L_j)
                k, flip, minimum = 0, 0, 0
                while k < self.nG and L_j[indices[k]] < 0 and flip < self.delta_l[i] and flip < self.delta_g:
                    if indices[k] == i:
                        k += 1
                        continue
                    minimum += L_j[indices[k]]
                    flip += 1
                    k += 1

                k, flip, maximum = self.nG-1, 0, 0
                while k > -1 and L_j[indices[k]] > 0 and flip < self.delta_l[i] and flip < self.delta_g:
                    if indices[k] == i:
                        k -= 1
                        continue
                    maximum += L_j[indices[k]]
                    flip += 1
                    k -= 1
                lb[i, j] = minimum + const[j] + self.XW[i, j]
                ub[i, j] = maximum + const[j] + self.XW[i, j]
        return lb, ub

    def doubleL_lb_coefficient(self):
        # g_i_hat = (A_i+e_i)*Q_i + p_i
        rep_U = np.tile(self.U, (self.nG, 1))

        # linear term in I_plus: (A_pert + I) .* Q1
        Q1 = (rep_U * self.I_plus) @ self.XW.T

        # linear term in I_mix: (A_pert + I) .* Q2
        a, b = self.ub, self.ub - self.lb
        tmp1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        Q2 = (rep_U * self.I_mix * tmp1) @ self.XW.T

        # remainder term in S_minus
        p = -np.sum(rep_U * self.S_minus * self.lb * tmp1, axis=1)
        return Q1+Q2, p

    def doubleL_ub_coefficient(self):
        # g_i_hat = (A_i+e_i)*Q_i + p_i
        rep_U = np.tile(self.U, (self.nG, 1))

        # linear term in I_plus: (A_pert + I) .* Q1
        Q1 = (rep_U * self.I_plus) @ self.XW.T

        # linear term in I_mix: (A_pert + I) .* Q2
        a, b = self.ub, self.ub - self.lb
        tmp1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        Q2 = (rep_U * self.I_mix * tmp1) @ self.XW.T

        # remainder term in S_plus
        S_plus = (self.lb*self.ub < 0) & np.tile((self.U > 0), (self.nG, 1))
        p = -np.sum(rep_U * S_plus * self.lb * tmp1, axis=1)
        return Q1+Q2, p

    # def F_relaxation_2(self, A_pert):
    #     # Convex envelop (38) by solving eq (39) element-wise
    #     F, G = 0, np.zeros((self.nG, self.nG))
    #     for i in range(self.nG):
    #         f, g = self.f_i_relaxation_2(A_pert[i], i)
    #         F += f
    #         G[i] = g
    #     return F, G

    # def f_i_relaxation_2(self, z, i):
    #     r = np.zeros(self.nG)
    #     r_iters = 10
    #     f_r = [0]*r_iters
    #     for j in range(r_iters):
    #         pass
    #         # alpha_iters = 10
    #         # f_alpha = [0]*alpha_iters
    #         # alpha = np.sum(A_org[i])
    #         # alpha_min, alpha_max = max(0, alpha-delta_l), min(self.nG, alpha+delta_l)
    #         # for k in range(alpha_iters):
    #         #     w_iters = 10
    #         #     f_w = [0]*w_iters
    #         #     # given alpha, find optimal w
    #         #     w = A_org[i].copy()
    #         #     for q in range(w_iters):
    #         #         f, g = g_i_hat(w, i)
    #         #         f = f/(1+alpha) - r @ w
    #         #         g = g/(1+alpha) - r
    #         #         f_w[q] = f
    #         #         w = w - 0.01*g
    #         #         w = projection_coPi_and_affine(np.delete(w, i), np.delete(A_org[i], i), delta_l, alpha)
    #         #         w = np.insert(w, i, 0)
    #         #         # w = projection_coPi_and_affine(w - 0.1*g, A_org[i], delta_l, alpha)
    #         #     # print(f_w)

    #         #     # given optimal w, take the gradient over alpha via eq (45)
    #         #     kappa, grad_kappa = g_i_hat(w, i)
    #         #     f_alpha[k] = kappa/(1+alpha) - r@w
    #         #     theta = np.abs(w - A_org[i])
    #         #     beta = (1-2*A_org[i])
    #         #     gamma = r*beta
    #         #     grad_kappa = grad_kappa * beta
    #         #     slack_idx = np.argwhere((theta<0.99) & (theta>0.01))
    #         #     if len(slack_idx) > 1:
    #         #         if np.sum(theta) < delta_l:
    #         #             si = slack_idx[0]
    #         #             mu = -(grad_kappa[si]/(1+alpha) - gamma[si])/beta[si]
    #         #         else:
    #         #             si, sj = slack_idx[0], slack_idx[1]
    #         #             if beta[si] == beta[sj]:
    #         #                 mu = -(grad_kappa[si]/(1+alpha) - gamma[si])/beta[si]
    #         #             else:
    #         #                 mu = ((grad_kappa[si]-grad_kappa[sj])/(1+alpha) + (gamma[sj]-gamma[si]))/(beta[sj]-beta[si])
    #         #     else:
    #         #         print('less than 2 indices satify theta_j \in (0, 1)!')
    #         #     grad_on_alpha = -kappa/(1+alpha)**2 - mu

    #         #     # update alpha
    #         #     alpha -= 0.1*grad_on_alpha.item()
    #         #     alpha = max(alpha_min, alpha)
    #         #     alpha = min(alpha_max, alpha)
    #         # # print(f_alpha)

    #         # f_r[j] = kappa/(1+alpha) + r@(z-w)

    # def f_i(self, z, i):
    #     before_relu = z @ self.XW + self.XW[i]
    #     after_relu = np.maximum(before_relu, 0)
    #     term1 = self.U @ after_relu
    #     g_on_term1 = self.XW @ ( self.U*(after_relu > 0) )

    #     return term1, g_on_term1
