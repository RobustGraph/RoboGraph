import numpy as np
from robograph.attack.utils import calculate_Fc

class Greedy_Attack(object):
    """
    Greedy algorithm (upper bound) for solving min_{A_G^{1+2+3}} F_c(A)

    param: 
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        delta_l:        row budgets (vector)
        delta_g:        global budget (scalar)
        activation:     'linear' or 'relu'
    """

    def __init__(self, A_org, XW, U, delta_l, delta_g, activation='linear'):
        self.A_org = A_org
        self.XW, self.U = XW, U
        self.XWU = XW @ U
        self.delta_l = delta_l
        self.delta_g = delta_g
        self.activation = activation
        self.nG = A_org.shape[0]
    
    def attack(self, A_pert):
        """
        Greedy searching minimal F_c(A) (upper bound)

        param: 
            A_pert:         Starting point for greedy search
        
        return a dict with keywords:
            opt_A:          optimal perturbed matrix
            opt_f:          optimal objective value
        """
        # Precompution based on initial A
        self.A_pert = A_pert
        self.edge_coors, self.no_edge_coors = [], []
        self.Flip = np.logical_xor(self.A_org, self.A_pert)
        self.local_budget = np.sum(self.Flip, axis=1)
        self.global_budget = np.sum(self.local_budget)
        self.Denominator, self.Numerator = np.zeros(self.nG), np.zeros(self.nG)
        for i in range(self.nG):
            self.Denominator[i] = np.sum(A_pert[i]) + 1
            if self.activation == 'linear':
                self.Numerator[i] = A_pert[i] @ self.XWU + self.XWU[i]
            else:
                self.Numerator[i] = np.maximum(A_pert[i] @ self.XW + self.XW[i], 0) @ self.U
            for j in range(i+1, self.nG):
                if A_pert[i][j]: self.edge_coors.append((i, j))
                else: self.no_edge_coors.append((i, j))

        # Prepruning if not satisfing local/global budget
        if self.global_budget > self.delta_g or np.any(self.local_budget > self.delta_l):
            self.preprocess(self.A_pert)

        # Attack
        A_start = self.A_pert.copy()
        while True:
            A_end = self.pruning(A_start)
            A_end = self.adding(A_end)
            if np.array_equal(A_start, A_end):  break
            else:   A_start = A_end
        sol = {
            'opt_A': A_end,
            'opt_f': calculate_Fc(A_end, self.XW, self.U, self.activation)
        }
        return sol



    def preprocess(self, A_pert):
        while np.any(self.local_budget > self.delta_l):
            max_decrement, change_coordinate = np.inf, None
            for i in range(self.nG):
                for j in range(i+1, self.nG):
                    if self.Flip[i][j] and (self.local_budget[i] > self.delta_l[i] or self.local_budget[j] > self.delta_l[j]):
                        if A_pert[i][j]:
                            nume_i = self.Numerator[i] - self.XWU[j]
                            deno_i = self.Denominator[i] - 1
                            nume_j = self.Numerator[j] - self.XWU[i]
                            deno_j = self.Denominator[j] - 1
                        else:
                            nume_i = self.Numerator[i] + self.XWU[j]
                            deno_i = self.Denominator[i] + 1
                            nume_j = self.Numerator[j] + self.XWU[i]
                            deno_j = self.Denominator[j] + 1
                        decrement = nume_i/deno_i - self.Numerator[i]/self.Denominator[i]
                        decrement += nume_j/deno_j - self.Numerator[j]/self.Denominator[j]
                        if decrement < max_decrement:
                            max_decrement = decrement
                            change_coordinate = (i, j)

            if change_coordinate:
                i, j = change_coordinate
                A_pert[i][j], A_pert[j][i] = self.A_org[i][j], self.A_org[j][i]
                self.local_budget[i] -= 1
                self.local_budget[j] -= 1
                self.global_budget -= 2
                self.Flip[i][j], self.Flip[j][i] = 0, 0
                if A_pert[i][j]:
                    self.Numerator[i] += self.XWU[j]
                    self.Numerator[j] += self.XWU[i]
                    self.Denominator[i] += 1
                    self.Denominator[j] += 1
                else:
                    self.Numerator[i] -= self.XWU[j]
                    self.Numerator[j] -= self.XWU[i]
                    self.Denominator[i] -= 1
                    self.Denominator[j] -= 1
                if np.any(self.Denominator == 0):
                    aa = 1

        while self.global_budget > self.delta_g:
            max_decrement, change_coordinate = np.inf, None
            for i in range(self.nG):
                for j in range(i+1, self.nG):
                    if self.Flip[i][j]:
                        if A_pert[i][j]:
                            nume_i = self.Numerator[i] - self.XWU[j]
                            deno_i = self.Denominator[i] - 1
                            nume_j = self.Numerator[j] - self.XWU[i]
                            deno_j = self.Denominator[j] - 1
                        else:
                            nume_i = self.Numerator[i] + self.XWU[j]
                            deno_i = self.Denominator[i] + 1
                            nume_j = self.Numerator[j] + self.XWU[i]
                            deno_j = self.Denominator[j] + 1
                        decrement = nume_i/deno_i - self.Numerator[i]/self.Denominator[i]
                        decrement += nume_j/deno_j - self.Numerator[j]/self.Denominator[j]
                        if decrement < max_decrement:
                            max_decrement = decrement
                            change_coordinate = (i, j)

            if change_coordinate:
                i, j = change_coordinate
                A_pert[i][j], A_pert[j][i] = self.A_org[i][j], self.A_org[j][i]
                self.local_budget[i] -= 1
                self.local_budget[j] -= 1
                self.global_budget -= 2
                self.Flip[i][j], self.Flip[j][i] = 0, 0
                if A_pert[i][j]:
                    self.Numerator[i] += self.XWU[j]
                    self.Numerator[j] += self.XWU[i]
                    self.Denominator[i] += 1
                    self.Denominator[j] += 1
                else:
                    self.Numerator[i] -= self.XWU[j]
                    self.Numerator[j] -= self.XWU[i]
                    self.Denominator[i] -= 1
                    self.Denominator[j] -= 1
                if np.any(self.Denominator == 0):
                    aa = 1
        if self.global_budget > self.delta_g or np.any(self.local_budget > self.delta_l):
            raise AssertionError('still does not satisfy budgets after prepruning')
        
        self.edge_coors, self.no_edge_coors = [], []
        self.Flip = np.logical_xor(self.A_org, self.A_pert)
        self.local_budget = np.sum(self.Flip, axis=1)
        self.global_budget = np.sum(self.local_budget)
        self.Denominator, self.Numerator = np.zeros(self.nG), np.zeros(self.nG)
        for i in range(self.nG):
            self.Denominator[i] = np.sum(A_pert[i]) + 1
            self.Numerator[i] = np.dot(A_pert[i], self.XWU) + self.XWU[i]
            for j in range(i+1, self.nG):
                if A_pert[i][j]: self.edge_coors.append((i, j))
                else: self.no_edge_coors.append((i, j))
            
    def pruning(self, A):
        max_decrement = 0 
        for idx, coodinate in enumerate(self.edge_coors):
            i, j = coodinate
            local_budget_i = self.local_budget[i] + 2*~(self.Flip[i][j]) - 1
            local_budget_j = self.local_budget[j] + 2*~(self.Flip[j][i]) - 1
            global_budget_ij = self.global_budget + 2*~(self.Flip[i][j]) - 1 + 2*~(self.Flip[j][i]) - 1
            if local_budget_i <= self.delta_l[i] and local_budget_j <= self.delta_l[j] \
                and global_budget_ij <= self.delta_g:
                # if satisfy local/global budgets, removing edge
                if self.activation == 'linear':
                    nume_i = self.Numerator[i] - self.XWU[j]
                    nume_j = self.Numerator[j] - self.XWU[i]
                else:
                    nume_i = np.maximum(A[i] @ self.XW + self.XW[i] - self.XW[j], 0) @ self.U
                    nume_j = np.maximum(A[j] @ self.XW + self.XW[j] - self.XW[i], 0) @ self.U
                deno_i = self.Denominator[i] - 1
                deno_j = self.Denominator[j] - 1
                decrement = nume_i/deno_i - self.Numerator[i]/self.Denominator[i]
                decrement += nume_j/deno_j - self.Numerator[j]/self.Denominator[j]
                if decrement < max_decrement:
                    max_decrement = decrement
                    pruning_coordinate = coodinate
                    pruning_idx = idx
        
        # update and return
        A_opt = A.copy()
        if max_decrement < 0:
            i, j = pruning_coordinate
            A_opt[i][j], A_opt[j][i] = 0, 0
            self.local_budget[i] += 2*~(self.Flip[i][j]) - 1
            self.local_budget[j] += 2*~(self.Flip[j][i]) - 1
            self.global_budget += 2*~(self.Flip[i][j]) - 1 + 2*~(self.Flip[j][i]) - 1
            self.Flip[i][j], self.Flip[j][i] = ~(self.Flip[i][j]), ~(self.Flip[j][i])
            if self.activation == 'linear':
                self.Numerator[i] -= self.XWU[j]
                self.Numerator[j] -= self.XWU[i]
            else:
                self.Numerator[i] = np.maximum(A_opt[i] @ self.XW + self.XW[i], 0) @ self.U
                self.Numerator[j] = np.maximum(A_opt[j] @ self.XW + self.XW[j], 0) @ self.U
            self.Denominator[i] -= 1
            self.Denominator[j] -= 1
            self.edge_coors.pop(pruning_idx)
            self.no_edge_coors.append(pruning_coordinate)
        if np.any(self.Denominator == 0):
                    aa = 1
        return A_opt

    def adding(self, A):
        max_decrement = 0 
        for idx, coordinate in enumerate(self.no_edge_coors):
            i, j = coordinate
            local_budget_i = self.local_budget[i] + 2*~(self.Flip[i][j]) - 1
            local_budget_j = self.local_budget[j] + 2*~(self.Flip[j][i]) - 1
            global_budget_ij = self.global_budget + 2*~(self.Flip[i][j]) - 1 + 2*~(self.Flip[j][i]) - 1
            if local_budget_i <= self.delta_l[i] and local_budget_j <= self.delta_l[j] \
                and global_budget_ij <= self.delta_g:
                # if satisfy local/global budgets, adding edge
                if self.activation == 'linear':
                    nume_i = self.Numerator[i] + self.XWU[j]
                    nume_j = self.Numerator[j] + self.XWU[i]
                else:
                    nume_i = np.maximum(A[i] @ self.XW + self.XW[i] + self.XW[j], 0) @ self.U
                    nume_j = np.maximum(A[j] @ self.XW + self.XW[j] + self.XW[i], 0) @ self.U
                deno_i = self.Denominator[i] + 1
                deno_j = self.Denominator[j] + 1
                decrement = nume_i/deno_i - self.Numerator[i]/self.Denominator[i]
                decrement += nume_j/deno_j - self.Numerator[j]/self.Denominator[j]
                if decrement < max_decrement:
                    max_decrement = decrement
                    removing_coordinate = coordinate
                    removing_idx = idx
        
        # update and return
        A_opt = A.copy()
        if max_decrement < 0:
            i, j = removing_coordinate
            A_opt[i][j], A_opt[j][i] = 1, 1
            self.local_budget[i] += 2*~(self.Flip[i][j]) - 1
            self.local_budget[j] += 2*~(self.Flip[j][i]) - 1
            self.global_budget += 2*~(self.Flip[i][j]) - 1 + 2*~(self.Flip[j][i]) - 1
            self.Flip[i][j], self.Flip[j][i] = ~(self.Flip[i][j]), ~(self.Flip[j][i])
            if self.activation == 'linear':
                self.Numerator[i] += self.XWU[j]
                self.Numerator[j] += self.XWU[i]
            else:
                self.Numerator[i] = np.maximum(A_opt[i] @ self.XW + self.XW[i], 0) @ self.U
                self.Numerator[j] = np.maximum(A_opt[j] @ self.XW + self.XW[j], 0) @ self.U
            self.Denominator[i] += 1
            self.Denominator[j] += 1
            self.no_edge_coors.pop(removing_idx)
            self.edge_coors.append(removing_coordinate)
        if np.any(self.Denominator == 0):
            aa = 1
        return A_opt
