import numpy as np
from sympy.utilities.iterables import variations, cartes
from robograph.attack.utils import calculate_Fc, calculate_doubleL_Fc, fill_diagonal

def possible_matrix_with_delta_l(A_org, delta_l):
    nG = A_org.shape[0]
    possible_rows = list(variations([0,1], nG-1, True))
    possible_all_A = []
    for i in range(nG):
        A_i = A_org[i]
        possible_A_i_new = []
        for rows in possible_rows:
            A_i_new = list(rows)
            A_i_new.insert(i, 0)
            diff = sum(abs(A_i_new - A_i))
            if diff <= delta_l[i]:
                possible_A_i_new.append(A_i_new)
        possible_all_A.append(possible_A_i_new)

    return list(cartes(*possible_all_A))


def sanity_check_dp(A_org, XW, U, L, delta_l, delta_g, check_symmetry=True, \
                                                    activation='linear'):
    """
    Sanity approach for solving min_{A_G^{1+2+3}} F_c(A) + np.sum(A.*L)

    param: 
        A_org:          original adjacency matrix
        XW:             XW
        U:              (u_y-u_c)/nG
        L:              L
        delta_l:        row budgets
        delta_g:        global budgets
        check_symmetry: If True, optA is symmtric
        activation:     'linear' or 'relu'
    
    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        opt_f:          optimal dual objective
    """
    nG = A_org.shape[0]
    if nG > 6 and delta_g > 2:
        print("Sanity check only support nG < 7, return None!")
    else:
        if delta_g == 2:
            Flip_idx = []
            for row in range(nG):
                for col in range(row+1, nG):
                    if delta_l[row] > 0 and delta_l[col] > 0: 
                        Flip_idx.append([(row, col), (col, row)])

            minimum = np.inf
            for idx in Flip_idx:
                A = A_org.copy()
                for s in idx:
                    A[s] = 1-A[s]
                val = calculate_Fc(A, XW, U, activation) + np.sum(L*A)
                if val < minimum:
                    minimum = val
                    A_final = A
            
        else:
            all_possible_adjacency_matrices = possible_matrix_with_delta_l(A_org, delta_l)
            print('# matrice satisfing delta_l: ', len(all_possible_adjacency_matrices))
            
            XWU = XW @ U
            minimum = np.inf
            for possible_matrix in all_possible_adjacency_matrices:
                possible_matrix = np.asarray(possible_matrix)
                
                symmetry = np.allclose(possible_matrix, possible_matrix.T) if check_symmetry else True
                if symmetry and np.sum(np.abs(A_org-possible_matrix)) <= delta_g:
                    val = calculate_Fc(possible_matrix, XW, U, activation) + np.sum(L*possible_matrix)
                    if val < minimum:
                        minimum = val
                        A_final = possible_matrix

        sol = {
            'opt_A': A_final,
            'opt_f': minimum
        }
        return sol

def sanity_check_doubleL_relax(A_org, Q, p, L, delta_l, delta_g, check_symmetry=True):
    """
    Sanity approach for solving min_{A_G^{1+2+3}} F_c(A) + np.sum(A.*L)

    param: 
        A_org:          original adjacency matrix
        Q:              Q
        p:              p
        L:              L
        delta_l:        row budgets
        delta_g:        global budgets
        check_symmetry: If True, optA is symmtric
        activation:     'linear' or 'relu'
    
    return a dict with keywords:
        opt_A:          optimal perturbed matrix
        opt_f:          optimal dual objective
    """

    nG = A_org.shape[0]
    if nG > 6:
        print("Sanity check only support nG < 7, return None!")
    else:
        all_possible_adjacency_matrices = possible_matrix_with_delta_l(A_org, delta_l)
        print('# matrice satisfing delta_l: ', len(all_possible_adjacency_matrices))
        
        minimum = np.inf
        for possible_matrix in all_possible_adjacency_matrices:
            possible_matrix = np.asarray(possible_matrix)
            
            symmetry = np.allclose(possible_matrix, possible_matrix.T) if check_symmetry else True
            if symmetry and np.sum(np.abs(A_org-possible_matrix)) <= delta_g:
                val = calculate_doubleL_Fc(possible_matrix, Q, p) + np.sum(L*possible_matrix)
                if val < minimum:
                    minimum = val
                    A_final = possible_matrix

        sol = {
            'opt_A': A_final,
            'opt_f': minimum
        }
        return sol


def sanity_check_polar_operator(A_from_dp, A_org, R, delta_l, delta_g):
    nG = A_org.shape[0]
    all_possible_adjacency_matrices = possible_matrix_with_delta_l(A_org, delta_l)
    print('# matrice satisfing delta_l: ', len(all_possible_adjacency_matrices))
    
    A_final = np.array([])
    m = -float('inf')
    for possible_matrix in all_possible_adjacency_matrices:
        possible_matrix = np.array([np.array(i) for i in possible_matrix])
        
        # if np.allclose(possible_matrix,possible_matrix.T) and check_global_budget(possible_matrix, A_org, delta_l, delta_g):
        if check_global_budget(possible_matrix, A_org, delta_l, delta_g):
            if np.array_equal(possible_matrix,A_from_dp):
                print('--- Match Found ---')
            val = np.sum(R*possible_matrix)
            if val > m:
                m = val
                A_final = possible_matrix.copy()

    print("---DP Solution---")
    print(A_from_dp)
    print(calculate_budget(A_from_dp, A_org, delta_l, delta_g))
    print(np.sum(R*A_from_dp))

    print('---Brute Force Solution----')
    print(A_final)
    print(calculate_budget(A_final, A_org, delta_l, delta_g))
    print(m)
    print('-------------------------------------------------------')