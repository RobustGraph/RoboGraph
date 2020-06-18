import numpy as np
import time
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
# @njit(parallel=True, fastmath=True)
# @njit
def local_solver_linear_term(A_org, XWU, delta_l, L):
    """
    solver of equation 8&11 of the paper when activation is identity, max_margin loss and average pooling    
    """
    nG = A_org.shape[0]
    a = np.zeros((nG+1, delta_l+1))     # matrix a for described in equation 6 
    add_edge_matrix = np.zeros((nG+1, delta_l+1))
    for i in prange(1, nG+1):    # looping each row of A
        A_i = A_org[i-1,:]
        A_i_edges = int(np.sum(A_i))
        L_i = L[:, i-1]
        max_edges = min(A_i_edges + delta_l + 1, nG)
        min_edges = max(A_i_edges - delta_l + 1, 1)
        possible_denomi = max_edges - min_edges + 1
        chunk_edges_mtx, chunk_no_edges_mtx = np.zeros((possible_denomi,delta_l+1)), np.zeros((possible_denomi,delta_l+1))
        for x in range(min_edges, max_edges+1):  # looping all possible (1'A_i + 1)
            V_L = XWU + L_i.T*x
            indices = np.argsort(V_L)
            chunk_edges, chunk_no_edges = [0.0]*(delta_l+1), [0.0]*(delta_l+1)
            temp_idx = 1
            for y in indices:
                if temp_idx > delta_l: break
                if y == i-1: continue    # excluding self edge
                if A_i[y] == 0:
                    chunk_no_edges[temp_idx] = V_L[y] + chunk_no_edges[temp_idx-1]
                    temp_idx += 1

            temp_idx = 1
            for y in indices[::-1]:
                if temp_idx > delta_l: break
                if y == i-1: continue    # excluding self edge
                if A_i[y] == 1:
                    chunk_edges[temp_idx] = V_L[y] + chunk_edges[temp_idx-1]
                    temp_idx += 1

            chunk_edges_mtx[x - min_edges] = chunk_edges
            chunk_no_edges_mtx[x - min_edges] = chunk_no_edges
        

        A_V_i = np.dot(A_i, np.ascontiguousarray(XWU)) + XWU[i-1]
        A_L_i = np.dot(A_i, np.ascontiguousarray(L_i))
        a[i,0] = A_V_i/(A_i_edges+1) + A_L_i
        for j in range(1,delta_l+1):    # looping each possible local constraint
            min_f = np.inf
            for k in range(j+1):  # looping different combinations of adding/removing
                add_edges, remove_edges = k, j-k
                if A_i_edges+add_edges > nG-1 or A_i_edges-remove_edges < 0:
                    continue

                new_edges = A_i_edges+add_edges-remove_edges + 1
                f = A_V_i + A_L_i*new_edges

                # adding k edges from chunk of A_i=0 in ascent order
                if add_edges > 0:
                    # print(chunk_no_edges_mtx[new_edges][add_edges])
                    f += chunk_no_edges_mtx[new_edges - min_edges][add_edges]

                # removing j-k edges from chunk of A_i=1 in descent order
                if remove_edges > 0:
                    # print(chunk_edges_mtx[new_edges][remove_edges])
                    f -= chunk_edges_mtx[new_edges - min_edges][remove_edges]

                final_f = f/new_edges
                if final_f < min_f:
                    min_f = final_f
                    sol = (min_f, add_edges)
            a[i,j], add_edge_matrix[i,j] = sol
    return a, add_edge_matrix

@njit(cache=True)
def get_A_opt(XWU, A_i, L_i, nG, delta_l, i, j, add_edges):
    A_i_edges = np.sum(A_i)
    remove_edges = j - add_edges
    new_edges = A_i_edges+add_edges-remove_edges + 1
    V_L = XWU + L_i.T*new_edges
    indices = np.argsort(V_L)
    
    A_new_i = A_i.copy()
    added_edges = 0
    for y in indices:
        if added_edges == add_edges: break
        if y == i-1: continue    # excluding self edge
        if A_i[y] == 0:
            A_new_i[y] = 1
            added_edges += 1
    
    removed_edges = 0    
    for y in indices[::-1]:
        if removed_edges == remove_edges: break
        if y == i-1: continue    # excluding self edge
        if A_i[y] == 1:
            A_new_i[y] = 0
            removed_edges += 1

    return A_new_i

@njit(cache=True)
def first_loop(a, delta_l, delta_g):
    nG = a.shape[0] - 1
    c = np.array([delta_l*x if delta_l*x < delta_g else delta_g for x in range(nG+1)])
    # s = [np.array([0.0]*(i+1)) for i in c]
    s = np.zeros((nG+1, min(nG*delta_l, delta_g)+1))
    for t in range(1, nG+1):
        st_1, st, at = s[t-1], s[t], a[t]
        for j in range(0,c[t]+1):
            m = np.inf
            for k in range(max(0, j-c[t-1]), min(j, delta_l)+1):
                m = min(st_1[j-k]+at[k], m)    # accessing s seems costly
            st[j] = m
    return c, s

@njit(cache=True)
def second_loop(A_org, XWU, delta_l, linear_term, s, c, a, add_edge_matrix):
    nG = A_org.shape[0]
    A_pert = np.zeros((nG,nG))
    j = np.argmin(s[nG])     # this sort takes nG*delta_l log(nG*delta_l)
    opt_val = s[nG][j]
    unpert_val = s[nG][0]
    for t in range(nG,0,-1):
        temp = np.ones(delta_l+1)*np.inf
        st_1, at = s[t-1], a[t]
        for k in range(max(0, j-c[t-1]), min(j, delta_l)+1):
            temp[k] = st_1[j-k] + at[k]
        kt = np.argmin(temp)
        j = j - kt
        A_pert[t-1,:] = get_A_opt(XWU, A_org[t-1,:], linear_term[:, t-1], nG, delta_l,\
                                     t, kt, add_edge_matrix[t][kt])
    
    return (unpert_val, opt_val, A_pert)


def dp_solver_1(A_org, XWU, delta_l, delta_g, linear_term):
    """
    DP for min_{A_G^{1+2}} F_c(A)  (Algorithm 1) 
    # 1. Precomputing matrix a
    # 2. DP to get matrix s
    # 3. Tracing back
    Complexity: nG^2*delta_l*log(nG) + nG*delta_l^2 + nG^2*delta_l^2

    param: 
        A_org:          original adjacency matrix
        XWU:            XW*(u_y-u_c)/nG
        delta_l:        row budgets
        delta_g:        global budgets
        linear_term:    linear term on A
        
    return:
        unpert_val:     function value under A_org     
        opt_val:        function value under A_pert
        A_pert:         optimal attacking adjacency matrix
    """
    
    # start = time.time()
    a, add_edge_matrix = local_solver_linear_term(A_org, XWU, delta_l, linear_term)
    # print(f'Precomputation of matrix a: {time.time() - start}')

    # start = time.time()
    c, s = first_loop(a, delta_l, delta_g)
    # print(f'First loop in DP          : {time.time() - start}')

    # start = time.time()
    sol = second_loop(A_org, XWU, delta_l, linear_term, s, c, a, add_edge_matrix)
    # print(f'Second loop in DP         : {time.time() - start}')

    return sol


def po_dp_solver(A_org, R, delta_l, delta_g):
    nG = A_org.shape[0]
    
    # precomputing a matrix
    J = R*(-2*A_org + 1)
    a = po_local_solver(J, nG, delta_l)

    A_pert = np.zeros((nG,nG))
    V_pert = np.zeros((nG,nG))
    c, s = first_loop(a, delta_l, delta_g)
    j = np.argmin(s[nG])
    unpert_val = s[nG][0]
    opt_val = s[nG][j]
    for t in range(nG,0,-1):
        temp = np.ones(delta_l+1)*np.inf
        st_1, at = s[t-1], a[t]
        for k in range(max(0, j-c[t-1]), min(j, delta_l)+1):
            temp[k] = st_1[j-k] + at[k]
        kt = np.argmin(temp)
        j = j - kt
        V_pert[t-1,:] = optVt_from_a_tj(J[t-1, :], t, kt, delta_l)
        A_pert[t-1,:] = ((2*A_org[t-1, :] - 1)*(-2*V_pert[t-1,:]+1)+1)/2
    
    return A_pert

def po_local_solver(J, nG, delta_l):
    a = np.zeros((nG+1, delta_l+1))
    
    for i in range(1, nG+1):    # looping each row of A
        J_i = J[i-1, :].copy()
        J_i = -np.delete(J_i, i-1)
        indices = np.argsort(J_i)

        for j in range(1,delta_l+1):    # looping each possible local constraints
            a[i,j] = J_i[indices[j-1]] + a[i,j-1]
    return a

def optVt_from_a_tj(J_t, t, j, delta_l):
    V = np.zeros(J_t.shape)
    indices = np.argsort(-J_t)
    changed_edges = 0
    for i in range(j+1):
        if indices[i] == t-1: continue 
        V[indices[i]] = 1
        changed_edges += 1
        if changed_edges >= j: break
    return V
