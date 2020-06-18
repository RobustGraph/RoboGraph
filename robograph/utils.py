import pickle as pkl
import os
import numpy as np
import scipy.sparse as sp
from numba import jit, prange
from itertools import combinations


def load_graph(filename):
    """ Load data from pickle file

    Parameters
    ----------
    filename : str

    Returns
    -------
    data : a dictionary with graph and learned parameters
    """
    if os.path.exists(filename):
        if filename.endswith('pickle'):
            data = pkl.load(open(filename, 'rb'))
    return data


def process_data(data):
    """ Process DataLoader data into A, X, y

    Parameters
    ----------
    data : torch_geometric data

    Returns
    -------
    A : sp.csr_matrix
    X : np.ndarray()
    y : int
    """
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    assert len(row) == len(col)
    conn = len(row)
    size = data.x.shape[0]
    d = np.ones(conn)
    A = sp.csr_matrix((d, (row, col)), shape=(size, size), dtype=np.float32)
    A = np.float64(A.toarray())
    X = np.float64(data.x.numpy())
    y = data.y
    return A, X, y


@jit
def attack_global(G, delta=5):
    """ Random global attack

    Parameters
    ----------
    G : networkx graph
    delta : global budget

    Returns
    -------
    edge_index : np.array
    """

    size = len(G.nodes)
    all_edge = list(combinations(np.arange(size), 2))
    picked_edge = np.random.choice(all_edge, delta)
    for edge in picked_edge:
        if edge in G.edges:
            G.remove_edge(edge[0], edge[1])
        else:
            G.add_edge(edge[0], edge[1])
    edge_idx = np.array(G.edges).T
    return edge_idx


@jit
def attack_local(G, u, delta=5):
    """ Random global attack

    Parameters
    ----------
    G : networkx graph
    u : target node
    delta : global budget

    Returns
    -------
    edge_index : np.array
    """
    edge = list(G.edges[u])
    candidate_edge = [(u, i) for i in range(G.size()) if i != u]
    picked_edge = np.random.choice(candidate_edge, delta)
    for edge in picked_edge:
        if edge in G.edges:
            G.remove_edge(edge[0], edge[1])
        else:
            G.add_edge(edge[0], edge[1])
    edge_idx = np.array(G.edges).T
    return edge_idx


def cal_logits(A, XW, U, act='linear'):
    """ Return logits

    Parameters
    ----------
    A : np.array with dimension (nG, nG)
    XW : np.array with dimension (nG, d)
    U : np.array with dimension (d, c)

    Returns
    -------
    np.array with dimension (1, c)
    """
    # relu = lambda x: x * (x > 0)
    _A = A + sp.eye(A.shape[0])
    deg = _A.sum(1).A1
    D_inv = sp.diags(np.power(deg, -1))
    if act == 'relu':
        P = np.maximum(D_inv @ _A @ XW, 0)
    else:
        P = D_inv @ _A @ XW
    logits = np.mean(P, axis=0) @ U.T
    return logits


def cal_fc(A, XW, u, act='linear'):
    """ Return fc_val

    Parameters
    ----------
    A : np.array with dimension (nG, nG)
    XW : np.array with dimension (nG, d)
    u : np.array with dimension (1, c)

    Returns
    -------
    float : f_c val
    """
    _A = A + sp.eye(A.shape[0])
    deg = _A.sum(1).A1
    D_inv = sp.diags(np.power(deg, -1))
    if act == 'relu':
        P = np.maximum(D_inv @ _A @ XW, 0)
    else:
        P = D_inv @ _A @ XW
    f_val = np.mean(P, axis=0) @ u.T
    return f_val.item()
