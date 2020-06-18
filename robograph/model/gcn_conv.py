import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_adj


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def normalize_adj(A, c=1):
    """ Normalize of adj

    Parameters
    ----------
    A : torch

    Returns
    -------
    row-wise normalization : torch
    """
    _device = A.device
    A = A + c * torch.eye(A.shape[0]).to(_device)
    deg = A.sum(1)
    D_inv = torch.diag(torch.pow(deg, -1))
    return D_inv @ A


class GCNConv(MessagePassing):
    r"""
    .. math: D^-1 A X W
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight=None):
        """ Simple forward implementation D^-1 A X W
        """
        # REVIEW: new simplied implmentation
        x = torch.matmul(x, self.weight)
        A = to_dense_adj(edge_index).squeeze()
        A_norm = normalize_adj(A)
        if A_norm.shape[1] != x.shape[0]:
            return A_norm @ x[:A_norm.shape[0]]
        return A_norm @ x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
