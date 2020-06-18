import os.path as osp
import torch
import torch.nn.functional as F

from torch.nn import Module, Linear, Dropout
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GlobalAttention
from torch_scatter import scatter_add
from robograph.model.gcn_conv import GCNConv
from robograph.attack.dual import dual_solver, dual_solver_doubleL
from robograph.attack.greedy_attack import Greedy_Attack
from robograph.utils import process_data
import numpy as np


class GC_NET(Module):
    """ Graph classification using single layer GCN
    """

    def __init__(self, hidden, n_features, n_classes, act='relu', pool='avg', dropout=0.):
        """ Model init

        Parameters
        ----------
        hidden: int
            Size of hidden layer
        n_features: int
            Size of feature dimension
        n_classes: int
            Number of classes
        act: str in ['relu', 'linear']
            Default: 'relu'
        pool: str in ['avg', 'max', 'att_h', 'att_x']
            Default: 'avg'
        dropout: float
            Dropout rate in training. Default: 0.
        """
        super(GC_NET, self).__init__()

        self.hidden = hidden
        self.n_features = n_features
        self.n_classes = n_classes
        self.act = act
        self.pool = pool

        # GCN layer
        self.conv = GCNConv(self.n_features, self.hidden, bias=False)
        # pooling
        if self.pool == 'att_x':
            self.att_x = Linear(self.n_features, self.n_features, bias=False)
        elif self.pool == 'att_h':
            self.att_h = GlobalAttention(torch.nn.Linear(self.hidden, 1))
        # linear output
        self.lin = Linear(self.hidden, self.n_classes, bias=False)
        # dropout
        self.dropout = Dropout(dropout)

    def forward(self, data):
        """ Forward computation of GC_NET model, computes the logits for each
        graph.

        Parameters
        ----------
        data: ptg.Data

        Returns
        -------
        logits: torch.tensor float32 [B, K]
            Logits of prediction for each label
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.training:
            x = self.dropout(x)
        if self.pool == 'att_x':
            P = torch.matmul(self.att_x(x), x.T)
            P = torch.sigmoid(P)
            P = F.softmax(scatter_add(P, batch, dim=0), dim=-1)

        if self.act == 'relu':
            x = F.relu(self.conv(x, edge_index))
        else:
            x = self.conv(x, edge_index)

        if self.training:
            x = self.dropout(x)

        if self.pool == 'avg':
            x = gap(x, batch)
        elif self.pool == 'max':
            x = gmp(x, batch)
        elif self.pool == 'att_h':
            x = self.att_h(x, batch)
        else:
            x = torch.matmul(P, x)

        logits = self.lin(x)
        return logits

    def predict(self, data):
        """ Predcit the classes of the input data

        Parameters
        ----------
        data: ptg.Data

        Returns
        -------
        labels: list
            Labels of the predicted data [B, ]
        """
        return self.forward(data).argmax(1).detach()


def train(model, loader, robust=False, adv_loader=None, lamb=0):
    """ Train GC_Net

    Parameters
    ----------
    model: GC_NET instance
    loader: torch.util.data.DataLoader
        DataLoader with each data in torch.Data
    robust: bool
        Flag for robust training. Defualt: False

    Returns
    -------
    loss: float
        Averaged loss on loader.
    """
    model.train()
    _device = next(model.parameters()).device

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_all = 0
    loader = loader if not robust else adv_loader

    for idx, data in enumerate(loader):
        data = data.to(_device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, data.y)

        if robust:
            ''' robust training with greedy attack '''
            # _W = model.conv.weight.detach().cpu().numpy()
            # _U = model.lin.weight.detach().cpu().numpy()

            # for _ in range(20):
            #     idx = np.random.randint(len(loader.dataset))
            #     _g_data = loader.dataset[idx]
            #     A, X, y = process_data(_g_data)

            #     deg = A.sum(1)
            #     # local budget
            #     delta_l = np.minimum(np.maximum(deg - np.max(deg) + 2,
            #                                     0), data.x.shape[0] - 1).astype(int)
            #     # global budget
            #     delta_g = 4

            #     fc_vals_greedy = []
            #     for c in range(model.n_classes):
            #         if c != y:
            #             u = _U[y] - _U[c]
            #             attack = Greedy_Attack(A, X@_W, u.T / data.x.shape[0], delta_l, delta_g,
            #                                 activation=model.act)
            #             greedy_sol = attack.attack(A)
            #             fc_vals_greedy.append(-greedy_sol['opt_f'])
            #     loss += max(max(fc_vals_greedy) + 1, 0) / 20
            # for adv in adv_loader:
            #     adv = adv.to(_device)
            #     output = model(adv)
            #     loss = F.hinge_embedding_loss(output, torch.eye(output.shape[1])[data.y].to(_device), margin=0.5)
            # loss
            loss += lamb * F.hinge_embedding_loss(output, torch.eye(output.shape[1])[data.y].to(_device))
        # loss = F.multilabel_margin_loss(output.argmax(1), data.y).to(_device)
        loss.backward()
        optimizer.step()
        loss_all += data.num_graphs * loss.item()

    return loss_all / len(loader.dataset)


def eval(model, loader, testing=False, save_path=None, robust=False):
    """ Evaluate model with dataloader

    Parameters
    ----------
    model: GC_NET instance
    loader: torch.util.data.DataLoader
        DataLoader with each data in torch.Data
    testing: bool
        Flag for testing. Default: False
    save_path: str
        Load model from saved path. Default: None
    robust: bool
        Flag for robust training. Defualt: False

    Returns
    -------
    accuracy: float
        Accuracy with loader
    """
    model.eval()
    _device = next(model.parameters()).device
    if testing:
        result_fn = 'result_robust.pk' if robust else 'result.pk'
        model.load_state_dict(torch.load(osp.join(save_path, result_fn)))

    correct = 0
    for data in loader:
        data = data.to(_device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
