import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets import MLP


def loss_fun_mse(xr, x):
    return F.mse_loss(xr, x)


def loss_fun_rec(xs, xrs):
    loss_rec = 0
    for iv, x in enumerate(xs):
        loss_rec += F.mse_loss(xrs[iv], xs[iv])
    return loss_rec


def loss_fun_orth(latents, target=None):
    sim = torch.matmul(latents.t(), latents)
    if target is None:
        target = torch.eye(latents.shape[1], device=latents.device)
    else:
        target = torch.diag(target)
    return F.mse_loss(sim, target)


def loss_fun_adaptive_kernel_alignment(xs, latents):
    KAs, vs = [], []
    y = latents
    for iv, x in enumerate(xs):
        ka = torch.pow(torch.norm(x.T.matmul(y), 'fro'), 2) / (torch.norm(x.T.matmul(x), 'fro') * torch.norm(y.T.matmul(y), 'fro'))
        KAs.append(ka)
        vs.append(ka.item())
    vnorm = np.linalg.norm(vs, ord=2)
    ws = [v / vnorm for v in vs]
    loss_KA = 0
    for w, KA in zip(ws, KAs):
        loss_KA -= w * KA
    return loss_KA


class DMCL(nn.Module):
    '''
    Deep Multi-view Clustering on Latent Variables
    '''
    def __init__(self,
                 dim_latents: int,
                 dims_mapping: list,
                 n_samples: int,
                 device='cpu'):
        super(DMCL, self).__init__()

        self.device = device
        self.dim_latents = dim_latents
        self.dims_mapping = dims_mapping
        self.n_views = len(dims_mapping)

        # check the input
        for iv, dim_mapping in enumerate(dims_mapping):
            assert dim_mapping[0] == dim_latents

        # init the latents
        self.latents = nn.Parameter(1.0e-8 * torch.ones((n_samples, self.dim_latents), device=device), requires_grad=True)
        nn.init.xavier_uniform_(self.latents)

        # init the mapping nets
        self.mappings = []
        for iv in range(self.n_views):
            self.mappings.append(MLP(dims_mapping[iv]).to(device))

        # for extra regularization of latents
        self.target_diag = nn.Parameter(torch.ones((self.dim_latents,), device=device), requires_grad=True)

    def fit(self, idx):

        # mapping to observations
        xrs = []
        for mapping in self.mappings:
            xrs.append(mapping(self.latents[idx, :]))

        return xrs, self.latents[idx, :]
