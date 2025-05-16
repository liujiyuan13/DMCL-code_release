import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Linear
from collections import OrderedDict


class MLP(nn.Module):

    def __init__(self, dims, device='cpu'):
        super(MLP, self).__init__()

        self.device = device

        layer_dict = OrderedDict()
        if len(dims) < 2:
            raise ValueError
        elif len(dims) == 2:
            layer_dict['fc'] = Linear(dims[0], dims[1], bias=True)
        else:
            for i in range(len(dims) - 2):
                layer_dict['fc{}'.format(i)] = Linear(dims[i], dims[i + 1], bias=True)
                layer_dict['relu{}'.format(i)] = nn.ReLU(inplace=True)
            layer_dict['fc{}'.format(i + 1)] = Linear(dims[i + 1], dims[i + 2], bias=True)

        self.dims = dims
        self.mlp = nn.Sequential(layer_dict)

        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
