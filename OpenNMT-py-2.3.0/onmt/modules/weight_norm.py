"""  Weights normalization modules  """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def get_var_maybe_avg(namespace, var_name, training, polyak_decay):
    """ utility for retrieving polyak averaged params
        Update average
    """
    v = getattr(namespace, var_name)
    v_avg = getattr(namespace, var_name + '_avg')
    v_avg -= (1 - polyak_decay) * (v_avg - v.data)

    if training:
        return v
    else:
        return v_avg


def get_vars_maybe_avg(namespace, var_names, training, polyak_decay):
    """ utility for retrieving polyak averaged params """
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(
            namespace, vn, training, polyak_decay))
    return vars


class WeightNormLinear(nn.Linear):
    """
    Implementation of "Weight Normalization: A Simple Reparameterization
    to Accelerate Training of Deep Neural Networks"
    :cite:`DBLP:journals/corr/SalimansK16`

    As a reparameterization method, weight normalization is same
    as BatchNormalization, but it doesn't depend on minibatch.

    NOTE: This is used nowhere in the code at this stage
          Vincent Nguyen 05/18/2018
    """

    def __init__(self, in_features, out_features,
                 init_scale=1., polyak_decay=0.9995):
        super(WeightNormLinear, self).__init__(
            in_features, out_features, bias=True)

        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_features))
        self.b = self.bias

        self.register_buffer(
            'V_avg', torch.zeros(out_features, in_features))
        self.register_buffer('g_avg', torch.zeros(out_features))
        self.register_buffer('b_avg', torch.zeros(out_features))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            # out_features * in_features
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(
                self.V.data) * 0.05)
            # norm is out_features * 1
            v_norm = self.V.data / \
                self.V.data.norm(2, 1).expand_as(self.V.data)
            # batch_size * out_features
            x_init = F.linear(x, v_norm).data
            # out_features
            m_init, v_init = x_init.mean(0).squeeze(
                0), x_init.var(0).squeeze(0)
            # out_features
            scale_init = self.init_scale / \
                torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            x_init = scale_init.view(1, -1).expand_as(x_init) \
                * (x_init - m_init.view(1, -1).expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return x_init
        else:
            v, g, b = get_vars_maybe_avg(self, ['V', 'g', 'b'],
                                         self.training,
                                         polyak_decay=self.polyak_decay)
            # batch_size * out_features
            x = F.linear(x, v)
            scalar = g / torch.norm(v, 2, 1).squeeze(1)
            x = scalar.view(1, -1).expand_as(x) * x + \
                b.view(1, -1).expand_as(x)
            return x
