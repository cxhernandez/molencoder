import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        size = x.size()  # read in N, C, H, W
        return x.view(size[0], -1)


class Repeat(nn.Module):
    def __init__(self, rep):
        super(Repeat, self).__init__()

        self.rep = rep

    def forward(self, x):
        size = (1,) + tuple(x.size())
        x_expanded =  x.view(*size)
        n = [1 for _ in size]
        n[0] = self.rep
        return x_expanded.repeat(*n)


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(1), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
