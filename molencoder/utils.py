import h5py
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
        x_expanded = x.view(*size)
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
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(1), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def train(loader_train, encoder, decoder, optimizer, dtype):
    model.train()
    for t, x in enumerate(loader_train):
        x_var = Variable(x.type(dtype))

        y_var = encoder(x_var)
        z_var = decoder(y_var)

        loss = encoder.vae_loss(x_var, z_var.detach())
        if (t + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def load_dataset(filename, split=True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)
