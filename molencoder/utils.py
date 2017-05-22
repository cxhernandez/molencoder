import h5py
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer


class ReduceLROnPlateau(object):

    def __init__(self, optimizer, mode='min', factor=0.5, patience=5,
                 verbose=True, epsilon=1E-4, min_lr=0.):

        if factor <= 0.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor <= 0.0')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self.reset()

    def reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError(
                'Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: a < (b - self.epsilon)
            self.best = 1E12
        else:
            self.monitor_op = lambda a, b: a > (b + self.epsilon)
            self.best = -1E12
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1E-4

    def step(self, metric, epoch):
        if self.monitor_op(metric, self.best):
            self.best = metric
            self.wait = 0

        elif self.wait >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = float(param_group['lr'])
                if old_lr > (self.min_lr + self.lr_epsilon):
                    new_lr = old_lr * self.factor
                    param_group['lr'] = max(new_lr, self.min_lr)
                    if self.verbose:
                        print('Reducing learning rate to %s.' % new_lr)
                    self.wait = 0
        else:
            self.wait += 1


class Flatten(nn.Module):

    def forward(self, x):
        size = x.size()  # read in N, C, H, W
        return x.view(size[0], -1)


class Repeat(nn.Module):

    def __init__(self, rep):
        super(Repeat, self).__init__()

        self.rep = rep

    def forward(self, x):
        size = tuple(x.size())
        size = (size[0], 1) + size[1:]
        x_expanded = x.view(*size)
        n = [1 for _ in size]
        n[1] = self.rep
        return x_expanded.repeat(*n)


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
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
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def train_model(train_loader, encoder, decoder, optimizer, dtype,
                print_every=100):
    encoder.train()
    decoder.train()

    for t, (x, y) in enumerate(train_loader):
        x_var = Variable(x.type(dtype))

        y_var = encoder(x_var)
        z_var = decoder(y_var)

        loss = encoder.vae_loss(x_var, z_var)
        if (t + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_model(val_loader, encoder, decoder, dtype):
    encoder.eval()
    decoder.eval()

    avg_val_loss = 0.
    for t, (x, y) in enumerate(val_loader):
        x_var = Variable(x.type(dtype))

        y_var = encoder(x_var)
        z_var = decoder(y_var)

        avg_val_loss += encoder.vae_loss(x_var, z_var).data
    avg_val_loss /= t
    print('average validation loss: %.4f' % avg_val_loss[0])
    return avg_val_loss[0]


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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
