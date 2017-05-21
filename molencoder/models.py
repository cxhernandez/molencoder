import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils import Flatten, Repeat, TimeDistributed

__all__ = ['MolEncoder', 'MolDecoder']


def ConvBNReLU(i, o, kernel_size=3, padding=0, p=0.):
    model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
             nn.BatchNorm1d(o),
             nn.LeakyReLU(inplace=True)
             ]
    if p > 0.:
        model += [nn.Dropout2d(p)]
    return nn.Sequential(*model)


class Lambda(nn.Module):

    def forward(self, x, y):
        eps = Variable(torch.randn(*x.size())).type_as(x)
        return x + torch.exp(y / 2.) * eps


class MolEncoder(nn.Module):

    def __init__(self, i=120, o=292, c=35):
        super(MolEncoder, self).__init__()

        self.conv_1 = ConvBNReLU(i, 9, kernel_size=9)
        self.conv_2 = ConvBNReLU(9, 9, kernel_size=9)
        self.conv_3 = ConvBNReLU(9, 10, kernel_size=11)
        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435),
                                     nn.BatchNorm1d(435),
                                     nn.LeakyReLU(inplace=True))

        self.z_mean = nn.Linear(435, o)
        self.z_log_var = nn.Linear(435, o)
        self.z = (torch.zeros(435), torch.zeros(435))

        self.lmbd = Lambda()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = Flatten()(out)
        out = self.dense_1(out)

        self.z = (self.z_mean(out), self.z_log_var(out))

        return self.lmbd(*self.z)

    def vae_loss(self, x, x_decoded_mean):
        z_mean, z_log_var = self.z

        bce = nn.BCELoss(size_average=True)
        xent_loss = bce(x_decoded_mean, x)
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))

        return kl_loss + xent_loss


class MolDecoder(nn.Module):

    def __init__(self, i=292, o=120, c=35):
        super(MolDecoder, self).__init__()

        self.latent_input = nn.Sequential(nn.Linear(i, i),
                                          nn.BatchNorm1d(i),
                                          nn.LeakyReLU(inplace=True))
        self.repeat_vector = Repeat(o)
        self.gru_1 = nn.GRU(i, 501)
        self.gru_2 = nn.GRU(501, 501)
        self.gru_3 = nn.GRU(501, 501)
        self.softmax = nn.Sequential(nn.Linear(501, c), nn.Softmax())
        self.decoded_mean = TimeDistributed(self.softmax, batch_first=True)

    def forward(self, x):
        out = self.latent_input(x)
        out = self.repeat_vector(out)
        out, h = self.gru_1(out)
        out, h = self.gru_2(out)
        out, h = self.gru_3(out)
        return self.decoded_mean(out)
