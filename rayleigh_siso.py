# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy

NUM_EPOCHS = 45
BATCH_SIZE = 128
n = 2
k = 4
M = 2 ** k
R = k / n
train_num = 5120
test_num = 10000

fd = 926
Ts = 1e-6
Ns = 50000


def Jakes_Flat(fd, Ts, Ns, t0=0, E0=1, phi_N=0):
    '''
    Inputs:
    fd      : Doppler frequency
    Ts      : sampling period
    Ns      : number of samples
    t0      : initial time
    E0      : channel power
    phi_N   : inital phase of the maximum doppler frequency sinusoid
    Outputs:
    h       : complex fading vector
    t_state : current time
    '''
    N0 = 8
    N = 4 * N0 + 2
    wd = 2 * np.pi * fd
    t = t0 + np.asarray([i for i in range(0, Ns)]) * Ts
    H = np.ones((2, Ns))
    coff = E0 / np.sqrt(2 * N0 + 1)
    phi_n = np.asarray([np.pi * i / (N0 + 1) for i in range(1, N0 + 1)])
    phi_N = 0
    w_n = np.asarray([wd * np.cos(2 * np.pi * i / N) for i in range(1, N0 + 1)])
    h_i = np.ones((N0 + 1, Ns))
    for i in range(N0):
        h_i[i, :] = 2 * np.cos(phi_n[i]) * np.cos(w_n[i] * t)
    h_i[N0, :] = np.sqrt(2) * np.cos(phi_N) * np.cos(wd * t)
    h_q = np.ones((N0 + 1, Ns))
    for i in range(N0):
        h_q[i, :] = 2 * np.sin(phi_n[i]) * np.cos(w_n[i] * t)
    h_q[N0, :] = np.sqrt(2) * np.sin(phi_N) * np.cos(wd * t)
    h_I = coff * np.sum(h_i, 0)
    h_Q = coff * np.sum(h_q, 0)
    H[0, :] = h_I
    H[1, :] = h_Q
    return H


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


h = Jakes_Flat(fd, Ts, Ns)


class RTN(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(RTN, self).__init__()
        self.in_channels = in_channels
        self.compressed_dim = compressed_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, compressed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def R2C(self, x):
        return x.view(x.size()[0], -1, 2)

    def C2R(self, x):
        return x.view(x.size()[0], -1)

    def channel(self, x, H):
        H = Variable(torch.from_numpy(H).float())
        real = H[0] * x[:, :, 0] - H[1] * x[:, :, 1]
        real = torch.unsqueeze(real, 2)
        imag = H[0] * x[:, :, 1] + H[1] * x[:, :, 0]
        imag = torch.unsqueeze(imag, 2)
        return torch.cat([real, imag], 2)

    def encode_signal(self, x):
        return self.encoder(x)

    def decode_signal(self, x):
        return self.decoder(x)

    def normalization(self, x):
        x = (self.compressed_dim ** 0.5) * (x / x.norm(dim=-1)[:, None])
        return x

    def AWGN(self, x, ebno):
        communication_rate = R
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * ebno) ** 0.5))
        x += noise
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.normalization(x)
        x = self.R2C(x)
        H = h[:, np.random.randint(0, Ns)]
        x = self.channel(x, H)
        training_signal_noise_ratio = 7  # dB
        training_signal_noise_ratio = 10.0 ** (training_signal_noise_ratio / 10.0)
        communication_rate = R
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        x += noise
        x = self.C2R(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = RTN(M, n)
    train_labels = (torch.rand(train_num) * M).long()
    train_data = torch.sparse.torch.eye(M).index_select(dim=0,
                                                        index=train_labels)
    test_labels = (torch.rand(test_num) * M).long()
    test_data = torch.sparse.torch.eye(M).index_select(dim=0,
                                                       index=test_labels)
    dataset = Data.TensorDataset(data_tensor=train_data,
                                 target_tensor=train_labels)
    datasettest = Data.TensorDataset(data_tensor=test_data,
                                     target_tensor=test_labels)
    train_loader = Data.DataLoader(dataset=dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=2)
    test_loader = Data.DataLoader(dataset=datasettest,
                                  batch_size=test_num,
                                  shuffle=True,
                                  num_workers=2)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(NUM_EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(x)
            b_label = Variable(y)
            decoded = model(b_x)
            loss = loss_fn(decoded, b_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

    if 1:
        EbNodB_range = list(frange(-4, 8.5, 0.5))
        ber = [None] * len(EbNodB_range)
        for i in range(0, len(EbNodB_range)):
            EbNo = 10.0 ** (EbNodB_range[i] / 10.0)
            for step, (x, y) in enumerate(test_loader):
                b_x = Variable(x)
                b_y = Variable(x)
                b_label = Variable(y)
                encoder = model.encode_signal(b_x)
                encoder = model.normalization(encoder)
                encoder = model.R2C(encoder)
                H = h[:, np.random.randint(0, Ns)]
                encoder = model.channel(encoder, H)
                encoder = model.AWGN(encoder, EbNo)
                encoder = model.C2R(encoder)
                decoder = model.decode_signal(encoder)
                pred = decoder.data.numpy()
                label = b_label.data.numpy()
                pred_output = np.argmax(pred, axis=1)
                no_errors = (pred_output != label)
                no_errors = no_errors.astype(int).sum()
                ber[i] = no_errors / test_num
                print('SNR:', EbNodB_range[i], 'BLER:', ber[i])
        plt.plot(EbNodB_range, ber, 'bo', label='Autoencoder(%d,%d)' % (n, k))
        plt.yscale('log')
        plt.xlabel('SNR Range')
        plt.ylabel('Block Error Rate')
        plt.grid()
        plt.legend(loc='upper right', ncol=1)
        plt.show()

#        test_labels = torch.linspace(0, M-1, steps=M).long()
#        test_data = torch.sparse.torch.eye(M).index_select(dim=0, index=test_labels)
#        test_data = Variable(test_data)
#        x = model.encode_signal(test_data)
#        x = model.normalization(x)
#        plot_data = x.data.numpy()
