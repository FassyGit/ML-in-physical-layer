# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


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
    # tf = t0 + Ns * Ts
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

    return h_I, h_Q


# test
fd = 926
Ts = 1e-6
Ns = 50000
t0 = 0
E0 = 1
phi_N = 0

h_I, h_Q = Jakes_Flat(fd, Ts, Ns)

plt.figure("amplitude")
plt.title("amplitude")
plt.plot(np.sqrt(h_Q * h_Q + h_I * h_I))
plt.yscale('log')
plt.show()
plt.figure("hist")
plt.title("hist")
n, bins, patches = plt.hist(np.sqrt(h_Q * h_Q + h_I * h_I), bins=50, normed=0)
plt.show()


