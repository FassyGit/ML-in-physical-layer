from numpy import genfromtxt
from math import pow
import  numpy as np
from matplotlib import pyplot as plt
k=2
uncodedbpsk_bers = genfromtxt('/data/uncodedbpsk.csv',delimiter=',')
uncodedbpsk_bers = 1- uncodedbpsk_bers
uncodedbpsk_blers = uncodedbpsk_bers
for i,uncodedbpsk_ber in enumerate(uncodedbpsk_bers):
    uncodedbpsk_blers[i] = 1 - pow(uncodedbpsk_ber,k)

EbNodB_range = list(np.linspace(-4, 8.5, 13))
plt.plot(EbNodB_range, uncodedbpsk_blers)
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.show()
