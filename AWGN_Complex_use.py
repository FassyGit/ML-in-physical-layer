import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from AWGN_ComplexChannel import AutoEncoder_C
from AutoEncoder_BasicModel import AutoEncoder_R

EbNodB_range = list(np.linspace(-4,8.5,26))
k=4
bers = genfromtxt('data/hamming74bpsk.csv',delimiter=',')
bers = 1- bers
blers = bers
for i,ber in enumerate(bers):
    blers[i] = 1 - pow(ber,k)
plt.plot(EbNodB_range, blers,'r.-' ,label= 'hamming74bpsk(7,4)')

EbNodB_train = 0
model_test = AutoEncoder_C(ComplexChannel=True,CodingMeth='Embedding',
                          M = 16, n_channel=7, k = 4, emb_k=16,
                          EbNodB_train = EbNodB_train,train_data_size=10000)
model_test.Initialize()
print("Initialization of the complex model Finished")
#model_test3.Draw_Constellation()
model_test.Cal_BLER(EbNodB_low=-4,EbNodB_high=8.5,EbNodB_num=26,bertest_data_size= 50000)
EbNodB_range = list(np.linspace(-4,8.5,26))
plt.plot(EbNodB_range, model_test.ber,'b.-',label='AE_AWGN_RESHAPE(7,4)')

model_real = AutoEncoder_R(CodingMeth='Embedding',M=16, n_channel=7, k=4,emb_k=16, EbNodB_train=EbNodB_train,train_data_size=10000)
model_real.Initialize()
print("Initialization of the real model Finished")
model_real.Cal_BLER(bertest_data_size=50000,EbNodB_low=-4,EbNodB_high=8.5,EbNodB_num=26)
plt.plot(EbNodB_range, model_real.ber,'y.-',label='AE_AWGN(7,4)')

plt.legend(loc='upper right')
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.title('awgnChannel(7,4),EnergyConstraint,EbdB_train:%f'%EbNodB_train)
plt.grid()

fig = plt.gcf()
fig.set_size_inches(16,12)
fig.savefig('graph/0506/AE_AWGN_RESHAPE(7,4)5.png',dpi=100)
plt.show()