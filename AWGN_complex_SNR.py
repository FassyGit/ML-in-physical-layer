from AWGN_ComplexChannel import AutoEncoder_C
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

EbNodB_range = list(np.linspace(-4,8.5,26))
k=2
bers = genfromtxt('data/uncodedbpsk.csv',delimiter=',')
bers = 1- bers
blers = bers
for i,ber in enumerate(bers):
    blers[i] = 1 - pow(ber,k)
plt.plot(EbNodB_range, blers,'r.-' ,label= 'uncodedbpsk(2,2)')

EbNodB_low= -4
EbNodB_high= 8.5
EbNodB_num= 26
M=4
n_channel=2
k=2
emb_k=4
#EbNodB_train=7
train_data_size=10000
bertest_data_size=50000
number = 7

#Train_EbNodB_range = list(np.linspace(start=5, stop=8, num=number))
Train_EbNodB_range = list(np.linspace(start=-10, stop=20, num=number))
EbNodB_range = list(np.linspace(start=EbNodB_low, stop=EbNodB_high, num=EbNodB_num))
for (i,train_ebnodb) in enumerate(Train_EbNodB_range):
    print('train_ebnodb',train_ebnodb)
    model_complex = AutoEncoder_C(ComplexChannel=True,CodingMeth='Embedding',M=M,n_channel=n_channel,k=k,
                                  emb_k=emb_k,EbNodB_train=train_ebnodb,train_data_size=train_data_size)
    model_complex.Initialize()
    model_complex.Cal_BLER(bertest_data_size=bertest_data_size,EbNodB_low=EbNodB_low,
                           EbNodB_high=EbNodB_high,EbNodB_num=EbNodB_num)
    plt.plot(EbNodB_range, model_complex.ber,linestyle='-.', label = 'Train_SNR:%f' % (train_ebnodb))

plt.legend(loc = 'lower left')
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.title('awgnChannel(2,2),EnergyConstraint,SNRComparison')
plt.grid()

fig = plt.gcf()
fig.set_size_inches(16,12)
fig.savefig('graph/0506/AE_AWGN_RESHAPE(2,2)5.png',dpi=100)
plt.show()
