from AutoEncoder_BasicModel import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np


EbNodB_low= -4
EbNodB_high= 8.5
EbNodB_num= 26
M=4
n_channel=2
k=2
emb_k=4
#EbNodB_train=7
train_data_size=10000
bertest_data_size=70000

#Train_EbNodB_range = list(np.linspace(start=-4, stop=8, num=13))
Train_EbNodB_range = list(np.linspace(start=5, stop=8, num=4))
EbNodB_range = list(np.linspace(start=EbNodB_low, stop=EbNodB_high, num=EbNodB_num))
for train_EnNodB in Train_EbNodB_range:
    model_test3 = AutoEncoder(CodingMeth='Embedding', M=M, n_channel=n_channel, k=k, emb_k=emb_k, EbNodB_train=train_EnNodB,
                              train_data_size=train_data_size)
    model_test3.Initialize()
    model_test3.Cal_BLER(bertest_data_size=bertest_data_size,EbNodB_low=EbNodB_low ,EbNodB_high=EbNodB_high ,
                         EbNodB_num=EbNodB_num )
    print(model_test3.EbNodB_train)
    plt.plot(EbNodB_range, model_test3.ber,label = 'Train_SNR:%f' % (train_EnNodB)
             )
    #label = 'Train_SNR:%f' % (train_EnNodB)
    plt.yscale('log')

plt.legend(fontsize='xx-small')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.grid()
plt.show()

