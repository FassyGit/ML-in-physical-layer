from TwoUserBasicModel import TwoUserEncoder
import matplotlib.pyplot as plt
import numpy as np

M= 4
n_channel = 2
k=2
emb_k=4
u1_EbNodB_train=7
u2_EbNodB_train=7
train_datasize=10000
alpha=0.5
beta=0.5
bertest_data_size=50000
EbNodB_low=0
EbNodB_high=14
EbNodB_num=28
EbNodB_range = list(np.linspace(EbNodB_low, EbNodB_high, EbNodB_num))
testmodel = TwoUserEncoder(ComplexChannel=True,M=M,n_channel=n_channel,
                           k=k,emb_k=emb_k,
                           u1_EbNodB_train=u1_EbNodB_train,
                           u2_EbNodB_train=u2_EbNodB_train,
                           train_datasize=train_datasize,
                           alpha=alpha,beta=beta)
testmodel.Initialize()
testmodel.CalBLER(bertest_data_size=bertest_data_size,
                  EbNodB_low=EbNodB_low,
                  EbNodB_high=EbNodB_high,
                  EbNodB_num=EbNodB_num)
plt.plot(EbNodB_range, testmodel.ber ,label = 'TwoUserSNR(2,2),emb_k=4,')
