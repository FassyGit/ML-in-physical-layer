from AutoEncoder_BasicModel import AutoEncoder
import  numpy as np
import matplotlib.pyplot as plt

M = 16
n_channel=7
k = 4
emb_k=16
EbNodB_train = 7
train_data_size=10000
model_test3 = AutoEncoder(CodingMeth='Onehot',M = M, n_channel=n_channel, k = k, emb_k=emb_k, EbNodB_train = EbNodB_train,train_data_size=train_data_size)
model_test3.Initialize()
print("Initialization Finished")
#model_test3.Draw_Constellation()
model_test3.Cal_BLER(bertest_data_size= 50000)
EbNodB_range = list(np.linspace(-4, 8.5, 26))
plt.figure(figsize=(16,12),dpi=100)
plt.plot(EbNodB_range, model_test3.ber,'bo')
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend()
#plt.savefig('AutoEncoder,test,Embedding,(%d,%d)emb_k:%d.png'%(n_channel,k, emb_k))
plt.savefig('graph/test2.png')
plt.show()