import numpy as np
import keras
from keras.layers import Input, LSTM,Dense,GaussianNoise, Lambda, Dropout, embeddings,Flatten
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.np_utils import to_categorical
from math import pow
from numpy import genfromtxt

#import the uncodedbpskbler
k=2
uncodedbpsk_bers = genfromtxt('data/uncodedbpsk.csv',delimiter=',')
uncodedbpsk_bers = 1- uncodedbpsk_bers
uncodedbpsk_blers = uncodedbpsk_bers
for i,uncodedbpsk_ber in enumerate(uncodedbpsk_bers):
    uncodedbpsk_blers[i] = 1 - pow(uncodedbpsk_ber,k)


#set the random state to generate the same/different  train data
from numpy.random import seed
seed(1)
from  tensorflow import set_random_seed
set_random_seed(3)

M = 4
k = np.log2(M)
k = int(k)
emb_k = 4
n_channel = 2
R = k / n_channel
print('M:', M, 'k:', k,'emb_k:',emb_k,'n:', n_channel)
#EbNodB_trains = [-10,-5,0,5,7.5,10,15]
EbNodB_trains = [15,10,7.5,5,0,-5,-10]

#generating train data
N = 10000
label =  np.random.randint(M, size = N)
label_out = label.reshape((-1,1))

#defining an auto encoder
# Embedding Layer
input_signal = Input( shape = (1, ) )
encoded = embeddings.Embedding(input_dim=M, output_dim = emb_k,input_length= 1 )(input_signal)
encoded1 = Flatten()(encoded)
encoded2 = Dense(M,activation= 'relu')(encoded1)
#encoded2 = LSTM(n_channel, dropout=0.2, recurrent_dropout=0.2)(encoded)
encoded3 = Dense(n_channel, activation= 'linear')(encoded2)
#encoded4 = Lambda(lambda x: np.sqrt(n_channel)* K.l2_normalize(x, axis=1))(encoded3)
encoded4 = Lambda(lambda x: np.sqrt(n_channel) * K.l2_normalize(x, axis=1))(encoded3)

EbNodB_train = K.variable([7])
EbNo_train = 10 ** (EbNodB_train / 10.0)
# EbNo_train = 5.01187
channel_out = GaussianNoise(K.sqrt(1 / (2 * R * EbNo_train)))(encoded4)

decoded = Dense(M, activation='relu')(channel_out)
decoded1 = Dense(M, activation='softmax')(decoded)
#decoded1 = Dense(M, activation= 'sigmoid')(decoded)
#?? why softmax?

auto_encoder_embedding = Model(input_signal, decoded1)
adam= Adam(lr= 0.01)
auto_encoder_embedding.compile(optimizer= adam,
                               loss= 'sparse_categorical_crossentropy',
                               )
print(auto_encoder_embedding.summary())

for val in EbNodB_trains:
    K.set_value(EbNodB_train,[val])
    print('EbNodB_train',K.get_value(EbNodB_train))
    print('EbNo_train', K.get_value(EbNo_train))
    auto_encoder_embedding.fit(label, label_out,
                           epochs=10,
                           batch_size=32,
                            verbose= 2)

#K.set_value(EbNodB_train,[-10])
#auto_encoder_embedding.fit(label, label_out,epochs=23, batch_size=32)

encoder = Model(input_signal, encoded4)
encoded_input = Input(shape=(n_channel,))

deco = auto_encoder_embedding.layers[-2](encoded_input)
deco = auto_encoder_embedding.layers[-1](deco)
decoder  = Model(encoded_input, deco)


#generating test data

N = 50000
test_label = np.random.randint(M, size=N)
test_label_out = test_label.reshape((-1,1))

#plotting constellation diagram for embedding
scatter_plot = []
for i in range (0,M):
    scatter_plot.append(encoder.predict(np.expand_dims(i, axis=0)))
scatter_plot = np.array(scatter_plot)
print(scatter_plot.shape)

import matplotlib.pyplot as plt
scatter_plot = scatter_plot.reshape(M, 2, 1)
plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1])
plt.legend(['embedding_constellation(2,2),emb_k=2'],loc='upper left')
plt.axis((-2.5, 2.5, -2.5, 2.5))
plt.grid()
fig = plt.gcf()
fig.set_size_inches(16,12)
fig.savefig('graph/MulSNRcons(2,2)0318_0.png',dpi=100)
plt.show()


# use this function for ploting constellation for higher dimenson like 7-D for (7,4) autoencoder
'''
x_emb = encoder.predict(test_data)
noise_std = np.sqrt(1/(2*R*EbNo_train))
noise = noise_std * np.random.randn(N,n_channel)
x_emb = x_emb + noise
from sklearn.manifold import TSNE
X_embedded = TSNE(learning_rate=700, n_components=2,n_iter=35000, random_state=0, perplexity=60).fit_transform(x_emb)
print (X_embedded.shape)
X_embedded = X_embedded / 7
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:,0],X_embedded[:,1])
#plt.axis((-2.5,2.5,-2.5,2.5)) 
plt.grid()
plt.show()
'''

#ccalculating BER for embedding
EbNodB_range = list(np.linspace(-4, 8.5 ,26))
ber = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)
    noise_std = np.sqrt(1 / (2 * R * EbNo))
    noise_mean  = 0
    no_errors = 0
    nn = N
    noise = noise_std * np.random.randn(nn, n_channel)
    encoded_signal = encoder.predict(test_label)
    final_signal = encoded_signal + noise
    pred_final_signal = decoder.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    ber[n] = no_errors/nn
    print('SNR:', EbNodB_range[n], 'BER:', ber[n])

plt.plot(EbNodB_range, ber )
plt.plot(EbNodB_range, uncodedbpsk_blers,'bo' )
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(['Autoencoeder_embedding(2,2),emb_k=2','uncodedbpsk'],loc = 'upper left')
plt.legend(loc='upper right',ncol= 1)

fig = plt.gcf()
fig.set_size_inches(16,12)
fig.savefig('graph/MulSNR(2,2)0318_0.png',dpi=100)
plt.show()