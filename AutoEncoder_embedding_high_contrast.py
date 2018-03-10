#This code is written for testing the BER of onehot encoding and embedding layer when (n,k) is higher than 2
import numpy as np
import keras
from keras.layers import Input, LSTM,Dense,GaussianNoise, Lambda, Dropout, embeddings,Flatten
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.np_utils import to_categorical

#set the random state to generate the same/different  train data
from numpy.random import seed
seed(1)
from  tensorflow import set_random_seed
set_random_seed(3)

M = 16
k_r = np.log2(M)
k_r = int(k_r)
k = 16
n_channel = 7
R = k_r / n_channel
print('M:', M, 'k:', k, 'n:', n_channel)

#generating train data
N = 10000
label =  np.random.randint(M, size = N)
label_out = label.reshape((-1,1))
#defining an auto encoder

# creating one hot encoded vectors
data = []
for i in label:
    temp = np.zeros(M)
    temp[i] = 1
    data.append(temp)

# checking data shape
data = np.array(data)
print(data.shape)

# checking generated data with it's label
temp_check = [17, 23, 45, 67, 89, 96, 72, 250, 350]
for i in temp_check:
    print(label[i], data[i])

# defining autoencoder and it's layer (onehot)
input_signal = Input(shape=(M,))
encoded_n = Dense(M, activation='relu')(input_signal)
encoded1_n = Dense(n_channel, activation='linear')(encoded_n)
encoded2_n = Lambda(lambda x: np.sqrt(n_channel) * K.l2_normalize(x, axis=1))(encoded1_n)
"""
K.l2_mormalize 二阶约束（功率约束）
"""
EbNo_train = 5.01187  # coverted 7 db of EbNo
encoded3_n = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(encoded2_n)

decoded_n = Dense(M, activation='relu')(encoded3_n)
decoded1_n = Dense(M, activation='softmax')(decoded_n)
autoencoder_n = Model(input_signal, decoded1_n)
adam = Adam(lr=0.01)
autoencoder_n.compile(optimizer=adam, loss='categorical_crossentropy')

# printing summary of layers and it's trainable parameters
print(autoencoder_n.summary())

# for tensor board visualization
# tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# traning auto encoder

autoencoder_n.fit(data, data,
                epochs=45,
                batch_size=32)


encoder_n = Model(input_signal, encoded2_n)
encoded_input_n = Input(shape=(n_channel,))

deco_n = autoencoder_n.layers[-2](encoded_input_n)
deco_n = autoencoder_n.layers[-1](deco_n)
decoder_n = Model(encoded_input_n, deco_n)

# Embedding Layer
input_signal = Input( shape = (1, ) )
encoded = embeddings.Embedding(input_dim=M, output_dim = k,input_length= 1 )(input_signal)
encoded1 = Flatten()(encoded)
encoded2 = Dense(M,activation= 'relu')(encoded1)
#encoded2 = LSTM(n_channel, dropout=0.2, recurrent_dropout=0.2)(encoded)
encoded3 = Dense(n_channel, activation= 'linear')(encoded2)
#encoded4 = Lambda(lambda x: np.sqrt(n_channel)* K.l2_normalize(x, axis=1))(encoded3)
encoded4 = Lambda(lambda x: np.sqrt(n_channel) * K.l2_normalize(x, axis=1))(encoded3)

EbNodB_train = 7
EbNo_train = 10 ** (EbNodB_train / 10.0)
# EbNo_train = 5.01187
channel_out = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(encoded4)

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
auto_encoder_embedding.fit(label, label_out,
                           epochs=45,
                           batch_size=32)
encoder = Model(input_signal, encoded4)
encoded_input = Input(shape=(n_channel,))

deco = auto_encoder_embedding.layers[-2](encoded_input)
deco = auto_encoder_embedding.layers[-1](deco)
decoder  = Model(encoded_input, deco)


#generating test data

N = 1500
test_label = np.random.randint(M, size=N)
test_label_out = test_label.reshape((-1,1))
test_data = []
for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)

test_data = np.array(test_data)
import matplotlib.pyplot as plt
"""
#plotting constellation diagram for embedding
scatter_plot = []
for i in range (0,M):
    scatter_plot.append(encoder.predict(np.expand_dims(i, axis=0)))
scatter_plot = np.array(scatter_plot)
print(scatter_plot.shape)


scatter_plot = scatter_plot.reshape(M, 2, 1)
plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1])
plt.legend(['embedding_constellation'],loc='upper left')
plt.axis((-2.5, 2.5, -2.5, 2.5))
plt.grid()
plt.show()

#plotting constellation diagram for one-hot
scatter_plot = []
for i in range(0, M):
    temp = np.zeros(M)
    temp[i] = 1
    scatter_plot.append(encoder_n.predict(np.expand_dims(temp, axis=0)))
scatter_plot = np.array(scatter_plot)
print(scatter_plot.shape)
scatter_plot = scatter_plot.reshape(M, 2, 1)


plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1], )
plt.legend(['onehot_constellation'],loc='upper left')
plt.axis((-2.5, 2.5, -2.5, 2.5))
plt.grid()
plt.show()
"""
# use this function for ploting constellation for higher dimenson like 7-D for (7,4) autoencoder
# generating data for checking BER
# if you're not using t-sne for visulation than set N to 70,000 for better result
# for t-sne use less N like N = 1500

x_emb = encoder.predict(test_label)
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

#ccalculating BER for onehot
ber_n = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
    noise_std = np.sqrt(1 / (2 * R * EbNo))
    noise_mean = 0
    no_errors = 0
    nn = N
    noise = noise_std * np.random.randn(nn, n_channel)
    encoded_signal = encoder_n.predict(test_data)
    final_signal = encoded_signal + noise
    pred_final_signal = decoder_n.predict(final_signal)
    pred_output = np.argmax(pred_final_signal, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    ber_n[n] = no_errors / nn
    print('SNR:', EbNodB_range[n], 'BER_N:', ber_n[n])

plt.plot(EbNodB_range, ber )
plt.plot(EbNodB_range, ber_n )
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(['Autoencoeder_embedding(7,4),emb_k=32 ','Autoencoeder_onehot(7,4)'],loc = 'lower left')
plt.legend(loc='upper right',ncol= 1)

plt.show()
