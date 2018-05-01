import numpy as np
import keras
from keras.layers import Input, LSTM, Dense, GaussianNoise, Lambda, Dropout,embeddings, Flatten, Add, Conv1D,Reshape,concatenate
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.callbacks import Callback
import pydot
#import graphviz
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from numpy import sqrt
from numpy import genfromtxt
from math import pow
#set the random state to generate the same/different  train data
from numpy.random import seed
seed(1)
from  tensorflow import set_random_seed
set_random_seed(3)


#parameters setting
NUM_EPOCHS = 100
BATCH_SIZE = 32
k = 2
M = 2**k
n_channel_c  = 2
n_channel_r = n_channel_c * 2
emb_k = M
R = k / n_channel_c
train_data_size=10000
bertest_data_size=50000
EbNodB_train = 7
EbNo_train = 10 ** (EbNodB_train / 10.0)
noise_std= np.sqrt( 1/ (2 * R * EbNo_train))
alpha = K.variable(0.5)
beta = K.variable(0.5)
N_sample = M * n_channel_c



class Rayleigh_SISO(object):
    """

    """
    def __init__(self,ComplexChannel=True,M = 4,n_channel = 2, k = 2,
                 emb_k=4, EbNodB_train = 7 , train_data_size = 10000,N_sample = 8):
        """

        :param ComplexChannel: True/False, whether to use complex representation in the channel
        :param M: number of symbols
        :param n_channel: use of channels
        :param k:
        :param emb_k:
        :param EbNodB_train:
        :param train_data_size:
        :param N_sample: N_sample = n_channel * M
        """
        assert ComplexChannel in (True, False)
        assert M > 1
        assert n_channel > 1
        assert emb_k > 1
        assert k > 1
        self.M = M
        self.ComplexChannel = ComplexChannel
        self.n_channel = n_channel
        self.k = k
        self.emb_k = emb_k
        self.train_data_size = train_data_size
        self.EbNodB_train = EbNodB_train
        self.R = k/n_channel_c
        EbNo_train = 10 ** (self.EbNodB_train / 10.0)
        self.noise_std = np.sqrt(1 / (2 * R * EbNo_train))
        if ComplexChannel== True:
            self.n_channel_r = self.n_channel * 2
            self.n_channel_c = self.n_channel
        if ComplexChannel == False:
            self.n_channel_r = self.n_channel
            self.n_channel_c = self.n_channel

        self.N_sample = N_sample

    def Rayleigh_Channel(self, x, n_sample):
       """

       :param x:
       :param n_sample:
       :return:
       """
       H_R = np.random.normal(0,1, n_sample)
       H_I = np.random.normal(0,1, n_sample)
       real = H_R * x[:,:,0] - H_I* x[:,:,1]
       imag = H_R * x[:,:,1]+ H_I* x[:,:,0]
       print('realshape',K.shape(real))
       noise_r = K.random_normal(K.shape(real),
                          mean=0,
                          stddev=self.noise_std)
       noise_i = K.random_normal(K.shape(imag),
                          mean=0,
                          stddev=self.noise_std)
       real = Add()([real, noise_r])
       imag = Add()([imag, noise_i])
       x = K.stack([real, imag], axis=2)
       return x

    def Rayleigh_Channel_test(self, x, n_sample, noise_std,test_datasize):
        """

        :param x:
        :param H:
        :return:
        """
        #print('x_shape',x.shape)
        #print('x[:,:,1]',K.shape(x[:,1]))
        #print('x',K.shape(x))
        #print('H',K.shape(self.H))
        #print('H[0,:]', K.shape(self.H[0,:]))
        H_R = np.random.normal(0, 1, n_sample*test_datasize)
        H_I = np.random.normal(0, 1, n_sample*test_datasize)
        H_R = np.reshape(H_R,(test_datasize,2,-1))
        H_I = np.reshape(H_I,(test_datasize,2,-1))
        np.random.shuffle(H_R)
        np.random.shuffle(H_I)
        real = H_R[:,0]*x[:,:,0] - H_I[:,1]*x[:,:,1]
        imag = H_R[:,0]*x[:,:,1] + H_I[:,1]*x[:,:,1]
        noise_r = K.random_normal(K.shape(real),
                                mean=0,
                                stddev=noise_std)
        noise_i = K.random_normal(K.shape(imag),
                                mean=0,
                                stddev=noise_std)
        real = real+ noise_r
        imag = imag+ noise_i
        #print('realshape',real.shape)
        #print('imagshape',imag.shape)
        x = K.stack([real, imag],axis=2)
        x = tf.Session().run(x)
        #print(x.shape)
        return x

    def Initialize(self):
        train_label =  np.random.randint(self.M, size= ( self.train_data_size, self.M))
        train_label_out = train_label.reshape((-1, self.M,1))
        input_signal = Input(shape=(self.M,))
        encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k,input_length=self.M)(input_signal)
        encoded1 = Conv1D(filters=self.M, kernel_size=1, activation='relu')(encoded)
        encoded2 = Conv1D(filters=self.M, kernel_size=1, activation='linear')(encoded1)
        encoded3 = LSTM(units=self.n_channel_r, input_shape=(self.M, self.M),return_sequences= True)(encoded2)
        encoded4 = BatchNormalization(momentum=0, center=False, scale=False)(encoded3)
        #encoded5 = Reshape((self.M*2, self.n_channel_c))(encoded4)
        #IQ两路串行发送
        encoded5 = Reshape((-1, 2))(encoded4)
        channel_out = Lambda(lambda x:self.Rayleigh_Channel(x,self.N_sample))(encoded5)
        decoded = Reshape((self.M,self.n_channel_r),name='pre_reshape')(channel_out)
        decoded1 = Conv1D(filters=self.M, kernel_size=1, activation='relu',name='pre_receiver')(decoded)
        decoded2 = Conv1D(filters=self.M, kernel_size=1, activation='softmax',name='receiver')(decoded1)


        self.rayleigh_channel_encoder = Model(inputs=input_signal,
                                              outputs= decoded2
                                              )
        adam = Adam(lr=0.005)
        self.rayleigh_channel_encoder.compile(optimizer=adam,
                                              loss ='sparse_categorical_crossentropy')
        print(self.rayleigh_channel_encoder.summary())
        self.encoder = Model(input_signal, encoded5)
        #encoded_input = Input(shape=(int(self.n_channel_r*self.M),))
        channel_shape = (self.n_channel_r*self.M) / 2
        encoded_input = Input(shape = (channel_shape,2,))
        deco = self.rayleigh_channel_encoder.get_layer('pre_reshape')(encoded_input)
        deco1 = self.rayleigh_channel_encoder.get_layer('pre_receiver')(deco)
        deco2 = self.rayleigh_channel_encoder.get_layer('receiver')(deco1)
        self.decoder = Model(encoded_input, deco2)
        self.rayleigh_channel_encoder.fit(train_label, train_label_out,
                                          epochs=5,
                                          batch_size=32,
                                          verbose=2)

    def Cal_Ber(self, bertest_datasize = 50000,EbNodB_low=-4, EbNodB_high=8, EbNodB_num=26):
        """

        :param bertest_datasize:
        :return:
        """
        test_label = np.random.randint(self.M, size=(bertest_datasize, self.M))
        EbNodB_range = list(np.linspace(EbNodB_low, EbNodB_high, EbNodB_num))
        ber = [None] * len(EbNodB_range)
        self.ber = ber
        for n in range(0, len(EbNodB_range)):
            EbNo  = 10 ** (EbNodB_range[n] / 10.0)
            noise_std = np.sqrt(1 / (2 * self.R * EbNo))
            nn = M * bertest_datasize
            #print(test_label.shape)
            encoded_signal = self.encoder.predict(test_label)
            #print(encoded_signal.shape)
            final_signal = self.Rayleigh_Channel_test(x=encoded_signal,n_sample=self.N_sample*2,
                                                      noise_std=noise_std,
                                                      test_datasize=bertest_datasize)
            #print(final_signal.shape)
            pred_final_signal = self.decoder.predict(final_signal)
            #??
            pred_output = np.argmax(pred_final_signal, axis=2)
            no_errors = (pred_output != test_label)
            no_errors = no_errors.astype(int).sum()
            print(no_errors)
            ber[n] = no_errors / nn
            print('SNR:', EbNodB_range[n], 'BER:', ber[n])
        return bertest_data_size

EbNodB_range = list(np.linspace(0, 20, 21))
k=2
bers = genfromtxt('data/uncodedbpskrayleigh.csv',delimiter=',')
bers = 1- bers
blers = bers
for i,ber in enumerate(bers):
    blers[i] = 1 - pow(ber,k)
plt.plot(EbNodB_range, blers,label= 'uncodedrayleigh(2,2)')

K.clear_session()
test = Rayleigh_SISO(ComplexChannel= True, M = M, n_channel=n_channel_c,k=k,emb_k=emb_k,
                     EbNodB_train=EbNodB_train, train_data_size=train_data_size, N_sample=N_sample)
test.Initialize()
test.Cal_Ber(bertest_datasize= 50000,EbNodB_low=0, EbNodB_high=20,EbNodB_num=21)
plt.plot(EbNodB_range, test.ber,'bo')
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.title('Rayleigh_Channel(2,2),PlanB,EnergyConstraint，EbdB_train:%f'%EbNodB_train)
plt.grid()

fig = plt.gcf()
fig.set_size_inches(16,12)
fig.savefig('graph/0501/B_rayleighBLER0.png',dpi=100)
plt.show()
