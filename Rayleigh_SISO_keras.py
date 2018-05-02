import numpy as np
import keras
from keras.layers import Input, LSTM, Dense, GaussianNoise, Lambda, Dropout,embeddings, Flatten, Add, Conv1D,Reshape,concatenate
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from  keras.optimizers import Adam, SGD
from keras import  backend as K
from keras.callbacks import Callback
import pydot
#import graphviz
import matplotlib.pyplot as plt
import tensorflow as tf

fd = 926
Ts = 1e-6
Ns = 50000


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
    H = np.ones((2, Ns))
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
    H[0, :] = h_I
    H[1, :] = h_Q
    return H

#h = Jakes_Flat(fd, Ts, Ns)

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

class Rayleigh_SISO(object):
    """

    """
    def __init__(self,ComplexChannel=True,M = 4,n_channel = 2, k = 2,
                 emb_k=4, EbNodB_train = 7 , train_data_size = 10000,
                 fd = 926,Ts = 1e-6,Ns = 50000):
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
        self.H = K.variable(Jakes_Flat(fd, Ts, Ns))

    def Rayleigh_Channel(self, x, H):
        """

        :param x:
        :param H:
        :return:
        """
        print('x[:,:,1]',K.shape(x[:,1]))
        print('x',K.shape(x))
        print('H',K.shape(self.H))
        print('H[0,:]', K.shape(self.H[0,:]))
        real = H[0,:]*x[:,:,0] - H[1,:]*x[:,:,1]
        imag = H[0,:]*x[:,:,1] + H[0,:]*x[:,:,1]
        noise_r = K.random_normal(K.shape(real),
                                mean=0,
                                stddev=self.noise_std)
        noise_i = K.random_normal(K.shape(imag),
                                mean=0,
                                stddev=self.noise_std)
        real = Add()([real, noise_r])
        imag = Add()([imag, noise_i])
        #x = concatenate([real, imag])
        x = K.stack([real,imag], axis=2)
        print(x.shape)
        return x
    def Rayleigh_Channel_test(self, x, H):
        """

        :param x:
        :param H:
        :return:
        """
        print('x_shape',x.shape)
        print('x[:,:,1]',K.shape(x[:,1]))
        print('x',K.shape(x))
        print('H',K.shape(self.H))
        print('H[0,:]', K.shape(self.H[0,:]))
        real = H[0,:]*x[:,:,0] - H[1,:]*x[:,:,1]
        imag = H[0,:]*x[:,:,1] + H[0,:]*x[:,:,1]
        noise_r = K.random_normal(K.shape(real),
                                mean=0,
                                stddev=self.noise_std)
        noise_i = K.random_normal(K.shape(imag),
                                mean=0,
                                stddev=self.noise_std)
        real = real+ noise_r
        imag = imag+ noise_i
        print('realshape',real.shape)
        print('imagshape',imag.shape)
        x = K.stack([real, imag],axis=2)
        x = tf.Session().run(x)
        print(x.shape)
        return x

    def R2C(self, x):
        return x.view(x.size()[0], -1, 2)

    def C2R(self, x):
        return x.view(x.size()[0], -1)

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
        channel_out = Lambda(lambda x:self.Rayleigh_Channel(x,self.H))(encoded5)
        decoded = Reshape((self.M,self.n_channel_r),name='pre_reshape')(channel_out)
        decoded1 = Conv1D(filters=self.M, kernel_size=1, activation='relu',name='pre_receiver')(decoded)
        decoded2 = Conv1D(filters=self.M, kernel_size=1, activation='softmax',name='receiver')(decoded1)


        self.rayleigh_channel_encoder = Model(inputs=input_signal,
                                              outputs= decoded2
                                              )
        adam = Adam(lr=0.01)
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
                                          epochs=1,
                                          batch_size=16,
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
            print(test_label.shape)
            encoded_signal = self.encoder.predict(test_label)
            print(encoded_signal.shape)
            H = Jakes_Flat(fd=926, Ts=1e-6,Ns=4*2)
            final_signal = self.Rayleigh_Channel_test(encoded_signal, H)
            print(final_signal.shape)
            pred_final_signal = self.decoder.predict(final_signal)
            #??
            pred_output = np.argmax(pred_final_signal, axis=2)
            no_errors = (pred_output != test_label)
            no_errors = no_errors.astype(int).sum()
            print(no_errors)
            ber[n] = no_errors / nn
            print('SNR:', EbNodB_range[n], 'BER:', ber[n])
        return bertest_data_size

test = Rayleigh_SISO(ComplexChannel= True, M = 4, n_channel=2,k=2,emb_k=4,
                     EbNodB_train=7, train_data_size=10000,fd=926, Ts=1e-6,
                     Ns=4*2)
test.Initialize()
test.Cal_Ber(bertest_datasize= 50000,EbNodB_low=-4, EbNodB_high=8,EbNodB_num=26)
