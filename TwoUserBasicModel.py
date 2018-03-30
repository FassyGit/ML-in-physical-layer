# -*- coding: utf-8 -*-
"""
created on Fri, 30 Mar
two user with complex number, Functional API
"""
import numpy as np
import keras
from keras.layers import Input, LSTM, Dense, GaussianNoise, Lambda, Dropout,embeddings, Flatten, Add
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from  keras.optimizers import Adam, SGD
from keras import  backend as K
from keras.callbacks import Callback
import pydot
import graphviz
import matplotlib.pyplot as plt
from numpy.random import seed

#define the dynamic loss weights
class Mycallback(Callback):
    def __init__(self,a, b):
        self.a = a
        self.b = b
        self.epoch_num = 0
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_num = self.epoch_num + 1
        loss1 = logs.get('u1_receiver_loss')
        loss2 = logs.get('u2_receiver_loss')
        print("epoch %d" %self.epoch_num)
        print("total_loss%f" %logs.get('loss'))
        print("u1_loss %f"%(loss1))
        print("u2_loss %f" % (loss2))
        u1_ls = loss1 / (loss1 + loss2)
        u2_ls = 1 - u1_ls
        K.set_value(self.a, u1_ls)
        K.set_value(self.b, u2_ls)
        #print("alpha %f" %K.get_value(alpha))
        #print("beta %f" % K.get_value(beta))
        print("selfalpha %f" % K.get_value(self.a))
        print("selfbeta %f" % K.get_value(self.b))

class TwoUserEncoder(object):
    """
    This is an API
    """
    def __init__(self, ComplexChannel = True, M=4, n_channel=2,k=2,emb_k=4,u1_EbNodB_train=7,u2_EbNodB_train = 7,train_datasize=10000):
        seed(1)
        from tensorflow import  set_random_seed
        set_random_seed(3)

        assert  ComplexChannel in (True, False)
        assert M > 1
        assert n_channel >1
        assert emb_k >1
        assert k > 1
        self.M = M
        self.ComplexChannel = ComplexChannel
        self.n_channel = n_channel
        self.k = k
        self.train_datasize = train_datasize
        self.u1_EbNodB_train  =u1_EbNodB_train
        self.u2_EbNodB_train = u2_EbNodB_train
        self.R = self.k / float(self.n_channel)
        self.n_channel_r = self.n_channel * 2
        self.u1_noise_std = np.sqrt(1 / (2 * self.R * self.u1_EbNo_train))
        self.u2_noise_std = np.sqrt(1 / (2 * self.R * self.u2_EbNo_train))

    # define the function for mixed AWGN channel
    def mixed_AWGN(self,x,User='u1'):
        assert User in ('u1','u2')
        signal = x[0]
        interference = x[1]
        if User == 'u1':
            noise = K.random_normal(K.shape(signal),
                                mean=0,
                                stddev=self.u1_noise_std)
        if User == 'u2':
            noise = K.random_normal(K.shape(signal),
                                    mean=0,
                                    stddev=self.u2_noise_std)
        signal = Add()([signal, interference])
        signal = Add()([signal, noise])
        return signal
    def Initialize(self):
        """

        :return:
        """

