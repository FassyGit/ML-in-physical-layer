import numpy as np
import keras
from keras.layers import Input, LSTM,Dense,GaussianNoise, Lambda, Dropout, embeddings,Flatten
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.np_utils import to_categorical

class AutoEncoder(object):

    def __init__(self, CodingMeth = 'Embedding',M = 4,n_channel = 2, emb_k=4, EbNodB_train = 7 , train_data = []):
        assert CodingMeth in ('Embedding','Onehot')
        assert M > 1
        assert n > 1
        assert emb_k > 1
        self.CodingMeth = CodingMeth
        self.n_channel = n_channel
        self.emb_k = emb_k
        self.train_data = train_data
        self.EbNodB_train = EbNodB_train

    def Initialize(self):
        if self.CodingMeth == 'Embedding':
            print("This model used Embedding layer")
            # Embedding Layer
            input_signal = Input(shape=(1,))
            encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k, input_length=1)(input_signal)
            encoded1 = Flatten()(encoded)
            encoded2 = Dense(self.M, activation='relu')(encoded1)
            encoded3 = Dense(self.n_channel, activation='linear')(encoded2)
            encoded4 = Lambda(lambda x: np.sqrt(self.n_channel) * K.l2_normalize(x, axis=1))(encoded3)

            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            channel_out = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(encoded4)

            decoded = Dense(self.M, activation='relu')(channel_out)
            decoded1 = Dense(self.M, activation='softmax')(decoded)
