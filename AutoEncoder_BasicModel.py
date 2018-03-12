import numpy as np
import keras
from keras.layers import Input, LSTM,Dense,GaussianNoise, Lambda, Dropout, embeddings,Flatten
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.np_utils import to_categorical
# for reproducing reslut
from numpy.random import seed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class AutoEncoder(object):
    """
    This is an API for the use of NN of an end to end communication system,
    AutoEncoder.Initialize():
        Model Building and Training
    Draw_Constellation()
        Constellation Graph of the transmitted signal

    """
    def __init__(self, CodingMeth = 'Embedding',M = 4,n_channel = 2, k = 2, emb_k=4, EbNodB_train = 7 , train_data_size = 10000):
        """

        :param CodingMeth: 'Embedding' or ' Onehot'
        :param M: The total number of symbol
        :param n_channel: bits of channel
        :param k: int(log(M))
        :param emb_k: output dimension of the first embedding layer if using the CodingMeth 'Embedding'
        :param EbNodB_train: SNR(dB) of the AWGN channel in train process
        :param train_data_size: size of the train data
        """
        assert CodingMeth in ('Embedding','Onehot')
        assert M > 1
        assert n_channel > 1
        assert emb_k > 1
        assert k >1
        self.M = M
        self.CodingMeth = CodingMeth
        self.n_channel = n_channel
        self.emb_k = emb_k
        self.k = k
        self.train_data_size = train_data_size
        self.EbNodB_train = EbNodB_train
        self.R = self.n_channel / float(self.k)

    def Initialize(self):
        """

        :return:
        """
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(3)

        if self.CodingMeth == 'Embedding':
            print("This model used Embedding layer")
            #Generating train_data
            train_data = np.random.randint(self.M, size=self.train_data_size)
            train_data_pre = train_data.reshape((-1,1))
            # Embedding Layer
            input_signal = Input(shape=(1,))
            encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k, input_length=1)(input_signal)
            encoded1 = Flatten()(encoded)
            encoded2 = Dense(self.M, activation='relu')(encoded1)
            encoded3 = Dense(self.n_channel, activation='linear')(encoded2)
            encoded4 = Lambda(lambda x: np.sqrt(self.n_channel) * K.l2_normalize(x, axis=1))(encoded3)

            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            channel_out = GaussianNoise(np.sqrt(1 / (2 * self.R * EbNo_train)))(encoded4)

            decoded = Dense(self.M, activation='relu')(channel_out)
            decoded1 = Dense(self.M, activation='softmax')(decoded)

            self.auto_encoder = Model(input_signal, decoded1)
            adam = Adam(lr=0.01)
            self.auto_encoder.compile(optimizer=adam,
                                           loss='sparse_categorical_crossentropy',
                                           )
            print(self.auto_encoder.summary())
            self.auto_encoder.fit(train_data, train_data_pre,
                                       epochs=45,
                                       batch_size=32,
                                        verbose=0)
            self.encoder = Model(input_signal, encoded4)
            encoded_input = Input(shape=(self.n_channel,))

            deco = self.auto_encoder.layers[-2](encoded_input)
            deco = self.auto_encoder.layers[-1](deco)
            self.decoder = Model(encoded_input, deco)

        if self.CodingMeth == 'Onehot':
            print("This is the model using Onehot")

            # Generating train_data
            train_data = np.random.randint(self.M, size=self.train_data_size)
            data = []
            for i in train_data:
                temp = np.zeros(self.M)
                temp[i] = 1
                data.append(temp)
            train_data = np.array(data)

            input_signal = Input(shape=(self.M,))
            encoded = Dense(self.M, activation='relu')(input_signal)
            encoded1 = Dense(self.n_channel, activation='linear')(encoded)
            encoded2 = Lambda(lambda x: np.sqrt(self.n_channel) * K.l2_normalize(x, axis=1))(encoded1)
            """
            K.l2_mormalize 二阶约束（功率约束）
            """
            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            encoded3 = GaussianNoise(np.sqrt(1 / (2 * self.R * EbNo_train)))(encoded2)

            decoded = Dense(self.M, activation='relu')(encoded3)
            decoded1 = Dense(self.M, activation='softmax')(decoded)
            self.autoencoder = Model(input_signal, decoded1)
            adam = Adam(lr=0.01)
            self.autoencoder.compile(optimizer=adam, loss='categorical_crossentropy')

            print(self.autoencoder.summary())

            # for tensor board visualization
            # tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
            # traning auto encoder

            self.autoencoder.fit(train_data, train_data,
                            epochs=45,
                            batch_size=32,
                            verbose = 0)

            # saving keras model
            from keras.models import load_model

            # if you want to save model then remove below comment
            # autoencoder.save('autoencoder_v_best.model')

            # making encoder from full autoencoder
            self.encoder = Model(input_signal, encoded2)

            # making decoder from full autoencoder
            encoded_input = Input(shape=(self.n_channel,))

            deco = self.autoencoder.layers[-2](encoded_input)
            deco = self.autoencoder.layers[-1](deco)
            self.decoder = Model(encoded_input, deco)

    def Draw_Constellation(self, test_data_size = 1500):
        """

        :param test_data_size:
        :return:
        """
        import matplotlib.pyplot as plt
        test_label = np.random.randint(self.M, size=test_data_size)
        test_data = []
        for i in test_label:
            temp = np.zeros(self.M)
            temp[i] = 1
            test_data.append(temp)
        test_data = np.array(test_data)

        if self.n_channel == 2:
            scatter_plot = []
            if self.CodingMeth == 'Embedding':
                print("Embedding,Two Dimension")
                for i in range(0, self.M):
                    scatter_plot.append(self.encoder.predict(np.expand_dims(i, axis=0)))
                scatter_plot = np.array(scatter_plot)
            if self.CodingMeth == 'Onehot':
                print("Onehot,Two Dimension")
                for i in range(0, self.M):
                    temp = np.zeros(self.M)
                    temp[i] = 1
                    scatter_plot.append(self.encoder.predict(np.expand_dims(temp, axis=0)))
                scatter_plot = np.array(scatter_plot)
            scatter_plot = scatter_plot.reshape(self.M, 2, 1)
            plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1],label= '%s,(%d, %d), %d'%(self.CodingMeth,self.n_channel, self.k, self.emb_k) )
            plt.legend()
            plt.axis((-2.5, 2.5, -2.5, 2.5))
            plt.grid()
            plt.show()
        if self.n_channel > 2 :
            if self.CodingMeth == 'Embedding':
                x_emb = self.encoder.predict(test_label)
                print("Embedding,High Dimension")
            if self.CodingMeth == 'Onehot':
                x_emb = self.encoder.predict(test_data)
                print("Onehot,High Dimension")

            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            noise_std = np.sqrt(1 / (2 * self.R * EbNo_train))
            noise = noise_std * np.random.randn(test_data_size, self.n_channel)
            x_emb = x_emb + noise
            X_embedded = TSNE(learning_rate=700, n_components=2, n_iter=35000, random_state=0,
                              perplexity=60).fit_transform(x_emb)
            print(X_embedded.shape)
            X_embedded = X_embedded / 7
            import matplotlib.pyplot as plt
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1],label= '%s,(%d, %d), %d'%(self.CodingMeth,self.n_channel, self.k, self.emb_k))
            # plt.axis((-2.5,2.5,-2.5,2.5))
            plt.legend()
            plt.grid()
            plt.show()

    def Cal_BLER(self, bertest_data_size = 50000, EbNodB_low = -4, EbNodB_high = 8.5, EbNodB_num =26):
        test_label = np.random.randint(self.M, size=bertest_data_size)
        test_data = []
        for i in test_label:
            temp = np.zeros(self.M)
            temp[i] = 1
            test_data.append(temp)
        test_data = np.array(test_data)

        EbNodB_range = list(np.linspace(EbNodB_low, EbNodB_high, EbNodB_num))
        ber = [None] * len(EbNodB_range)
        self.ber = ber
        for n in range(0, len(EbNodB_range)):
            EbNo = 10 ** (EbNodB_range[n] / 10.0)
            noise_std = np.sqrt(1 / (2 * self.R * EbNo))
            noise_mean = 0
            no_errors = 0
            nn = bertest_data_size
            noise = noise_std * np.random.randn(nn, self.n_channel)
            if self.CodingMeth == 'Embedding':
                encoded_signal = self.encoder.predict(test_label)
            if self.CodingMeth == 'Onehot':
                encoded_signal = self.encoder.predict(test_data)
            final_signal = encoded_signal + noise
            pred_final_signal = self.decoder.predict(final_signal)
            pred_output = np.argmax(pred_final_signal, axis=1)
            no_errors = (pred_output != test_label)
            no_errors = no_errors.astype(int).sum()
            ber[n] = no_errors / nn
            print('SNR:', EbNodB_range[n], 'BER:', ber[n])
        self.ber = ber

"""
model_test = AutoEncoder(CodingMeth='Onehot',M = 16, n_channel=7, k = 4, emb_k=16,EbNodB_train = 7,train_data_size=10000)
model_test.Initialize()
print("Initialization Finished")
model_test.Draw_Constellation(1500)


model_test2 = AutoEncoder(CodingMeth='Embedding',M = 16, n_channel=7, k = 4, emb_k=16,EbNodB_train = 7,train_data_size=10000)
model_test2.Initialize()
print("Initialization Finished")
model_test2.Draw_Constellation(1000)
model_test2.Cal_BLER(test_data_size= 50000)

model_test3 = AutoEncoder(CodingMeth='Embedding',M = 16, n_channel=7, k = 4, emb_k=16,EbNodB_train = 7,train_data_size=10000)
model_test3.Initialize()
print("Initialization Finished")
#model_test3.Draw_Constellation()
model_test3.Cal_BLER(bertest_data_size= 70000)
EbNodB_range = list(np.linspace(-4, 8.5, 26))
plt.plot(EbNodB_range, model_test3.ber,'bo')
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.show()
"""
"""
model_test3 = AutoEncoder(CodingMeth='Embedding',M = 16, n_channel=7, k = 4, emb_k=16, EbNodB_train = 7,train_data_size=10000)
model_test3.Initialize()
print("Initialization Finished")
#model_test3.Draw_Constellation()
model_test3.Cal_BLER(bertest_data_size= 70000)
EbNodB_range = list(np.linspace(-4, 8.5, 26))
plt.plot(EbNodB_range, model_test3.ber,'bo')
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.show()
"""