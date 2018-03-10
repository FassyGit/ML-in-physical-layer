# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
traindata = pd.read_csv('stock_train_data_20171111.csv')
testdata = pd.read_csv('stock_test_data_20171111.csv')
X_train=traindata[:]

del X_train['label']
del X_train['id']
del X_train['era']
del X_train['weight']
del X_train['group1']
del X_train['group2']
del X_train['code_id']
weight=traindata['weight'].values
y_train=traindata['label']
X_test=testdata[:]
del X_test['id']
del X_test['group1']
del X_test['group2']
del X_test['code_id']
X_train=X_train.values
X_test=X_test.values
Y_train=y_train.values
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train) 
X_test=scaler.transform(X_test)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print('preproc.....')
weight=weight/np.mean(weight)
batch_size = 64 #mini_batch_size
nb_epoch = 10 #大循环次数
nb_classes=2
Y_train = np_utils.to_categorical(y_train, nb_classes)
print('Y_train shape:', Y_train.shape)
print('build model')
model = Sequential()
model.add(Dense(256, input_shape=(98,))) #输入维度, 101==输出维度
model.add(BatchNormalization())
model.add(Activation('relu')) #激活函数
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu')) #激活函数
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )
print('train')
history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,shuffle=True,
                    verbose=1, validation_split=0.2,sample_weight=weight)
print('predited')
fitted_test = model.predict_proba(X_test)[:, 1]
save = pd.DataFrame({'id':testdata['id'],'proba':fitted_test})  
save.to_csv('result.csv',index=False,sep=',') 
