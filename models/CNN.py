
#coding=utf-8
'''
    FileName      ：CNN.py
    Author        ：@zch0423
    Date          ：Jun 25, 2021
    Description   ：
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPool1D, Input, Flatten, concatenate, SpatialDropout1D
from keras import initializers
from keras.callbacks import EarlyStopping
from tqdm import tqdm
import pickle

dir_path = "/Users/zch/Desktop/IM319_NLP.nosync/project/data/CNN/"
def loadData(dir_path):
    '''
    @Description
    导入CNN所需数据，包括X、y和embedding
    ------------
    @Params
    dir_path, str, 文件夹路径
    ------------
    @Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test, train_w,valid_w, test_w, embedding
    '''
    X_train = np.load(dir_path+"X_train.npy")
    X_valid = np.load(dir_path+"X_valid.npy")
    X_test = np.load(dir_path+"X_test.npy")
    train_w = np.load(dir_path+"trn_w.npy")
    valid_w = np.load(dir_path+"dev_w.npy")
    test_w = np.load(dir_path+"tst_w.npy")
    embedding = np.load(dir_path+"index2emb.npy")
    y_train = np.load(dir_path+"trn_labels.npy")
    y_valid = np.load(dir_path+"dev_labels.npy")
    y_test = np.load(dir_path+"tst_labels.npy")
    return X_train, X_valid, X_test, y_train, y_valid, y_test,train_w, valid_w, test_w, embedding
def constructCNN(size_input, size_output, embedding, dim_embedding, kernel_size, filters):
    '''
    构建CNN模型
    '''
    ipt = Input(shape=(size_input,))

    emb = Embedding(len(embedding), dim_embedding, 
                    embeddings_initializer=initializers.Constant(embedding), 
                    mask_zero=True, trainable=True)(ipt)
    cnn = Conv1D(filters=filters, kernel_size=(kernel_size, ), activation='relu')(emb)
    pool = MaxPool1D(pool_size=size_input - kernel_size + 1)(cnn)
    flat = Flatten()(pool)
    dense = Dense(40, activation='tanh')(flat)
    dense_drop = Dropout(0.2)(dense)
    output = Dense(size_output, activation='softmax')(dense_drop)  
    model = Model(inputs=ipt, outputs=output)
    # config model
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
def transformY(y_train, y_valid, y_test, yidx):
    train = keras.utils.to_categorical(y_train[yidx], 2)
    valid = keras.utils.to_categorical(y_valid[yidx], 2)
    test = keras.utils.to_categorical(y_test[yidx], 2)
    return train, valid, test
def oneFit(X_train, y_train1, X_valid, y_valid1, train_w, valid_w, emb, kernel=3, filters=30, batch=128, patience=2):
    model = constructCNN(len(X_train[0]), 2, emb, len(emb[0]), kernel_size=kernel, filters=filters)
    es = EarlyStopping(patience = patience)
    history = model.fit(X_train, y_train1, batch_size=batch,epochs=15, validation_data=(X_valid,y_valid1), callbacks=(es), verbose=0, sample_weight=train_w)
    loss, acc = model.evaluate(X_valid, y_valid1, sample_weight=valid_w)
    return acc, model

X_train, X_valid, X_test, y_train, y_valid, y_test,train_w, valid_w, test_w, embedding = loadData(dir_path)
# yidx = 1 # intent
# yidx = 0 # target
yidx = 2 # lewd
# yidx = 3 # offensive
y_train1, y_valid1, y_test1 = transformY(y_train, y_valid, y_test, yidx)
train_w = train_w[yidx]
valid_w = valid_w[yidx]
test_w = test_w[yidx]
filters = [10, 20, 30, 40, 50, 60, 70]
acc = []
for f in tqdm(filters):
    temp = []
    for i in range(5):
        tempAcc, model = oneFit(X_train, y_train1, X_valid, y_valid1, train_w, valid_w, embedding, filters=f)
        temp.append(tempAcc)
    acc.append(np.mean(temp))
    # print(f, acc[-1])

print(acc)