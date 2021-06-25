import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
import random
import pickle
import cnn_train

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# cls_dev = np.load("npy_files/CLS_dev_last.npy")
# cls_trn = np.load("npy_files/CLS_trn_last.npy")
# cls_tst = np.load("npy_files/CLS_tst_last.npy")

avg_dev = np.load("npy_files/avg_dev_last.npy")
avg_trn = np.load("npy_files/avg_trn_last.npy")
avg_tst = np.load("npy_files/avg_tst_last.npy")

dev_label = np.load("npy_files/dev.npy")
trn_label = np.load("npy_files/trn.npy")
tst_label = np.load("npy_files/tst.npy")

# cls_dev = cls_dev.reshape(cls_dev.shape[0], cls_dev.shape[-1])
# cls_trn = cls_trn.reshape(cls_trn.shape[0], cls_trn.shape[-1])
# cls_tst = cls_tst.reshape(cls_tst.shape[0], cls_tst.shape[-1])

avg_dev = avg_dev.reshape(avg_dev.shape[0], avg_dev.shape[-1])
avg_trn = avg_trn.reshape(avg_trn.shape[0], avg_trn.shape[-1])
avg_tst = avg_tst.reshape(avg_tst.shape[0], avg_tst.shape[-1])

# YN_trn = pd.DataFrame(cls_trn)
# YN_dev = pd.DataFrame(cls_dev)
# YN_tst = pd.DataFrame(cls_tst)

YN_trn = pd.DataFrame(avg_trn)
YN_dev = pd.DataFrame(avg_dev)
YN_tst = pd.DataFrame(avg_tst)

column_index = 0
arg_trn = pd.DataFrame(trn_label[column_index])
arg_dev = pd.DataFrame(dev_label[column_index])
arg_tst = pd.DataFrame(tst_label[column_index])


# arg_trn.fillna(0)
# arg_dev.fillna(0)
# arg_tst.fillna(0)


def foo(z: float):
    if np.isnan(z):
        return 1
    t = z
    if z > 0:
        t = 1
    return t


def foo2(z: float):
    if np.isnan(z):
        return 0
    t = z
    if z == 0:
        t = 1
    return t


label_dev = arg_dev[0].apply(lambda x: foo(x))
weight_dev = arg_dev[0].apply(lambda x: foo2(x))

label_trn = arg_trn[0].apply(lambda x: foo(x))
weight_trn = arg_trn[0].apply(lambda x: foo2(x))

label_tst = arg_tst[0].apply(lambda x: foo(x))
weight_tst = arg_tst[0].apply(lambda x: foo2(x))

# lr
# model = LogisticRegression(max_iter=10000, verbose=True)

# svm
# c = 20
# model = SVC(C=c, verbose=True)

# model = Sequential()
# model.add(Input(shape=768))
# model.add(Dense(units=600, activation='sigmoid'))
# model.add(Dense(units=300, activation='tanh'))
# model.add(Dense(units=300, activation='tanh'))
# model.add(Dense(units=20, activation='sigmoid'))
# model.add(Dense(units=120, activation='tanh'))
# model.add(Dense(units=120, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Dense(units=2, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(YN_trn, label_trn, sample_weight=weight_trn, epochs=6, batch_size=32, verbose=1,
#           validation_data=(YN_dev, label_dev))

# loss, acc = model.evaluate(YN_dev, label_dev, sample_weight=weight_dev)
# print('Test Accuracy on the dev set: %.3f' % acc)

# model.fit(YN_trn, label_trn, sample_weight=weight_trn)

# probs = model.predict_proba(sex_YN_dev)
#
# print("training set accuracy  ", model.score(YN_trn, label_trn, sample_weight=weight_trn))
# #
# print("validation set accuracy  ", model.score(YN_dev, label_dev, sample_weight=weight_dev))
# #
# print("test set accuracy  ", model.score(YN_tst, label_tst, sample_weight=weight_tst))
# loss, acc = model.evaluate(YN_tst, label_tst, sample_weight=weight_tst)




