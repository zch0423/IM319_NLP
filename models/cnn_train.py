import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPool1D, Input, Flatten, concatenate, SpatialDropout1D
from keras import initializers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import pickle
import numpy as np
import pandas as pd


def transform_feature(X_train, X_valid, X_test, MAX_LENGTH, pre_trained_w2v, dim_embedding):
    print('transforming data feature...')
    tokenizer = Tokenizer(filters='', lower=True)  # 考虑标点符号，忽略大小写
    tokenizer.fit_on_texts(X_train)

    # word2index = tokenizer.word_index
    index2word = tokenizer.index_word  # 从1开始，0预留给了 <pad>

    X_train = tokenizer.texts_to_sequences(X_train)
    X_valid = tokenizer.texts_to_sequences(X_valid)
    X_test = tokenizer.texts_to_sequences(X_test)

    # embedding
    index2emb = [np.zeros(dim_embedding)]  # index 0 means <pad>

    for i in range(1, len(index2word) + 1):
        if index2word[i] in pre_trained_w2v:
            index2emb.append(pre_trained_w2v[index2word[i]])
        else:
            index2emb.append(np.random.uniform(-0.25, 0.25, dim_embedding))

    index2emb = np.array(index2emb)

    max_length = min(max(len(x) for x in X_train), MAX_LENGTH)

    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_valid = pad_sequences(X_valid, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

    return X_train, X_valid, X_test, index2emb


def construct_CNN(size_input, size_output, word_emb, dim_embedding, kernel_size):
    print('constructing model...')

    input_ = Input(shape=(size_input,))

    emb = Embedding(len(word_emb), dim_embedding,
                    embeddings_initializer=initializers.Constant(word_emb),
                    mask_zero=True, trainable=False)(input_)

    cnn_ = Conv1D(filters=40, kernel_size=(kernel_size,), activation='relu')(emb)
    pool = MaxPool1D(pool_size=size_input - kernel_size + 1)(cnn_)
    flat = Flatten()(pool)

    dense = Dense(40, activation='tanh')(flat)
    dense_drop = Dropout(0.2)(dense)

    output = Dense(size_output, activation='softmax')(dense_drop)

    model = Model(inputs=input_, outputs=output)

    # config model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # text
    with open('datan/SBIC.v2.trn.csv.txt', 'r', encoding='utf-8', errors='ignore') as f:
        YN_trn = [line.strip() for line in f.readlines()]
    with open('datan/SBIC.v2.dev.csv.txt', 'r', encoding='utf-8', errors='ignore') as f:
        YN_dev = [line.strip() for line in f.readlines()]
    with open('datan/SBIC.v2.tst.csv.txt', 'r', encoding='utf-8', errors='ignore') as f:
        YN_tst = [line.strip() for line in f.readlines()]

    # label
    dev_label = np.load("npy_files/dev.npy")
    trn_label = np.load("npy_files/trn.npy")
    tst_label = np.load("npy_files/tst.npy")


    def foo(z: float):
        if np.isnan(z):
            return 0
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


    # CNN
    pre_trained_word2vec = pickle.load(open("model/glove_twitter_200.bin", "rb"))
    dim_embedding = len(pre_trained_word2vec['the'])
    # kernel_size = 3
    # max_length = 50
    length_list = [40, 50, 60, 70]
    kernel_size_list = [2, 3, 4, 5, 6, 7]
    acc_list = {'dev': [], 'tst': []}
    # reload
    column_index = 0

    arg_trn = pd.DataFrame(trn_label[column_index])
    arg_dev = pd.DataFrame(dev_label[column_index])
    arg_tst = pd.DataFrame(tst_label[column_index])

    label_dev = arg_dev[0].apply(lambda x: foo(x))
    weight_dev = arg_dev[0].apply(lambda x: foo2(x))

    label_trn = arg_trn[0].apply(lambda x: foo(x))
    weight_trn = arg_trn[0].apply(lambda x: foo2(x))

    label_tst = arg_tst[0].apply(lambda x: foo(x))
    weight_tst = arg_tst[0].apply(lambda x: foo2(x))

    for max_length in length_list:

        X_train_, X_valid_, X_test_, word_embedding = transform_feature(YN_trn, YN_dev, YN_tst, max_length,
                                                                        pre_trained_word2vec, dim_embedding)
        for kernel_size in kernel_size_list:
            part_1 = []
            part_2 = []
            for times in range(5):

                cnn = construct_CNN(len(X_train_[0]), 2, word_embedding, dim_embedding, kernel_size)
                cnn.summary()
                es = EarlyStopping(patience=2)
                cnn.fit(X_train_, label_trn, sample_weight=weight_trn, epochs=6, batch_size=32, verbose=1,
                        validation_data=(X_valid_, label_dev), callbacks=es)
                part_1.append(cnn.evaluate(X_valid_, label_dev, sample_weight=weight_dev)[1])
                part_2.append(cnn.evaluate(X_test_, label_tst, sample_weight=weight_tst)[1])
            acc_list['dev'].append(np.average(part_1))
            acc_list['tst'].append(np.average(part_2))
            print(len(acc_list['dev']), '\n\n\n\n')

