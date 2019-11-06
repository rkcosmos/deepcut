#!/usr/bin/env python
# encoding: utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, \
    Concatenate, Flatten, SpatialDropout1D, \
    BatchNormalization, Conv1D, Maximum, ZeroPadding1D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam

def conv_unit(inp, n_gram, no_word=200, window=2):
    out = Conv1D(no_word, window, strides=1, padding="valid", activation='relu')(inp)
    out = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(out)
    out = ZeroPadding1D(padding=(0, window - 1))(out)
    return out

def get_convo_nn2(no_word=200, n_gram=21, no_char=178):
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))

    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.15)(a)
    a = BatchNormalization()(a)

    a_concat = []
    for i in range(1,9):
        a_concat.append(conv_unit(a, n_gram, no_word, window=i))
    for i in range(9,12):
        a_concat.append(conv_unit(a, n_gram, no_word - 50, window=i))
    a_concat.append(conv_unit(a, n_gram, no_word - 100, window=12))
    a_sum = Maximum()(a_concat)

    b = Embedding(12, 12, input_length=n_gram)(input2)
    b = SpatialDropout1D(0.15)(b)

    x = Concatenate(axis=-1)([a, a_sum, b])
    #x = Concatenate(axis=-1)([a_sum, b])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy', metrics=['acc'])
    return model
