from keras.models import Model
from keras.layers import Input, Dense, Embedding, \
    Concatenate, Flatten, SpatialDropout1D, \
    BatchNormalization, Conv1D, Maximum, ZeroPadding1D
from keras.layers import TimeDistributed
from keras.optimizers import Adam


def get_convo_nn2(no_word=200, n_gram=21, no_char=178):
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))

    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.2)(a)

    a2 = Conv1D(no_word, 2, strides=1, padding="valid", activation='relu')(a)
    a2 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a2)
    a2 = ZeroPadding1D(padding=(0, 1))(a2)

    a3 = Conv1D(no_word, 3, strides=1, padding="valid", activation='relu')(a)
    a3 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a3)
    a3 = ZeroPadding1D(padding=(0, 2))(a3)

    a4 = Conv1D(no_word, 4, strides=1, padding="valid", activation='relu')(a)
    a4 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a4)
    a4 = ZeroPadding1D(padding=(0, 3))(a4)

    a5 = Conv1D(no_word, 5, strides=1, padding="valid", activation='relu')(a)
    a5 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a5)
    a5 = ZeroPadding1D(padding=(0, 4))(a5)

    a6 = Conv1D(no_word, 6, strides=1, padding="valid", activation='relu')(a)
    a6 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a6)
    a6 = ZeroPadding1D(padding=(0, 5))(a6)

    a7 = Conv1D(no_word, 7, strides=1, padding="valid", activation='relu')(a)
    a7 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a7)
    a7 = ZeroPadding1D(padding=(0, 6))(a7)

    a8 = Conv1D(no_word, 8, strides=1, padding="valid", activation='relu')(a)
    a8 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a8)
    a8 = ZeroPadding1D(padding=(0, 7))(a8)

    a9 = Conv1D(no_word - 50, 9, strides=1, padding="valid", activation='relu')(a)
    a9 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a9)
    a9 = ZeroPadding1D(padding=(0, 8))(a9)

    a10 = Conv1D(no_word - 50, 10, strides=1, padding="valid", activation='relu')(a)
    a10 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a10)
    a10 = ZeroPadding1D(padding=(0, 9))(a10)

    a11 = Conv1D(no_word - 50, 11, strides=1, padding="valid", activation='relu')(a)
    a11 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a11)
    a11 = ZeroPadding1D(padding=(0, 10))(a11)

    a12 = Conv1D(no_word - 100, 12, strides=1, padding="valid", activation='relu')(a)
    a12 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a12)
    a12 = ZeroPadding1D(padding=(0, 11))(a12)

    a_concat = [a2, a3, a4, a5,
                a6, a7, a8, a9,
                a10, a11, a12]
    a_sum = Maximum()(a_concat)

    b = Embedding(12, 12, input_length=n_gram)(input2)
    b = SpatialDropout1D(0.2)(b)

    x = Concatenate(axis=-1)([a, a_sum, b])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model
