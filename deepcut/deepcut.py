import os
import pickle
import pandas as pd

from keras.models import Model
from keras.layers import Input, Dense, Embedding, \
    Concatenate, Flatten, Dropout, \
    SpatialDropout1D, BatchNormalization, \
    Conv1D, MaxPooling1D, Maximum, ZeroPadding1D
from keras.layers import TimeDistributed
from keras.optimizers import Adam

module_path = os.path.dirname(__file__)
object_path = os.path.join(module_path, 'weight', 'object.pk')
weight_path = os.path.join(module_path, 'weight', 'best_cnn3.h5')

with open(object_path, 'rb') as handle:
    char_le, type_le, listed_char = pickle.load(handle)

CHAR_TYPE = {
    'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    'ฅฉผฟฌหฮ': 'n',
    'ะาำิีืึุู': 'v',  # า ะ ำ ิ ี ึ ื ั ู ุ
    'เแโใไ': 'w',
    '่้๊๋': 't', # วรรณยุกต์ ่ ้ ๊ ๋
    '์ๆฯ.': 's', # ์  ๆ ฯ .
    '0123456789๑๒๓๔๕๖๗๘๙': 'd',
    '"': 'q',
    "'": 'q',
    ' ': 'p',
    'abcdefghijklmnopqrstuvwxyz': 's_e',
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}

CHAR_TYPE_FLATTEN = {}
for k, v in CHAR_TYPE.items():
    for k_ in k:
        CHAR_TYPE_FLATTEN[k_] = v


def create_n_gram_df(df, n_gram=11):
    n = int((n_gram - 1)/2)

    for i in range(n):
        df['char-{}'.format(i+1)] = df['char'].shift(i + 1)
        df['type-{}'.format(i+1)] = df['type'].shift(i + 1)
        df['char{}'.format(i+1)] = df['char'].shift(-i - 1)
        df['type{}'.format(i+1)] = df['type'].shift(-i - 1)

    return df[n:-n].copy()


def create_char_dict(text):
    char_dict = []
    for char in text:
        if char in listed_char:
            char_dict.append({'char': char,
                              'type': CHAR_TYPE_FLATTEN.get(char, 'o'),
                              'target': True})
        else:
            char_dict.append({'char': 'other',
                              'type': CHAR_TYPE_FLATTEN.get(char, 'o'),
                              'target': True})
    return char_dict


def pad_dict(char_dict, n_pad=11):
    n_pad_half = int((n_pad - 1)/2)
    pad = [{'char': ' ', 'type': 'p', 'target': True}]
    char_dict_pad = (pad * n_pad_half) + char_dict + (pad * n_pad_half)
    return char_dict_pad


def get_convo_nn2(no_word=200, n_gram=21, no_char=178):
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))

    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.2)(a)

    a2 = Conv1D(no_word, 2, strides=1, padding="valid", activation='relu')(a)
    a2 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a2)
    a2 = ZeroPadding1D(padding=(0,1))(a2)

    a3 = Conv1D(no_word, 3, strides=1, padding="valid", activation='relu')(a)
    a3 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a3)
    a3 = ZeroPadding1D(padding=(0,2))(a3)

    a4 = Conv1D(no_word, 4, strides=1, padding="valid", activation='relu')(a)
    a4 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a4)
    a4 = ZeroPadding1D(padding=(0,3))(a4)

    a5 = Conv1D(no_word, 5, strides=1, padding="valid", activation='relu')(a)
    a5 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a5)
    a5 = ZeroPadding1D(padding=(0,4))(a5)

    a6 = Conv1D(no_word, 6, strides=1, padding="valid", activation='relu')(a)
    a6 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a6)
    a6 = ZeroPadding1D(padding=(0,5))(a6)

    a7 = Conv1D(no_word, 7, strides=1, padding="valid", activation='relu')(a)
    a7 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a7)
    a7 = ZeroPadding1D(padding=(0,6))(a7)

    a8 = Conv1D(no_word, 8, strides=1, padding="valid", activation='relu')(a)
    a8 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a8)
    a8 = ZeroPadding1D(padding=(0,7))(a8)

    a9 = Conv1D(no_word-50, 9, strides=1, padding="valid", activation='relu')(a)
    a9 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a9)
    a9 = ZeroPadding1D(padding=(0,8))(a9)

    a10 = Conv1D(no_word-50, 10, strides=1, padding="valid", activation='relu')(a)
    a10 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a10)
    a10 = ZeroPadding1D(padding=(0,9))(a10)

    a11 = Conv1D(no_word-50, 11, strides=1, padding="valid", activation='relu')(a)
    a11 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a11)
    a11 = ZeroPadding1D(padding=(0,10))(a11)

    a12 = Conv1D(no_word-100, 12, strides=1, padding="valid", activation='relu')(a)
    a12 = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(a12)
    a12 = ZeroPadding1D(padding=(0,11))(a12)

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

# load model before hand
model = get_convo_nn2()
model.load_weights(weight_path)

def tokenize(text):

    n_gram = 21
    n_gram_2 = int((n_gram - 1)/2)

    char_dict = create_char_dict(text)
    char_dict_pad = pad_dict(char_dict, n_pad=n_gram)
    char_df = pd.DataFrame(char_dict_pad)
    char_df['char'] = char_le.transform(char_df['char'])
    char_df['type'] = type_le.transform(char_df['type'])
    char_df_ngram = create_n_gram_df(char_df, n_gram=n_gram)

    char_row = ['char'+str(i+1) for i in range(n_gram_2)] + \
               ['char-'+str(i+1) for i in range(n_gram_2)] + ['char']
    type_row = ['type'+str(i+1) for i in range(n_gram_2)] + \
               ['type-'+str(i+1) for i in range(n_gram_2)] + ['type']

    x_char = char_df_ngram[char_row].as_matrix()
    x_type = char_df_ngram[type_row].as_matrix()

    y_predict = model.predict([x_char, x_type])
    y_predict = (y_predict.ravel() > 0.5).astype(int)
    word_end = list(y_predict[1:]) + [1]

    tokens = []
    word = ''
    for char, w_e in zip(text, word_end):
        word += char
        if w_e:
            tokens.append(word)
            word = ''
    return tokens
