import os
import pickle
import pandas as pd
from .model import get_convo_nn2

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
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v

# load model when importing library
model = get_convo_nn2()
model.load_weights(weight_path)


def create_n_gram_df(df, n_pad):
    """
    Given input dataframe, create feature dataframe of shifted characters
    """
    n_pad_2 = int((n_pad - 1)/2)
    for i in range(n_pad_2):
        df['char-{}'.format(i+1)] = df['char'].shift(i + 1)
        df['type-{}'.format(i+1)] = df['type'].shift(i + 1)
        df['char{}'.format(i+1)] = df['char'].shift(-i - 1)
        df['type{}'.format(i+1)] = df['type'].shift(-i - 1)
    return df[n_pad_2: -n_pad_2]


def create_char_dict(text):
    """
    Transform input text into list of character feature
    """
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


def pad_dict(char_dict, n_pad):
    """
    Pad list of dictionary with empty character of size n_pad
    (Pad half before and half after the list)
    """
    n_pad_2 = int((n_pad - 1)/2)
    pad = [{'char': ' ', 'type': 'p', 'target': True}]
    char_dict_pad = (pad * n_pad_2) + char_dict + (pad * n_pad_2)
    return char_dict_pad


def tokenize(text):
    """
    Tokenize Thai text string

    Input
    =====
    text: str, Thai text string

    Output
    ======
    tokens: list, list of tokenized words
    """
    n_pad = 21
    n_pad_2 = int((n_pad - 1)/2)

    char_dict = create_char_dict(text)
    char_dict_pad = pad_dict(char_dict, n_pad=n_pad)
    char_df = pd.DataFrame(char_dict_pad)
    char_df['char'] = char_le.transform(char_df['char'])
    char_df['type'] = type_le.transform(char_df['type'])
    char_df_ngram = create_n_gram_df(char_df, n_pad=n_pad)

    char_row = ['char' + str(i + 1) for i in range(n_pad_2)] + \
               ['char-' + str(i + 1) for i in range(n_pad_2)] + ['char']
    type_row = ['type' + str(i + 1) for i in range(n_pad_2)] + \
               ['type-' + str(i + 1) for i in range(n_pad_2)] + ['type']

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
