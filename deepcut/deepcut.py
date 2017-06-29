import os
import pandas as pd
from .model import get_convo_nn2

module_path = os.path.dirname(__file__)
weight_path = os.path.join(module_path, 'weight', 'best_cnn3.h5')


CHAR_TYPE = {
    'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    'ฅฉผฟฌหฮ': 'n',
    'ะาำิีืึุู': 'v',  # า ะ ำ ิ ี ึ ื ั ู ุ
    'เแโใไ': 'w',
    '่้๊๋': 't', # วรรณยุกต์ ่ ้ ๊ ๋
    '์ๆฯ.': 's', # ์  ๆ ฯ .
    '0123456789๑๒๓๔๕๖๗๘๙': 'd',
    '"': 'q',
    "‘": 'q',
    "’": 'q',
    "'": 'q',
    ' ': 'p',
    'abcdefghijklmnopqrstuvwxyz': 's_e',
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}

CHAR_TYPE_FLATTEN = {}
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v

# create map of dictionary to character
CHARS = [
    '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'other', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z', '}', '~', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
    'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท',
    'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ',
    'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า',
    'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ', 'เ', 'แ', 'โ', 'ใ', 'ไ',
    'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๐', '๑', '๒', '๓',
    '๔', '๕', '๖', '๗', '๘', '๙', '‘', '’', '\ufeff'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}

CHAR_TYPES = [
    'b_e', 'c', 'd', 'n', 'o',
    'p', 'q', 's', 's_e', 't',
    'v', 'w'
]
CHAR_TYPES_MAP = {v: k for k, v in enumerate(CHAR_TYPES)}

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
        if char in CHARS:
            char_dict.append({'char': char,
                              'type': CHAR_TYPE_FLATTEN.get(char, 'o')})
        else:
            char_dict.append({'char': 'other',
                              'type': CHAR_TYPE_FLATTEN.get(char, 'o')})
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
    char_df['char'] = char_df['char'].map(lambda x: CHARS_MAP.get(x, 80))
    char_df['type'] = char_df['type'].map(lambda x: CHAR_TYPES_MAP.get(x, 4))
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
