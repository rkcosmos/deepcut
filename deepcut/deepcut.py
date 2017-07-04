import os
import pandas as pd

from .model import get_convo_nn2
from .utils import create_n_gram_df, create_char_dict, pad_dict, CHARS_MAP, CHAR_TYPES_MAP

module_path = os.path.dirname(__file__)
weight_path = os.path.join(module_path, 'weight', 'cnn_without_ne_ab.h5')

# load model when importing library
model = get_convo_nn2()
model.load_weights(weight_path)

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
