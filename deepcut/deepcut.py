import os
import pandas as pd
import scipy.sparse as sp
from itertools import chain

from .model import get_convo_nn2
from .utils import create_n_gram_df, create_char_dict, pad_dict, CHARS_MAP, CHAR_TYPES_MAP

module_path = os.path.dirname(__file__)
weight_path = os.path.join(module_path, 'weight', 'cnn_without_ne_ab.h5')

# load model when importing library
model = get_convo_nn2()
model.load_weights(weight_path)

def tokenize(text):
    """
    Tokenize given Thai text string

    Input
    =====
    text: str, Thai text string

    Output
    ======
    tokens: list, list of tokenized words

    Example
    =======
    >> deepcut.tokenize('ตัดคำได้ดีมาก')
    >> ['ตัด','คำ','ได้','ดี','มาก']

    """
    n_pad = 21
    n_pad_2 = int((n_pad - 1)/2)

    if len(text) == 0:
        return [''] # case of empty string

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


class DeepcutTokenizer(object):
    """
    Class for tokenizing given documents using deepcut library

    Example
    =======
    raw_documents = ['ฉันอยากกินข้าวของฉัน',
                     'ฉันอยากกินไก่',
                     'อยากนอนอย่างสงบ']
    tokenizer = DeepcutTokenizer(ngram_range=(1, 1))
    X = tokenizer.fit_tranform(raw_documents) # document-term matrix in sparse CSR format

    >> X.todense()
    >> [[0, 0, 1, 0, 1, 0, 2, 1],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 0, 0]]
    >> tokenizer.vocabulary_
    >> {'นอน': 0, 'ไก่': 1, 'กิน': 2, 'อย่าง': 3, 'อยาก': 4, 'สงบ': 5, 'ฉัน': 6, 'ข้าว': 7}

    """
    def __init__(self, ngram_range=(1, 1), stop_words=set()):
        self.vocabulary_ = {}
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def _word_ngrams(self, tokens, ngram_range=(1, 1), stop_words={}):
        """
        Turn tokens into a tokens of n-grams

        ref: https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/feature_extraction/text.py#L124-L153
        """
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def transform(self, raw_documents, new_document=False):
        n_documents = len(raw_documents)
        tokenized_documents = []
        for doc in raw_documents:
            tokens = tokenize(doc) # method in this file
            tokens = self._word_ngrams(tokens,
                                       ngram_range=self.ngram_range,
                                       stop_words=self.stop_words)
            tokenized_documents.append(tokens)

        if new_document:
            self.vocabulary_ = {v: k for k, v in enumerate(set(chain.from_iterable(tokenized_documents)))}

        values, row_indices, col_indices = [], [], []
        for r, tokens in enumerate(tokenized_documents):
            tokens = self._word_ngrams(tokens,
                                       ngram_range=self.ngram_range,
                                       stop_words=self.stop_words)
            feature = {}
            for token in tokens:
                word_index = self.vocabulary_.get(token)
                if word_index is not None:
                    if word_index not in feature.keys():
                        feature[word_index] = 1
                    else:
                        feature[word_index] += 1
            for c, v in feature.items():
                values.append(v)
                row_indices.append(r)
                col_indices.append(c)

        # document-term matrix in CSR format
        X = sp.csr_matrix((values, (row_indices, col_indices)),
                          shape=(n_documents, len(self.vocabulary_)))
        return X

    def fit_tranform(self, raw_documents):
        """
        Transform given list of raw_documents to document-term matrix in
        sparse CSR format (see scipy)
        """
        X = self.transform(raw_documents, new_document=True)
        return X
