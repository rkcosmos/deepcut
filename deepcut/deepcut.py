#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import six
import numbers
import numpy as np
import pandas as pd
import re
import scipy.sparse as sp
from itertools import chain

from .model import get_convo_nn2
from .utils import create_n_gram_df, create_char_dict, pad_dict, CHARS_MAP, CHAR_TYPES_MAP
from .stop_words import THAI_STOP_WORDS

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
    >> ['ตัดคำ','ได้','ดี','มาก']

    """
    n_pad = 21
    n_pad_2 = int((n_pad - 1)/2)

    if len(text) == 0:
        return [''] # case of empty string

    if isinstance(text, str) and sys.version_info.major == 2:
        text = text.decode('utf-8')

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

    try:
        with open('custom_dict.txt') as f:
            word_list = f.readlines()
        for word in word_list:
            if isinstance(word, str) and sys.version_info.major == 2:
                word = word.decode('utf-8')
            word = word.strip('\n')
            word_end = _custom_dict(word, text, word_end)
    except:
        pass

    tokens = []
    word = ''
    for char, w_e in zip(text, word_end):
        word += char
        if w_e:
            tokens.append(word)
            word = ''
    return tokens


def _custom_dict(word, text, word_end):
    word_length = len(word)
    initial_loc = 0

    while True:
        try:
            start_char = re.search(word, text).start()
            first_char = start_char+initial_loc
            last_char = first_char+word_length-1

            initial_loc += start_char+word_length
            text = text[start_char+word_length:]
            word_end[first_char:last_char] = (word_length-1)*[0]
            word_end[last_char] = 1
        except:
            break
    return word_end


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


def _check_stop_list(stop):
    """
    Check stop words list
    ref: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L87-L95
    """
    if stop == "thai":
        return THAI_STOP_WORDS
    elif isinstance(stop, six.string_types):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:               # assume it's a collection
        return frozenset(stop)


class DeepcutTokenizer(object):
    """
    Class for tokenizing given Thai text documents using deepcut library

    Parameters
    ==========
    ngram_range : tuple, tuple for ngram range for vocabulary, (1, 1) for unigram
        and (1, 2) for bigram
    stop_words : list or set, list or set of stop words to be removed
        if None, max_df can be set to value [0.7, 1.0) to automatically remove
        vocabulary. If using "thai", this will use list of pre-populated stop words
    max_features : int or None, if provided, only consider number of vocabulary
        ordered by term frequencies
    max_df : float in range [0.0, 1.0] or int, default=1.0
        ignore terms that have a document frequency higher than the given threshold
    min_df : float in range [0.0, 1.0] or int, default=1
        ignore terms that have a document frequency lower than the given threshold
    dtype : type, optional


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

    def __init__(self, ngram_range=(1, 1), stop_words=None,
                 max_df=1.0, min_df=1, max_features=None, dtype=np.float64):
        self.vocabulary_ = {}
        self.ngram_range = ngram_range
        self.dtype = dtype
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.stop_words = _check_stop_list(stop_words)


    def _word_ngrams(self, tokens):
        """
        Turn tokens into a tokens of n-grams

        ref: https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/feature_extraction/text.py#L124-L153
        """
        # handle stop words
        if self.stop_words is not None:
            tokens = [w for w in tokens if w not in self.stop_words]

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


    def _limit_features(self, X, vocabulary,
                        high=None, low=None, limit=None):
        """Remove too rare or too common features.

        ref: https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/feature_extraction/text.py#L734-L773
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], vocabulary, removed_terms


    def transform(self, raw_documents, new_document=False):
        n_doc = len(raw_documents)
        tokenized_documents = []
        for doc in raw_documents:
            tokens = tokenize(doc) # method in this file
            tokens = self._word_ngrams(tokens)
            tokenized_documents.append(tokens)

        if new_document:
            self.vocabulary_ = {v: k for k, v in enumerate(set(chain.from_iterable(tokenized_documents)))}

        values, row_indices, col_indices = [], [], []
        for r, tokens in enumerate(tokenized_documents):
            tokens = self._word_ngrams(tokens)
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
                          shape=(n_doc, len(self.vocabulary_)),
                          dtype=self.dtype)

        # truncate vocabulary by max_df and min_df
        max_df = self.max_df
        min_df = self.min_df
        max_doc_count = (max_df
                         if isinstance(max_df, numbers.Integral)
                         else max_df * n_doc)
        min_doc_count = (min_df
                         if isinstance(min_df, numbers.Integral)
                         else min_df * n_doc)
        if max_doc_count < min_doc_count:
            raise ValueError(
                "max_df corresponds to < documents than min_df")
        X, vocabulary, _ = self._limit_features(X, self.vocabulary_,
                                                max_doc_count,
                                                min_doc_count,
                                                self.max_features)
        self.vocabulary_ = vocabulary

        return X


    def fit_tranform(self, raw_documents):
        """
        Transform given list of raw_documents to document-term matrix in
        sparse CSR format (see scipy)
        """
        X = self.transform(raw_documents, new_document=True)
        return X
