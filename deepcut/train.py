import os
import re
from glob import glob
import pandas as pd

from .deepcut import create_n_gram_df, CHAR_TYPE_FLATTEN
from .model import get_convo_nn2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score


article_types = ['article', 'encyclopedia', 'news', 'novel']
char_labels = ['c', 'n', 'v', 'w', 't', 's',
               'd', 'q', 'p', 's_e', 'b_e', 'o']


def generate_words(files):
    """
    Transform list of files to list of words,
    removing new line character
    and replace name entity '<NE>...</NE>' symbol
    """
    words_all = []
    for i, file in enumerate(files):
        lines = open(file, 'r')
        for line in lines:
            words = [word.replace('<NE>', '').replace('</NE>', '') for word in line.split("|") if word is not '\n']
            words_all.extend(words)
    return words_all


def create_char_dataframe(words):
    """
    Give list of input tokenized words,
    create dataframe of characters where first character of
    the word is tagged as 1, otherwise 0

    Example
    =======
    ['กิน', 'หมด'] to dataframe of
    [{'char': 'ก', 'type': ..., 'target': 1}, ...,
     {'char': 'ด', 'type': ..., 'target': 0}]
    """
    char_dict = []
    for word in words:
        for i, char in enumerate(word):
            if i == 0:
                char_dict.append({'char': char,
                                  'type': CHAR_TYPE_FLATTEN.get(char, 'o'),
                                  'target': True})
            else:
                char_dict.append({'char': char,
                                  'type': CHAR_TYPE_FLATTEN.get(char, 'o'),
                                  'target': False})
    return pd.DataFrame(char_dict)


def generate_best_dataset(best_path, output_path='cleaned_data'):
    """
    Generate CSV file for training and testing data

    Input
    =====
    best_path: str, path to BEST folder which contains unzipped subfolder
        'article', 'encyclopedia', 'news', 'novel'

    cleaned_data: str, path to output folder, the cleaned data will be saved
        in the given folder name
    """
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for article_type in article_types:
        files = glob(os.path.join(best_path, article_type, '*.txt'))
        files_train, files_test = train_test_split(files, random_state=0)
        train_words = generate_words(files_train)
        test_words = generate_words(files_test)
        train_df = create_char_dataframe(train_words)
        test_df = create_char_dataframe(test_words)
        train_df.to_csv(os.path.join(output_path, 'df_best_{}_train.csv'.format(article_type)), index=False)
        test_df.to_csv(os.path.join(output_path, 'df_best_{}_test.csv'.format(article_type)), index=False)
        print(article_type)


def train_model(best_processed_path):
    """
    Given path to processed BEST dataset,
    train CNN model for words beginning alongside with
    character label encoder and character type label encoder

    Output
    ======
    model: keras model
    char_le: character label encoder
    type_le: character type label encoder
    """
    # padding for training and testing set
    n_pad = 21
    n_pad_2 = int((n_pad - 1)/2)
    pad = [{'char': ' ', 'type': 'p', 'target': True}]
    df_pad = pd.DataFrame(pad * n_pad_2)

    # read and concat all characters
    df_train, df_test = [], []
    for article_type in article_types:
        df_train.append(pd.read_csv(os.path.join(best_processed_path, 'df_best_{}_train.csv'.format(article_type))))
    df_train = pd.concat(df_train)
    df_train = pd.concat((df_pad, df_train, df_pad)) # pad with empty string feature

    # tranform to
    char_le = LabelEncoder()
    char_le.fit(chars)
    chars = list(df_train.char.unique()) + ['other']
    char_le.fit(chars)
    type_le.fit(char_labels)

    df_train['char'] = char_le.transform(df_train['char'].astype(str))
    df_train['type'] = type_le.transform(df_train['type'].astype(str))
    df_train_pad = create_n_gram_df(df_train, n_pad=n_pad)

    char_row = ['char' + str(i + 1) for i in range(n_pad_2)] + \
               ['char-' + str(i + 1) for i in range(n_pad_2)] + ['char']
    type_row = ['type' + str(i + 1) for i in range(n_pad_2)] + \
               ['type-' + str(i + 1) for i in range(n_pad_2)] + ['type']

    x_train1 = df_train_pad[char_row].as_matrix()
    x_train2 = df_train_pad[type_row].as_matrix()
    y_train = df_train_pad['target']

    model = get_convo_nn2()
    model.fit([x_train1, x_train2], y_train, epochs=10, batch_size=256, verbose=2)
    model.fit([x_train1, x_train2], y_train, epochs=3, batch_size=512, verbose=2)
    model.fit([x_train1, x_train2], y_train, epochs=3, batch_size=2048, verbose=2)
    model.fit([x_train1, x_train2], y_train, epochs=3, batch_size=4096, verbose=2)
    model.fit([x_train1, x_train2], y_train, epochs=3, batch_size=8192, verbose=2)

    return model, char_le, type_le
