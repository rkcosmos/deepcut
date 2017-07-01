import os
from glob import glob
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

if __package__ != 'deepcut':
    from utils import create_n_gram_df, CHAR_TYPE_FLATTEN, CHARS_MAP, CHAR_TYPES_MAP
    from model import get_convo_nn2
else:
    from .utils import create_n_gram_df, CHAR_TYPE_FLATTEN, CHARS_MAP, CHAR_TYPES_MAP
    from .model import get_convo_nn2

article_types = ['article', 'encyclopedia', 'news', 'novel']

def generate_words(files):
    """
    Transform list of files to list of words,
    removing new line character
    and replace name entity '<NE>...</NE>' and abbreviation '<AB>...</AB>' symbol
    """
    
    repls = {'<NE>' : '','</NE>' : '','<AB>': '','</AB>': ''}
    
    words_all = []
    for i, file in enumerate(files):
        lines = open(file, 'r')
        for line in lines:
            line = reduce(lambda a, kv: a.replace(*kv), repls.items(), line)
            words = [word for word in line.split("|") if word is not '\n']
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
        files_train, files_test = train_test_split(files, random_state=0, test_size=0.1)
        train_words = generate_words(files_train)
        test_words = generate_words(files_test)
        train_df = create_char_dataframe(train_words)
        test_df = create_char_dataframe(test_words)
        train_df.to_csv(os.path.join(output_path, 'df_best_{}_train.csv'.format(article_type)), index=False)
        test_df.to_csv(os.path.join(output_path, 'df_best_{}_test.csv'.format(article_type)), index=False)
        print("Save {} to CSV file".format(article_type))


def prepare_feature(best_processed_path, option='train'):
    """
    Transform processed path into

    Input
    =====
    best_processed_path: str, path to processed BEST dataset

    option: str, 'train' or 'test'
    """
    # padding for training and testing set
    n_pad = 21
    n_pad_2 = int((n_pad - 1)/2)
    pad = [{'char': ' ', 'type': 'p', 'target': True}]
    df_pad = pd.DataFrame(pad * n_pad_2)

    df = []
    for article_type in article_types:
        df.append(pd.read_csv(os.path.join(best_processed_path, 'df_best_{}_{}.csv'.format(article_type, option))))
    df = pd.concat(df)
    df = pd.concat((df_pad, df, df_pad)) # pad with empty string feature

    df['char'] = df['char'].map(lambda x: CHARS_MAP.get(x, 80))
    df['type'] = df['type'].map(lambda x: CHAR_TYPES_MAP.get(x, 4))
    df_pad = create_n_gram_df(df, n_pad=n_pad)

    char_row = ['char' + str(i + 1) for i in range(n_pad_2)] + \
               ['char-' + str(i + 1) for i in range(n_pad_2)] + ['char']
    type_row = ['type' + str(i + 1) for i in range(n_pad_2)] + \
               ['type-' + str(i + 1) for i in range(n_pad_2)] + ['type']

    x_char = df_pad[char_row].as_matrix()
    x_type = df_pad[type_row].as_matrix()
    y = df_pad['target'].astype(int)

    return x_char, x_type, y


def train_model(best_processed_path):
    """
    Given path to processed BEST dataset,
    train CNN model for words beginning alongside with
    character label encoder and character type label encoder

    Input
    =====
    best_processed_path: str, path to processed BEST dataset

    Output
    ======
    model: keras model, keras model for tokenize prediction
    """

    x_train_char, x_train_type, y_train = prepare_feature(best_processed_path, option='train')
    x_test_char, x_test_type, y_test = prepare_feature(best_processed_path, option='test')

    # train model
    model = get_convo_nn2()
    model.fit([x_train_char, x_train_type], y_train, epochs=10, batch_size=256, verbose=2,\
              validation_data = ([x_test_char, x_test_type], y_test))
    model.fit([x_train_char, x_train_type], y_train, epochs=3, batch_size=512, verbose=2,\
              validation_data = ([x_test_char, x_test_type], y_test))
    model.fit([x_train_char, x_train_type], y_train, epochs=3, batch_size=2048, verbose=2,\
              validation_data = ([x_test_char, x_test_type], y_test))
    model.fit([x_train_char, x_train_type], y_train, epochs=3, batch_size=4096, verbose=2,\
              validation_data = ([x_test_char, x_test_type], y_test))
    model.fit([x_train_char, x_train_type], y_train, epochs=3, batch_size=8192, verbose=2,\
              validation_data = ([x_test_char, x_test_type], y_test))

    return model


def evaluate(best_processed_path, model):
    """
    Evaluate model with splitted testing set
    """
    x_test_char, x_test_type, y_test = prepare_feature(best_processed_path, option='test')

    y_predict = model.predict([x_test_char, x_test_type])
    y_predict = (y_predict.ravel() > 0.5).astype(int)

    f1score = f1_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)

    return f1score, precision, recall

