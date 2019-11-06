"""
This script is a simplified script for True-voice Intent classification problem using a dataset from https://github.com/PyThaiNLP/truevoice-intent

The script is simplified and adapted based on https://github.com/PyThaiNLP/truevoice-intent/blob/master/classification.ipynb
"""
import pandas as pd
import deepcut
from deepcut import DeepcutTokenizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


if __name__ == '__main__':
    df = pd.read_csv('mari-intent/mari_train.csv') # load training data
    tokenizer = DeepcutTokenizer(ngram_range=(1,1), max_df=0.8, min_df=2)
    tokenizer_bigram = DeepcutTokenizer(ngram_range=(1,2), max_df=0.85, min_df=2)
    predictors = [
        (LogisticRegression(solver='lbfgs'), 'logistic_regression'),
        (LinearSVC(max_iter=3000), 'linear_svc')
    ]

    print('tokenizing dataset (unigram)...')
    X = tokenizer.fit_tranform(df.texts.map(lambda x: x.strip()).values)
    y = pd.get_dummies(df.action).values
    print('done tokenization!')

    print('"Prediction result on bag of words unigram matrix"')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1412)
    for predictor, predictor_name in predictors:
        pred_model = OneVsRestClassifier(predictor)
        pred_model.fit(X_train, y_train)
        y_val_pred = pred_model.predict(X_val)
        print('Predictor model: {}'.format(predictor_name))
        print('Accuracy = {}'.format(accuracy_score(y_val, y_val_pred)))
        print('Precision, Recall, F-Score (micro) = ', precision_recall_fscore_support(y_val, y_val_pred, average='micro'))

    print('\n\n')
    print('"Prediction result on tf-idf of bag of words matrix"')
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    X_val_tfidf = tfidf_transformer.transform(X_val)
    for predictor, predictor_name in predictors:
        pred_model = OneVsRestClassifier(predictor)
        pred_model.fit(X_train_tfidf, y_train)
        y_val_pred = pred_model.predict(X_val_tfidf)
        print('Predictor model: {}'.format(predictor_name))
        print('Accuracy = {}'.format(accuracy_score(y_val, y_val_pred)))
        print('Precision, Recall, F-Score (micro) = ', precision_recall_fscore_support(y_val, y_val_pred, average='micro'))

    print('tokenizing dataset (bigram)...')
    X_bigram = tokenizer_bigram.fit_tranform(df.texts.map(lambda x: x.strip()).values)
    print('done tokenization!')

    print('"Prediction result on bigram count vectorizer matrix"')
    X_train, X_val, y_train, y_val = train_test_split(X_bigram, y, test_size=0.15, random_state=1412)
    for predictor, predictor_name in predictors:
        pred_model = OneVsRestClassifier(predictor)
        pred_model.fit(X_train, y_train)
        y_val_pred = pred_model.predict(X_val)
        print('Predictor model: {}'.format(predictor_name))
        print('Accuracy = {}'.format(accuracy_score(y_val, y_val_pred)))
        print('Precision, Recall, F-Score (micro) = ', precision_recall_fscore_support(y_val, y_val_pred, average='micro'))

    print('\n\n')
    print('"Prediction result on tf-idf of bigram bag of words matrix"')
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    X_val_tfidf = tfidf_transformer.transform(X_val)
    for predictor, predictor_name in predictors:
        pred_model = OneVsRestClassifier(predictor)
        pred_model.fit(X_train_tfidf, y_train)
        y_val_pred = pred_model.predict(X_val_tfidf)
        print('Predictor model: {}'.format(predictor_name))
        print('Accuracy = {}'.format(accuracy_score(y_val, y_val_pred)))
        print('Precision, Recall, F-Score (micro) = ', precision_recall_fscore_support(y_val, y_val_pred, average='micro'))