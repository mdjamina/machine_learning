# -*- coding: utf-8 -*-
import os

import joblib
import pandas as pd
from joblib import Memory
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def load_data(datafile: str):
    tmp = pd.read_pickle(datafile)

    content = tmp.docs.apply(
        lambda x: ' '.join([w['lemma'] for w in x if not w['is_stop'] and len(w['lemma']) > 3])).to_numpy()
    labels = tmp.parti.to_numpy()

    return content, labels


def pipline_svc(lowercase=False, max_df=0.95, min_df=0.05, sublinear_tf=True):
    """Pipeline pour le modèle LinearSVC"""
    return [
        ('tfidf', TfidfVectorizer(lowercase=lowercase, max_df=max_df, min_df=min_df, sublinear_tf=sublinear_tf)),
        ('clf', LinearSVC())
    ]


def pipline_nb(lowercase=False, max_df=0.95, min_df=0.05, sublinear_tf=True):
    """Pipeline pour le modèle MultinomialNB"""
    return [
        ('tfidf', TfidfVectorizer(lowercase=lowercase, max_df=max_df, min_df=min_df, sublinear_tf=sublinear_tf)),
        ('clf', MultinomialNB())
    ]


def pipline_rfc(lowercase=False, max_df=0.95, min_df=0.05, sublinear_tf=True):
    """Pipeline pour le modèle RandomForestClassifier"""
    return [
        ('tfidf', TfidfVectorizer(lowercase=lowercase, max_df=max_df, min_df=min_df, sublinear_tf=sublinear_tf)),
        ('clf', RandomForestClassifier())
    ]


def pipline_dtc(lowercase=False, max_df=0.5, min_df=0.05, sublinear_tf=True):
    """Pipeline pour le modèle DecisionTreeClassifier"""
    return [
        ('tfidf', TfidfVectorizer(lowercase=lowercase, max_df=max_df, min_df=min_df, sublinear_tf=sublinear_tf)),
        ('clf', DecisionTreeClassifier())
    ]


def model_training(x_train, y_train
                   , pipeline=pipline_svc
                   , lowercase=False
                   , max_df=0.95
                   , min_df=0.05
                   , sublinear_tf=True
                   , memory=Memory(location="/tmp/cachedir", verbose=1)):
    """Entraîner le modèle"""

    model = Pipeline(
        steps=pipeline(lowercase=lowercase
                       , max_df=max_df
                       , min_df=min_df
                       , sublinear_tf=sublinear_tf)
        , verbose=True, memory=memory)

    model.fit(x_train, y_train)

    return model


def save_model(model):
    """Sauvegarder le modèle"""

    pathfile = os.path.join(os.getcwd(), 'deft09_model.pkl')

    print(pathfile)
    joblib.dump(model, pathfile)


def main():
    # data directory
    datafile = '/home/amina/workspace/github/machine_learning/data/deft09_parlement_appr_fr_pre.pkl'

    # load data
    x_train, y_train = load_data(datafile)

    # x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, train_size=0.8,
    #                                                    random_state=44)

    model = model_training(x_train, y_train, pipeline=pipline_svc)

    # sauvgarder le modèle
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    main()
