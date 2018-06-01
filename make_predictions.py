"""Combine models to predict sentiment."""

import sys

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, Word2Vec
from scipy.sparse import hstack
from sklearn.datasets import load_svmlight_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

from svm_model import generate_svm_model
from utilities import get_clean_labeled_reviews, review_to_wordlist

TRAIN = sys.argv[1]
TEST = sys.argv[2]
UNLABELED = sys.argv[3]
OUTPUT = sys.argv[4]


def create_feature_vector(words, model, num_features):
    """Generate feature vector as a numpy array"""
    feature_vec = np.zeros((num_features,), dtype="float32")
    ctr = 0

    # adds all unique features to the feature vector
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            feature_vec = np.add(feature_vec, model[word])
            ctr += 1

    if ctr != 0:
        feature_vec /= ctr
    return feature_vec


def get_average_vectors(reviews, model, num_features):
    """Generate feature vectors for a set of reviews"""
    ctr = 0
    feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    # creates feature vectors for each review
    for review in reviews:
        feature_vecs[ctr] = create_feature_vector(
            review,
            model,
            num_features
        )
        ctr += 1

    return feature_vecs


def get_clean_reviews(reviews):
    """Clean and return reviews as a list of wordlists"""
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(review_to_wordlist(review, True))

    return clean_reviews


def get_vectors(reviews, model, num_features):
    """Populate numpy array with reviews pulled from model"""
    feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
    ctr = -1

    # converts reviews to nparray for processing
    for review in reviews:
        try:
            feature_vecs[ctr] = np.array(
                model[review.labels[0]]).reshape((1, num_features))
        except:
            continue
        ctr += 1

    return feature_vecs


def main():
    """Program entry point"""

    # read in data
    train = pd.read_csv(
        'data/labeledTrainData.tsv',
        header=0,
        delimiter="\t",
        quoting=3
    )
    test = pd.read_csv(
        'data/testData.tsv',
        header=0,
        delimiter="\t",
        quoting=3
    )

    print("Cleaning and parsing the datasets...\n")

    # tokenize and clean reviews
    clean_train_reviews = []
    for review in train['review']:
        clean_train_reviews.append(
            " ".join(review_to_wordlist(review))
        )

    clean_test_reviews = []
    for review in test['review']:
        clean_test_reviews.append(
            " ".join(review_to_wordlist(review))
        )

    print("Creating the bag of words...\n")

    # create and train bag-of-words model
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        sublinear_tf=True
    )

    # bow = Bag of Words
    bow_train = vectorizer.fit_transform(clean_train_reviews)
    del clean_train_reviews
    bow_test = vectorizer.transform(clean_test_reviews)
    del clean_test_reviews

    del vectorizer

    print("Cleaning and labeling the data sets...\n")

    # clean and label reviews for paragraph vectors
    train_reviews = get_clean_labeled_reviews(train)
    test_reviews = get_clean_labeled_reviews(test)

    n_dim = 400  # number of feature dimensions for Doc2Vec

    print('Loading the Doc2Vec model...\n')

    # load paragraph vector model
    model_dm = Doc2Vec.load(f"{n_dim}features_1minwords_10context_dm")
    model_dbow = Doc2Vec.load(f"{n_dim}features_1minwords_10context_dbow")

    print("Extracting Doc2Vec features...\n")

    doc2vec_dm_train = get_vectors(train_reviews, model_dm, n_dim)
    doc2vec_dbow_train = get_vectors(train_reviews, model_dbow, n_dim)
    doc2vec_train = np.hstack((doc2vec_dm_train, doc2vec_dbow_train))

    del doc2vec_dm_train
    del doc2vec_dbow_train

    doc2vec_dm_test = get_vectors(test_reviews, model_dm, n_dim)
    doc2vec_dbow_test = get_vectors(test_reviews, model_dbow, n_dim)
    doc2vec_test = np.hstack((doc2vec_dm_test, doc2vec_dbow_test))

    # delete references to trigger garbage collection
    del doc2vec_dm_test
    del doc2vec_dbow_test
    del model_dm
    del model_dbow

    n_dim = 5000  # number of feature dimensions for Word2Vec

    print('Loading Word2Vec model...\n')

    # load Word2Vec model
    model = Word2Vec.load(f"{n_dim}features_40minwords_10context")

    print("Extracting Word2Vec features...\n")

    word2vec_train = scale(get_average_vectors(
        get_clean_reviews(train), model, n_dim))
    word2vec_test = scale(get_average_vectors(
        get_clean_reviews(test), model, n_dim))
    del model

    print("Creating the SVM model...\n")

    # create SVM model
    generate_svm_model(train, test, '123', 'data/svm')

    files = ("data/svm-train.txt", "data/svm-test.txt")

    svm_train, _, svm_test, _ = load_svmlight_files(files)
    del _

    # moving feature vectors to hstack object
    # to be used for logistic regression
    print("Generating sparse representation for BoW + Word2Vec...\n")

    word2vec_train_sparse = hstack([bow_train, word2vec_train])
    word2vec_test_sparse = hstack([bow_test, word2vec_test])
    del word2vec_train
    del word2vec_test

    print("Generating sparse representation for BoW + Doc2Vec...\n")

    doc2vec_train_sparse = hstack([bow_train, doc2vec_train])
    doc2vec_test_sparse = hstack([bow_test, doc2vec_test])
    del doc2vec_train
    del doc2vec_test

    train_sent = train['sentiment']  # training sentiment values
    del train

    print("Predicting with Bag-of-words model...\n")

    # regression function
    clf = LogisticRegression(class_weight="balanced")

    # fit regression object to all models
    clf.fit(bow_train, train_sent)
    bow_probs = clf.predict_proba(bow_test)

    print("Predicting with SVM...\n")

    clf.fit(svm_train, train_sent)
    svm_probs = clf.predict_proba(svm_test)

    print("Predicting with Bag-of-words model and Word2Vec model...\n")

    clf.fit(word2vec_train_sparse, train_sent)
    word2vec_probs = clf.predict_proba(word2vec_test_sparse)

    print("Predicting with Bag-of-words model and Doc2Vec model...\n")

    clf.fit(doc2vec_train_sparse, train_sent)
    doc2vec_probs = clf.predict_proba(doc2vec_test_sparse)
    del clf

    print("\nCalculating averages\n")

    bow_weight = 0.15
    doc2vec_weight = 0.3
    svm_weight = 0.55

    # weights applied to nparray using scalar multiplication
    pred = bow_weight * bow_probs + \
        (1 - bow_weight - doc2vec_weight - svm_weight) * word2vec_probs + \
        doc2vec_weight * doc2vec_probs + svm_weight * svm_probs

    mean = (bow_probs + word2vec_probs + doc2vec_probs + svm_probs) / 4
    mean_scores = []

    # determine mean scores
    ctr = 0
    for row in mean:
        if row[1] > 0.5:
            score = max(bow_probs[ctr, 1], word2vec_probs[ctr, 1],
                        doc2vec_probs[ctr, 1], svm_probs[ctr, 1])
            mean_scores.append(score)
        elif row[1] < 0.5:
            score = min(bow_probs[ctr, 1], word2vec_probs[ctr, 1],
                        doc2vec_probs[ctr, 1], svm_probs[ctr, 1])
            mean_scores.append(score)
        else:
            mean_scores.append(pred[ctr, 1])
        ctr += 1

    best_scores = []

    # determine weighted average means
    print("\nMaking predictions\n")
    ctr = 0
    for row in pred:
        if row[1] > 0.5:
            score = max(bow_probs[ctr, 1], word2vec_probs[ctr, 1],
                        doc2vec_probs[ctr, 1], svm_probs[ctr, 1])
            best_scores.append(score)
        elif row[1] < 0.5:
            score = min(bow_probs[ctr, 1], word2vec_probs[ctr, 1],
                        doc2vec_probs[ctr, 1], svm_probs[ctr, 1])
            best_scores.append(score)
        else:
            best_scores.append(pred[ctr, 1])
        ctr += 1

    # create numpy arrays for weighting
    weighted_avgs = np.array([row[1] for row in pred])
    average_means = np.array(mean_scores)
    weighted_avg_means = np.array(best_scores)

    alpha1 = 0.6
    alpha2 = 0.4

    # weights applied to mean score nparrays
    # scalar multiple
    sent_probs = alpha1 * weighted_avgs + \
        (1 - alpha1 - alpha2) * average_means + alpha2 * weighted_avg_means

    # output results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": sent_probs})
    output.to_csv(
        OUTPUT, index=False, quoting=3
    )

    print("Wrote results to output file")


if __name__ == '__main__':
    main()
