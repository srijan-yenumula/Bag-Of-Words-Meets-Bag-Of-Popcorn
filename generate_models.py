"""Creates the deep-learning neural networks.

Produces bag-of-words and paragraph vector models.
"""

import logging
import sys
from random import shuffle

import nltk.data
import pandas as pd
from gensim.models import Doc2Vec, Word2Vec

from utilities import get_clean_labeled_reviews, review_to_sentences

TRAIN = sys.argv[1]
TEST = sys.argv[2]
UNLABELED = sys.argv[3]


def create_word2vec_model(doclist):
    """Train Word2Vec model on input data"""
    n_dim = 5000  # number of dimensions

    model_name = f"{n_dim}features_40minwords_10context"

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    num_features = n_dim    # Word vector dimensionality
    min_word_count = 5   # Minimum word count
    num_workers = 11       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    print("Training Word2Vec model...")
    model = Word2Vec(
        doclist,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling,
        seed=1
    )

    model.init_sims(replace=True)
    model.save(model_name)


def create_doc2vec_model(train, test, unlabeled_train):
    """Train Doc2Vec model on input data"""

    print("Cleaning and labeling all data sets...\n")
    train_reviews = get_clean_labeled_reviews(train)
    test_reviews = get_clean_labeled_reviews(test)
    unlabeled_train_reviews = get_clean_labeled_reviews(unlabeled_train)

    n_dim = 400  # number of dimensions

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    num_features = n_dim    # Word vector dimensionality
    min_word_count = 1   # Minimum word count
    num_workers = 11       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    print("Training Doc2Vec model...")
    model_dm = Doc2Vec(
        min_count=min_word_count,
        window=context,
        size=num_features,
        sample=downsampling,
        workers=num_workers
    )
    model_dbow = Doc2Vec(
        min_count=min_word_count,
        window=context,
        size=num_features,
        sample=downsampling,
        workers=num_workers,
        dm=0
    )

    review_list = train_reviews + test_reviews + unlabeled_train_reviews
    model_dm.build_vocab(review_list)
    model_dbow.build_vocab(review_list)

    alpha, min_alpha, passes = (0.025, 0.001, 10)
    alpha_delta = (alpha - min_alpha) / passes

    # train Doc2Vec model, making multiple passes
    epoch = 0
    while epoch < passes:
        shuffle(review_list)  # shuffles documents

        model_dm.alpha, model_dm.min_alpha = alpha, alpha
        model_dm.train(
            review_list,
            total_examples=len(review_list),
            epochs=1
        )

        model_dbow.alpha, model_dbow.min_alpha = alpha, alpha
        model_dbow.train(
            review_list,
            total_examples=len(review_list),
            epochs=1
        )

        alpha -= alpha_delta
        epoch += 1

    model_dm.save(f"{n_dim}features_1minwords_10context_dm")
    model_dbow.save(f"{n_dim}features_1minwords_10context_dbow")


def main():
    """Reads and tokenizes data files"""

    # parse TSV data files into dataframes
    train = pd.read_csv(
        TRAIN,
        header=0,
        delimiter="\t",
        quoting=3
    )
    unlabeled_train = pd.read_csv(
        UNLABELED,
        header=0,
        delimiter="\t",
        quoting=3
    )
    test = pd.read_csv(
        TEST,
        header=0,
        delimiter="\t",
        quoting=3
    )

    # load punkt tokenizer from NLTK
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []

    # create corpus from labeled training data
    print("Reading labeled training dataset")
    for review in train["review"]:
        sentences += review_to_sentences(
            review,
            tokenizer
        )

    # append unlabeled training data to corpus
    print("Reading unlabeled dataset")
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(
            review,
            tokenizer
        )

    # generate models
    create_word2vec_model(sentences)
    create_doc2vec_model(train, test, unlabeled_train)


if __name__ == '__main__':
    main()
