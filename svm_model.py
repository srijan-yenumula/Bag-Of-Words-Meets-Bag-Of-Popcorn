"""Generate Naive-Bayes SVM model."""

import numpy as np
import pandas as pd
from collections import Counter

from utilities import review_to_wordlist


def tokenize(sentence, grams):
    """Break words into tokens for SVM processing"""
    words = review_to_wordlist(sentence)
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i: i + gram])]
    return tokens


def build_dict(data, grams):
    """Count frequences and return Counter object"""
    counter = Counter()
    for token_list in data:
        counter.update(token_list)
    return counter


def compute_ratios(pos_counts, neg_counts, alpha=1):
    """Compute ratio of positive to negative tokens"""
    # create aggregate list of unique tokens
    tokens = list(set(list(pos_counts.keys()) + list(neg_counts.keys())))

    # switch index and value
    dict_ = dict((token, index) for index, token in enumerate(tokens))
    length = len(dict_)

    print("Computing ratios...\n")

    # scalar multiplication on nparray
    positive, negative = np.ones(length) * alpha, np.ones(length) * alpha
    for token in tokens:
        positive[dict_[token]] += pos_counts[token]
        negative[dict_[token]] += neg_counts[token]

    positive /= abs(positive).sum()
    negative /= abs(negative).sum()
    ratio = np.log(positive / negative)

    return dict_, ratio


def build_svm_content(data, dict_, ratios, grams):
    """Populate SVM model"""
    output = []

    for _, row in data.iterrows():
        tokens = tokenize(row['review'], grams)
        indexes = []
        for token in tokens:
            try:
                indexes += [dict_[token]]
            except KeyError:
                continue

        indexes = list(set(indexes))
        indexes.sort()

        # match index to ratio
        if 'sentiment' in row:
            line = [str(row['sentiment'])]
        else:
            line = ['0']
        for index in indexes:
            line += ["{}:{}".format(index + 1, ratios[index])]
        output += [" ".join(line)]

    return '\n'.join(output)


def generate_svm_model(train, test, grams, outfn):
    """Output SVM model"""
    ngram = [int(gram) for gram in grams]
    pos_train = []
    neg_train = []

    print("Reading training data...\n")

    for _, row in train.iterrows():
        if row['sentiment'] == 1:
            pos_train.append(tokenize(row['review'], ngram))
        elif row['sentiment'] == 0:
            neg_train.append(tokenize(row['review'], ngram))

    # build positive and negative Counters
    pos_counts = build_dict(pos_train, ngram)
    neg_counts = build_dict(neg_train, ngram)

    dict_, ratios = compute_ratios(pos_counts, neg_counts)

    # write model to files
    with open(outfn + '-train.txt', 'w') as file_:
        file_.writelines(
            build_svm_content(train, dict_, ratios, ngram)
        )
    with open(outfn + '-test.txt', 'w') as file_:
        file_.writelines(
            build_svm_content(test, dict_, ratios, ngram)
        )

    print("SVM model has been generated.")
