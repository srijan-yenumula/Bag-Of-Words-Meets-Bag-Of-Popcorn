import re
from random import shuffle

import pandas as pd
from bs4 import BeautifulSoup
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords

with open('data/negator.txt', 'r') as file_:
    NEGATORS = [line.strip() for line in file_]


def review_to_wordlist(review, remove_stopwords=False):
    """Convert a document to a sequence of words.

    Return the sequence as a list."""

    # Remove HTML
    review_text = BeautifulSoup(review, 'lxml').get_text()

    # Extract negators
    review_text = re.sub('n\'t', ' not', review_text)

    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # Convert words to lower case and split into list
    words = review_text.lower().split()

    # Optionally remove stop words except for negators (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [
            w for w in words if not w in stops or w in NEGATORS]

    # Return a list of words
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    """Split a review into parsed sentences.

    Return a list of sentences."""
    # Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    # Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(
                raw_sentence,
                remove_stopwords)
            )

    # Return the list of sentences
    return sentences


def get_clean_labeled_reviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(review_to_wordlist(review))

    labelized = []
    for i, id_label in enumerate(reviews["id"]):
        labelized.append(LabeledSentence(clean_reviews[i], [id_label]))
    return labelized
