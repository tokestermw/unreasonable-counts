import re
import io
import plac
import pdb
from collections import Counter

import numpy as np

from ngrams import NGRAM_SEPARATOR

DATA_PATH = "tinyshakespeare.txt"

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

MAX_NGRAM_ORDER = 5
SKIP = 1

OOV_ID = 0  # padding should be 0

class DataConfig:
    token_type = "char"  # [char, word]


def loop_data(data_path=DATA_PATH, p=0.01):
    with io.open(data_path, 'r', encoding='utf8') as f:
        entire_text = f.read()
        length_text = len(entire_text)
        split = int((1 - p) * length_text)
        train_text = entire_text[:split]
        valid_text = entire_text[split:]

        if DataConfig.token_type == "char":
            return train_text, valid_text
        # elif DataConfig.token_type == "word":
            # return f.readlines()


def normalize(text):
    return text.lower()


def char_tokenize(text):
    return list(normalize(text))


def word_tokenize(sentence):
    words = []
    for space_separated_fragment in normalize(sentence).split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def ignore(text):
    if len(text) == 0:
        return True
    if text.endswith(':'):
        return True
    return False


def build_vocabulary(text):
    counts = Counter()
    # for text in data:
    if DataConfig.token_type == "char":
        tokenize = char_tokenize
    elif DataConfig.token_type == "word":
        tokenize = word_tokenize

    for t in tokenize(text):
        counts[t] += 1

    vocab = {k: idx + 1 for idx, (k, v) in enumerate(counts.most_common())}
    vocab["OOV"] = OOV_ID
    rev_vocab = {v: k for k, v in vocab.iteritems()}
    return vocab, rev_vocab


def vectorize(text, vocab):
    if DataConfig.token_type == "char":
        tokenize = char_tokenize
    elif DataConfig.token_type == "word":
        tokenize = word_tokenize
    ids = [vocab.get(t, "OOV") for t in tokenize(text)]
    return np.array(ids)


def batchify(text, vocab, batch_size, n_steps):
    n_plus_one = n_steps + 1
    N = len(text)
    skip = 10
    for i in xrange(0, N, batch_size * n_plus_one):
        if i % skip != 0:
            continue
        partial_text = vectorize(text[i : i + batch_size * n_plus_one], vocab)
        if len(partial_text) != batch_size * n_plus_one:
            continue # ignore for now
        partial_text = np.reshape(partial_text, [-1, n_plus_one])
        input_ids = partial_text[:, :n_steps]
        target_ids = partial_text[:, 1:n_plus_one]
        yield input_ids, target_ids


def test(verbose=True):
    batch_size = 32
    n_steps = 20

    train_text, valid_text = loop_data()
    vocab, rev_vocab = build_vocabulary(train_text)
    print len(vocab)
    print sorted(vocab.iteritems(), key=lambda x: x[1])[:50]

    for epoch in range(5):
        print "epoch", epoch
        for x, y in batchify(train_text, vocab, batch_size, n_steps):
            if verbose:
                print 'x', ''.join(map(lambda c: rev_vocab[c], x[0]))
                print 'y', ''.join(map(lambda c: rev_vocab[c], y[0]))

                print x.shape, y.shape
        # break


if __name__ == '__main__':
    test(verbose=True)
