import re
import io
import plac
from collections import Counter

import numpy as np

from ngrams import NGRAM_SEPARATOR

DATA_PATH = "tinyshakespeare.txt"

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

MAX_NGRAM_ORDER = 5
SKIP = 1


def loop_data(data_path=DATA_PATH):
    with io.open(data_path, 'r', encoding='utf8') as f:
        return f.read()
        # for text in f:
            # yield text.strip()


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
    for c in char_tokenize(text):
        counts[c] += 1

    vocab = {k: idx for idx, (k, v) in enumerate(counts.most_common())}
    rev_vocab = {v: k for k, v in vocab.iteritems()}
    return vocab, rev_vocab


def vectorize(text, vocab):
    ids = [vocab[c] for c in char_tokenize(text)]
    return np.array(ids)


def batchify(text, vocab, batch_size, n_steps):
    n_plus_one = n_steps + 1
    N = len(text)
    for i in xrange(0, N, batch_size * n_plus_one):
        partial_text = vectorize(text[i : i + batch_size * n_plus_one], vocab)
        if len(partial_text) != batch_size * n_plus_one:
            continue # ignore for now
        partial_text = np.reshape(partial_text, [-1, n_plus_one])
        input_ids = partial_text[:, :n_steps]
        target_ids = partial_text[:, 1:n_plus_one]
        yield input_ids, target_ids


def test():
    batch_size = 32
    n_steps = 10

    text = loop_data()
    vocab, rev_vocab = build_vocabulary(text)
    print len(vocab)
    print sorted(vocab.iteritems(), key=lambda x: x[1])[:]

    for x, y in batchify(text, vocab, batch_size, n_steps):
        # for x_, y_ in zip(x, y):
            # print x_, y_
            # print ''.join([rev_vocab[i] for i in x_])
            # print ''.join([rev_vocab[i] for i in y_])
        print x.shape, y.shape
        # break


if __name__ == '__main__':
    test()
