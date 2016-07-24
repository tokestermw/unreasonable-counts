from __future__ import division

import plac
from tqdm import tqdm

import numpy as np
import tensorflow as tf

BATCH_SIZE = 16
N_STEPS = 40
EMBEDDINGS_SIZE = 50
RNN_HIDDEN_SIZE = 32


class RNNModel(object):
    def __init__(self, vocab_size, rnn_type="rnn",
                 n_steps=N_STEPS, embeddings_size=EMBEDDINGS_SIZE, rnn_hidden_size=RNN_HIDDEN_SIZE):
        """ RNN Model object """

        """ The char or word IDs for input and target. """
        self.input_ids = tf.placeholder(tf.int32, [None, n_steps])
        self.target_ids = tf.placeholder(tf.int32, [None, n_steps])

        batch_size = tf.shape(self.input_ids)[0]

        """ set up embeddings """
        with tf.device("/cpu:0"):
            self.embeddings = tf.get_variable("embeddings", [vocab_size, embeddings_size])
            # [batch_size, n_steps, embeddings_size]
            self.inputs = tf.nn.embedding_lookup(self.embeddings, self.input_ids)

        """ set up RNN cells """
        with tf.variable_scope("rnn_model"):
            from fast_and_slow import TraceRNNCell, SCRNNCell

            if rnn_type == "rnn":
                self.cell = tf.nn.rnn_cell.BasicRNNCell(rnn_hidden_size)
            elif rnn_type == "lstm":
                self.cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
            elif rnn_type == "trace":
                self.cell = TraceRNNCell(rnn_hidden_size)
            elif rnn_type == "scrnn":
                self.cell = SCRNNCell(rnn_hidden_size, state_is_tuple=True)

            self.softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, vocab_size])
            self.softmax_b = tf.get_variable("softmax_b", [vocab_size])

            # [batch_size, n_steps, rnn_hidden_size]
            self._outputs, _ = tf.nn.dynamic_rnn(self.cell, self.inputs,
                time_major=False, dtype=tf.float32)

        """ set up loss """
        # [n_steps * batch_size, rnn_hidden_size]
        self.outputs = tf.reshape(self._outputs, [-1, rnn_hidden_size])
        # [n_steps * batch_size, vocab_size]
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        # [n_steps * batch_size, vocab_size]
        self.probs = tf.nn.softmax(self.logits)

        # [n_steps * batch_size]
        self.targets = tf.reshape(self.target_ids, [-1])
        # [n_steps * batch_size]
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.targets)
        # [1]
        self.cost = tf.reduce_sum(self.loss) / tf.to_float(batch_size) / tf.to_float(n_steps)

        self.lr = 0.01 # tf.Variable(0.0, trainable=False)
        
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.cost)

    def step(self, session, input_ids, target_ids, verbose=False):
        feed_dict = {
            self.input_ids: input_ids,
            self.target_ids: target_ids,
        }
        cost, _ = session.run([self.cost, self.train_op], feed_dict=feed_dict)
        return cost

    def decode(self, session, inputs, temperature=1.0):
        pass


@plac.annotations(
    rnn_type=("RNN type."),
)
def test(rnn_type="rnn"):
    from data import loop_data, build_vocabulary, batchify

    batch_size = 32
    n_steps = 20

    text = loop_data()
    vocab, rev_vocab = build_vocabulary(text)
    vocab_size = len(vocab)
    print vocab_size

    model = RNNModel(vocab_size, n_steps=n_steps, rnn_type=rnn_type)

    # TODO: sample decoded sentence
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for epoch in range(5):
            print "epoch", epoch
            list_of_costs = []
            for idx, (x, y) in tqdm(enumerate(batchify(text, vocab, batch_size, n_steps))):
                list_of_costs.append(model.step(sess, x, y))
                if idx % 100 == 0:
                    print "cost", np.mean(list_of_costs)
                    list_of_costs = []


if __name__ == '__main__':
    plac.call(test)


