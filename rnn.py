from __future__ import division

import numpy as np
import tensorflow as tf

BATCH_SIZE = 16
N_STEPS = 40
EMBEDDINGS_SIZE = 50
RNN_HIDDEN_SIZE = 32


class RNNModel(object):
    def __init__(self, vocab_size, 
        n_steps=N_STEPS, embeddings_size=EMBEDDINGS_SIZE, rnn_hidden_size=RNN_HIDDEN_SIZE):

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
            self.cell = tf.nn.rnn_cell.BasicRNNCell(rnn_hidden_size)
            self.softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, vocab_size])
            self.softmax_b = tf.get_variable("softmax_b", [vocab_size])

            # [batch_size, n_steps, rnn_hidden_size]
            self._outputs, _ = tf.nn.dynamic_rnn(self.cell, self.inputs,
                time_major=False, dtype=tf.float32)

        """ set up loss """
        # [n_steps, batch_size, rnn_hidden_size]
        # outputs = tf.transpose(self._outputs, [1, 0, 2])
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

        self.lr = 0.03 # tf.Variable(0.0, trainable=False)
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                # 5.0)
        # grads = tf.gradients(self.cost, tvars)
        # self.optimizer = tf.train.AdamOptimizer(self.lr)
        # self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        # grads_and_vars = self.optimizer.compute_gradients(self.cost)
        # self.train_op = self.optimizer.apply_gradients(grads_and_vars)
        # print "kdfjkdfjdkfj", type(grads)

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.cost)

    def step(self, session, input_ids, target_ids, verbose=False):
        feed_dict = {
            self.input_ids: input_ids,
            self.target_ids: target_ids,
        }
        if verbose:
            cost, loss, probs, inputs, _outputs, outputs, targets, logits, softmax_w, softmax_b = session.run([self.cost, self.loss, self.probs, self.inputs, self._outputs, self.outputs, self.targets, self.logits, self.softmax_w, self.softmax_b], feed_dict=feed_dict)
            print 'cost', cost
            print 'loss', loss.shape
            print 'probs', probs.shape
            print 'inputs', inputs.shape
            print '_outputs', _outputs.shape
            print 'outputs', outputs.shape
            print 'targets', targets.shape
            print 'logits', logits.shape        
            print 'softmax_w', softmax_w.shape
            print 'softmax_b', softmax_b.shape
        else:
            cost, _ = session.run([self.cost, self.train_op], feed_dict=feed_dict)
        return cost


def test():
    from data import loop_data, build_vocabulary, batchify

    batch_size = 16
    n_steps = 50

    text = loop_data()
    vocab, rev_vocab = build_vocabulary(text)
    vocab_size = len(vocab)
    print vocab_size

    model = RNNModel(vocab_size, n_steps=n_steps)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for epoch in range(10):
            print "epoch", epoch
            list_of_costs = []
            for idx, (x, y) in enumerate(batchify(text, vocab, batch_size, n_steps)):
                list_of_costs.append(model.step(sess, x, y))
                if idx % 100 == 0:
                    print "cost", np.mean(list_of_costs)
                    # print "x", ''.join([rev_vocab[i] for i in x[0]])
                    # print "y", ''.join([rev_vocab[i] for i in y[0]])
                    list_of_costs = []


if __name__ == '__main__':
    test()


