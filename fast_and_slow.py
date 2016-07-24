"""
https://github.com/facebook/SCRNNs
http://arxiv.org/abs/1412.7753
"""
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid


class TraceRNNCell(tf.nn.rnn_cell.BasicRNNCell):
  """Exponentially Decaying Trace Recurrent Neural Network"""

  def __init__(self, num_units, alpha=0.95, **kwargs):
    assert 0 < alpha < 1
    self._alpha = alpha
    super(TraceRNNCell, self).__init__(num_units, **kwargs)    

  def __call__(self, inputs, state, scope=None):  
    with vs.variable_scope(scope or type(self).__name__):  # "TraceRNNCell"
      new_state = tf.nn.rnn_cell._linear([inputs], self._num_units, True)
      output = (1 - self._alpha) * new_state + self._alpha * state  # no weights on state, and no non-tf.nn.rnn_cell._linearity
    return output, output


class SCRNNCell(tf.nn.rnn_cell.BasicLSTMCell):
  """Structurally Constrained Recurrent Neural Network"""

  def __init__(self, num_units, alpha=0.95, **kwargs):
    assert 0 < alpha < 1
    self._alpha = alpha
    super(SCRNNCell, self).__init__(num_units, **kwargs)

  def __call__(self, inputs, state, scope=None):
    with vs.variable_scope(scope or type(self).__name__):  # "SCRNNCell"

      if self._state_is_tuple:
        s, h = state
      else:
        s, h = array_ops.split(1, 2, state)
      
      new_s = tf.nn.rnn_cell._linear([(1 - self._alpha) * inputs, self._alpha * s], self._num_units, True, scope="SlowLinear")  
      new_h = sigmoid(tf.nn.rnn_cell._linear([inputs, new_s, h], self._num_units, True, scope="FastLinear"))

      if self._state_is_tuple:
        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_s, new_h)
      else:
        new_state = array_ops.concat(1, [new_s, new_h])

      return new_h, new_state


if __name__ == '__main__':
    pass
