import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from typing import List, Union, Optional


def rnn_cell(cell: str,
             hidden: int,
             keep_prob: Union[float, tf.Tensor, tf.Variable],
             output_drop: bool,
             state_drop: bool):
    if cell == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(hidden, state_is_tuple=True)
    elif cell == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(hidden)
    elif cell == 'RNN':
        cell = tf.nn.rnn_cell.RNNCell(hidden)
    else:
        raise TypeError('cell name error')

    if output_drop and state_drop:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             output_keep_prob=keep_prob,
                                             state_keep_prob=keep_prob,
                                             variational_recurrent=True,
                                             dtype=tf.float32)
    elif output_drop:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             output_keep_prob=keep_prob,
                                             dtype=tf.float32)
    elif state_drop:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             state_keep_prob=keep_prob,
                                             variational_recurrent=True,
                                             dtype=tf.float32)
    else:
        pass
    return cell


def rnn_cells(cell_type: str,
              hidden_list: List[int],
              keep_prob: Union[float, tf.Tensor, tf.Variable],
              output_drop=True,
              state_drop=True):
    multi_cell = []
    for hidden in hidden_list:
        cell = rnn_cell(cell_type, hidden, keep_prob, output_drop, state_drop)
        multi_cell.append(cell)
    return multi_cell


def get_tf_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    # config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def clip_gradients(grads_and_vars, clip_norm=5.0):
    """
    :param grads_and_vars: A list of (gradients and variables) pairs
    :param clip_norm: clip norm
    :return: A list of (clipped gradients and variables) pairs
    """
    # gvs = [('g1', 'v1'), ('g2', 'v2'), ('g3', 'v3')]
    # zip(*gvs)
    # gradients = ('g1', 'g2', 'g3')
    # variables = ('v1', 'v2', 'v3')
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
    return list(zip(clipped_gradients, variables))


def merge_gradients(towers_gvs):
    """
    Merge every grads_and_vars of each tower
    :param towers_gvs: [each tower's grads and vars]
    :return: merged grads_and_vars
    """
    average_grads = []
    for grad_and_vars in zip(*towers_gvs):
        # iteration through all grads and vars
        grads = []
        for g, _ in grad_and_vars:
            # iteration through all towers
            ex_g = tf.expand_dims(g, 0)
            grads.append(ex_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        # variables of first tower
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads


def entry_stop_gradients(embedding: tf.Variable, mask):
    mask_n = tf.logical_not(mask)

    mask = tf.cast(mask, dtype=embedding.dtype)
    mask_n = tf.cast(mask_n, dtype=embedding.dtype)

    return tf.stop_gradient(mask_n * embedding) + mask + embedding


class OutputLayer(tf.layers.Layer):
    def __init__(self, num_inputs, embedding_matrix):
        super(OutputLayer, self).__init__()
        self.num_inputs = num_inputs
        self.embedding_matrix = embedding_matrix
        self.shape = embedding_matrix.get_shape()       # (vocab_size, embedding_dim)

    def build(self, _):
        embedding_size = self.embedding_matrix.get_shape()[1]
        self.kernel = self.add_variable("kernel", [self.num_inputs, embedding_size])

    def call(self, inputs, **kwargs):
        # inputs
        # if Greedy, [batch, hidden]
        # if Beam, [batch, beam, hidden]

        # layer hidden
        # if Greedy, [batch, kernel second dim]
        # if Beam, [batch, beam, kernel second dim]
        hidden = tf.tensordot(inputs, self.kernel, axes=((-1,), (0,)))  # Remove 'a' last dim, 'b' first dim

        # layer hidden
        # if Greedy, [batch, vocab size]
        # if Beam, [batch, beam, vocab size]
        return tf.tensordot(hidden, tf.transpose(self.embedding_matrix), axes=((-1,), (0,)))

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.shape[0])
