import tensorflow as tf


def rnn_cell(cell, hidden, keep_prob, output_drop, state_drop):
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


def rnn_cells(cell_type,
              hidden_list,
              keep_prob,
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


def print_write(s, file, mode=None):
    if isinstance(file, str):
        if mode is None:
            mode = 'a'
        f = open(file, mode)
        print(s, end='')
        f.write(s)
        f.close()
    else:
        print(s, end='')
        file.write(s)
