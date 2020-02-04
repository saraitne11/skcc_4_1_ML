import csv
import random
import numpy as np
import tensorflow as tf

filePath = '../_data/ml_6_spacing_train.csv'

ALPHABET = {c: i for i, c in enumerate('Pabcdefghijklmnopqrstuvwxyz')}
PAD = 'P'

def word2idx(word, max_len, dtype=np.uint8):
    word = word + PAD * (max_len-len(word))
    return np.array(list(map(lambda c: ALPHABET[c], word)), dtype=dtype)


def readData(fileName):
    f = open(fileName, 'r', encoding='utf-8')
    lines = list(csv.reader(f))
    f.close()

    max_len = 0
    compounds = [i[0] for i in lines[1:]]
    for word in compounds:
        if max_len < len(word):
            max_len = len(word)

    flag = True
    if len(lines[0]) == 1:
        flag = False

    alpha_indices = np.ones((len(compounds), max_len), dtype=np.uint8)
    split_indices = np.zeros((len(compounds), max_len), dtype=np.uint8)
    word_compound = []

    index = 0
    for line in lines[1:]:
        word_compound.append([line[0], []])
        alpha_indices[index] = word2idx(line[0], max_len)
        if flag:
            inner_index = 0
            for data in line[1:]:
                if len(data) == 0:
                    break
                else:
                    split_indices[index][inner_index] = 1
                    word_compound[index][1].append(data)
                    inner_index += len(data)
                    split_indices[index][inner_index - 1] = 2
        index += 1
    return alpha_indices, split_indices, word_compound


class DataSet:
    def __init__(self, train_path, test_path):
        # self.train_path = '../_data/ml_6_spacing_train.csv'
        # self.test_path = '../_data/ml_6_spacing_test.csv'
        self.train_path = train_path
        self.test_path = test_path

        self.x, self.y, self.compound = readData(self.train_path)
        self.seq_len = np.array([len(i[0]) for i in self.compound])
        self.num_data = len(self.x)

        self.test_x, self.test_y, self.test_compound = readData(self.test_path)
        self.test_seq_len = np.array([len(i[0]) for i in self.test_compound])
        self.test_num_data = len(self.test_x)

        self.sequential_index = 0

    def random_batch(self, batch_size):
        choose = random.sample(list(range(0, self.num_data)), batch_size)
        x = self.x[choose]
        y = self.y[choose]
        seq_len = self.seq_len[choose]
        return x, y, seq_len

    def sequential_batch(self, batch_size):
        x = self.test_x[self.sequential_index: self.sequential_index + batch_size]
        seq_len = self.test_seq_len[self.sequential_index: self.sequential_index + batch_size]
        self.sequential_index += batch_size
        if self.sequential_index >= self.test_num_data:
            return x, seq_len, True
        return x, seq_len, False


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


def idx2chr(index):
    return chr(ord('a') + index - 1)


def idx2word(indices):
    word = ''
    for char in indices:
        word += idx2chr(char)
    return word


def expectation_to_compound(word, location):
    word = np.array(word)
    length = np.where(word == 0)[0]
    if len(length) == 0:
        length = len(word)
    else:
        length = length[0]
    compound = idx2word(word[:length])
    compounds = ['', '', '']

    location = np.array(location)
    indices = np.where(location == 1)[0]
    idx = 0
    iidx = 0
    for index in indices[1:]:
        compounds[iidx] = (compound[idx:index])
        idx = index
        iidx += 1
    compounds[iidx] = (compound[idx:length])

    return compound, compounds


def csv_save(csv_name, predictions):
    f = open(csv_name, 'w')
    f.write('compound,n1,n2,n3\n')
    for x, p in predictions:
        comp, comps = expectation_to_compound(x, p)
        f.write('%s,' % comp)
        for w in comps[:-1]:
            f.write('%s,' % w)
        f.write('%s' % comps[-1])
        f.write('\n')


if __name__ == '__main__':
    dataSet = DataSet('../_Data/ml_6_spacing_train.csv', '../_Data/ml_6_spacing_test.csv')
    print(dataSet.random_batch(16))

