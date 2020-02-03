import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import tensorflow.contrib as tf_contrib     # noqa
from tensorflow.contrib import rnn          # noqa
from CompSplit.utils import *               # noqa
import time                                 # noqa


LOG_DIR = 'Logs'
PARAMS_DIR = 'Params'
PARAMS = 'params'
TRAIN_LOG = 'Train_log.txt'


class Arguments:
    """
    Network hyper-parameter arguments set
    """
    def __init__(self,
                 cell_type,
                 num_hiddens,
                 output_drop,
                 state_drop,
                 embed_dim):
        """
        :param cell_type: RNN cell type
        :param num_hiddens: The number of hidden units of RNN cell, length of list means the number of layers
        :param output_drop: Whether to use RNN cell output drop out or not
        :param state_drop: Whether to use RNN cell state drop out or not
        :param embed_dim: Character embedding dimension
        """
        self.cell_type = cell_type
        self.num_hiddens = num_hiddens
        self.output_drop = output_drop
        self.state_drop = state_drop
        self.embed_dim = embed_dim

    def print(self):
        s = """
             ==========Arguments==========
             cell_type: %s
             num_hidden: %s
             output_drop: %s
             state_drop: %s
             embed_dim: %s
             =============================
             \n\r""" % (self.cell_type,
                        self.num_hiddens,
                        self.output_drop,
                        self.state_drop,
                        self.embed_dim)
        return s


class BiLstmCrf:
    def __init__(self, name, args):
        self.log_dir = os.path.join(LOG_DIR, name)
        self.param_dir = os.path.join(PARAMS_DIR, name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)

        self.hyper_params = args

        self.input_dim = len(ALPHABET)
        self.output_dim = len([0, 1])

        # [batch size, sequence length]
        self.x = tf.placeholder(tf.int32, [None, None], name='x')
        self.y = tf.placeholder(tf.int32, [None, None], name='y')
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        self.global_step = tf.get_variable('global_step', [],
                                           dtype=tf.int32,
                                           initializer=tf.initializers.constant(0),
                                           trainable=False)

        self.keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')
        self.lr = tf.placeholder(tf.float32, [], 'learning_rate')

        self.embed_inp, self.embed_matrix = self.embedding(self.x)
        self.logit = self.bi_lstm(self.embed_inp)
        self.prediction, self.loss = self.crf(self.logit)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))
        self._loss, self._loss_op = tf.metrics.mean(self.loss, name='Loss_mean')
        self._acc, self._acc_op = tf.metrics.mean(self.acc, name='Accuracy_mean')

        self.train_merged, self.test_merged = self._summaries()

        self.train_op = self.optimizer(self.loss)

        self.saver = tf.train.Saver()
        self.get_trainable_var = tf.trainable_variables()
        self.global_var_init = tf.global_variables_initializer()
        self.local_var_init = tf.local_variables_initializer()

        return

    def embedding(self, inp, scope='Embedding'):
        with tf.variable_scope(scope):
            embed_init = tf_contrib.layers.xavier_initializer()
            embedding = tf.get_variable('embedding_matrix',
                                        shape=[self.input_dim, self.hyper_params.embed_dim],
                                        dtype=tf.float32,
                                        initializer=embed_init,
                                        trainable=True)
            embed_x = tf.nn.embedding_lookup(embedding, inp)
        return embed_x, embedding

    def bi_lstm(self, inp: tf.Tensor, scope='BiLSTM'):
        with tf.variable_scope(scope):
            cells_fw = rnn_cells(self.hyper_params.cell_type,
                                 self.hyper_params.num_hiddens,
                                 self.keep_prob,
                                 self.hyper_params.output_drop,
                                 self.hyper_params.state_drop)

            cells_bw = rnn_cells(self.hyper_params.cell_type,
                                 self.hyper_params.num_hiddens,
                                 self.keep_prob,
                                 self.hyper_params.output_drop,
                                 self.hyper_params.state_drop)

            outputs, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                cells_bw=cells_bw,
                                                                                inputs=inp,
                                                                                sequence_length=self.seq_len,
                                                                                dtype=tf.float32)
            logit = tf.layers.dense(outputs, self.output_dim, activation=None)
        return logit

    def crf(self, logit: tf.Tensor, scope='CRF'):
        with tf.variable_scope(scope):
            log_likelihood, trans_params = tf_contrib.crf.crf_log_likelihood(logit, self.y, self.seq_len)
            loss = tf.reduce_mean(-log_likelihood)

            pred, scores = tf_contrib.crf.crf_decode(logit, trans_params, self.seq_len)
        return pred, loss

    def _summaries(self):
        test_summaries = list()
        train_summaries = list()

        with tf.name_scope('general'):
            summary = tf.summary.scalar('loss', self._loss)
            train_summaries.append(summary)
            test_summaries.append(summary)
            summary = tf.summary.scalar('accuracy', self._acc)
            train_summaries.append(summary)
            test_summaries.append(summary)
            train_summaries.append(tf.summary.scalar('learning rate', self.lr))

        return tf.summary.merge(train_summaries), tf.summary.merge(test_summaries)

    def optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grad_and_vars = optimizer.compute_gradients(loss)
        return optimizer.apply_gradients(grad_and_vars, global_step=self.global_step)

    def init_model(self, sess, ckpt=None):
        if ckpt is not None:
            self.saver.restore(sess, os.path.join(self.param_dir, ckpt, PARAMS))
            print_write('model loaded from file: %s\n' % os.path.join(self.log_dir, ckpt),
                        os.path.join(self.log_dir, TRAIN_LOG), 'a')
        else:
            f = open(os.path.join(self.log_dir, TRAIN_LOG), 'w')
            sess.run(self.global_var_init)
            print_write('global_variables_initialize\n', f)
            writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), filename_suffix='-graph')
            writer.add_graph(sess.graph)

            print_write('============================================================\n', f)
            count_vars = 0
            for var in self.get_trainable_var:
                name = var.name
                shape = var.shape.as_list()
                num_elements = var.shape.num_elements()
                print_write('Variable name: %s\n' % name, f)
                print_write('Placed device: %s\n' % var.device, f)
                print_write('Shape : %s  Elements: %d\n' % (str(shape), num_elements), f)
                print_write('============================================================\n', f)
                count_vars = count_vars + num_elements
            print_write('Total number of trainalbe variables %d\n' % count_vars, f)
            print_write('============================================================\n', f)
            f.close()
            writer.close()
        return

    def train(self, sess, train_data, train_step, lr, batch_size, keep_prob, ckpt=None, summary_step=100):
        self.init_model(sess, ckpt)
        global_step = sess.run(self.global_step)
        base_step = global_step

        s = time.time()
        for i in range(train_step // (summary_step * 10)):
            sess.run(self.local_var_init)
            writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'),
                                           filename_suffix='-step-%d' % global_step)
            for j in range(summary_step * 10):
                x, y, seq_len = train_data.random_batch(batch_size)
                fetch = [self.train_op, self._loss_op, self._acc_op]
                feed = {self.x: x, self.y: y, self.seq_len: seq_len, self.lr: lr, self.keep_prob: keep_prob}
                _, loss, acc = sess.run(fetch, feed_dict=feed)
                global_step = sess.run(self.global_step)
                print('\rTraining - Loss: %0.3f, Accuracy: %0.3f, step: %d/%d'
                      % (loss, acc, global_step, base_step+train_step), end='')

                if global_step % summary_step == 0 and global_step != 0:
                    fetch = [self.train_merged, self._loss, self._acc]
                    merged, loss, acc = sess.run(fetch, feed_dict={self.lr: lr, self.keep_prob: keep_prob})
                    writer.add_summary(merged, global_step)
                    print('\r', end='')
                    print_write('Training - Loss: %0.3f, Accuracy: %0.3f step: %d/%d  %0.3f sec/step\n'
                                % (loss, acc, global_step, base_step+train_step, (time.time() - s) / summary_step),
                                os.path.join(self.log_dir, TRAIN_LOG), 'a')
                    s = time.time()
                    sess.run(self.local_var_init)

            print_write('global step: %d, model save %s, time: %s\n'
                        % (global_step,
                           os.path.join(self.param_dir, 'step-%d' % global_step, PARAMS),
                           time.strftime('%y-%m-%d %H:%M:%S')),
                        os.path.join(self.log_dir, TRAIN_LOG), 'a')
            writer.close()

            self.saver.save(sess, os.path.join(self.param_dir, 'step-%d' % global_step, PARAMS))
            s = time.time()
        return

    def runs(self, sess, test_data, ckpt, batch_size, csv_name='result.csv'):
        s = time.time()
        self.init_model(sess, ckpt)
        predictions = []
        num_data = test_data.num_data
        end = False
        while not end:
            x, seq_len, end = test_data.sequential_batch(batch_size)
            pred_idx = sess.run(self.prediction,
                                feed_dict={self.x: x, self.seq_len: seq_len, self.keep_prob: 1.0})

            if len(x) != len(pred_idx):
                print('len(file_names) %d != len(pred_idx) %d' % (len(x), len(pred_idx)))
                return -1

            print('\rTest - %d/%d' % (len(predictions), num_data), end='')

            for x, p in zip(x, pred_idx):
                predictions.append([x, p])

        csv_save(csv_name, predictions)
        print()
        print('total time: %0.3f sec, %0.3f sec/image' % (time.time()-s, (time.time()-s/num_data)))
        return
