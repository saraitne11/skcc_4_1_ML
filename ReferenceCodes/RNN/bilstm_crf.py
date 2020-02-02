import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import *                         # noqa
from tf_utils import *                      # noqa
import tensorflow.contrib as tf_contrib     # noqa
from tensorflow.contrib import rnn          # noqa
import time                                 # noqa


class Arguments:
    """
    Network hyper-parameter arguments set
    """
    def __init__(self,
                 cell_type: str,
                 num_hiddens: List[int],
                 output_drop: bool,
                 state_drop: bool,
                 char2id: Dict,
                 char_embed_vectors: Optional[np.ndarray],
                 char_embed_dim: Optional[int]):
        """
        :param cell_type: RNN cell type
        :param num_hiddens: The number of hidden units of RNN cell, length of list means the number of layers
        :param output_drop: Whether to use RNN cell output drop out or not
        :param state_drop: Whether to use RNN cell state drop out or not
        :param char2id: character-to-ID dictionary
        :param char_embed_vectors: Character embedding initialization vectors
        :param char_embed_dim: Character embedding dimension (if None - No use this embedding)
        """
        self.cell_type = cell_type
        self.num_hiddens = num_hiddens
        self.output_drop = output_drop
        self.state_drop = state_drop
        self.char2id = char2id
        self.char_embed_vectors = char_embed_vectors
        self.char_embed_dim = char_embed_dim

    def print(self):
        if self.char_embed_vectors is not None:
            char_embed_vectors_init = 'word2vec'
        else:
            char_embed_vectors_init = 'random'
        s = """
             ==========Arguments==========
             cell_type: %s
             num_hidden: %s
             output_drop: %s
             state_drop: %s
             char2id: %s 
                %s
             char_embed_vectors: %s
             char_embed_dim: %s
             =============================
             \n\r""" % (self.cell_type,
                        self.num_hiddens,
                        self.output_drop,
                        self.state_drop,
                        len(self.char2id),
                        random.sample(list(self.char2id.items()), 10),
                        char_embed_vectors_init,
                        self.char_embed_dim)
        return s


class BiLstmCrf:
    """
    Model Description
    Bidirectional LSTM-CRF Named Entity Recognition Model
    """
    def __init__(self,
                 name: str,
                 args: Optional[Arguments] = None,
                 config_rewrite=False):
        self.log_dir = os.path.join(LOG_DIR, name)
        self.param_dir = os.path.join(PARAMS_DIR, name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)

        if args is None:
            args = pickle_load(os.path.join(self.log_dir, CONFIG_PICKLE))

        if not os.path.exists(os.path.join(self.log_dir, CONFIG_PICKLE)) or config_rewrite:
            pickle_store(args, os.path.join(self.log_dir, CONFIG_PICKLE))
            with open(os.path.join(self.log_dir, CONFIG_TXT), 'w') as f:
                print_write(self.__doc__, f)
                print_write(args.__init__.__doc__, f)
                print_write(args.print(), f)

        self.hyper_params = args

        self.char2id = self.hyper_params.char2id
        self.id2char = dict_swap(self.hyper_params.char2id)

        self.vocab_size = len(self.hyper_params.char2id)
        self.out_dim = len(BIO2IDX)

        # [batch size, sequence length]
        self._placeholders()

        self.embed_inp, self.embed_matrix = self.embedding(self.x, scope='ChineseCharEmb')

        self.logit = self.bi_lstm(self.embed_inp)
        self.pred, self.loss = self.crf(self.logit)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))
        self._loss, self._loss_op = tf.metrics.mean(self.loss, name='Loss_mean')
        self._acc, self._acc_op = tf.metrics.mean(self.acc, name='Accuracy_mean')

        self.train_merged, self.test_merged = self._summaries()

        self.train_op = self.optimizer(self.loss)

        self.saver = tf.train.Saver()
        self.get_trainable_var = tf.trainable_variables()
        self.global_var_init = tf.global_variables_initializer()
        self.local_var_init = tf.local_variables_initializer()
        return

    def _placeholders(self):
        # [batch size, sequence length]
        self.x = tf.placeholder(tf.int32, [None, None], name='x')
        self.y = tf.placeholder(tf.int32, [None, None], name='y')
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        self.precision_entity = tf.placeholder(tf.float32, [])
        self.recall_entity = tf.placeholder(tf.float32, [])
        self.f1_score_entity = tf.placeholder(tf.float32, [])

        self.precision_surface = tf.placeholder(tf.float32, [])
        self.recall_surface = tf.placeholder(tf.float32, [])
        self.f1_score_surface = tf.placeholder(tf.float32, [])

        self.global_step = tf.get_variable('global_step', [],
                                           dtype=tf.int32,
                                           initializer=tf.initializers.constant(0),
                                           trainable=False)

        self.keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')

        self.lr = tf.get_variable('learning_rate', [],
                                  dtype=tf.float32,
                                  initializer=tf.initializers.zeros(),
                                  trainable=False)
        self.lr_placeholder = tf.placeholder(tf.float32)
        self.lr_update = tf.assign(self.lr, self.lr_placeholder)
        return

    def embedding(self, inp, scope='ChineseCharEmb') -> Tuple[tf.Tensor, tf.Variable]:
        with tf.variable_scope(scope):
            if self.hyper_params.char_embed_vectors is not None:
                embed_init = tf.constant_initializer(self.hyper_params.char_embed_vectors, verify_shape=True)
            else:
                embed_init = tf_contrib.layers.xavier_initializer()
            embedding = tf.get_variable('embedding_matrix',
                                        shape=[self.vocab_size, self.hyper_params.char_embed_dim],
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
            logit = tf.layers.dense(outputs, self.out_dim, activation=None)
        return logit

    def crf(self, logit: tf.Tensor, scope='CRF'):
        with tf.variable_scope(scope):
            log_likelihood, trans_params = tf_contrib.crf.crf_log_likelihood(logit, self.y, self.seq_len)
            loss = tf.reduce_mean(-log_likelihood)

            pred, scores = tf_contrib.crf.crf_decode(self.logit, trans_params, self.seq_len)
        return pred, loss

    def optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grad_and_vars = optimizer.compute_gradients(loss)
        return optimizer.apply_gradients(grad_and_vars, global_step=self.global_step)

    def _summaries(self):
        test_summaries = list()
        train_summaries = list()

        with tf.name_scope('entity_from'):
            test_summaries.append(tf.summary.scalar('precision', self.precision_entity))
            test_summaries.append(tf.summary.scalar('recall', self.recall_entity))
            test_summaries.append(tf.summary.scalar('f1-score', self.f1_score_entity))

        with tf.name_scope('surface_from'):
            test_summaries.append(tf.summary.scalar('precision', self.precision_surface))
            test_summaries.append(tf.summary.scalar('recall', self.recall_surface))
            test_summaries.append(tf.summary.scalar('f1-score', self.f1_score_surface))

        with tf.name_scope('general'):
            summary = tf.summary.scalar('loss', self._loss)
            train_summaries.append(summary)
            test_summaries.append(summary)
            summary = tf.summary.scalar('accuracy', self._acc)
            train_summaries.append(summary)
            test_summaries.append(summary)
            train_summaries.append(tf.summary.scalar('learning rate', self.lr))

        return tf.summary.merge(train_summaries), tf.summary.merge(test_summaries)

    def init_model(self, sess, log_file=None, ckpt=None, variables_rewrite=False):
        if ckpt:
            self.saver.restore(sess, os.path.join(self.param_dir, ckpt, PARAMS))
            if log_file:
                print_write('model loaded from file: %s\n' % os.path.join(self.log_dir, ckpt),
                            os.path.join(self.log_dir, log_file), 'a')
        else:
            sess.run(self.global_var_init)
            if log_file:
                with open(os.path.join(self.log_dir, log_file), 'w') as f:
                    print_write('global variable initialize\n', f)

            writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), filename_suffix='-graph')
            writer.add_graph(sess.graph)
            writer.close()

        if not ckpt or variables_rewrite:
            with open(os.path.join(self.log_dir, CONFIG_TXT), 'a') as f:
                # print_write('input embed: %s\n' % self.enc_emb_inp, f)
                print_write('===============================================================================\n', f)
                print_write('Input: %s\n' % self.embed_inp.__str__(), f)
                print_write('===============================================================================\n', f)
                count_vars = 0
                for var in self.get_trainable_var:
                    name = var.name
                    shape = var.shape.as_list()
                    num_elements = var.shape.num_elements()
                    print_write('Variable name: %s\n' % name, f)
                    print_write('Placed device: %s\n' % var.device, f)
                    print_write('Shape : %s  Elements: %d\n' % (str(shape), num_elements), f)
                    print_write('===============================================================================\n', f)
                    count_vars = count_vars + num_elements
                print_write('Total number of trainable variables %d\n' % count_vars, f)
                print_write('===============================================================================\n', f)
        return

    def build_dataset(self, data_list: List[ParaData], max_seq_len: int, train_phase: bool):
        dataset = DataSetNER(max_seq_len=max_seq_len,
                             char2id=self.hyper_params.char2id)
        dataset.make(paras=data_list,
                     train_phase=train_phase)
        return dataset

    def runs(self, sess, sentences, batch_size):
        # sentences_id = list(map(lambda x: seq2ids(x, self.hyper_params.char2id), sentences))
        seq_len = np.array(list(map(lambda x: len(x), sentences)), dtype=np.uint16)
        sentences_id = np.zeros(shape=[batch_size, np.max(seq_len)], dtype=np.uint16)
        for i in range(batch_size):
            ids = seq2ids(sentences[i], self.hyper_params.char2id)
            sentences_id[i, :seq_len[i]] = ids

        feed = {self.x: sentences_id, self.seq_len: seq_len, self.keep_prob: 1.0}
        predictions = sess.run(self.pred, feed_dict=feed)
        # print(sentences_id.shape)
        # print(predictions.shape)
        pred_ne = []
        for i in range(batch_size):
            pred_ne.append(pred2ne(''.join(sentences[i]), predictions[i, :seq_len[i]]))
        return pred_ne

    def train(self,
              sess: tf.Session,
              train_step: int,
              lr: float,
              train_data: List[ParaData],
              max_seq_len: int,
              test_data: Optional[List[ParaData]] = None,
              batch_size=256,
              keep_prob=0.7,
              ckpt=None,
              summary_step=1000,
              variable_rewrite=False):
        assert (train_step % summary_step) == 0
        self.init_model(sess, TRAIN_LOG, ckpt, variable_rewrite)
        dataset = self.build_dataset(train_data, max_seq_len, train_phase=True)

        global_step = sess.run(self.global_step)
        base_step = global_step
        sess.run([self.local_var_init, self.lr_update], feed_dict={self.lr_placeholder: lr})

        self.write_configure(dataset, TRAIN_LOG,
                             global_step=base_step,
                             batch_size=batch_size,
                             train_phase=True,
                             train_step=train_step,
                             lr=sess.run(self.lr))

        s = time.time()
        for i in range(train_step // (summary_step * 10)):
            writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'),
                                           filename_suffix='-step-%d' % global_step)
            for j in range(summary_step * 10):
                feed = self.get_train_feed(dataset, batch_size, keep_prob)
                fetch = [self.train_op, self._loss_op, self._acc_op]
                _, loss, acc = sess.run(fetch, feed_dict=feed)
                global_step = sess.run(self.global_step)

                print('\rTraining - Loss: %0.3f, Accuracy: %0.3f, step %d/%d'
                      % (loss, acc, global_step, train_step + base_step), end='')

                if global_step % summary_step == 0:
                    fetch = [self.train_merged, self._loss, self._acc]
                    merged, loss, acc = sess.run(fetch)
                    writer.add_summary(merged, global_step)

                    print('\r', end='')
                    print_write('Training - Loss: %0.3f, Accuracy: %0.3f, step: %d, %0.2f sec/step\n'
                                % (loss, acc, global_step, (time.time() - s) / summary_step),
                                os.path.join(self.log_dir, TRAIN_LOG), 'a')

                    s = time.time()
                    sess.run(self.local_var_init)

            print_write('global step: %d, model save, time: %s\n'
                        % (global_step, time.strftime('%y-%m-%d %H:%M:%S')),
                        os.path.join(self.log_dir, TRAIN_LOG), 'a')
            writer.close()

            self.saver.save(sess, os.path.join(self.param_dir, 'step-%d' % global_step, PARAMS))

            if test_data is not None:
                self.eval(sess, test_data, max_seq_len, batch_size)

            s = time.time()
        return

    def get_train_feed(self, dataset, batch_size, keep_prob):
        x, y, seq_len = dataset.get_train_batch(batch_size)
        feed = {self.x: x,
                self.y: y,
                self.seq_len: seq_len,
                self.keep_prob: keep_prob}
        return feed

    def eval(self,
             sess: tf.Session,
             data_list: List[ParaData],
             max_seq_len: int,
             batch_size: int,
             ckpt=None,
             num_examples=50,
             seed=777):
        eval_start_time = time.time()
        if ckpt:
            test_log = ckpt + '.txt'
            self.init_model(sess, test_log, ckpt)
            global_step = sess.run(self.global_step)
        else:
            global_step = sess.run(self.global_step)
            test_log = 'step-%s.txt' % global_step

        dataset = self.build_dataset(data_list, max_seq_len, train_phase=False)
        sess.run(self.local_var_init)

        src_sent = []
        pred_ne = []
        true_ne = []
        cnt = 0
        dataset.sequential_indices = 0

        self.write_configure(dataset, test_log,
                             global_step=global_step,
                             batch_size=batch_size,
                             train_phase=False,
                             train_step=None,
                             lr=None)
        s = time.time()
        end = False
        while not end:
            feed, tokens, n_data, end = self.get_eval_feed(dataset, batch_size, keep_prob=1.0)
            cnt += n_data
            fetch = [self.pred, self._loss_op, self._acc_op]
            prediction, loss, acc = sess.run(fetch, feed_dict=feed)

            for i in range(n_data):
                ch = ''.join(tokens[i])
                seq_len = feed[self.seq_len][i]
                src_sent.append(ch)
                pred_ne.append(pred2ne(ch, prediction[i, :seq_len]))
                true_ne.append(pred2ne(ch, feed[self.y][i, :seq_len]))

            print('\rEvaluation - Loss: %0.3f, Accuracy: %0.3f, step: %d, data %d/%d, %0.2f sec/batch'
                  % (loss, acc, global_step, cnt, dataset.num_data, time.time()-s), end='')
            s = time.time()
        print()

        self.write_eval_result(sess, global_step, test_log, src_sent, true_ne, pred_ne,
                               eval_start_time, num_examples, seed)

    def write_eval_result(self, sess, global_step, test_log, src_sent, true_ne, pred_ne,
                          eval_start_time, num_examples=50, seed=777):
        t = 'len(src_sent) = %d, len(true_ne) = %d, len(pred_ne) = %d' % (len(src_sent), len(true_ne), len(pred_ne))
        assert len(src_sent) == len(true_ne) == len(pred_ne), t

        num_results = len(src_sent)

        f1_et, prcs_et, rcll_et = entity_form(true_ne, pred_ne)
        f1_sf, prcs_sf, rcll_sf = surface_form(true_ne, pred_ne)

        feed = {self.f1_score_entity: f1_et,
                self.precision_entity: prcs_et,
                self.recall_entity: rcll_et,
                self.f1_score_surface: f1_sf,
                self.precision_surface: prcs_sf,
                self.recall_surface: rcll_sf}

        fetch = [self.test_merged, self._loss, self._acc]
        merged, loss, acc = sess.run(fetch, feed_dict=feed)
        writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'test'),
                                       filename_suffix='-step-%d' % global_step)

        writer.add_summary(merged, global_step)
        writer.close()

        sess.run(self.local_var_init)

        f = open(os.path.join(self.log_dir, test_log), 'a')

        print('\r', end='')
        print_write(
            """
            Evaluation - Num_Result: %d, Loss: %0.3f, Accuracy: %0.3f
            entity form[f1: %0.3f, precision: %0.3f, recall: %0.3f]
            surface form[f1: %0.3f, precision: %0.3f, recall: %0.3f]
            step: %d, time: %s\n
            """
            % (num_results, loss, acc, f1_et, prcs_et, rcll_et, f1_sf, prcs_sf, rcll_sf,
               global_step, time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - eval_start_time))), f)

        random.seed(seed)  # Always print same examples of result
        for i in range(num_examples):
            n = random.randint(0, num_results - 1)
            s = """
                  source[%d]: %s
              prediction[%d]: %s
            ground truth[%d]: %s
                """ % (n, ' '.join(src_sent[n]), n, pred_ne[n], n, true_ne[n])
            print_write(s, f)
        f.close()
        return

    def get_eval_feed(self, dataset, batch_size, keep_prob):
        ch_toks, x, y, seq_len = dataset.get_eval_batch(batch_size)
        feed = {self.x: x,
                self.y: y,
                self.seq_len: seq_len,
                self.keep_prob: keep_prob}
        n = y.shape[0]
        end = False
        if dataset.num_data <= dataset.sequential_indices:
            end = True
        return feed, ch_toks, n, end

    def write_configure(self, dataset, log_file, global_step, batch_size, train_phase: bool,
                        train_step: Optional[int], lr: Optional[float], num_examples=3):
        xs, ys, seq_lens = dataset.get_train_batch(num_examples)

        with open(os.path.join(self.log_dir, log_file), 'a') as f:
            if train_phase:
                f.write('==============================Training Configure==============================\n')
                f.write('Training Step: %s ~ %s\n' % (global_step, global_step + train_step))
                f.write('Learning Rate: %s\n' % lr)
                f.write('Start Time: %s\n' % time.strftime('%y-%m-%d %H:%M:%S'))
                f.write('The Number of Data: %s\n' % dataset.num_data)
                f.write('Max Sequence Length: %s\n' % dataset.max_seq_len)
                f.write('Batch Size: %s\n' % batch_size)
            else:
                f.write('==============================Evaluation Configure==============================\n')
                f.write('Evaluation Step: %s\n' % global_step)
                f.write('Start Time: %s\n' % time.strftime('%y-%m-%d %H:%M:%S'))
                f.write('The Number of Data: %s\n' % dataset.num_data)
                f.write('Max Sequence Length: %s\n' % dataset.max_seq_len)
                f.write('Batch Size: %s\n' % batch_size)

            f.write('\n')
            f.write('Data Examples\n')
            for i in range(num_examples):
                x = xs[i]
                y = ys[i]
                seq_len = seq_lens[i]
                f.write('\n\n')
                f.write('Input\n')
                f.write('   %s\n' % ' '.join(map(lambda k: str(k), x[:seq_len])))
                f.write('   %s\n' % ids2seq(x, self.id2char, PAD))
                f.write('\n')
                f.write('Target y\n')
                f.write('   %s\n' % ' '.join(map(lambda k: str(k), y[:seq_len])))
                f.write('\n')
                f.write('seq_len: %s\n' % seq_len)
                f.write('\n\n')
            f.write('==============================================================================\n')
        return


def my_test():
    word2vec_model = 'Data/split_1/word2vec/Chinese_d300_w04_mc05.embed'
    char2id, char_embed_vectors, char_embed_dim = load_word2vec(word2vec_model, [PAD, UNK])

    args = Arguments(cell_type='LSTM',
                     num_hiddens=[512, 512],
                     output_drop=True,
                     state_drop=True,
                     char2id=char2id,
                     char_embed_vectors=char_embed_vectors,
                     char_embed_dim=char_embed_dim)

    detector = BiLstmCrf('test', args)

    sess = tf.Session(config=get_tf_config())

    detector.init_model(sess)


if __name__ == '__main__':
    my_test()
