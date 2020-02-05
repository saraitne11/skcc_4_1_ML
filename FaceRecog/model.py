import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from FaceRecog.cnn_models import *      # noqa
from FaceRecog.utils import *       # noqa
import time                         # noqa

LOG_DIR = 'Logs'
PARAMS_DIR = 'Params'
PARAMS = 'params'
TRAIN_LOG = 'Train_log.txt'


class CNN:
    def __init__(self, name, num_class, input_size,
                 weight_decay=1e-3):
        self.log_dir = os.path.join(LOG_DIR, name)
        self.param_dir = os.path.join(PARAMS_DIR, name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)

        self.input_size = input_size
        self.num_class = num_class

        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.x = tf.placeholder(tf.float32, [None] + self.input_size, name='X')
        self.y = tf.placeholder(tf.int32, [None], name='Y')

        # self.x = tf.cond(self.is_train, lambda: images_augment(self.x), lambda: self.x)

        if 'resnet17' in name:
            logit = build_resnet17(self.x, self.is_train, self.num_class)
        elif 'resnet13' in name:
            logit = build_resnet13(self.x, self.is_train, self.num_class)
        elif 'resnet18' in name:
            logit = build_resnet18(self.x, self.is_train, self.num_class)
        elif 'vggnet18' in name:
            logit = build_vggnet18(self.x, self.is_train, self.num_class)
        elif 'resnet26' in name:
            logit = build_resnet26(self.x, self.is_train, self.num_class)
        elif 'vggnet8' in name:
            logit = build_vggnet8(self.x, self.is_train, self.num_class)
        else:
            print('Check model name')
            sys.exit()

        y_one_hot = tf.one_hot(self.y, self.num_class)
        cross_entropy = tf.losses.softmax_cross_entropy(y_one_hot, logit)
        l2_norm = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                           for v in tf.trainable_variables()
                                           if exclude_batch_norm(v.name)])

        loss = tf.add(cross_entropy, l2_norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradient = optimizer.compute_gradients(loss)

        self.train_op = optimizer.apply_gradients(gradient, global_step=self.global_step)
        self.prediction = tf.argmax(logit, 1, output_type=tf.int32)
        self.top1 = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))
        # self.top2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.prediction, self.y, k=2), tf.float32))

        self._top1, self._top1_op = tf.metrics.mean(self.top1, name='top1_mean')
        # self._top2, self._top2_op = tf.metrics.mean(self.top2, name='top2_mean')
        self._cross_entropy, self._cross_entropy_op = tf.metrics.mean(cross_entropy, name='cross_entroy_mean')
        self._l2_loss, self._l2_loss_op = tf.metrics.mean(l2_norm, name='l2_loss_mean')
        self._loss, self._loss_op = tf.metrics.mean(loss, name='loss_mean')

        tf.summary.scalar('learning_rate', self.lr)
        with tf.name_scope('losses'):
            tf.summary.scalar('cross_entropy', self._cross_entropy)
            tf.summary.scalar('l2_loss', self._l2_loss)
            tf.summary.scalar('loss', self._loss)

        with tf.name_scope('accuracy'):
            tf.summary.scalar('top1', self._top1)
            # tf.summary.scalar('top2', self._top2)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=256)
        self.get_trainable_var = tf.trainable_variables()
        self.global_var_init = tf.global_variables_initializer()
        self.local_var_init = tf.local_variables_initializer()
        return

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

    def train(self, sess, train_data, val_data, train_step, lr, batch_size, ckpt=None, summary_step=100):
        self.init_model(sess, ckpt)
        global_step = sess.run(self.global_step)
        base_step = global_step

        s = time.time()
        for i in range(train_step // (summary_step * 10)):
            sess.run(self.local_var_init)
            writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'),
                                           filename_suffix='-step-%d' % global_step)
            for j in range(summary_step * 10):
                x, y = train_data.train_batch(batch_size)
                fetch = [self.train_op, self._cross_entropy_op, self._l2_loss_op, self._loss_op, self._top1_op]
                feed = {self.x: x, self.y: y, self.lr: lr, self.is_train: True}
                _, c, l2, l, t1 = sess.run(fetch, feed_dict=feed)
                global_step = sess.run(self.global_step)
                print('\rTraining - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                      'top1: %0.4f, step: %d/%d'
                      % (c, l2, l, t1, global_step, base_step+train_step), end='')

                if global_step % summary_step == 0 and global_step != 0:
                    fetch = [self.merged, self._cross_entropy, self._l2_loss, self._loss, self._top1]
                    merged, c, l2, l, t1 = sess.run(fetch, feed_dict={self.lr: lr, self.is_train: True})
                    writer.add_summary(merged, global_step)
                    print('\r', end='')
                    print_write('Training - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                                'top1: %0.4f, step: %d/%d  %0.3f sec/step\n'
                                % (c, l2, l, t1, global_step, base_step+train_step, (time.time() - s) / summary_step),
                                os.path.join(self.log_dir, TRAIN_LOG), 'a')
                    s = time.time()
                    if val_data:
                        self.validation(sess, val_data, batch_size, None)
                    sess.run(self.local_var_init)

            print_write('global step: %d, model save, time: %s\n'
                        % (global_step, time.strftime('%y-%m-%d %H:%M:%S')),
                        os.path.join(self.log_dir, TRAIN_LOG), 'a')
            writer.close()

            self.saver.save(sess, os.path.join(self.param_dir, 'step-%d' % global_step, PARAMS))
            s = time.time()
        return

    def validation(self, sess, val_data, batch_size, ckpt):
        if ckpt:
            self.init_model(sess, ckpt)
        global_step = sess.run(self.global_step)

        s = time.time()
        sess.run(self.local_var_init)
        val_data.sequential_idx_reset()
        writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'validation'),
                                       filename_suffix='-step-%d' % global_step)
        end = False
        cnt = 0
        while not end:
            x, y, end = val_data.val_batch(batch_size)
            fetch = [self._cross_entropy_op, self._l2_loss_op, self._loss_op, self._top1_op]
            feed = {self.x: x, self.y: y, self.lr: 0, self.is_train: False}
            c, l2, l, t1 = sess.run(fetch, feed_dict=feed)
            cnt += len(x)
            print('\rValidation - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                  'top1: %0.4f, step: %d, images %d/%d'
                  % (c, l2, l, t1, global_step, cnt, val_data.numData), end='')

        fetch = [self.merged, self._cross_entropy, self._l2_loss, self._loss, self._top1]
        merged, c, l2, l, t1 = sess.run(fetch, feed_dict={self.lr: 0, self.is_train: False})
        writer.add_summary(merged, global_step)
        print('\r', end='')
        print_write('Validation - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                    'top1: %0.4f, step: %d, images %d/%d  %0.3f sec\n'
                    % (c, l2, l, t1, global_step, cnt, val_data.numData, (time.time() - s)),
                    os.path.join(self.log_dir, TRAIN_LOG), 'a')

        sess.run(self.local_var_init)
        return

    def runs(self, sess, test_data, ckpt, batch_size, csv_name='result.csv'):
        s = time.time()
        self.init_model(sess, ckpt)
        predictions = []
        num_data = test_data.numData
        end = False
        while not end:
            x, file_names, end = test_data.infer_batch(batch_size)
            pred_idx = sess.run(self.prediction, feed_dict={self.x: x, self.is_train: False})

            if len(file_names) != len(pred_idx):
                print('len(file_names) %d != len(pred_idx) %d' % (len(file_names), len(pred_idx)))
                return -1

            print('\rTest - %d/%d' % (len(predictions), num_data), end='')

            for f, p in zip(file_names, pred_idx):
                predictions.append([f, p])

        csv_save(csv_name, predictions)
        print()
        print('total time: %0.3f sec, %0.3f sec/image' % (time.time()-s, (time.time()-s)/num_data))
        return
