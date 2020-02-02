import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from FaceRecog.resnet import *      # noqa
import time                         # noqa

TRAIN_LOG = 'Train-log.txt'
PARAMS = 'params'


class CNN:
    def __init__(self, logdir, num_class, input_size, num_gpu,
                 weight_decay=1e-4, optimizer_momentum=0.9):
        self.logdir = logdir + '/' if logdir[-1] != '/' else logdir
        os.makedirs(self.logdir, exist_ok=True)

        self.input_size = input_size
        self.num_class = num_class
        self.num_gpu = num_gpu

        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)
        optimizer = tf.train.MomentumOptimizer(self.lr, momentum=optimizer_momentum)

        self.x = tf.placeholder(tf.float32, [None] + self.input_size, name='X')
        self.y = tf.placeholder(tf.float32, [None], name='Y')

        logit = build_resnet34(self.x, self.is_train, self.num_class)
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
        self.prediction = tf.argmax(logit, 1, output_type=tf.int8)
        self.top1 = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))
        self.top2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logit, self.y, k=2), tf.float32))

        self._top1, self._top1_op = tf.metrics.mean(self.top1, name='top1_mean')
        self._top2, self._top2_op = tf.metrics.mean(self.top2, name='top2_mean')
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
            tf.summary.scalar('top2', self._top2)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=256)
        self.get_trainable_var = tf.trainable_variables()
        self.global_var_init = tf.global_variables_initializer()
        self.local_var_init = tf.local_variables_initializer()
        return

    def init_model(self, sess, ckpt=None):
        if ckpt is not None:
            f = open(self.logdir + TRAIN_LOG, 'a')
            self.saver.restore(sess, self.logdir + ckpt)
            print_write('model loaded from file: %s\n' % (self.logdir + ckpt), f)
            f.close()
        else:
            f = open(self.logdir + TRAIN_LOG, 'w')
            sess.run(self.global_var_init)
            print_write('global_variables_initialize\n', f)
            writer = tf.summary.FileWriter(self.logdir + 'train', filename_suffix='-graph')
            writer.add_graph(sess.graph)

            print_write('============================================================\n', f)
            count_vars = 0
            for var in self.get_trainable_var:
                name = var.name
                shape = var.shape.as_list()
                num_elements = var.shape.num_elements()
                print_write('Variable name: %s\n' % (name), f)
                print_write('Placed device: %s\n' % (var.device), f)
                print_write('Shape : %s  Elements: %d\n' % (str(shape), num_elements), f)
                print_write('============================================================\n', f)
                count_vars = count_vars + num_elements
            print_write('Total number of trainalbe variables %d\n' % (count_vars), f)
            print_write('============================================================\n', f)
            f.close()
            writer.close()
        return

    def train(self, sess, train_data, train_step, lr, batch_size, ckpt=None, summary_step=1000):
        self.init_model(sess, ckpt)
        global_step = sess.run(self.global_step)
        base_step = global_step

        s = time.time()
        for i in range(train_step // (summary_step * 10)):
            sess.run(self.local_var_init)
            writer = tf.summary.FileWriter(os.path.join(self.logdir, 'train'),
                                           filename_suffix='-step-%d' % global_step)
            for j in range(summary_step * 10):
                x, y = train_data.get_batch(batch_size)
                fetch = [self.train_op, self._cross_entropy_op, self._l2_loss_op, self._loss_op,
                         self._top1_op, self._top2_op]
                feed = {self.x: x, self.y: y, self.lr: lr, self.is_train: True}
                _, c, l2, l, t1, t2 = sess.run(fetch, feed_dict=feed)
                global_step = sess.run(global_step)
                print('\rTraining - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                      'top1: %0.4f, top2: %0.4f, step: %d/%d'
                      % (c, l2, l, t1, t2, base_step, global_step), end='')

                if global_step % summary_step == 0 and global_step != 0:
                    fetch = [self.merged, self._cross_entropy, self._l2_loss, self._loss,
                             self._top1, self._top2]
                    merged, c, l2, l, t1, t2 = sess.run(fetch)
                    writer.add_summary(merged, global_step)
                    print('\r', end='')
                    print_write('Train summary write  cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                                'top1: %0.4f, top2: %0.4f, step: %d/%d  %0.3f sec/step\n'
                                % (c, l2, l, t1, t2, base_step, global_step, (time.time() - s) / summary_step),
                                os.path.join(self.logdir, TRAIN_LOG), 'a')
                    s = time.time()
                    sess.run(self.local_var_init)

            print_write('global step: %d, model save, time: %s\n'
                        % (global_step, time.strftime('%y-%m-%d %H:%M:%S')),
                        os.path.join(self.logdir, TRAIN_LOG), 'a')
            writer.close()

            self.saver.save(sess, os.path.join(self.logdir, 'step-%d' % global_step, PARAMS))
            s = time.time()
        return

    def runs(self, sess, data, ckpt, batch_size):
        s = time.time()
        self.init_model(sess, ckpt)




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


def get_tf_config():
    #config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def exclude_batch_norm(name):
    return 'batch_normalization' not in name
