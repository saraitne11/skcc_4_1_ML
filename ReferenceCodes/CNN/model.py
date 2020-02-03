from utils import *
from dataset import *
from resnet34 import *
import os
import time


class CNN:
    def __init__(self, logdir, num_class, input_size, num_gpu):
        with tf.device('/cpu:0'):
            self.logdir = logdir + '/' if logdir[-1] != '/' else logdir
            os.makedirs(self.logdir, exist_ok=True)
            self.input_size = input_size
            self.num_class = num_class
            self.num_gpu = num_gpu
            
            self.global_step = tf.Variable(0, trainable=False)

            self.epoch = tf.Variable(0, trainable=False)
            self.epoch_update = tf.assign_add(self.epoch, 1)

            self.is_train = tf.Variable(True, trainable=False, dtype=tf.bool)
            self.is_train_placeholder = tf.placeholder(tf.bool)
            self.is_train_update = tf.assign(self.is_train, self.is_train_placeholder)

            # self.lr = tf.train.exponential_decay(lr, self.global_step, 10000, 0.99, staircase=True)
            self.lr = tf.Variable(0, trainable=False, dtype=tf.float32)
            self.lr_placeholder = tf.placeholder(tf.float32)
            self.lr_update = tf.assign(self.lr, self.lr_placeholder)
        return


    def build_input(self, mode, tfrecord, prefetch=20000, batch_size=256):
        with tf.device('/cpu:0'):
            if mode == 'train':
                self.iter = train_dataset(tfrecord, self.input_size, batch_size, prefetch)
                self.X, self.Y = self.iter.get_next()
            elif mode == 'eval':
                self.iter = eval_dataset(tfrecord, self.input_size, prefetch//50)
                self.X, self.Y = self.iter.get_next()
            else:
                raise NameError('Check model name')
        return


    def build_train(self,
                    model,
                    tfrecord,
                    batch_size,
                    weight_decay=1e-4,
                    optimizer_momentum=0.9):
        self.build_input('train', tfrecord, batch_size)
        with tf.device('/cpu:0'):
            split_X = tf.split(self.X, self.num_gpu)
            split_Y = tf.split(self.Y, self.num_gpu)
            build_resnet34(self.X, self.is_train, self.num_class)
            tower_gradient = []
            tower_logit = []
            tower_cross_entropy = []
            tower_l2_norm = []
            tower_loss = []

            self.optimizer = tf.train.MomentumOptimizer(self.lr, momentum=optimizer_momentum)

            for i in range(self.num_gpu):
                with tf.device('/gpu:%d' %(i)):
                    with tf.name_scope('tower_%d' %(i)) as scope:
                        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                            if model == 'resnet34':
                                logit = build_resnet34(split_X[i], self.is_train, self.num_class)
                            else:
                                raise NameError('Check model name')
                            tower_logit.append(logit)

                        Y_one_hot = tf.one_hot(split_Y[i], self.num_class)
                        cross_entropy = tf.losses.softmax_cross_entropy(Y_one_hot,
                                                                        logit,
                                                                        scope=scope)
                        l2_norm = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                                           for v in tf.trainable_variables()
                                                           if exclude_batch_norm(v.name)])
                        loss = tf.add(cross_entropy, l2_norm)

                        tower_l2_norm.append(l2_norm)
                        tower_cross_entropy.append(cross_entropy)
                        tower_loss.append(loss)

                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                        # for op in update_ops:
                        #     print op
                        with tf.control_dependencies(update_ops):
                            gradient = self.optimizer.compute_gradients(loss)
                        tower_gradient.append(gradient)
        
            self.cross_entropy = tf.reduce_mean(tower_cross_entropy)
            self.l2_loss = tf.reduce_mean(tower_l2_norm)
            self.loss = tf.reduce_mean(tower_loss)
            self.logit = tf.concat(tower_logit, axis=0)
            self.gradient = merge_gradients(tower_gradient)
            self.train_op = self.optimizer.apply_gradients(self.gradient, global_step=self.global_step)

            self.pred_label = tf.argmax(self.logit, 1, output_type=tf.int32)
            self.top1 = tf.reduce_mean(tf.cast(tf.equal(self.pred_label, self.Y), tf.float32))
            self.top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logit, self.Y, k=5), tf.float32))

            self._top1, self._top1_op = tf.metrics.mean(self.top1, name='top1_mean')
            self._top5, self._top5_op = tf.metrics.mean(self.top5, name='top5_mean')
            self._cross_entropy, self._cross_entropy_op = tf.metrics.mean(self.cross_entropy, name='cross_entroy_mean')
            self._l2_loss, self._l2_loss_op = tf.metrics.mean(self.l2_loss, name='l2_loss_mean')
            self._loss, self._loss_op = tf.metrics.mean(self.loss, name='loss_mean')

            tf.summary.scalar('learning_rate', self.lr)
            with tf.name_scope('losses'):
                tf.summary.scalar('cross_entropy', self._cross_entropy)
                tf.summary.scalar('l2_loss', self._l2_loss)
                tf.summary.scalar('loss', self._loss)

            with tf.name_scope('accuracy'):
                tf.summary.scalar('top1', self._top1)
                tf.summary.scalar('top5', self._top5)

            self.merged = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=256)
            self.get_trainable_var = tf.trainable_variables()
            self.global_var_init = tf.global_variables_initializer()
            self.local_var_init = tf.local_variables_initializer()
        return

    # def build_eval(self, model, tfrecord,
    #                prefetch=400,
    #                weight_decay=1e-4):
    #     self.eval_input(tfrecord, prefetch)
    #     if model == 'resnet34':
    #         self.build_resnet34()
    #     elif model == 'alexnet':
    #         self.build_alexnet()
    #     else:
    #         raise NameError('Check model name')
    #     self.logit = tf.reduce_mean(self.logit, axis=0, keepdims=True)
    #     self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit,
    #                                                                                        labels=self.Y,
    #                                                                                        name='cross_entropy'))
    #     self.l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
    #                                             for v in tf.trainable_variables() if exclude_batch_norm(v.name)])
    #     self.loss = self.cross_entropy + self.l2_loss
    #
    #     self.pred_label = tf.argmax(self.logit, 1, output_type=tf.int32)
    #     self.top1 = tf.reduce_mean(tf.cast(tf.equal(self.pred_label, self.Y), tf.float32))
    #     self.top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logit, self.Y, k=5), tf.float32))
    #
    #     self._top1, self._top1_op = tf.metrics.mean(self.top1, name='top1_mean')
    #     self._top5, self._top5_op = tf.metrics.mean(self.top5, name='top5_mean')
    #     self._cross_entropy, self._cross_entropy_op = tf.metrics.mean(self.cross_entropy, name='cross_entroy_mean')
    #     self._l2_loss, self._l2_loss_op = tf.metrics.mean(self.l2_loss, name='l2_loss_mean')
    #     self._loss, self._loss_op = tf.metrics.mean(self.loss, name='loss_mean')
    #
    #     tf.summary.scalar('learning_rate', self.lr)
    #     with tf.name_scope('losses'):
    #         tf.summary.scalar('cross_entropy', self._cross_entropy)
    #         tf.summary.scalar('l2_loss', self._l2_loss)
    #         tf.summary.scalar('loss', self._loss)
    #
    #     with tf.name_scope('accuracy'):
    #         tf.summary.scalar('top1', self._top1)
    #         tf.summary.scalar('top5', self._top5)
    #
    #     self.merged = tf.summary.merge_all()
    #     self.saver = tf.train.Saver(max_to_keep=256)
    #     self.local_var_init = tf.local_variables_initializer()
    #     return

    def init_model(self, sess, ckpt=None):
        if ckpt != None:
            f = open(self.logdir + 'Train-log.txt', 'a')
            self.saver.restore(sess, self.logdir + ckpt)
            print_write('model loaded from file: %s\n' % (self.logdir + ckpt), f)
            f.close()
        else:
            f = open(self.logdir + 'Train-log.txt', 'w')
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

    def train(self, sess, epoch, lr, ckpt=None, summary_step=1000):
        self.init_model(sess, ckpt)
        sess.run([self.local_var_init, self.lr_update, self.is_train_update],
                 feed_dict={self.lr_placeholder: lr, self.is_train_placeholder: True})
        s = time.time()
        for i in range(epoch):
            e = sess.run(self.epoch)
            writer = tf.summary.FileWriter(self.logdir + 'train', filename_suffix='-epoch-%03d' % (e))
            f = open(self.logdir + 'Train-log.txt', 'a')
            sess.run([self.iter.initializer, self.local_var_init, self.lr_update, self.is_train_update],
                     feed_dict={self.lr_placeholder: lr, self.is_train_placeholder: True})
            print_write('Epoch %d, learning rate: %f, start time: %s\n'
                        % (e, sess.run(self.lr), time.strftime("%y-%m-%d %H:%M:%S")), f)
            while True:
                try:
                    _, c, l2, l, t1, t5 = sess.run([self.train_op, self._cross_entropy_op, self._l2_loss_op,
                                                    self._loss_op, self._top1_op, self._top5_op])
                    global_step = sess.run(self.global_step)
                    print('\rTraining - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                          'top1: %0.4f, top5: %0.4f, step: %d'
                          % (c, l2, l, t1, t5, global_step), end='')

                    if global_step % summary_step == 0 and global_step != 0:
                        merged, c, l2, l, t1, t5 = sess.run([self.merged, self._cross_entropy,
                                                             self._l2_loss, self._loss, self._top1, self._top5])
                        writer.add_summary(merged, global_step)
                        print('\r', end='')
                        print_write('Train summary write  cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                                    'top1: %0.4f, top5: %0.4f, step: %d  %0.3f sec/step\n'
                                    % (c, l2, l, t1, t5, global_step, (time.time() - s) / summary_step), f)
                        s = time.time()
                        sess.run(self.local_var_init)

                except tf.errors.OutOfRangeError:
                    print()
                    print_write('Epoch %d, end time: %s\n' % (e, time.strftime("%y-%m-%d %H:%M:%S")), f)
                    f.close()
                    writer.close()
                    sess.run(self.epoch_update)
                    self.saver.save(sess, self.logdir + 'epoch-%d-weights' % (e), sess.run(self.global_step))
                    break
        return

    def validation(self, sess, ckpt):
        s = time.time()
        f = open(self.logdir + 'Validation-log.txt', 'a')
        self.saver.restore(sess, self.logdir + ckpt)
        print_write('model loaded from file: %s\n' % (self.logdir + ckpt), f)
        global_step = sess.run(self.global_step)
        epoch = sess.run(self.epoch) - 1  # indicate end of i epoch not start
        writer = tf.summary.FileWriter(self.logdir + 'val', filename_suffix='-epoch-%03d' % (epoch))
        sess.run([self.iter.initializer, self.local_var_init, self.is_train_update],
                 feed_dict={self.is_train_placeholder: False})
        while True:
            try:
                c, l2, l, t1, t5 = sess.run([self._cross_entropy_op, self._l2_loss_op,
                                             self._loss_op, self._top1_op, self._top5_op])
                print('\rValidation - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                      'top1: %0.4f, top5: %0.4f, step: %d'
                      % (c, l2, l, t1, t5, global_step), end='')
            except tf.errors.OutOfRangeError:
                merged, c, l2, l, t1, t5 = sess.run([self.merged, self._cross_entropy,
                                                     self._l2_loss, self._loss, self._top1, self._top5])
                writer.add_summary(merged, global_step)
                print('\r', end='')
                # print_write('Validation summary write  cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                #             'top1: %0.4f, top5: %0.4f, epoch: %d, step: %d, total validation time: %0.3f sec\n'
                #             % (c, l2, l, t1, t5, epoch, global_step, time.time()-s), f)
                print_write('Validation summary write  cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                            'top1: %0.4f, top5: %0.4f, epoch: %d, step: %d, total validation time %s\n'
                            % (c, l2, l, t1, t5, epoch, global_step,
                               time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - s))), f)
                sess.run(self.local_var_init)
                break
        f.close()
        writer.close()
        return

    def test(self, sess, ckpt):
        s = time.time()
        f = open(self.logdir + 'Test-log.txt', 'a')
        self.saver.restore(sess, self.logdir + ckpt)
        print_write('model loaded from file: %s\n' % (self.logdir + ckpt), f)
        global_step = sess.run(self.global_step)
        epoch = sess.run(self.epoch) - 1  # indicate end of i epoch not start
        writer = tf.summary.FileWriter(self.logdir + 'test', filename_suffix='-epoch-%03d' % (epoch))
        sess.run([self.iter.initializer, self.local_var_init, self.is_train_update],
                 feed_dict={self.is_train_placeholder: False})
        while True:
            try:
                c, l2, l, t1, t5 = sess.run([self._cross_entropy_op, self._l2_loss_op,
                                             self._loss_op, self._top1_op, self._top5_op])
                print('\rTest - cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                      'top1: %0.4f, top5: %0.4f, step: %d'
                      % (c, l2, l, t1, t5, global_step), end='')
            except tf.errors.OutOfRangeError:
                merged, c, l2, l, t1, t5 = sess.run([self.merged, self._cross_entropy,
                                                     self._l2_loss, self._loss, self._top1, self._top5])
                writer.add_summary(merged, global_step)
                print('\r', end='')
                # print_write('Test summary write  cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                #             'top1: %0.4f, top5: %0.4f, epoch: %d, step: %d, total test time: %0.3f sec\n'
                #             % (c, l2, l, t1, t5, epoch, global_step, time.time()-s), f)
                print_write('Test summary write  cross_entropy: %0.4f, l2_loss: %0.4f, loss: %0.4f, '
                            'top1: %0.4f, top5: %0.4f, epoch: %d, step: %d, total test time %s\n'
                            % (c, l2, l, t1, t5, epoch, global_step,
                               time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - s))), f)
                sess.run(self.local_var_init)
                break
        f.close()
        writer.close()
        return
