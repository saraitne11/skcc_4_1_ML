from utils import *
import os
import time

class CNN:
    def __init__(self, logdir, num_class, input_size):
        self.logdir = logdir + '/' if logdir[-1] != '/' else logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.input_size = input_size
        self.num_class = num_class

        self.global_step = tf.Variable(0, trainable=False)

        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_update = tf.assign_add(self.epoch, 1)

        self.is_train = tf.Variable(True, trainable=False, dtype=tf.bool)
        self.is_train_placeholder = tf.placeholder(tf.bool)
        self.is_train_update = tf.assign(self.is_train, self.is_train_placeholder)

        #self.lr = tf.train.exponential_decay(lr, self.global_step, 10000, 0.99, staircase=True)
        self.lr = tf.Variable(0, trainable=False, dtype=tf.float32)
        self.lr_placeholder = tf.placeholder(tf.float32)
        self.lr_update = tf.assign(self.lr, self.lr_placeholder)
        return

    def train_input(self, tfrecord, batch_size, prefetch):
        with tf.device('/cpu:0'):
            self.dataset = tf.data.TFRecordDataset(tfrecord)
            self.dataset = self.dataset.map(lambda x: parse_train(x, self.input_size), num_parallel_calls=8)
            self.dataset = self.dataset.shuffle(buffer_size=prefetch)
            self.dataset = self.dataset.batch(batch_size)
            self.dataset = self.dataset.prefetch(prefetch)
            #self.dataset = self.dataset.repeat()
            self.iter = self.dataset.make_initializable_iterator()
            self.X, self.Y = self.iter.get_next()
        return
    
    def eval_input(self, tfrecord, prefetch):
        with tf.device('/cpu:0'):
            self.dataset = tf.data.TFRecordDataset(tfrecord)
            self.dataset = self.dataset.map(lambda x: parse_eval(x, self.input_size), num_parallel_calls=8)
            # self.dataset = self.dataset.shuffle(buffer_size=batch_size)
            # self.dataset = self.dataset.batch(batch_size)
            self.dataset = self.dataset.prefetch(prefetch)
            # self.dataset = self.dataset.repeat()
            self.iter = self.dataset.make_initializable_iterator()
            self.X, self.Y = self.iter.get_next()
        return

    def build_resnet34(self):
        batch_norm_decay = 0.997
        batch_norm_epsilon = 1e-5
        net = tf.layers.conv2d(self.X, 64, [7, 7], strides=[2, 2], padding='SAME', use_bias=False,
                               kernel_initializer=tf.initializers.variance_scaling(),
                               name='stem_conv1')
        net = tf.layers.batch_normalization(net, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                            training=self.is_train, fused=True, name='stem_batch_normalization')
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2], padding='SAME', name='stem_pool')

        net = residual_block(net, 64, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block1')
        net = residual_block(net, 64, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block2')
        net = residual_block(net, 64, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block3')

        net = residual_block_down_sample(net, 128, self.is_train,
                                         batch_norm_decay, batch_norm_epsilon, name='res_block4')
        net = residual_block(net, 128, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block5')
        net = residual_block(net, 128, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block6')
        net = residual_block(net, 128, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block7')

        net = residual_block_down_sample(net, 256, self.is_train,
                                         batch_norm_decay, batch_norm_epsilon, name='res_block8')
        net = residual_block(net, 256, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block9')
        net = residual_block(net, 256, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block10')
        net = residual_block(net, 256, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block11')
        net = residual_block(net, 256, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block12')
        net = residual_block(net, 256, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block13')

        net = residual_block_down_sample(net, 512, self.is_train,
                                         batch_norm_decay, batch_norm_epsilon, name='res_block14')
        net = residual_block(net, 512, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block15')
        net = residual_block(net, 512, self.is_train, batch_norm_decay, batch_norm_epsilon, name='res_block16')

        net = tf.reduce_mean(net, [1, 2])  # global average pooling
        self.logit = tf.layers.dense(net, self.num_class,
                                     kernel_initializer=tf.initializers.variance_scaling(), name='logit')
        return

    def build_vgg(self):
        net = conv_block(self.X, 32, [3, 3], [2, 2], 'SAME', self.is_train, 'conv1')    #[N, 112, 112, 32]
        #print(net, 'conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool1')        #[N, 56, 56, 64]
        #print(net, 'pool1')
        net = conv_block(net, 64, [3, 3], [1, 1], 'SAME', self.is_train, 'conv2')       #[N, 56, 56, 64]
        #print(net, 'conv2')
        net = conv_block(net, 64, [3, 3], [1, 1], 'SAME', self.is_train, 'conv3')       #[N, 56, 56, 64]
        #print(net, 'conv3')
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool2')        #[N, 28, 28, 64]
        #print(net, 'pool2')
        net = conv_block(net, 128, [3, 3], [1, 1], 'SAME', self.is_train, 'conv4')      #[N, 28, 28, 128]
        #print(net, 'conv4')
        net = conv_block(net, 128, [3, 3], [1, 1], 'SAME', self.is_train, 'conv5')      #[N, 28, 28, 128]
        #print(net, 'conv5')
        net = conv_block(net, 128, [3, 3], [1, 1], 'SAME', self.is_train, 'conv6')      #[N, 28, 28, 128]
        #print(net, 'conv6')
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool3')        #[N, 14, 14, 128]
        #print(net, 'pool3')
        net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', self.is_train, 'conv7')      #[N, 14, 14, 256]
        #print(net, 'conv7')
        net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', self.is_train, 'conv8')      #[N, 14, 14, 256]
        #print(net, 'conv8')
        net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', self.is_train, 'conv9')      #[N, 14, 14, 256]
        #print(net, 'conv9')
        net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', self.is_train, 'conv10')     #[N, 14, 14, 256]
        #print(net, 'conv10')
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool4')        #[N, 7, 7, 256]
        #print(net, 'pool4')
        net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', self.is_train, 'conv11')     #[N, 7, 7, 512]
        #print(net, 'conv11')
        net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', self.is_train, 'conv12')     #[N, 7, 7, 512]
        #print(net, 'conv12')
        net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', self.is_train, 'conv13')     #[N, 7, 7, 512]
        #print(net, 'conv13')
        net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', self.is_train, 'conv14')     #[N, 7, 7, 512]
        #print(net, 'conv14')
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool5')        #[N, 4, 4, 512]
        #print(net, 'pool5')
        net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', self.is_train, 'conv15')    #[N, 4, 4, 1024]
        #print(net, 'conv15')

        net = tf.layers.flatten(net)
        net = dense_block(net, 4096, self.is_train, 'fc1')
        self.logit = tf.layers.dense(net, self.num_class,
                                     kernel_initializer=tf.initializers.variance_scaling(), name='logit')
        return

    def build_train(self, model, tfrecord, batch_size,
                    prefetch=10000,
                    weight_decay=2e-4,
                    optimizer_momentum=0.9):
        self.train_input(tfrecord, batch_size, prefetch)
        if model == 'resnet34':
            self.build_resnet34()
        elif model == 'vgg17':
            self.build_vgg()
        else:
            raise NameError('Check model name')
        Y_one_hot = tf.one_hot(self.Y, self.num_class)
        self.cross_entropy = tf.losses.softmax_cross_entropy(Y_one_hot, self.logit)

        self.l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                                for v in tf.trainable_variables() if exclude_batch_norm(v.name)])
        self.loss = self.cross_entropy + self.l2_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)
            self.optimizer = tf.train.MomentumOptimizer(self.lr, momentum=optimizer_momentum)
            self.train_op = self.optimizer.minimize(self.loss, self.global_step)

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

    def build_eval(self, model, tfrecord,
                   prefetch=200,
                   weight_decay=2e-4):
        self.eval_input(tfrecord, prefetch)
        if model == 'resnet34':
            self.build_resnet34()
        elif model == 'vgg17':
            self.build_vgg()
        else:
            raise NameError('Check model name')
        self.logit = tf.reduce_mean(self.logit, axis=0, keepdims=True)
        Y_one_hot = tf.one_hot(self.Y, self.num_class)
        self.cross_entropy = tf.losses.softmax_cross_entropy(Y_one_hot, self.logit)

        self.l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                                for v in tf.trainable_variables() if exclude_batch_norm(v.name)])
        self.loss = self.cross_entropy + self.l2_loss

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
        self.local_var_init = tf.local_variables_initializer()
        return

    def init_model(self, sess, ckpt=None):
        if ckpt != None:
            f = open(self.logdir + 'Train-log.txt', 'a')
            self.saver.restore(sess, self.logdir + ckpt)
            print_write('model loaded from file: %s\n' %(self.logdir + ckpt), f)
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
                print_write('Variable name: %s\n' %(name), f)
                print_write('Shape : %s  Elements: %d\n' %(str(shape), num_elements), f)
                print_write('============================================================\n', f)
                count_vars = count_vars + num_elements
            print_write('Total number of trainalbe variables %d\n' %(count_vars), f)
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
        epoch = sess.run(self.epoch)-1          #indicate end of i epoch not start
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
                               time.strftime('%Hh %Mm %Ss', time.gmtime(time.time()-s))), f)
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
        epoch = sess.run(self.epoch)-1          #indicate end of i epoch not start
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
                               time.strftime('%Hh %Mm %Ss', time.gmtime(time.time()-s))), f)
                sess.run(self.local_var_init)
                break
        f.close()
        writer.close()
        return
