import tensorflow as tf


DROP_RATE = 0.3


def conv_block(x,
               output_channel,
               kernel_size,
               strides,
               padding,
               is_train,
               name):
    with tf.variable_scope(name):
        h = tf.layers.conv2d(x, output_channel, kernel_size, strides, padding,
                             use_bias=False,
                             kernel_initializer=tf.initializers.variance_scaling())
        # h = tf.layers.batch_normalization(h, training=is_train, fused=True)
        h = tf.layers.dropout(h, rate=DROP_RATE, training=is_train)
    return tf.nn.relu(h)


def dense_block(x,
                output_unit,
                is_train,
                name):
    with tf.variable_scope(name):
        h = tf.layers.dense(x, output_unit,
                            use_bias=False,
                            kernel_initializer=tf.initializers.variance_scaling())
        # h = tf.layers.batch_normalization(h, training=is_train, fused=True)
        h = tf.layers.dropout(h, rate=DROP_RATE, training=is_train)
    return tf.nn.relu(h)


def residual_block(x,
                   output_channel,
                   is_train,
                   batch_norm_decay=0.99,
                   batch_norm_epsilon=1e-3,
                   name='res_block'):
    # input_channel = int(x.shape[-1])  # get # of input channels
    stride = 1
    with tf.variable_scope(name):
        with tf.variable_scope('conv1'):
            h1 = tf.layers.conv2d(x, output_channel, [3, 3], strides=stride, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling())
            h1 = tf.layers.batch_normalization(h1, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                               training=is_train, fused=True)
            h1 = tf.layers.dropout(h1, rate=DROP_RATE, training=is_train)
            h1 = tf.nn.relu(h1)

        with tf.variable_scope('conv2'):
            h2 = tf.layers.conv2d(h1, output_channel, [3, 3], strides=1, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling())
            h2 = tf.layers.batch_normalization(h2, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                               training=is_train, fused=True)
            h2 = tf.layers.dropout(h2, rate=DROP_RATE, training=is_train)
    return tf.nn.relu(h2 + x)


def residual_block_down_sample(x,
                               output_channel,
                               is_train,
                               batch_norm_decay=0.99,
                               batch_norm_epsilon=1e-3,
                               name='res_block'):
    # input_channel = int(x.shape[-1])  # get # of input channels
    stride = 2
    with tf.variable_scope(name):
        with tf.variable_scope('conv1'):
            h1 = tf.layers.conv2d(x, output_channel, [3, 3], strides=stride, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling())
            h1 = tf.layers.batch_normalization(h1, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                               training=is_train, fused=True)
            h1 = tf.layers.dropout(h1, rate=DROP_RATE, training=is_train)
            h1 = tf.nn.relu(h1)

        with tf.variable_scope('conv2'):
            h2 = tf.layers.conv2d(h1, output_channel, [3, 3], strides=1, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling())
            h2 = tf.layers.batch_normalization(h2, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                               training=is_train, fused=True)
            h2 = tf.layers.dropout(h2, rate=DROP_RATE, training=is_train)

        # option A - zero padding for extra dimension => no need extra params
        # if downsampling:
        #     pooled_x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='valid')
        #     padded_x = tf.pad(pooled_x, [[0,0], [0,0], [0,0], [input_channel // 2, input_channel // 2]])
        # else:
        #     padded_x = x

        # option B - projection with 1x1 strided conv
        with tf.variable_scope('conv_dim_match'):
            x = tf.layers.conv2d(x, output_channel, [1, 1], strides=stride, padding='SAME',
                                 kernel_initializer=tf.initializers.variance_scaling())

    return tf.nn.relu(h2 + x)


def build_vggnet18(X, is_train, num_class):
    net = conv_block(X, 128, [3, 3], [2, 2], 'SAME', is_train, 'conv1')         # [N, 64, 64, 128]
    # print(net, 'conv1')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool1')    # [N, 32, 32, 128]
    # print(net, 'pool1')
    net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', is_train, 'conv2')       # [N, 32, 32, 256]
    # print(net, 'conv2')
    net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', is_train, 'conv3')       # [N, 32, 32, 256]
    # print(net, 'conv3')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool2')    # [N, 16, 16, 256]
    # print(net, 'pool2')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv4')       # [N, 16, 16, 512]
    # print(net, 'conv4')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv5')       # [N, 16, 16, 512]
    # print(net, 'conv5')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv6')       # [N, 16, 16, 512]
    # print(net, 'conv6')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool3')    # [N, 8, 8, 512]
    # print(net, 'pool3')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv7')       # [N, 8, 8, 512]
    # print(net, 'conv7')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv8')       # [N, 8, 8, 512]
    # print(net, 'conv8')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv9')       # [N, 8, 8, 512]
    # print(net, 'conv9')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv10')      # [N, 8, 8, 512]
    # print(net, 'conv10')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool4')    # [N, 4, 4, 512]
    # print(net, 'pool4')
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv11')     # [N, 4, 4, 1024]
    # print(net, 'conv11')
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv12')     # [N, 4, 4, 1024]
    # print(net, 'conv12')
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv13')     # [N, 4, 4, 1024]
    # print(net, 'conv13')
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv14')     # [N, 4, 4, 1024]
    # print(net, 'conv14')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool5')    # [N, 2, 2, 1024]
    # print(net, 'pool5')
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv15')     # [N, 2, 2, 1024]
    # print(net, 'conv15')
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv16')  # [N, 2, 2, 1024]
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv17')  # [N, 2, 2, 1024]
    net = conv_block(net, 1024, [3, 3], [1, 1], 'SAME', is_train, 'conv18')  # [N, 2, 2, 1024]

    net = tf.layers.flatten(net)
    net = dense_block(net, 4096, is_train, 'fc1')
    logit = tf.layers.dense(net, num_class,
                            kernel_initializer=tf.initializers.variance_scaling(), name='logit')
    return logit


def build_vggnet8(X, is_train, num_class):
    net = conv_block(X, 128, [3, 3], [2, 2], 'SAME', is_train, 'conv1')         # [N, 64, 64, 128]
    # print(net, 'conv1')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool1')    # [N, 32, 32, 128]
    # print(net, 'pool1')
    net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', is_train, 'conv2')       # [N, 32, 32, 256]
    # print(net, 'conv2')
    net = conv_block(net, 256, [3, 3], [1, 1], 'SAME', is_train, 'conv3')       # [N, 32, 32, 256]
    # print(net, 'conv3')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool2')    # [N, 16, 16, 256]
    # print(net, 'pool2')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv4')       # [N, 16, 16, 512]
    # print(net, 'conv4')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv5')       # [N, 16, 16, 512]
    # print(net, 'conv5')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv6')       # [N, 16, 16, 512]
    # print(net, 'conv6')
    net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], 'SAME', name='pool3')    # [N, 8, 8, 512]
    # print(net, 'pool3')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv7')       # [N, 8, 8, 512]
    # print(net, 'conv7')
    net = conv_block(net, 512, [3, 3], [1, 1], 'SAME', is_train, 'conv8')       # [N, 8, 8, 512]
    # print(net, 'conv8')

    net = tf.layers.flatten(net)
    net = dense_block(net, 512, is_train, 'fc1')
    logit = tf.layers.dense(net, num_class,
                            kernel_initializer=tf.initializers.variance_scaling(), name='logit')
    return logit


def build_resnet18(X, is_train, num_class):
    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-5

    net = residual_block_down_sample(X, 128, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block1')
    net = residual_block(net, 128, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block2')

    net = residual_block_down_sample(net, 256, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block3')
    net = residual_block(net, 256, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block4')
    net = residual_block(net, 256, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block5')

    net = residual_block_down_sample(net, 512, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block6')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block7')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block8')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block9')

    net = tf.reduce_mean(net, [1, 2])  # global average pooling
    logit = tf.layers.dense(net, num_class, kernel_initializer=tf.initializers.variance_scaling(), name='logit')
    return logit


def build_resnet26(X, is_train, num_class):
    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-5

    net = residual_block_down_sample(X, 128, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block1')
    net = residual_block(net, 128, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block2')

    net = residual_block_down_sample(net, 256, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block3')
    net = residual_block(net, 256, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block4')
    net = residual_block(net, 256, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block5')

    net = residual_block_down_sample(net, 512, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block6')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block7')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block8')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block9')

    net = residual_block_down_sample(net, 1024, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block10')
    net = residual_block(net, 1024, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block11')
    net = residual_block(net, 1024, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block12')
    net = residual_block(net, 1024, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block13')

    net = tf.reduce_mean(net, [1, 2])  # global average pooling
    logit = tf.layers.dense(net, num_class, kernel_initializer=tf.initializers.variance_scaling(), name='logit')
    return logit


def build_resnet17(X, is_train, num_class):
    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-5
    net = tf.layers.conv2d(X, 256, [3, 3], strides=[2, 2], padding='SAME', use_bias=False,
                           kernel_initializer=tf.initializers.variance_scaling(),
                           name='stem_conv1')
    net = tf.layers.batch_normalization(net, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                        training=is_train, fused=True, name='stem_batch_normalization')
    net = tf.nn.relu(net, name='stem_relu')

    net = residual_block_down_sample(net, 512, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block1')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block2')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block3')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block4')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block5')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block6')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block7')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block8')

    net = tf.reduce_mean(net, [1, 2])  # global average pooling
    logit = tf.layers.dense(net, num_class, kernel_initializer=tf.initializers.variance_scaling(), name='logit')
    return logit


def build_resnet13(X, is_train, num_class):
    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-5
    net = tf.layers.conv2d(X, 256, [3, 3], strides=[2, 2], padding='SAME', use_bias=False,
                           kernel_initializer=tf.initializers.variance_scaling(),
                           name='stem_conv1')
    net = tf.layers.batch_normalization(net, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                        training=is_train, fused=True, name='stem_batch_normalization')
    net = tf.nn.relu(net, name='stem_relu')

    net = residual_block_down_sample(net, 512, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block1')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block2')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block3')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block4')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block5')
    net = residual_block(net, 512, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block6')

    net = tf.reduce_mean(net, [1, 2])  # global average pooling
    logit = tf.layers.dense(net, num_class, kernel_initializer=tf.initializers.variance_scaling(), name='logit')
    return logit
