import tensorflow as tf


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
            h1 = tf.nn.relu(h1)

        with tf.variable_scope('conv2'):
            h2 = tf.layers.conv2d(h1, output_channel, [3, 3], strides=1, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling())
            h2 = tf.layers.batch_normalization(h2, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                               training=is_train, fused=True)
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
            h1 = tf.nn.relu(h1)

        with tf.variable_scope('conv2'):
            h2 = tf.layers.conv2d(h1, output_channel, [3, 3], strides=1, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling())
            h2 = tf.layers.batch_normalization(h2, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                               training=is_train, fused=True)

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


def build_resnet(X, is_train, num_class):
    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-5
    net = tf.layers.conv2d(X, 64, [3, 3], strides=[2, 2], padding='SAME', use_bias=False,
                           kernel_initializer=tf.initializers.variance_scaling(),
                           name='stem_conv1')
    net = tf.layers.batch_normalization(net, momentum=batch_norm_decay, epsilon=batch_norm_epsilon,
                                        training=is_train, fused=True, name='stem_batch_normalization')
    net = tf.nn.relu(net, name='stem_relu')
    net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2], padding='SAME', name='stem_pool')

    net = residual_block(net, 64, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block1')

    net = residual_block_down_sample(net, 128, is_train,
                                     batch_norm_decay, batch_norm_epsilon, name='res_block2')
    net = residual_block(net, 128, is_train, batch_norm_decay, batch_norm_epsilon, name='res_block3')

    net = tf.reduce_mean(net, [1, 2])  # global average pooling
    logit = tf.layers.dense(net, num_class, kernel_initializer=tf.initializers.variance_scaling(), name='logit')
    return logit
