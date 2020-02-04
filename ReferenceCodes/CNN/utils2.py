import tensorflow as tf
import os


def print_write(s, f):
    print(s, end='')
    f.write(s)


def get_uneval_ckpt(log_dir, log_txt):
    log_dir = log_dir + '/' if log_dir[-1] != '/' else log_dir
    files = os.listdir(log_dir)
    all_ckpt = []
    for file in files:
        ext = os.path.splitext(file)
        if ext[-1] == '.index':
            all_ckpt.append(ext[0])
    try:
        f = open(log_dir + log_txt, 'r')
        lines = f.readlines()
        f.close()
        eval_ckpt = []
        for line in lines:
            if 'summary' not in line:
                s = line.index('epoch')
                e = line.index('\n')
                eval_ckpt.append(line[s:e])
        for ckpt in eval_ckpt:
            all_ckpt.remove(ckpt)
    except:
        pass
    all_ckpt.sort(key=lambda x: int(x.split('-')[-1]))
    return all_ckpt


def get_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def exclude_batch_norm(name):
    return 'batch_normalization' not in name


def parse_train(example_proto, crop_size):
    features = {'label': tf.FixedLenFeature([], tf.int64, default_value=0),
                'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                'width': tf.FixedLenFeature([], tf.int64, default_value=0),
                'image': tf.FixedLenFeature([], tf.string, default_value=''),
                'file_name': tf.FixedLenFeature([], tf.string, default_value='')}
    parsed_features = tf.parse_single_example(example_proto, features)

    h = tf.to_int32(parsed_features['height'])
    w = tf.to_int32(parsed_features['width'])
    img = tf.decode_raw(parsed_features['image'], tf.uint8)
    img = tf.reshape(img, (h, w, 3))
    ratio = tf.divide(tf.maximum(h, w), tf.minimum(h, w))
    new_shoter = tf.random_uniform([], 256, 481, dtype=tf.int32)
    new_longer = tf.to_int32(tf.multiply(tf.cast(new_shoter, tf.float64), ratio))
    img = tf.cond(tf.less(w, h),
                  lambda: tf.image.resize_images(img, (new_longer, new_shoter)),
                  lambda: tf.image.resize_images(img, (new_shoter, new_longer)))

    img = tf.random_crop(img, crop_size)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.random_brightness(img, 0.15)
    # img = tf.image.random_contrast(img, lower=0.6, upper=1.8)
    # img = tf.image.random_hue(img, 0.08)
    # img = tf.image.random_saturation(img, lower=0.2, upper=2.0)
    img = tf.divide(tf.cast(img, tf.float32), 255.)
    img = tf.image.per_image_standardization(img)
    label = tf.cast(parsed_features['label'], tf.int32)
    return img, label-1


def parse_eval(example_proto, crop_size):
    features = {'label': tf.FixedLenFeature([], tf.int64, default_value=0),
                'height': tf.FixedLenFeature([], tf.int64, default_value=0),
                'width': tf.FixedLenFeature([], tf.int64, default_value=0),
                'image': tf.FixedLenFeature([], tf.string, default_value=''),
                'file_name': tf.FixedLenFeature([], tf.string, default_value='')}
    parsed_features = tf.parse_single_example(example_proto, features)

    h = tf.to_int32(parsed_features['height'])
    w = tf.to_int32(parsed_features['width'])
    img = tf.decode_raw(parsed_features['image'], tf.uint8)
    img = tf.reshape(img, (h, w, 3))
    ratio = tf.divide(tf.maximum(h, w), tf.minimum(h, w))
    scale1 = _10_crop_func(img, h, w, ratio, 224, crop_size)
    scale2 = _10_crop_func(img, h, w, ratio, 256, crop_size)
    scale3 = _10_crop_func(img, h, w, ratio, 384, crop_size)
    scale4 = _10_crop_func(img, h, w, ratio, 480, crop_size)
    scale5 = _10_crop_func(img, h, w, ratio, 640, crop_size)

    multi_scale_crop = tf.concat([scale1, scale2, scale3, scale4, scale5], axis=0)

    label = tf.cast(parsed_features['label'], tf.int32)
    return multi_scale_crop, tf.expand_dims(label-1, 0)


def _10_crop_func(img, w, h, ratio, resize, crop_size):
    new_shoter = resize
    new_longer = tf.to_int32(tf.multiply(tf.cast(new_shoter, tf.float64), ratio))
    new_h = tf.cond(tf.less(w, h), lambda: new_longer, lambda: new_shoter)
    new_w = tf.cond(tf.less(w, h), lambda: new_shoter, lambda: new_longer)
    img = tf.image.resize_images(img, (new_h, new_w))
    img = tf.divide(tf.cast(img, tf.float32), 255.)
    img = tf.image.per_image_standardization(img)

    img_center = tf.image.crop_to_bounding_box(img,
                                               new_h // 2 - crop_size[0] // 2, new_w // 2 - crop_size[1] // 2,
                                               crop_size[0], crop_size[1])
    img_top_left = tf.image.crop_to_bounding_box(img,
                                                 0, 0, crop_size[0], crop_size[1])
    img_top_right = tf.image.crop_to_bounding_box(img,
                                                  0, new_w - crop_size[1], crop_size[0], crop_size[1])
    img_bot_left = tf.image.crop_to_bounding_box(img,
                                                 new_h - crop_size[0], 0, crop_size[0], crop_size[1])
    img_bot_right = tf.image.crop_to_bounding_box(img,
                                                  new_h - crop_size[0], new_w - crop_size[1], crop_size[0],
                                                  crop_size[1])
    _10_crop = tf.stack([img_center,
                         tf.image.flip_left_right(img_center),
                         img_top_left,
                         tf.image.flip_left_right(img_top_left),
                         img_top_right,
                         tf.image.flip_left_right(img_top_right),
                         img_bot_left,
                         tf.image.flip_left_right(img_bot_left),
                         img_bot_right,
                         tf.image.flip_left_right(img_bot_right)])
    return _10_crop


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
        h = tf.layers.batch_normalization(h, training=is_train, fused=True)
    return tf.nn.relu(h)


def dense_block(x,
                output_unit,
                is_train,
                name):
    with tf.variable_scope(name):
        h = tf.layers.dense(x, output_unit,
                             use_bias=False,
                             kernel_initializer=tf.initializers.variance_scaling())
        h = tf.layers.batch_normalization(h, training=is_train, fused=True)
    return tf.nn.relu(h)


def residual_block(x,
                   output_channel,
                   is_train,
                   batch_norm_decay,
                   batch_norm_epsilon,
                   name='res_block'):
    #input_channel = int(x.shape[-1])  # get # of input channels
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
                               batch_norm_decay,
                               batch_norm_epsilon,
                               name='res_block'):
    #input_channel = int(x.shape[-1])  # get # of input channels
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
