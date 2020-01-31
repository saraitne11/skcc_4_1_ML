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
    #config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def exclude_batch_norm(name):
    return 'batch_normalization' not in name


def merge_gradients(temp_grads):
    average_grads = []
    for grad_and_vars in zip(*temp_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads