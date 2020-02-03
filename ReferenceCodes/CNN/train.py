from model import *


tf_list = os.listdir('/home/saraitne/PycharmProjects/imagenet_2010/Data/')
tf_list.sort()
tf_list = list(map(lambda a: '/home/saraitne/PycharmProjects/imagenet_2010/Data/'+a, tf_list))

#test_tfrecord = tf_list[0:8]
#train_tfrecord = tf_list[8:16]
#train_tfrecord = ['/ssd/cache/train_00.tfrecord', '/ssd/cache/train_01.tfrecord']
#val_tfreocrd = tf_list[16:24]
train_tfrecord = '/home/saraitne/PycharmProjects/imagenet_2010/Data/train_*.tfrecord'


model = 'resnet34'

graph = tf.Graph()
with graph.as_default():
    cnn = CNN(model, 1000, (224, 224, 3), 2)
    cnn.build_train(model, train_tfrecord, 256)
sess = tf.Session(graph=graph, config=get_tf_config())
cnn.train(sess, 30, 1e-1, ckpt=None)
sess.close()
