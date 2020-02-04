import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from FaceRecog.model import *       # noqa


dataset = DataSet('../_Data/face_images_128x128/', '../_Data/ml_8_faceclassifier_train.csv')

classifier = CNN('resnet13', 5, [128, 128, 3])

sess = tf.Session(config=get_tf_config())
classifier.train(sess, dataset, 5000, 1e-4, 64)
classifier.train(sess, dataset, 10000, 1e-5, 64, ckpt='step-5000')

sess.close()
