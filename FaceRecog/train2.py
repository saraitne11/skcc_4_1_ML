import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from FaceRecog.model import *       # noqa


# train_data = DataSet('../_Data/face_images_128x128/', '../_Data/ml_8_faceclassifier_train_split.csv')
# val_data = DataSet('../_Data/face_images_128x128/', '../_Data/ml_8_faceclassifier_train_val.csv')
train_data = DataSet('../_Data/face_images_128x128/', '../_Data/ml_8_faceclassifier_train.csv')
val_data = None

classifier = CNN('resnet18_2', 6, [128, 128, 3])

sess = tf.Session(config=get_tf_config())
classifier.train(sess, train_data, val_data, 5000, 1e-4, 32)
classifier.train(sess, train_data, val_data, 10000, 1e-5, 32, ckpt='step-5000')
# classifier.train(sess, train_data, None, 2000, 1e-5, 32, ckpt='step-6000')

sess.close()
