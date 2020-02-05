import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from FaceRecog.model import *       # noqa


train_data = DataSet('../_Data/face_images_128x128/', '../_Data/ml_8_faceclassifier_train_split.csv')
val_data = DataSet('../_Data/face_images_128x128/', '../_Data/ml_8_faceclassifier_train_val.csv')


classifier = CNN('vggnet8', 6, [128, 128, 3])

sess = tf.Session(config=get_tf_config())
classifier.train(sess, train_data, val_data, 1000, 1e-4, 128)
classifier.train(sess, train_data, val_data, 3000, 1e-5, 128, ckpt='step-1000')
# classifier.train(sess, dataset, 5000, 1e-6, 16, ckpt='step-10000')

sess.close()
