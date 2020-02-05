import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from FaceRecog.model import *       # noqa


dataset = DataSet('../_Data/face_images_128x128/', '../_Data/ml_8_faceclassifier_test.csv')


classifier = CNN('resnet18_2', 6, [128, 128, 3])

sess = tf.Session(config=get_tf_config())
classifier.runs(sess, dataset, 'step-13000', 32, csv_name='resnet18_2.csv')

sess.close()
