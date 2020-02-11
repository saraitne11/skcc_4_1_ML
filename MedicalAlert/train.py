import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from MedicalAlert.model import *       # noqa


dataset = TrainData('../_Data/ml_10_medicalalert_train.csv')

args = Arguments('LSTM', [300, 300], True, True)
model = UniLSTM('model7', args)

sess = tf.Session(config=get_tf_config())
model.train(sess, dataset, 1000, 1e-4, 8, 0.7)
model.train(sess, dataset, 1000, 1e-5, 8, 0.7, ckpt='step-1000')

sess.close()
