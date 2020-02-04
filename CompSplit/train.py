import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from CompSplit.model import *       # noqa


dataset = DataSet('../_Data/ml_6_spacing_train.csv', '../_Data/ml_6_spacing_test.csv')


args = Arguments('LSTM', [300, 300], True, True, 100)
splitter = BiLstmCrf('model1', args)

sess = tf.Session(config=get_tf_config())
splitter.train(sess, dataset, 1000, 1e-3, 128, 0.7)
splitter.train(sess, dataset, 4000, 1e-4, 128, 0.7, ckpt='step-1000')

sess.close()
