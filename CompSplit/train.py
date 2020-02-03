import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from CompSplit.model import *       # noqa


dataset = DataSet('../_Data/ml_6_spacing_train.csv', '../_Data/ml_6_spacing_test.csv')

args = Arguments('LSTM', [300, 300], True, True, 100)
splitter = BiLstmCrf('model1', args)

sess = tf.Session()
splitter.train(sess, dataset, 10000, 1e-3, 128, 0.7)

sess.close()