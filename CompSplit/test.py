import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from CompSplit.model import *       # noqa


dataset = DataSet(train_path='../_Data/ml_6_spacing_train_split.csv',
                  test_path='../_Data/ml_6_spacing_test.csv',
                  val_path=None)


args = Arguments('LSTM', [300, 300, 300], True, True, 100)
splitter = BiLstmCrf('model3', args)

sess = tf.Session(config=get_tf_config())
splitter.runs(sess, dataset, 'step-5000', 128, csv_name='result3.csv')

sess.close()
