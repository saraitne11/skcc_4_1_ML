import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from MedicalAlert.model import *       # noqa


dataset = TrainData('../_Data/ml_10_medicalalert_train.csv')
test_dataset = TestData('../_Data/ml_10_medicalalert_test.csv', dataset)

args = Arguments('LSTM', [300, 300], True, True)
model = UniLSTM('model7', args)

sess = tf.Session(config=get_tf_config())
model.runs(sess, test_dataset, ckpt='step-2000', csv_name='result7.csv')

sess.close()
