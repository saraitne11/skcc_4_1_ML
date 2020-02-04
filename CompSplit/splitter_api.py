import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import splitter                     # noqa
from CompSplit.utils import *       # noqa


dataset = DataSet('../_Data/ml_6_spacing_train.csv', '../_Data/ml_6_spacing_test.csv')


for a in dataset.test_compound:
    print(splitter.split(a[0]))
