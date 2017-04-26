import pandas as pd
import numpy as np
from datetime import datetime
from subprocess import check_output
import util
from pprint import pprint
from predictor import ml
import scipy
import itertools
import os
import lambda_func
import file_handler as fh
from data import dataset


if __name__ == '__main__':

  ####### setting #######
  CUR_ML = ml.rf

  prefix = 'result/2016_04_25'
  submit_file_name = 'submit.csv'

  train_start_date = '2016-07-19'
  train_end_date = '2016-10-10'
  test_start_date = '2016-10-11'
  test_end_date = '2016-10-17'

  ####### check file path #######
  if not os.path.exists(prefix): os.mkdir(prefix)

  ####### prepare dataframe #######
  df_train = fh.grab_data_within_range('res/conclusion/training_20min_avg_travel_time.csv', \
          train_start_date, train_end_date)
  df_test_given = fh.grab_data_within_range('res/conclusion/testing_data_{}_{}.csv'.format(test_start_date, test_end_date), \
          test_start_date, test_end_date)
  df_test = util.generate_testing_dataframe(test_start_date, test_end_date, df_train)
  df = pd.concat([df_train, df_test_given, df_test], ignore_index=True)

  ####### generate X, y, test_X, test_y #######
  X, y, test_X, test_y, df_train, df_test = dataset.generate_data(df, train_start_date='2016-07-19', train_end_date='2016-10-10', \
                           test_start_date='2016-10-11', test_end_date='2016-10-17')

  ####### build model #######
  CUR_ML.train(X, y, prefix)
  CUR_ML.predict(test_X, test_y, prefix)

  fh.generate_submit_file(df_test, prefix, submit_file_name, CUR_ML.read_result)
  mape = util.evaluation('{}/{}'.format(prefix, submit_file_name), 'res/conclusion/testing_ans.csv')
  pprint("mape: " + str(mape))


