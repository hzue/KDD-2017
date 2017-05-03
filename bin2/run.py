import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from subprocess import check_output
import util
import math
from pprint import pprint
from predictor import ml
import scipy
import itertools
import os
import file_handler as fh
import dataset
import feature_selection
import sys

if __name__ == '__main__':

  ####### setting #######
  prefix = 'result/0503_select_from_all'
  submit_file_name = 'submit.csv'
  selection_mode = 'rf'

  ml.prefix = prefix

  train_start_date = '2016-07-19'
  train_end_date = '2016-10-10'
  test_start_date = '2016-10-11'
  test_end_date = '2016-10-17'

  # train_start_date = '2016-07-19'
  # train_end_date = '2016-10-17'
  # test_start_date = '2016-10-18'
  # test_end_date = '2016-10-24'

  ####### check file path #######
  if not os.path.exists(prefix): os.mkdir(prefix)

  ####### generate X, y, test_X, test_y #######
  X, y, train_info_map, test_X, test_y, test_info_map, feature_list = dataset.generate_data( \
          'res/conclusion/training_20min_avg_travel_time.csv', \
          'res/conclusion/testing_data_{}_{}.csv'.format(test_start_date, test_end_date), \
          train_start_date, train_end_date, test_start_date, test_end_date)

  ####### feature selection #######
  # selected_list = feature_selection.forward_selection(X, y, test_X, test_info_map, prefix, selection_mode)
  # print(selected_list)
  # feature_list = np.asarray(feature_list)
  # print(feature_list[selected_list])
  # exit()

  ####### build model #######
  # for CUR_ML in [ml.rf, ml.svr, RandomForestRegressor(n_estimators=400, max_features='sqrt'), SVR()]:
  CUR_ML = SVR()
  # CUR_ML = GridSearchCV(SVR(), cv=5, \
  #       param_grid={
  #         "C": np.logspace(-8, 8, num=6, base=2), \
  #         "gamma": np.logspace(-8, 8, num=6, base=2) \
  #       }
  # )
  CUR_ML.fit(X, y)
  test_y = CUR_ML.predict(test_X)

  fh.write_submit_file(test_info_map, test_y, prefix, submit_file_name)
  mape = util.evaluation('{}/{}'.format(prefix, submit_file_name), 'res/conclusion/testing_ans.csv')
  pprint("mape: {}".format(str(mape)))

