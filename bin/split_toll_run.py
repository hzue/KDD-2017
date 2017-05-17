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
from dataframe import dataframe

if __name__ == '__main__':

  ####### setting #######
  prefix = 'result/05_06_split_model_part_train_v2_submit'
  submit_file_name = 'submit.csv'

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
  X, y, train_info_map, test_X, test_y, test_info_map, feature_list, df_train, df_test = dataset.generate_data( \
          'res/conclusion/training_20min_avg_travel_time.csv', \
          'res/conclusion/testing_data_{}_{}.csv'.format(test_start_date, test_end_date), \
          train_start_date, train_end_date, test_start_date, test_end_date)

  libsvr_selected_list = {
  #   '1': [1, 61, 29, 24, 54, 57, 56, 48],
  #   '2': [57, 24, 58, 59],
  #   '3': [3, 60, 67, 56, 57]
  # }
    '1': [1, 24, 33, 54, 32, 52, 30, 49, 55, 59],
    '2': [57, 56, 58],
    '3': [56, 13, 67]
  }

  ####### build model #######
  # CUR_ML: [ml.rf, ml.svr, RandomForestRegressor(n_estimators=400, max_features='sqrt'), SVR()]:

  CUR_ML = RandomForestRegressor(n_estimators=400, max_features='sqrt')

  groups, iterable_groups, obj_groups = dataframe.groupby(df_train, ['tollgate_id'])
  test_groups, test_iterable_groups, test_obj_groups = dataframe.groupby(df_test, ['tollgate_id'])

  np_X = np.asarray(X)
  np_y = np.asarray(y)
  np_map = np.asarray(train_info_map)

  np_test_X = np.asarray(test_X)
  np_test_map = np.asarray(test_info_map)

  result_y = []
  result_map = []
  feature_selection_result = {}


  for group, data in iterable_groups:
    tmp_train_X = np_X[data,:]
    tmp_train_y = np_y[data]

    test_ind_list = test_obj_groups[group]
    tmp_test_map = np_test_map[test_ind_list]
    tmp_test_X = np_test_X[test_ind_list,:]

    # print(str(group))
    # sklearn_rf_selected_list = feature_selection.forward_selection(tmp_train_X, tmp_train_y, tmp_test_X, tmp_test_map.tolist(), prefix, 'libsvr')
    # print(sklearn_rf_selected_list)
    # feature_list = np.asarray(feature_list)
    # print(feature_list[sklearn_rf_selected_list])
    # feature_selection_result[str(group)] = sklearn_rf_selected_list

    test_y = None

    CUR_ML = ml.svr
    selected_list = libsvr_selected_list

    # transform feature
    tmp_train_X = tmp_train_X[:,selected_list[group]]
    tmp_test_X = tmp_test_X[:,selected_list[group]]

    CUR_ML.fit(tmp_train_X.tolist(), tmp_train_y.tolist())
    test_y = CUR_ML.predict(tmp_test_X.tolist())

    if type(test_y) == np.ndarray: test_y = test_y.tolist()
    result_map += np_test_map[test_ind_list].tolist()
    result_y += test_y

  fh.write_submit_file(result_map, result_y, prefix, submit_file_name)
  mape = util.evaluation('{}/{}'.format(prefix, submit_file_name), 'res/conclusion/testing_ans.csv')
  pprint("mape: {}".format(str(mape)))
  # pprint(feature_selection_result)

