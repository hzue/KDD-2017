from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.neural_network import MLPRegressor

from datetime import datetime
from subprocess import check_output
from predictor import ml
from pprint import pprint
import numpy as np
import os
import file_handler as fh
import dataset
import feature_selection
import sys
import util
import gc
import sys
np.seterr(all='ignore')

if __name__ == '__main__':

  ####### simple parse argv #######
  arg = {}
  for i in range(1, len(sys.argv) - 1, 2):
    arg[sys.argv[i].replace("-", "")] = sys.argv[i+1]

  ####### setting #######
  submit_file_name = 'submit.csv'
  if not 'prefix' in arg: arg['prefix'] = 'default'
  arg['prefix'] = 'result/' + arg['prefix']
  ml.prefix = arg['prefix']

  if arg['mode'] == 'validation':
    train_start_date = '2016-07-19'
    train_end_date   = '2016-10-10'
    test_start_date  = '2016-10-11'
    test_end_date    = '2016-10-17'

  elif arg['mode'] == 'submit':
    train_start_date = '2016-07-19'
    train_end_date   = '2016-10-17'
    test_start_date  = '2016-10-18'
    test_end_date    = '2016-10-24'

  ####### check file path #######
  if not os.path.exists(arg['prefix']): os.mkdir(arg['prefix'])

  ####### generate X, y, test_X, test_y #######
  X, y, train_info_map, test_X, test_y, test_info_map, df_train, df_test = dataset.generate_data( \
          'res/conclusion/training_20min_avg_travel_time.csv', \
          'res/conclusion/testing_data_{}_{}.csv'.format(test_start_date, test_end_date), \
          train_start_date, train_end_date, test_start_date, test_end_date)

  ####### scale feature #######
  scale = Scaler(feature_range=(-1,1))
  X = scale.fit_transform(X)
  test_X = scale.transform(test_X)

  ####### feature selection #######
  libsvr_selected_list = [58, 42, 59, 49, 52, 56, 29, 55, 50]
  knn_selected_list = [58, 64, 63, 31, 35, 66, 25]
  rf_selected_list = [40, 54, 24, 38, 37, 32, 15, 9]

  ####### build model #######
  # svr = ml.svr()
  # tmpX, test_tmpX = feature_selection.transform([X, test_X], libsvr_selected_list)
  # svr.fit(tmpX, y)
  # svr_pred_train_y = svr.predict(tmpX)
  # svr_pred_test_y = svr.predict(test_tmpX)

  knn = KNeighborsRegressor(n_neighbors=40)
  tmpX, test_tmpX = feature_selection.transform([X, test_X], knn_selected_list)
  knn.fit(tmpX, y)
  knn_pred_train_y = knn.predict(tmpX)
  knn_pred_test_y = knn.predict(test_tmpX)

  # rf = RandomForestRegressor(n_estimators=400, max_features=0.3)
  # tmpX, test_tmpX = feature_selection.transform([X, test_X], rf_selected_list)
  # rf.fit(tmpX, y)
  # rf_pred_train_y = rf.predict(tmpX)
  # rf_pred_test_y = rf.predict(test_tmpX)

  # agg_train_X = [[svr_pred_train_y[i], knn_pred_train_y[i]] for i in range(0, len(svr_pred_train_y))]
  # agg_test_X = [[svr_pred_test_y[i], knn_pred_test_y[i]] for i in range(0, len(svr_pred_test_y))]

  X, test_X = feature_selection.transform([X, test_X], libsvr_selected_list)
  for i in range(0, len(X)):
    X[i].append(knn_pred_train_y[i])
  for i in range(0, len(test_X)):
    test_X[i].append(knn_pred_test_y[i])

  agg_svr = ml.svr()
  agg_svr.fit(X, y)
  test_y = agg_svr.predict(test_X)

  fh.write_submit_file(test_info_map, test_y, arg['prefix'], submit_file_name)
  if arg['mode'] == 'validation':
    mape = util.evaluation('{}/{}'.format(arg['prefix'], submit_file_name), 'res/conclusion/testing_ans.csv')
    print("mape: {}".format(str(mape)))

  gc.collect()

