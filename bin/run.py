from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.preprocessing import RobustScaler
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
  arg = {'prefix': 'default'}
  for i in range(1, len(sys.argv) - 1, 2):
    arg[sys.argv[i].replace("-", "")] = sys.argv[i+1]

  ####### setting #######
  submit_file_name = 'submit.csv'
  arg['prefix'] = 'result/' + arg['prefix']
  ml.prefix = arg['prefix']

  if arg['mode'] == 'validation':
    train_start_date = '2016-08-01'
    train_end_date   = '2016-10-10'
    test_start_date  = '2016-10-11'
    test_end_date    = '2016-10-17'

  elif arg['mode'] == 'submit':
    train_start_date = '2016-07-19'
    train_end_date   = '2016-10-17'
    test_start_date  = '2016-10-18'
    test_end_date    = '2016-10-24'

  else: raise ValueError('No this mode!')

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
  ##################################################
  # libsvr [58, 42, 59, 49, 52, 56, 29, 55, 50]    #
  # KNN [60, 64, 63, 31, 35, 66, 25]               #
  # rf [61, 40, 30, 31, 0, 1, 42, 36]              #
  ##################################################

  selected_list = feature_selection.forward_selection(X, y, test_X, test_info_map, arg['prefix'], ml.svr())
  # selected_list = [58, 42, 59, 49, 52, 56, 29, 55, 50]
  X, test_X = feature_selection.transform([X, test_X], selected_list)

  ####### build model #######
  for CUR_ML in [ml.svr(), KNeighborsRegressor(n_neighbors=20)]:
    CUR_ML.fit(X, y)
    test_y = CUR_ML.predict(test_X)

    fh.write_submit_file(test_info_map, test_y, arg['prefix'], submit_file_name)
    if arg['mode'] == 'validation':
      mape = util.evaluation('{}/{}'.format(arg['prefix'], submit_file_name), 'res/conclusion/testing_ans.csv')
      print("mape: {}".format(str(mape)))

  gc.collect()

