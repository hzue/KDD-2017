import numpy as np
import file_handler as fh
import util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.stats import hmean
import os

def transform(X_arr, selected_list):
  return_list = []
  for X in X_arr:
    X = np.asarray(X)
    X = X[:, selected_list]
    X = X.tolist()
    return_list.append(X)
  return return_list

def MAPE(y_true, y_pred):
  return np.mean(np.abs((y_true - y_pred) / y_true))

@util.flow_logger
def forward_selection(X, y, reg, val_X=None, info_map=None, prefix=None, pre_feed=[]):
  # just conver from list to numpy array
  X = np.asarray(X); y = np.asarray(y)
  n_samples, n_features = X.shape

  F = pre_feed # result feature set
  count = 1
  while not len(F) == n_features:
    print("[forward selection] Start iteration: {}".format(count))
    min_mape = np.inf
    idx = None
    for i in range(n_features):
      if i not in F:
        F.append(i)

        #######################
        mape = None
        if not val and not info_map and not prefix:
          mape = []
          kf = KFold(n_splits=5)
          count2 = 0
          for train, test in kf.split(X):
            count2 += 1
            if not count2 == 5: continue
            reg.fit(X[:, F][train], y[train])
            y_pred = reg.predict(X[:, F][test])
            mape.append(MAPE(y[test], y_pred))
          mape = np.mean(mape)
        else:
          reg.fit(X[:, F], y)
          y_pred = reg.predict(val_X[:, F])
          mape = cal_mape(y_pred, info_map, prefix)
        #######################

        F.pop()

        if mape < min_mape:
          min_mape = mape
          idx = i

    F.append(idx)
    print("[forward selection] iteration {} min mape: {}".format(count, min_mape))
    print("[forward selection] iteration {} F: {}".format(count, F))
    count += 1
  return F

def cal_mape(y_predict, info_map, prefix):
  fh.write_submit_file(info_map, y_predict, prefix, 'tmp.csv')
  return util.evaluation('{}/tmp.csv'.format(prefix), 'res/conclusion/testing_ans_2016-10-18_2016-10-24.csv')
