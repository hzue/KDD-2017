import numpy as np
import file_handler as fh
import util
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from predictor import ml
from sklearn.model_selection import KFold
from pprint import pprint
from scipy.stats import hmean
np.set_printoptions(precision=4)


def transform(X_arr, selected_list):
  return_list = []
  for X in X_arr:
    X = np.asarray(X)
    X = X[:, selected_list]
    X = X.tolist()
    return_list.append(X)
  return return_list

@util.flow_logger
def forward_selection(X, y, val_X, info_map, prefix, reg):
  # just conver from list to numpy array
  X = np.asarray(X); y = np.asarray(y); val_X = np.asarray(val_X)
  n_samples, n_features = X.shape

  F = [] # result feature set
  count = 1
  global_min_mape = np.inf
  while True:
    print("[forward selection] Start iteration: {}".format(count))
    min_mape = np.inf
    idx = None
    for i in range(n_features):
      if i not in F:
        F.append(i)

        # cross-validation
        mape = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
          reg.fit(X[:, F][train], y[train])
          y_pred = reg.predict(X[:, F][test])
          mape.append(MAPE(y[test], y_pred))
        mape = np.mean(mape)

        reg.fit(X[:, F], y)
        y_pred = reg.predict(val_X[:, F])
        mape = cal_mape(y_pred, info_map, prefix)

        F.pop()
        if mape < min_mape:
          min_mape = mape
          idx = i
    if min_mape >= global_min_mape:
      print("[forward selection final] iteration {} F: {}".format(count, F))
      return F
    global_min_mape = min_mape
    F.append(idx)
    print("[forward selection] iteration {} min mape: {}".format(count, min_mape))
    print("[forward selection] iteration {} F: {}".format(count, F))
    count += 1
  return F

def MAPE(y_true, y_pred):
  return np.mean(np.abs((y_true - y_pred) / y_true))

def cal_mape(y_predict, info_map, prefix):
  fh.write_submit_file(info_map, y_predict, prefix, 'tmp.csv')
  return util.evaluation2('{}/tmp.csv'.format(prefix), 'res/conclusion/testing_ans.csv')
