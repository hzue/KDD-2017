import numpy as np
import util
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import file_handler as fh
from sklearn.svm import SVR
from predictor import ml
np.set_printoptions(precision=4)


#######################################################################################
@util.flow_logger
def forward_selection(X, y, val_X, info_map, prefix, selection_mode):
  # just conver from list to numpy array
  X = np.asarray(X); y = np.asarray(y); val_X = np.asarray(val_X)

  n_samples, n_features = X.shape

  # decide the wrapper
  reg = None
  if   selection_mode == 'rf'    : reg = RandomForestRegressor(n_estimators=400, max_features='sqrt')
  elif selection_mode == 'svr'   : reg = SVR()
  elif selection_mode == 'libsvr': reg = ml.svr
  elif selection_mode == 'Rrf'   : reg = ml.rf
  elif selection_mode == 'rvkde'   : reg = ml.rvkde

  F = [] # result feature set
  count = 1
  global_min_mape = np.inf

  while True:
    print("[{} forward selection] Start iteration: {}".format(selection_mode, count))
    min_mape = np.inf
    idx = None
    for i in range(n_features):
      if i not in F:
        F.append(i)
        reg.fit(X[:, F], y)
        y_predict = reg.predict(val_X[:, F])
        mape = cal_mape(y_predict, info_map, prefix)
        F.pop()
        if mape < min_mape:
          min_mape = mape
          idx = i
    if min_mape >= global_min_mape: return F
    global_min_mape = min_mape
    F.append(idx)
    print("[decision tree forward selection] iteration {} min mape: {}".format(count, min_mape))
    print("[decision tree forward selection] iteration {} F: {}".format(count, F))
    count += 1
  return F

def cal_mape(y_predict, info_map, prefix):
  fh.write_submit_file(info_map, y_predict, prefix, 'tmp.csv')
  return util.evaluation2('{}/tmp.csv'.format(prefix), 'res/conclusion/testing_ans.csv')

#######################################################################################

# def tree_based_importance(X, y):
#   regressor = RandomForestRegressor(n_estimators=800, max_features='sqrt')
#   model = regressor.fit(X, y)
#   score = model.feature_importances_
#   rank = sorted(range(len(score)), key=lambda k: score[k])
#   return rank

# def decision_tree_backward(X, y, val_X, info_map):
#
#     X = np.asarray(X)
#     y = np.asarray(y)
#     val_X = np.asarray(val_X)
#
#     n_samples, n_features = X.shape
#     reg = DecisionTreeRegressor()
#     reg = RandomForestRegressor(n_estimators=50, max_features='sqrt')
#     # reg = SVR()
#
#     F = [i for i in range(n_features)]
#     count = 1
#     global_min_mse = np.inf
#
#     while True:
#         print("[decision tree backward selection] Start iteration: {}".format(count))
#         min_mse = np.inf
#         idx = None
#         for i in range(n_features):
#             if i in F:
#                 F.remove(i)
#
#                 reg.fit(X[:, F], y)
#                 y_predict = reg.predict(val_X[:, F])
#                 # mse = accuracy_score(y[test], y_predict)
#                 mse = cal_mape(y_predict, info_map)
#
#                 F.append(i)
#                 if mse < min_mse:
#                     min_mse = mse
#                     idx = i
#         print("[decision tree backward selection] iteration {} min mse: {}".format(count, min_mse))
#         if min_mse >= global_min_mse: return F
#         global_min_mse = min_mse
#         F.remove(idx)
#         print("[decision tree backward selection] iteration {} F: {}".format(count, F))
#         count += 1
#     return F
